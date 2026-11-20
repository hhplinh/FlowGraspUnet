import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam3Model, Sam3Processor


class Sam3Grounder(nn.Module):
    """
    Frozen SAM 3 spatial grounding & conditioning bridge.
    Converts (img, query) ->
      - inst_mask: (B, 1, H, W) top-3 query-weighted & presence-normalized part mask in [0, 1]
      - sem_mask:  (B, 1, H, W) dense semantic segmentation probability map in [0, 1]
      - sam_vec:   (B, 272) [16-D geometric summary || 256-D L2-normalized SAM3 text embedding]
    """

    _LEAD_VERBS = re.compile(
        r"^(take\s+hold\s+of|pick\s+up|reach\s+for|grasp|grip|hold|lift|catch|fetch|grab|seize|clasp|clutch)\s+",
        flags=re.IGNORECASE,
    )
    _PART_PREP = re.compile(
        r"\s+(by|on|at|from|near|along)\s+(its|the|a|an)\s+",
        flags=re.IGNORECASE,
    )

    def __init__(self, model_path="/workspace/models/sam3", device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
        self.model = (
            Sam3Model.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Detect text embedding width dynamically (256 in SAM 3)
        self.text_dim = getattr(
            getattr(self.model.config, "text_config", None), "hidden_size", 256
        )
        self.sam_vec_dim = 16 + self.text_dim

    @classmethod
    def _simplify_prompt(cls, text: str) -> str:
        raw = str(text).strip().rstrip(".")
        simplified = cls._LEAD_VERBS.sub("", raw)
        simplified = cls._PART_PREP.sub(" ", simplified).strip()
        return simplified if simplified else raw

    @staticmethod
    def _recover_rgb_01(img: torch.Tensor) -> torch.Tensor:
        b_min = img.amin(dim=(1, 2, 3), keepdim=True)
        b_max = img.amax(dim=(1, 2, 3), keepdim=True)
        return ((img - b_min) / (b_max - b_min + 1e-6)).clamp(0.0, 1.0)

    def _extract_text_embedding(self, inputs: dict, batch_size: int) -> torch.Tensor:
        """Extracts pooled (B, text_dim) float32 L2-normalized text embedding from frozen SAM 3."""
        text_kwargs = {
            k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")
        }
        if not text_kwargs:
            return torch.zeros((batch_size, self.text_dim), device=self.device, dtype=torch.float32)

        if hasattr(self.model, "get_text_features"):
            feats = self.model.get_text_features(**text_kwargs)
        elif hasattr(self.model, "text_model"):
            feats = self.model.text_model(**text_kwargs)
        else:
            return torch.zeros((batch_size, self.text_dim), device=self.device, dtype=torch.float32)

        if not torch.is_tensor(feats):
            feats = feats.pooler_output if getattr(feats, "pooler_output", None) is not None else feats.last_hidden_state

        if feats.ndim == 3:
            if "attention_mask" in text_kwargs:
                mask = text_kwargs["attention_mask"].unsqueeze(-1).to(feats.dtype)
                feats = (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                feats = feats.mean(dim=1)

        feats = feats.float()
        if feats.shape[-1] != self.text_dim:
            feats = F.adaptive_avg_pool1d(feats.unsqueeze(1), self.text_dim).squeeze(1)
        return F.normalize(feats, p=2, dim=-1)

    @staticmethod
    def _extract_geometric_summary(
        inst_mask: torch.Tensor,
        sem_mask: torch.Tensor,
        query_probs: torch.Tensor,
        presence_prob: torch.Tensor,
        pred_boxes: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = inst_mask.shape
        device = inst_mask.device

        topk_probs, topk_idx = torch.topk(query_probs, k=min(3, query_probs.shape[1]), dim=1)
        if topk_probs.shape[1] < 3:
            topk_probs = F.pad(topk_probs, (0, 3 - topk_probs.shape[1]))
        top1_idx = topk_idx[:, 0].view(B, 1, 1).expand(B, 1, 4)
        top1_box = torch.gather(pred_boxes.float(), dim=1, index=top1_idx).squeeze(1)

        grid_y = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1)
        grid_x = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W)
        mass = inst_mask.sum(dim=(2, 3), keepdim=True) + 1e-6
        cx = (inst_mask * grid_x).sum(dim=(2, 3), keepdim=True) / mass
        cy = (inst_mask * grid_y).sum(dim=(2, 3), keepdim=True) / mass
        sx = torch.sqrt(((inst_mask * (grid_x - cx) ** 2).sum(dim=(2, 3), keepdim=True) / mass) + 1e-6)
        sy = torch.sqrt(((inst_mask * (grid_y - cy) ** 2).sum(dim=(2, 3), keepdim=True) / mass) + 1e-6)
        spatial_stats = torch.cat([cx, cy, sx, sy], dim=1).view(B, 4)

        inst_mean = inst_mask.mean(dim=(1, 2, 3)).unsqueeze(1)
        inst_max = inst_mask.amax(dim=(1, 2, 3)).unsqueeze(1)
        sem_mean = sem_mask.mean(dim=(1, 2, 3)).unsqueeze(1)
        sem_max = sem_mask.amax(dim=(1, 2, 3)).unsqueeze(1)

        return torch.cat(
            [presence_prob, topk_probs, top1_box, spatial_stats, inst_mean, inst_max, sem_mean, sem_max],
            dim=1,
        )

    @torch.no_grad()
    def forward(self, img: torch.Tensor, queries):
        B, _, H, W = img.shape
        text_list = [self._simplify_prompt(q) for q in queries]
        img_01 = self._recover_rgb_01(img)

        inputs = self.processor(
            images=img_01,
            text=text_list,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)

        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)

        outputs = self.model(**inputs)
        text_emb = self._extract_text_embedding(inputs, B)  # (B, 256)

        presence_prob = torch.sigmoid(outputs.presence_logits.float())  # (B, 1)
        query_probs = torch.sigmoid(outputs.pred_logits.float())      # (B, 200)
        mask_probs = torch.sigmoid(outputs.pred_masks.float())        # (B, 200, 288, 288)

        # Top-3 softmax-weighted query masks + conditional peak normalization
        topk_probs, topk_idx = torch.topk(query_probs, k=3, dim=1)
        topk_masks = torch.gather(
            mask_probs, 1, topk_idx[:, :, None, None].expand(-1, -1, 288, 288)
        )
        top3_weights = torch.softmax(topk_probs * 5.0, dim=1)[:, :, None, None]
        inst_mask_288 = (topk_masks * top3_weights).sum(dim=1, keepdim=True)

        peak = inst_mask_288.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        should_norm = (presence_prob[:, :, None, None] > 0.3) & (peak > 0.05)
        inst_mask_288 = torch.where(should_norm, inst_mask_288 / peak, inst_mask_288)

        sem_mask_288 = torch.sigmoid(outputs.semantic_seg.float())

        inst_mask = F.interpolate(inst_mask_288, size=(H, W), mode="bilinear", align_corners=False)
        sem_mask = F.interpolate(sem_mask_288, size=(H, W), mode="bilinear", align_corners=False)

        geom_vec = self._extract_geometric_summary(
            inst_mask=inst_mask,
            sem_mask=sem_mask,
            query_probs=query_probs,
            presence_prob=presence_prob,
            pred_boxes=outputs.pred_boxes,
        )
        sam_vec = torch.cat([geom_vec, text_emb], dim=1)  # (B, 272)
        return inst_mask, sem_mask, sam_vec
