import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam3Model, Sam3Processor


class Sam3Grounder(nn.Module):
    """
    Frozen SAM 3 spatial grounding & conditioning bridge.
    Converts (img, query) ->
      - inst_mask: (B, 1, H, W) confidence-weighted instance probability map in [0, 1]
      - sem_mask:  (B, 1, H, W) dense semantic segmentation probability map in [0, 1]
      - sam_vec:   (B, 16) compact geometric & confidence summary vector for FiLM
    """

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

    @staticmethod
    def _recover_rgb_01(img: torch.Tensor) -> torch.Tensor:
        """
        GraspAnywhereDataset normalizes RGB via: img = img / 255.0 - img.mean().
        Invert per-image range back to [0, 1] on GPU before passing to Sam3Processor.
        """
        b_min = img.amin(dim=(1, 2, 3), keepdim=True)
        b_max = img.amax(dim=(1, 2, 3), keepdim=True)
        return ((img - b_min) / (b_max - b_min + 1e-6)).clamp(0.0, 1.0)

    @staticmethod
    def _extract_geometric_summary(
        inst_mask: torch.Tensor,
        sem_mask: torch.Tensor,
        query_probs: torch.Tensor,
        presence_prob: torch.Tensor,
        pred_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds a version-safe (B, 16) global conditioning vector encoding:
          [0]    presence_prob
          [1:4]  top-3 query confidence scores
          [4:8]  top-1 predicted bounding box (normalized)
          [8:12] mask center-of-mass (cx, cy) and spatial spread (sx, sy)
          [12:16] mean & max activation of inst_mask and sem_mask
        """
        B, _, H, W = inst_mask.shape
        device = inst_mask.device

        # Top-3 query confidences & top-1 box
        topk_probs, topk_idx = torch.topk(query_probs, k=min(3, query_probs.shape[1]), dim=1)
        if topk_probs.shape[1] < 3:
            topk_probs = F.pad(topk_probs, (0, 3 - topk_probs.shape[1]))
        top1_idx = topk_idx[:, 0].view(B, 1, 1).expand(B, 1, 4)
        top1_box = torch.gather(pred_boxes.float(), dim=1, index=top1_idx).squeeze(1)

        # Spatial center of mass & spread of inst_mask
        grid_y = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1)
        grid_x = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W)
        mass = inst_mask.sum(dim=(2, 3), keepdim=True) + 1e-6
        cx = (inst_mask * grid_x).sum(dim=(2, 3), keepdim=True) / mass
        cy = (inst_mask * grid_y).sum(dim=(2, 3), keepdim=True) / mass
        sx = torch.sqrt(((inst_mask * (grid_x - cx) ** 2).sum(dim=(2, 3), keepdim=True) / mass) + 1e-6)
        sy = torch.sqrt(((inst_mask * (grid_y - cy) ** 2).sum(dim=(2, 3), keepdim=True) / mass) + 1e-6)
        spatial_stats = torch.cat([cx, cy, sx, sy], dim=1).view(B, 4)

        # Mask activation statistics
        inst_mean = inst_mask.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        inst_max = inst_mask.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        sem_mean = sem_mask.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        sem_max = sem_mask.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        return torch.cat(
            [presence_prob, topk_probs, top1_box, spatial_stats, inst_mean, inst_max, sem_mean, sem_max],
            dim=1,
        )  # (B, 16)

    @torch.no_grad()
    def forward(self, img: torch.Tensor, queries):
        """
        :param img: (B, 3, H, W) float tensor from dataloader
        :param queries: list/tuple of B strings
        :return: inst_mask (B, 1, H, W), sem_mask (B, 1, H, W), sam_vec (B, 16)
        """
        B, _, H, W = img.shape
        text_list = [str(q) for q in queries]
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

        # Query & presence probabilities: (B, 200) and (B, 1)
        presence_prob = torch.sigmoid(outputs.presence_logits.float())  # (B, 1)
        query_probs = torch.sigmoid(outputs.pred_logits.float()) * presence_prob  # (B, 200)

        # Confidence-weighted instance mask: max over 200 queries -> (B, 1, 288, 288)
        mask_probs = torch.sigmoid(outputs.pred_masks.float())  # (B, 200, 288, 288)
        weighted_masks = mask_probs * query_probs[:, :, None, None]
        inst_mask_288 = weighted_masks.max(dim=1, keepdim=True).values

        # Dense semantic segmentation map: (B, 1, 288, 288)
        sem_mask_288 = torch.sigmoid(outputs.semantic_seg.float())

        # Resize to match grasp map resolution (224 x 224)
        inst_mask = F.interpolate(inst_mask_288, size=(H, W), mode="bilinear", align_corners=False)
        sem_mask = F.interpolate(sem_mask_288, size=(H, W), mode="bilinear", align_corners=False)

        sam_vec = self._extract_geometric_summary(
            inst_mask=inst_mask,
            sem_mask=sem_mask,
            query_probs=query_probs,
            presence_prob=presence_prob,
            pred_boxes=outputs.pred_boxes,
        )
        return inst_mask, sem_mask, sam_vec