from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RectifiedGraspFlow(nn.Module):
    """
    Low-Variance Optimal Transport Conditional Flow Matching (OT-CFM) wrapper
    coupling Sam3Grounder (frozen) with FlowGraspUNet (trainable).
    """

    def __init__(
        self,
        sam_grounder: nn.Module,
        flow_net: nn.Module,
        prior_std: float = 0.15,
        fg_loss_weight: float = 15.0,
        mask_loss_weight: float = 3.0,
        dice_loss_weight: float = 0.5,
        bg_suppression_weight: float = 1.0,
        fg_pose_weight: float = 0.5,
        angle_norm_weight: float = 0.05,
    ):
        super().__init__()
        self.sam_grounder = sam_grounder
        self.flow_net = flow_net
        self.prior_std = prior_std
        self.fg_loss_weight = fg_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bg_suppression_weight = bg_suppression_weight
        self.fg_pose_weight = fg_pose_weight
        self.angle_norm_weight = angle_norm_weight

    @staticmethod
    def pack_targets(yc: List[torch.Tensor]) -> torch.Tensor:
        """Packs [pos_gt, cos_gt, sin_gt, width_gt] -> (B, 4, H, W)."""
        maps = [m.unsqueeze(1) if m.ndim == 3 else m for m in yc]
        return torch.cat(maps, dim=1)

    @staticmethod
    def _soft_dice_loss(pred_pos: torch.Tensor, target_pos: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        p = pred_pos.clamp(0.0, 1.0)
        inter = (p * target_pos).sum(dim=(1, 2, 3))
        denom = (p * p).sum(dim=(1, 2, 3)) + (target_pos * target_pos).sum(dim=(1, 2, 3)) + eps
        return (1.0 - (2.0 * inter + eps) / denom).mean()

    def compute_loss(
        self,
        img: torch.Tensor,
        yc: List[torch.Tensor],
        queries,
        cached_sam: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x1 = self.pack_targets(yc)  # (B, 4, H, W)
        pos_gt = x1[:, 0:1]
        B, _, H, W = x1.shape
        device = x1.device

        if cached_sam is not None:
            inst_mask, sem_mask, sam_vec = cached_sam
        else:
            inst_mask, sem_mask, sam_vec = self.sam_grounder(img, queries)

        # Low-variance Gaussian prior x0 ~ N(0, prior_std^2 I) prevents background noise spikes
        x0 = self.prior_std * torch.randn((B, 4, H, W), device=device, dtype=x1.dtype)
        t = torch.rand((B,), device=device)
        t_view = t.view(B, 1, 1, 1)

        xt = (1.0 - t_view) * x0 + t_view * x1
        target_v = x1 - x0

        pred_v = self.flow_net(xt, t, img, inst_mask, sem_mask, sam_vec)

        spatial_weight = (
            1.0
            + self.fg_loss_weight * pos_gt
            + self.mask_loss_weight * torch.maximum(inst_mask, sem_mask)
        )

        sq_err = (pred_v - target_v) ** 2
        loss_pos = (spatial_weight * sq_err[:, 0:1]).mean()
        loss_ang = (spatial_weight * sq_err[:, 1:3]).mean()
        loss_wid = (spatial_weight * sq_err[:, 3:4]).mean()
        loss_flow = loss_pos + loss_ang + loss_wid

        # Implied clean endpoint x1_hat = xt + (1 - t) * pred_v
        x1_hat = xt + (1.0 - t_view) * pred_v
        loss_dice = self._soft_dice_loss(x1_hat[:, 0:1], pos_gt)

        # Suppress false positive peaks on background pixels
        bg_mask = pos_gt <= 0.05
        loss_bg = (x1_hat[:, 0:1][bg_mask] ** 2).mean() if bg_mask.any() else torch.tensor(0.0, device=device)

        target_region = pos_gt > 0.1
        if target_region.any():
            fg_mask = target_region.expand_as(x1[:, 1:4])
            loss_fg_pose = F.smooth_l1_loss(x1_hat[:, 1:4][fg_mask], x1[:, 1:4][fg_mask])
            cos_hat, sin_hat = x1_hat[:, 1:2], x1_hat[:, 2:3]
            loss_norm = ((cos_hat[target_region] ** 2 + sin_hat[target_region] ** 2 - 1.0) ** 2).mean()
        else:
            loss_fg_pose = torch.tensor(0.0, device=device)
            loss_norm = torch.tensor(0.0, device=device)

        total_loss = (
            loss_flow
            + self.dice_loss_weight * loss_dice
            + self.bg_suppression_weight * loss_bg
            + self.fg_pose_weight * loss_fg_pose
            + self.angle_norm_weight * loss_norm
        )

        return {
            "loss": total_loss,
            "losses": {
                "flow_pos": loss_pos.detach(),
                "flow_ang": loss_ang.detach(),
                "flow_wid": loss_wid.detach(),
                "pos_dice": loss_dice.detach(),
                "bg_supp": loss_bg.detach(),
                "fg_pose": loss_fg_pose.detach(),
                "ang_norm": loss_norm.detach(),
            },
        }

    @torch.no_grad()
    def sample_ode(
        self,
        img: torch.Tensor,
        queries,
        num_steps: int = 8,
        solver: str = "euler",
        seed: Optional[int] = 42,
        return_masks: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Zero-leakage ODE integration from t=0 to t=1 using ONLY (img, queries).
        """
        B, _, H, W = img.shape
        device = img.device

        inst_mask, sem_mask, sam_vec = self.sam_grounder(img, queries)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)

        x = self.prior_std * torch.randn((B, 4, H, W), device=device, dtype=img.dtype, generator=gen)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_curr = torch.full((B,), step * dt, device=device)
            if solver == "euler":
                v = self.flow_net(x, t_curr, img, inst_mask, sem_mask, sam_vec)
                x = x + v * dt
            elif solver == "midpoint":
                v1 = self.flow_net(x, t_curr, img, inst_mask, sem_mask, sam_vec)
                x_mid = x + 0.5 * dt * v1
                t_mid = torch.full((B,), (step + 0.5) * dt, device=device)
                v2 = self.flow_net(x_mid, t_mid, img, inst_mask, sem_mask, sam_vec)
                x = x + v2 * dt
            else:
                raise ValueError(f"Unknown solver: {solver}")

        pred = {
            "pos": x[:, 0:1].clamp(0.0, 1.0),
            "cos": x[:, 1:2].clamp(-1.0, 1.0),
            "sin": x[:, 2:3].clamp(-1.0, 1.0),
            "width": x[:, 3:4].clamp(0.0, 1.0),
        }
        if return_masks:
            return pred, (inst_mask, sem_mask)
        return pred, None

    @torch.no_grad()
    def predict(self, img: torch.Tensor, queries, num_steps: int = 8) -> Dict[str, torch.Tensor]:
        """Drop-in replacement for GraspGenerator inference."""
        pred, _ = self.sample_ode(img, queries, num_steps=num_steps, solver="euler", seed=42)
        return pred
