from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class RectifiedGraspFlow(nn.Module):
    """
    Mask-Prior Optimal Transport Conditional Flow Matching (OT-CFM) wrapper
    coupling Sam3Grounder (frozen) with FlowGraspUNet (trainable).
    """

    def __init__(
        self,
        sam_grounder: nn.Module,
        flow_net: nn.Module,
        prior_noise_std: float = 0.25,
        mask_loss_weight: float = 4.0,
        angle_norm_weight: float = 0.05,
    ):
        super().__init__()
        self.sam_grounder = sam_grounder
        self.flow_net = flow_net
        self.prior_noise_std = prior_noise_std
        self.mask_loss_weight = mask_loss_weight
        self.angle_norm_weight = angle_norm_weight

    def _sample_prior(
        self,
        inst_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Constructs mask-biased initial state x_0 of shape (B, 4, H, W):
          channel 0 (pos): inst_mask + sigma * eps
          channels 1..3 (cos, sin, width): sigma * eps
        """
        B, _, H, W = inst_mask.shape
        eps = torch.randn(
            (B, 4, H, W),
            device=inst_mask.device,
            dtype=inst_mask.dtype,
            generator=generator,
        )
        prior_mean = torch.zeros_like(eps)
        prior_mean[:, 0:1] = inst_mask
        return prior_mean + self.prior_noise_std * eps

    @staticmethod
    def pack_targets(yc: List[torch.Tensor]) -> torch.Tensor:
        """Packs [pos_gt, cos_gt, sin_gt, width_gt] -> (B, 4, H, W)."""
        maps = [m.unsqueeze(1) if m.ndim == 3 else m for m in yc]
        return torch.cat(maps, dim=1)

    def compute_loss(
        self,
        img: torch.Tensor,
        yc: List[torch.Tensor],
        queries,
    ) -> Dict[str, torch.Tensor]:
        x1 = self.pack_targets(yc)  # (B, 4, H, W)
        B = x1.shape[0]
        device = x1.device

        # Stage 1: Frozen SAM 3 spatial grounding
        inst_mask, sem_mask, sam_vec = self.sam_grounder(img, queries)

        # Stage 2: Sample prior x0 and continuous time t in [0, 1]
        x0 = self._sample_prior(inst_mask)
        t = torch.rand((B,), device=device)
        t_view = t.view(B, 1, 1, 1)

        # Straight OT interpolation and target velocity
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_v = x1 - x0

        # Stage 3: Predict velocity
        pred_v = self.flow_net(xt, t, img, inst_mask, sem_mask, sam_vec)

        # Spatial attention weight: emphasizes target part while suppressing background
        spatial_weight = 1.0 + self.mask_loss_weight * torch.maximum(inst_mask, sem_mask)

        sq_err = (pred_v - target_v) ** 2
        loss_pos = (spatial_weight * sq_err[:, 0:1]).mean()
        loss_ang = (spatial_weight * sq_err[:, 1:3]).mean()
        loss_wid = (spatial_weight * sq_err[:, 3:4]).mean()
        loss_flow = loss_pos + loss_ang + loss_wid

        # Unit-circle regularization on implied endpoint x1_hat = xt + (1 - t) * pred_v
        x1_hat = xt + (1.0 - t_view) * pred_v
        cos_hat, sin_hat = x1_hat[:, 1:2], x1_hat[:, 2:3]
        target_region = (yc[0].unsqueeze(1) if yc[0].ndim == 3 else yc[0]) > 0.1
        if target_region.any():
            loss_norm = ((cos_hat[target_region] ** 2 + sin_hat[target_region] ** 2 - 1.0) ** 2).mean()
        else:
            loss_norm = torch.tensor(0.0, device=device)

        total_loss = loss_flow + self.angle_norm_weight * loss_norm

        return {
            "loss": total_loss,
            "losses": {
                "flow_pos": loss_pos.detach(),
                "flow_ang": loss_ang.detach(),
                "flow_wid": loss_wid.detach(),
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
        B = img.shape[0]
        device = img.device

        inst_mask, sem_mask, sam_vec = self.sam_grounder(img, queries)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)

        x = self._sample_prior(inst_mask, generator=gen)
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