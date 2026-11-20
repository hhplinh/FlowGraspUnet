import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch * 2),
        )

        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight spatial self-attention applied only at the 28x28 bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + attn_out.transpose(1, 2).view(B, C, H, W)


class FlowGraspUNet(nn.Module):
    """
    4-Channel Conditional Flow Matching UNet predicting velocity v_theta(x_t, t, cond)
    for joint grasp maps: [pos, cos(2*theta), sin(2*theta), width].
    """

    def __init__(self, base_ch: int = 64, cond_dim: int = 256, sam_vec_dim: int = 16):
        super().__init__()
        # Input channels: x_t (4) + img (3) + inst_mask (1) + sem_mask (1) + masked_img (3) = 12
        in_channels = 12

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.sam_mlp = nn.Sequential(
            nn.Linear(sam_vec_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # Encoder: 224 -> 112 -> 56 -> 28
        self.enc1 = FiLMResBlock(base_ch, base_ch, cond_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = FiLMResBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)

        self.enc3 = FiLMResBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck at 28x28
        self.mid1 = FiLMResBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.mid_attn = SelfAttention2d(base_ch * 4, num_heads=4)
        self.mid2 = FiLMResBlock(base_ch * 4, base_ch * 4, cond_dim)

        # Decoder: 28 -> 56 -> 112 -> 224
        self.dec3 = FiLMResBlock(base_ch * 8, base_ch * 4, cond_dim)
        self.dec2 = FiLMResBlock(base_ch * 6, base_ch * 2, cond_dim)
        self.dec1 = FiLMResBlock(base_ch * 3, base_ch, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 4, kernel_size=3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        img: torch.Tensor,
        inst_mask: torch.Tensor,
        sem_mask: torch.Tensor,
        sam_vec: torch.Tensor,
    ) -> torch.Tensor:
        masked_img = img * inst_mask
        inp = torch.cat([x_t, img, inst_mask, sem_mask, masked_img], dim=1)  # (B, 12, H, W)
        cond = self.time_mlp(t) + self.sam_mlp(sam_vec)  # (B, cond_dim)

        h0 = self.in_conv(inp)
        e1 = self.enc1(h0, cond)                          # (B, C, 224, 224)
        e2 = self.enc2(self.down1(e1), cond)              # (B, 2C, 112, 112)
        e3 = self.enc3(self.down2(e2), cond)              # (B, 4C, 56, 56)

        m = self.down3(e3)                                # (B, 4C, 28, 28)
        m = self.mid2(self.mid_attn(self.mid1(m, cond)), cond)

        u3 = F.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1), cond)  # (B, 4C, 56, 56)

        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)  # (B, 2C, 112, 112)

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)  # (B, C, 224, 224)

        return self.out_conv(F.silu(self.out_norm(d1)))