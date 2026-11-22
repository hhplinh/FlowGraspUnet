from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hardware.device import get_device
from inference.post_process import post_process_output
from models.flow_grasp_net.flow_grasp_net import FlowGraspUNet
from models.flow_grasp_net.sam3_grounder import Sam3Grounder
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp


def torch_load_ckpt(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_flow_model(ckpt_path: str, device: torch.device, base_ch: int = 64, sam_vec_dim: int | None = None) -> torch.nn.Module:
    """Load FlowGraspUNet (preferring EMA weights if present in the checkpoint)."""
    ckpt = torch_load_ckpt(ckpt_path, device)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        for key in ("ema_net", "ema_model", "ema", "flow_net", "model"):
            if key in ckpt and isinstance(ckpt[key], torch.nn.Module):
                model = ckpt[key]
                break
        else:
            state = None
            for key in ("ema_state_dict", "ema_net", "ema", "model_state_dict", "flow_net", "model", "state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
            if state is None:
                state = ckpt
            cleaned = {k.replace("module.", ""): v for k, v in state.items() if k != "n_averaged"}

            cond_dim = 256
            if "sam_mlp.0.weight" in cleaned:
                cond_dim, inferred_sam_dim = cleaned["sam_mlp.0.weight"].shape
                if sam_vec_dim is None:
                    sam_vec_dim = inferred_sam_dim
            if sam_vec_dim is None:
                sam_vec_dim = 1040

            model = FlowGraspUNet(base_ch=base_ch, cond_dim=cond_dim, sam_vec_dim=sam_vec_dim)
            model.load_state_dict(cleaned, strict=False)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    return model.to(device).eval()

def _to_corners_xy(bb) -> np.ndarray:
    """Convert Grasp or GraspRectangle into (4, 2) corners in (x, y) pixel order."""
    if hasattr(bb, "as_gr"):
        bb = bb.as_gr
    if hasattr(bb, "points"):
        return bb.points[:, ::-1].astype(np.int32)
    raise TypeError(f"Unsupported grasp object type: {type(bb)}")


def draw_grasps(
    img_rgb: np.ndarray,
    grasps,
    max_grasps: int = 10,
    jaw_color: tuple[int, int, int] = (30, 144, 255),
    span_color: tuple[int, int, int] = (40, 220, 90),
    gtbbs: grasp.GraspRectangles | None = None,
) -> np.ndarray:
    """Draw oriented gripper rectangles; optionally label each predicted grasp with its max IoU."""
    canvas = img_rgb.copy()
    items = list(grasps)
    if len(items) > max_grasps:
        idx = np.linspace(0, len(items) - 1, max_grasps, dtype=int)
        items = [items[i] for i in idx]

    for bb in items:
        pts = _to_corners_xy(bb)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), span_color, 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[2]), tuple(pts[3]), span_color, 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[1]), tuple(pts[2]), jaw_color, 2, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[3]), tuple(pts[0]), jaw_color, 2, cv2.LINE_AA)

        center_xy = tuple(pts.mean(axis=0).astype(int))
        cv2.circle(canvas, center_xy, 2, (255, 50, 50), -1, cv2.LINE_AA)

        if gtbbs is not None and hasattr(bb, "max_iou"):
            try:
                iou = bb.max_iou(gtbbs)
                cv2.putText(
                    canvas,
                    f"{iou:.2f}",
                    (center_xy[0] + 4, center_xy[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

    return canvas


def overlay_sam_mask(img_rgb: np.ndarray, inst_mask: np.ndarray) -> np.ndarray:
    """Overlay SAM 3 instance mask onto RGB image."""
    mask = np.clip(inst_mask.squeeze(), 0.0, 1.0)
    overlay = img_rgb.astype(np.float32).copy()
    tint = np.zeros_like(overlay)
    tint[..., 1] = 220.0
    tint[..., 2] = 255.0
    alpha = mask[..., None] * 0.55
    blended = overlay * (1.0 - alpha) + tint * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def get_sam3_features(
    sample_idx: int,
    img: torch.Tensor,
    prompt_str: str,
    args,
    device: torch.device,
    grounder_holder: list,
):
    """Load cached SAM 3 features if available, else run frozen Sam3Grounder."""
    cache_file = os.path.join(
        args.sam3_cache_dir,
        f"seen{args.seen}_sz{args.input_size}",
        f"{sample_idx:06d}.pt",
    )
    if os.path.isfile(cache_file):
        cached = torch_load_ckpt(cache_file, device)
        if isinstance(cached, dict):
            inst_mask, sem_mask, sam_vec = cached["inst_mask"], cached["sem_mask"], cached["sam_vec"]
        else:
            inst_mask, sem_mask, sam_vec = cached[0], cached[1], cached[2]

        inst_mask = inst_mask.to(device=device, dtype=torch.float32)
        sem_mask = sem_mask.to(device=device, dtype=torch.float32)
        sam_vec = sam_vec.to(device=device, dtype=torch.float32)

        if inst_mask.ndim == 2:
            inst_mask = inst_mask.unsqueeze(0).unsqueeze(0)
        elif inst_mask.ndim == 3:
            inst_mask = inst_mask.unsqueeze(0)

        if sem_mask.ndim == 2:
            sem_mask = sem_mask.unsqueeze(0).unsqueeze(0)
        elif sem_mask.ndim == 3:
            sem_mask = sem_mask.unsqueeze(0)

        if sam_vec.ndim == 1:
            sam_vec = sam_vec.unsqueeze(0)
    else:
        if not grounder_holder:
            grounder_holder.append(Sam3Grounder(model_path=args.sam3_path, device=str(device)))
        inst_mask, sem_mask, sam_vec = grounder_holder[0](img, [prompt_str])

    sam_vec = sam_vec.clone()
    sam_vec[:, 4:8] = 0.0
    return inst_mask, sem_mask, sam_vec

@torch.no_grad()
def predict_flow_grasp(
    flow_net: torch.nn.Module,
    img: torch.Tensor,
    inst_mask: torch.Tensor,
    sem_mask: torch.Tensor,
    sam_vec: torch.Tensor,
    num_steps: int = 16,
    prior_std: float = 0.15,
    seed: int = 123,
):
    """Run N-step Euler ODE integration from low-variance Gaussian prior."""
    gen = torch.Generator(device=img.device)
    gen.manual_seed(seed)
    B, _, H, W = img.shape
    x = torch.randn((B, 4, H, W), generator=gen, device=img.device, dtype=torch.float32) * prior_std

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((B,), step / num_steps, device=img.device, dtype=torch.float32)
        v_pred = flow_net(x, t, img, inst_mask, sem_mask, sam_vec)
        x = x + dt * v_pred

    pos_out = x[:, 0:1].clamp(0.0, 1.0)
    cos_out = x[:, 1:2].clamp(-1.0, 1.0)
    sin_out = x[:, 2:3].clamp(-1.0, 1.0)
    width_out = x[:, 3:4].clamp(0.0, 1.0)
    return post_process_output(pos_out, cos_out, sin_out, width_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FlowGraspUnet predictions on validation samples.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FlowGraspUnet checkpoint (.pt)")
    parser.add_argument("--sam3-path", type=str, default=None)
    parser.add_argument("--sam3-cache-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--add-file-path", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--seen", type=int, default=None)
    parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ds-shuffle", action="store_true", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--prior-std", type=float, default=0.15)
    parser.add_argument("--base-ch", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=4, help="Number of validation samples to plot")
    parser.add_argument("--val-offsets", type=int, nargs="*", default=None, help="Specific offsets in val split (e.g. 0 4 12)")
    parser.add_argument("--only-pass", action="store_true", help="Only keep samples where FlowGraspUnet passes IoU threshold")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K predicted grasps to draw")
    parser.add_argument("--cpu", dest="force_cpu", action="store_true", default=False)
    parser.add_argument("--save", type=str, default="figs/qualitative.pdf")
    return parser.parse_args()


def merge_with_saved_args(args):
    """Fill dataset/model flags from commandline_args.json in the checkpoint directory."""
    saved_args_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "commandline_args.json")
    saved = {}
    if os.path.isfile(saved_args_path):
        with open(saved_args_path, "r") as f:
            saved = json.load(f)

    defaults = {
        "sam3_path": "/workspace/models/sam3",
        "sam3_cache_dir": "data/sam3_cache",
        "dataset": "grasp-anywhere",
        "dataset_path": "data/grasp-anything++",
        "add_file_path": "data/grasp-anything++",
        "input_size": 224,
        "seen": 1,
        "split": 0.9,
        "max_samples": 2000,
        "ds_shuffle": True,
        "random_seed": 123,
        "num_steps": 16,
        "base_ch": 64,
    }
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, fallback))
    return args


def main():
    args = merge_with_saved_args(parse_args())
    device = get_device(args.force_cpu)

    Dataset = get_dataset(args.dataset)
    dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=0.0,
        random_rotate=False,
        random_zoom=False,
        include_depth=0,
        include_rgb=1,
        seen=args.seen,
        add_file_path=args.add_file_path,
        max_samples=None if args.max_samples == 0 else args.max_samples,
    )

    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:] if split < dataset.length else indices

    if args.val_offsets is not None:
        val_indices = [val_indices[i] for i in args.val_offsets if 0 <= i < len(val_indices)]

    val_sampler = torch.utils.data.SequentialSampler(val_indices)
    val_data = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, sampler=val_sampler)

    flow_net = load_flow_model(args.checkpoint, device, base_ch=args.base_ch)
    grounder_holder: list[Sam3Grounder] = []

    rows = []
    with torch.no_grad():
        for x, y, didx, rot, zoom_factor, prompt, _ in val_data:
            if len(rows) >= args.num_samples:
                break

            img = x.to(device)
            pos_gt = y[0].to(device)
            sample_idx = int(didx[0]) if isinstance(didx, torch.Tensor) else int(didx)
            prompt_str = prompt[0] if isinstance(prompt, (list, tuple)) else str(prompt)

            inst_mask, sem_mask, sam_vec = get_sam3_features(
                sample_idx, img, prompt_str, args, device, grounder_holder
            )
            q_out, ang_out, w_out = predict_flow_grasp(
                flow_net,
                img,
                inst_mask,
                sem_mask,
                sam_vec,
                num_steps=args.num_steps,
                prior_std=args.prior_std,
                seed=args.random_seed + sample_idx,
            )

            sample_idx = int(didx[0]) if isinstance(didx, torch.Tensor) else int(didx)
            rot_val = float(rot[0]) if isinstance(rot, torch.Tensor) else float(rot)
            zoom_val = float(zoom_factor[0]) if isinstance(zoom_factor, torch.Tensor) else float(zoom_factor)

            gtbbs = dataset.get_gtbb(sample_idx, rot_val, zoom_val)
            matched = evaluation.calculate_iou_match(
                q_out, ang_out, gtbbs, no_grasps=1, grasp_width=w_out, threshold=args.iou_threshold
            )
            if args.only_pass and not matched:
                continue

            pred_grasps = grasp.detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=args.top_k)

            gt_peak_idx = int(torch.argmax(pos_gt).item())
            gy, gx = divmod(gt_peak_idx, args.input_size)
            max_mask = torch.maximum(inst_mask[0, 0], sem_mask[0, 0])
            mask_hit = bool(max_mask[gy, gx].item() > 0.25)

            grasp_file = dataset.grasp_files[sample_idx]
            stem = os.path.splitext(os.path.basename(grasp_file))[0]
            rgb_name = re.sub(r"_\d+_\d+\.pt$", ".jpg", os.path.basename(grasp_file))
            rgb_path = os.path.join(dataset.rgb_dir, rgb_name)
            img_rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (args.input_size, args.input_size))

            rows.append((stem, prompt_str, img_rgb, gtbbs, inst_mask[0, 0].cpu().numpy(), mask_hit, q_out, pred_grasps, matched))

    n = len(rows)
    if n == 0:
        print("No samples matched the filter criteria.")
        return

    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n), squeeze=False)
    passes = 0

    for r, (stem, prompt_str, img_rgb, gtbbs, inst_np, mask_hit, q_out, pred_grasps, matched) in enumerate(rows):
        passes += int(matched)
        gt_vis = draw_grasps(img_rgb, gtbbs, max_grasps=10)
        sam_vis = overlay_sam_mask(img_rgb, inst_np)
        pred_vis = draw_grasps(
            img_rgb, pred_grasps, max_grasps=args.top_k, jaw_color=(0, 220, 90), span_color=(30, 144, 255), gtbbs=gtbbs
        )

        axes[r, 0].imshow(gt_vis)
        axes[r, 0].set_title(f'GT: "{prompt_str}"', fontsize=9, fontweight="bold")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(sam_vis)
        axes[r, 1].set_title(f"SAM 3 Mask (Hit={'Yes' if mask_hit else 'No'})", fontsize=9)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(img_rgb)
        im = axes[r, 2].imshow(q_out, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
        axes[r, 2].set_title(f"Quality Map (max={q_out.max():.2f})", fontsize=9)
        axes[r, 2].axis("off")
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)

        axes[r, 3].imshow(pred_vis)
        axes[r, 3].set_title(
            f"FlowGraspUnet ({args.num_steps}-step) [{'PASS' if matched else 'FAIL'}]",
            fontsize=9,
            fontweight="bold",
            color="green" if matched else "crimson",
        )
        axes[r, 3].axis("off")

    plt.tight_layout()
    out_path = os.path.abspath(args.save)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {n} FlowGraspUnet prediction(s) to: {out_path} ({passes}/{n} PASS)")


if __name__ == "__main__":
    main()

# Example usage:
# python visualize_flow.py \
#   --checkpoint logs_flow/261121_1917_sam3_flow_2k_diskcache_ema_ep400/best_epoch_169_iou_0.2150.pt \
#   --num-samples 3 \
#   --only-pass \
#   --save figs/qualitative.pdf