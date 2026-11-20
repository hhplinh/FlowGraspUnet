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
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.model_util import create_diffusion


def torch_load_ckpt(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _to_corners_xy(bb) -> np.ndarray:
    """Convert Grasp or GraspRectangle into (4, 2) corners in (x, y) pixel order."""
    if hasattr(bb, "as_gr"):
        bb = bb.as_gr
    if hasattr(bb, "points"):
        # bb.points is (4, 2) in (row, col) = (y, x)
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
        # Opening span
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), span_color, 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[2]), tuple(pts[3]), span_color, 1, cv2.LINE_AA)
        # Parallel gripper plates
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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LGD model predictions on validation samples.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model in logs/<run>/epoch_*")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (auto-loaded from commandline_args.json if omitted)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to Grasp-Anything++ directory")
    parser.add_argument("--add-file-path", type=str, default=None, help="Additional file path (same as dataset-path for GA++)")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--use-depth", type=int, default=None)
    parser.add_argument("--use-rgb", type=int, default=None)
    parser.add_argument("--seen", type=int, default=None)
    parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ds-shuffle", action="store_true", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--ddim-steps", type=str, default="ddim50", help="Timestep respacing, e.g. ddim50 or ddim20")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of validation samples to visualize")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K predicted grasps to draw")
    parser.add_argument("--cpu", dest="force_cpu", action="store_true", default=False)
    parser.add_argument("--save", type=str, default="vis_pred.png")
    return parser.parse_args()


def merge_with_saved_args(args):
    """Fill unspecified dataset/model flags from commandline_args.json in the checkpoint directory."""
    saved_args_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "commandline_args.json")
    saved = {}
    if os.path.isfile(saved_args_path):
        with open(saved_args_path, "r") as f:
            saved = json.load(f)

    defaults = {
        "dataset": "grasp-anywhere",
        "dataset_path": "data/grasp-anything++",
        "add_file_path": "data/grasp-anything++",
        "input_size": 224,
        "use_depth": 0,
        "use_rgb": 1,
        "seen": 1,
        "split": 0.9,
        "max_samples": 2000,
        "ds_shuffle": False,
        "random_seed": 123,
    }
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, fallback))
    return args


def main():
    args = merge_with_saved_args(parse_args())
    device = get_device(args.force_cpu)

    # Load Dataset & reproduce validation split (without random augs so images are unrotated)
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=0.0,
        random_rotate=False,
        random_zoom=False,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
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

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_data = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, sampler=val_sampler)

    # Load saved network & DDIM sampler
    net = torch_load_ckpt(args.checkpoint, device)
    net = net.to(device)
    net.eval()

    val_diffusion = create_diffusion(timestep_respacing=args.ddim_steps)

    rows = []
    with torch.no_grad():
        for x, y, didx, rot, zoom_factor, prompt, query in val_data:
            if len(rows) >= args.num_samples:
                break

            img = x.to(device)
            yc = [yy.to(device) for yy in y]
            pos_gt = yc[0]

            alpha = 0.4
            idx = torch.ones(img.shape[0], device=device)

            sample = val_diffusion.ddim_sample_loop(
                net,
                pos_gt.shape,
                pos_gt,
                img,
                query,
                alpha,
                idx,
            )

            pos_output = sample
            cos_output, sin_output, width_output = (
                net.cos_output_str,
                net.sin_output_str,
                net.width_output_str,
            )

            lossd = net.compute_loss(yc, pos_output, cos_output, sin_output, width_output)
            q_out, ang_out, w_out = post_process_output(
                lossd["pred"]["pos"],
                lossd["pred"]["cos"],
                lossd["pred"]["sin"],
                lossd["pred"]["width"],
            )

            gtbbs = dataset.get_gtbb(didx, rot, zoom_factor)
            matched = evaluation.calculate_iou_match(
                q_out,
                ang_out,
                gtbbs,
                no_grasps=1,
                grasp_width=w_out,
                threshold=args.iou_threshold,
            )
            pred_grasps = grasp.detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=args.top_k)

            # Load clean unnormalized RGB image matching didx
            sample_idx = int(didx[0]) if isinstance(didx, torch.Tensor) else int(didx)
            grasp_file = dataset.grasp_files[sample_idx]
            stem = os.path.splitext(os.path.basename(grasp_file))[0]
            rgb_name = re.sub(r"_\d+_\d+\.pt$", ".jpg", os.path.basename(grasp_file))
            rgb_path = os.path.join(dataset.rgb_dir, rgb_name)

            img_bgr = cv2.imread(rgb_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (args.input_size, args.input_size))

            prompt_str = prompt[0] if isinstance(prompt, (list, tuple)) else str(prompt)
            rows.append((stem, prompt_str, img_rgb, gtbbs, q_out, pred_grasps, matched))

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.4 * n), squeeze=False)

    for r, (stem, prompt_str, img_rgb, gtbbs, q_out, pred_grasps, matched) in enumerate(rows):
        gt_vis = draw_grasps(img_rgb, gtbbs, max_grasps=15)
        pred_vis = draw_grasps(
            img_rgb,
            pred_grasps,
            max_grasps=args.top_k,
            jaw_color=(255, 0, 180),
            span_color=(255, 215, 0),
            gtbbs=gtbbs,
        )

        axes[r, 0].imshow(gt_vis)
        axes[r, 0].set_title(f"GT ({stem}, n={len(list(gtbbs))})", fontsize=9)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(img_rgb)
        im = axes[r, 1].imshow(q_out, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
        axes[r, 1].set_title(f'"{prompt_str}"\nQuality Heatmap', fontsize=9, fontweight="bold")
        axes[r, 1].axis("off")
        plt.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.04)

        status = "PASS" if matched else "FAIL"
        axes[r, 2].imshow(pred_vis)
        axes[r, 2].set_title(f"Pred Top-{args.top_k} [{status} @ IoU>{args.iou_threshold}]", fontsize=9)
        axes[r, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.abspath(args.save)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {n} prediction comparison(s) to: {out_path}")


if __name__ == "__main__":
    main()


# python visualize_pred.py \
#   --checkpoint logs/261120_1300_50ddim_2k_bs32/epoch_59_iou_0.04 \
#   --num-samples 4 \
#   --top-k 3 \
#   --save vis_pred.png
