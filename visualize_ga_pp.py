from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure /workspace/LGD is on PYTHONPATH when run from any working directory
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.dataset_processing import grasp


def draw_grasp_rectangles(img_rgb: np.ndarray, gtbbs: grasp.GraspRectangles, max_grasps: int = 15) -> np.ndarray:
    """Draw oriented grasp rectangles using GraspRectangle.points (row, col -> x, y)."""
    canvas = img_rgb.copy()
    bbs = list(gtbbs)
    if len(bbs) > max_grasps:
        indices = np.linspace(0, len(bbs) - 1, max_grasps, dtype=int)
        bbs = [bbs[i] for i in indices]

    for bb in bbs:
        # bb.points has shape (4, 2) in (row, col) = (y, x) order
        pts = bb.points[:, ::-1].astype(np.int32)
        # Opening span (green)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (40, 220, 90), 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[2]), tuple(pts[3]), (40, 220, 90), 1, cv2.LINE_AA)
        # Parallel gripper plates (blue)
        cv2.line(canvas, tuple(pts[1]), tuple(pts[2]), (30, 144, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[3]), tuple(pts[0]), (30, 144, 255), 2, cv2.LINE_AA)
        # Center point (red)
        cy, cx = int(bb.center[0]), int(bb.center[1])
        cv2.circle(canvas, (cx, cy), 2, (255, 60, 60), -1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize Grasp-Anything++ samples using LGD loader.")
    parser.add_argument("--data-dir", type=str, default="data/grasp-anything++")
    parser.add_argument("--split", type=str, default=None, choices=["seen", "unseen"], help="Optional test split filter")
    parser.add_argument("--sample-id", type=str, default=None, help="Specific grasp stem (e.g. 0000123_0_1)")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random samples to plot")
    parser.add_argument("--max-grasps", type=int, default=15, help="Max grasps drawn per image")
    parser.add_argument("--img-size", type=int, default=416, help="Target square resolution (default 416)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="vis_ga_pp.png")
    args = parser.parse_args()

    grasp_files = sorted(glob.glob(os.path.join(args.data_dir, "positive_grasp", "*.pt")))
    if not grasp_files:
        raise FileNotFoundError(f"No .pt files found in {args.data_dir}/positive_grasp")

    if args.split is not None:
        split_file = os.path.join("split/grasp-anything++/test", f"{args.split}.obj")
        with open(split_file, "rb") as f:
            idxs = pickle.load(f)
        grasp_files = [x for x in grasp_files if os.path.basename(x)[:-5] in idxs]

    if args.sample_id:
        selected = [os.path.join(args.data_dir, "positive_grasp", f"{args.sample_id}.pt")]
    else:
        random.seed(args.seed)
        selected = random.sample(grasp_files, min(args.num_samples, len(grasp_files)))

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.6 * n), squeeze=False)

    for row, grasp_path in enumerate(selected):
        base_pt = os.path.basename(grasp_path)          # e.g., 0000123_0_1.pt
        stem = os.path.splitext(base_pt)[0]             # e.g., 0000123_0_1
        rgb_name = re.sub(r"_\d+_\d+\.pt$", ".jpg", base_pt)

        rgb_path = os.path.join(args.data_dir, "image", rgb_name)
        inst_path = os.path.join(args.data_dir, "grasp_instructions", f"{stem}.pkl")

        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (args.img_size, args.img_size))

        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(
            grasp_path, scale=args.img_size / 416.0
        )

        with open(inst_path, "rb") as f:
            instruction = pickle.load(f)
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0]

        vis_rgb = draw_grasp_rectangles(img_rgb, gtbbs, max_grasps=args.max_grasps)

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"{stem} -> {rgb_name}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vis_rgb)
        axes[row, 1].set_title(f'"{instruction}" (n={len(list(gtbbs))})', fontsize=10, fontweight="bold")
        axes[row, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.abspath(args.save)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {n} sample(s) to: {out_path}")


if __name__ == "__main__":
    main()