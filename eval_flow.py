import argparse
import time
import torch
from tqdm import tqdm

from flow.rectified_flow import RectifiedGraspFlow
from inference.post_process import post_process_output
from models.flow_grasp_net.flow_grasp_net import FlowGraspUNet
from models.flow_grasp_net.sam3_grounder import Sam3Grounder
from utils.data import get_dataset
from utils.dataset_processing import evaluation


@torch.no_grad()
def evaluate_checkpoint(model: RectifiedGraspFlow, loader, device, num_steps: int, solver: str, best_of_k: int, iou_threshold: float):
    model.flow_net.eval()
    top1_correct = 0
    bestk_correct = 0
    mask_hits = 0
    total = 0
    latencies_ms = []

    for x, y, didx, rot, zoom_factor, prompt, query in tqdm(loader, desc=f"Eval ({solver}, NFE={num_steps})"):
        img = x.to(device)
        gtbbs = loader.dataset.get_gtbb(int(didx.item()), float(rot.item()), float(zoom_factor.item()))

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Top-1 deterministic draw (seed=42)
        pred1, (inst_mask, sem_mask) = model.sample_ode(
            img, query, num_steps=num_steps, solver=solver, seed=42, return_masks=True
        )

        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # SAM 3 Mask Hit Rate
        gt_pos = y[0].to(device)
        if gt_pos.ndim == 3:
            gt_pos = gt_pos.unsqueeze(1)
        combined_mask = torch.maximum(inst_mask, sem_mask)
        peak_idx = gt_pos.view(1, -1).argmax(dim=1)
        if combined_mask.view(1, -1).gather(1, peak_idx.unsqueeze(1)).item() > 0.25:
            mask_hits += 1

        q1, ang1, w1 = post_process_output(pred1["pos"], pred1["cos"], pred1["sin"], pred1["width"])
        hit1 = evaluation.calculate_iou_match(q1, ang1, gtbbs, no_grasps=1, grasp_width=w1, threshold=iou_threshold)
        if hit1:
            top1_correct += 1
            bestk_correct += 1
        elif best_of_k > 1:
            # Draw K-1 additional stochastic samples in a single batch
            img_k = img.expand(best_of_k - 1, -1, -1, -1)
            query_k = [query[0]] * (best_of_k - 1)
            pred_k, _ = model.sample_ode(img_k, query_k, num_steps=num_steps, solver=solver, seed=123)
            for k in range(best_of_k - 1):
                qk, angk, wk = post_process_output(
                    pred_k["pos"][k : k + 1],
                    pred_k["cos"][k : k + 1],
                    pred_k["sin"][k : k + 1],
                    pred_k["width"][k : k + 1],
                )
                if evaluation.calculate_iou_match(qk, angk, gtbbs, no_grasps=1, grasp_width=wk, threshold=iou_threshold):
                    bestk_correct += 1
                    break

        total += 1

    return {
        "top1_iou": top1_correct / max(total, 1),
        "bestk_iou": bestk_correct / max(total, 1),
        "mask_hit_rate": mask_hits / max(total, 1),
        "latency_ms": sum(latencies_ms[2:]) / max(len(latencies_ms[2:]), 1),  # skip 2 warmup steps
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 + Flow Grasp Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam3-path", type=str, default="/workspace/models/sam3")
    parser.add_argument("--dataset-path", type=str, default="data/grasp-anything++")
    parser.add_argument("--add-file-path", type=str, default="data/grasp-anything++")
    parser.add_argument("--seen", type=int, default=1, help="1 for seen split, 0 for unseen split")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--steps-list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--best-of-k", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=0.25)

    parser.add_argument("--val-only", action="store_true", help="Evaluate only the held-out 10% validation split (seed=123)")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--start-idx", type=int, default=0, help="Start index in dataset (e.g., 2000 for disjoint seen test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    base_ch = ckpt.get("base_ch", 64)

    sam_grounder = Sam3Grounder(model_path=args.sam3_path, device=device)
    flow_net = FlowGraspUNet(base_ch=base_ch, sam_vec_dim=ckpt.get("sam_vec_dim", sam_grounder.sam_vec_dim)).to(device)
    flow_net.load_state_dict(ckpt["model_state_dict"])
    model = RectifiedGraspFlow(sam_grounder, flow_net).to(device)

    Dataset = get_dataset("grasp-anywhere")
    ds = Dataset(
        args.dataset_path,
        output_size=224,
        random_rotate=False,
        random_zoom=False,
        include_depth=0,
        include_rgb=1,
        seen=args.seen,
        add_file_path=args.add_file_path,
        max_samples=None if args.max_samples == 0 else args.max_samples,
    )

    indices = list(range(ds.length))
    if args.val_only:
        rng = np.random.RandomState(args.random_seed)
        rng.shuffle(indices)
        eval_indices = indices[int(np.floor(args.split * ds.length)):]
    else:
        eval_indices = indices[args.start_idx:]

    eval_subset = torch.utils.data.Subset(ds, eval_indices)
    loader = torch.utils.data.DataLoader(eval_subset, batch_size=1, shuffle=False, num_workers=2)
    # loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    split_name = "SEEN" if args.seen == 1 else "UNSEEN"
    print(f"\n=== Evaluating {args.checkpoint} on {split_name} ({ds.length} samples) ===")
    for steps in args.steps_list:
        res = evaluate_checkpoint(model, loader, device, steps, args.solver, args.best_of_k, args.iou_threshold)
        print(
            f"Solver={args.solver:8s} | NFE={steps:2d} | "
            f"Top-1 IoU@{args.iou_threshold}: {res['top1_iou']:.4f} | "
            f"Best-of-{args.best_of_k} IoU: {res['bestk_iou']:.4f} | "
            f"SAM3 Hit: {res['mask_hit_rate']:.4f} | "
            f"Latency: {res['latency_ms']:.1f} ms/img"
        )


if __name__ == "__main__":
    main()