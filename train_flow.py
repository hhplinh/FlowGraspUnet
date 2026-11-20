import argparse
import datetime
import json
import logging
import math
import os
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF
from tqdm import tqdm

from flow.rectified_flow import RectifiedGraspFlow
from inference.post_process import post_process_output
from models.flow_grasp_net.flow_grasp_net import FlowGraspUNet
from models.flow_grasp_net.sam3_grounder import Sam3Grounder
from utils.data import get_dataset
from utils.dataset_processing import evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM3 + Conditional Flow Matching Grasp Model")

    # Model & SAM 3
    parser.add_argument("--sam3-path", type=str, default="/workspace/models/sam3")
    parser.add_argument("--base-ch", type=int, default=64, help="Base channel width of FlowGraspUNet")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=8, help="Euler ODE steps for validation")
    parser.add_argument("--iou-threshold", type=float, default=0.25)

    # Dataset
    parser.add_argument("--dataset", type=str, default="grasp-anywhere")
    parser.add_argument("--dataset-path", type=str, default="data/grasp-anything++")
    parser.add_argument("--add-file-path", type=str, default="data/grasp-anything++")
    parser.add_argument("--seen", type=int, default=1)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--ds-shuffle", action="store_true", default=True)
    parser.add_argument("--ds-rotate", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=2000, help="0 for full dataset")
    parser.add_argument("--no-augment", action="store_true", default=False, help="Disable rotate/zoom")
    parser.add_argument("--no-cache-sam3", action="store_true", default=False, help="Run SAM3 live every batch")
    parser.add_argument("--num-workers", type=int, default=4)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs-per-eval", type=int, default=10)
    parser.add_argument("--eval-train", action="store_true", default=False, help="Also measure train IoU")

    # Logging
    parser.add_argument("--description", type=str, default="sam3_flow_2k")
    parser.add_argument("--logdir", type=str, default="logs_flow/")
    parser.add_argument("--random-seed", type=int, default=123)

    return parser.parse_args()


@torch.no_grad()
def build_sam3_cache(sam_grounder: Sam3Grounder, unaug_dataset, indices, batch_size: int, device):
    """
    Pre-computes (inst_mask, sem_mask, sam_vec) in float16 on CPU RAM for fast training.
    """
    subset = torch.utils.data.Subset(unaug_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    cache = {}
    for x, _, didx, _, _, _, query in tqdm(loader, desc="Caching SAM 3 masks"):
        inst_m, sem_m, sam_v = sam_grounder(x.to(device), query)
        inst_cpu = inst_m.half().cpu()
        sem_cpu = sem_m.half().cpu()
        vec_cpu = sam_v.half().cpu()
        for i, sample_idx in enumerate(didx.tolist()):
            cache[int(sample_idx)] = (inst_cpu[i], sem_cpu[i], vec_cpu[i])
    return cache


def fetch_aligned_cached_sam(cache, didx, rot, zoom_factor, device):
    """
    Retrieves cached SAM 3 masks for a batch and applies exact GPU rotate/zoom
    to stay aligned with augmented images and targets.
    """
    inst_list, sem_list, vec_list = [], [], []
    for i, sample_idx in enumerate(didx.tolist()):
        inst_m, sem_m, sam_v = cache[int(sample_idx)]
        inst_m = inst_m.to(device=device, dtype=torch.float32)
        sem_m = sem_m.to(device=device, dtype=torch.float32)
        sam_v = sam_v.to(device=device, dtype=torch.float32)

        r = float(rot[i].item())
        z = float(zoom_factor[i].item())
        _, H, W = inst_m.shape

        if abs(r) > 1e-6:
            angle_deg = r * 180.0 / math.pi
            inst_m = TF.rotate(inst_m, angle=angle_deg, interpolation=TF.InterpolationMode.BILINEAR)
            sem_m = TF.rotate(sem_m, angle=angle_deg, interpolation=TF.InterpolationMode.BILINEAR)

        if abs(z - 1.0) > 1e-6:
            sr = int(H * (1.0 - z)) // 2
            sc = int(W * (1.0 - z)) // 2
            if sr > 0 or sc > 0:
                inst_m = F.interpolate(
                    inst_m[:, sr : H - sr, sc : W - sc].unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                sem_m = F.interpolate(
                    sem_m[:, sr : H - sr, sc : W - sc].unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        if abs(r) > 1e-6 or abs(z - 1.0) > 1e-6:
            gy = torch.linspace(-1.0, 1.0, H, device=device).view(1, H, 1)
            gx = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, W)
            mass = inst_m.sum() + 1e-6
            cx = (inst_m * gx).sum() / mass
            cy = (inst_m * gy).sum() / mass
            sx = torch.sqrt(((inst_m * (gx - cx) ** 2).sum() / mass) + 1e-6)
            sy = torch.sqrt(((inst_m * (gy - cy) ** 2).sum() / mass) + 1e-6)
            sam_v = sam_v.clone()
            sam_v[4:8] = 0.0
            sam_v[8:12] = torch.stack([cx, cy, sx, sy])
            sam_v[12:16] = torch.stack([inst_m.mean(), inst_m.amax(), sem_m.mean(), sem_m.amax()])

        inst_list.append(inst_m)
        sem_list.append(sem_m)
        vec_list.append(sam_v)

    return torch.stack(inst_list, dim=0), torch.stack(sem_list, dim=0), torch.stack(vec_list, dim=0)


@torch.no_grad()
def validate(model: RectifiedGraspFlow, device, val_data, iou_threshold: float, num_steps: int = 8):
    model.flow_net.eval()
    results = {"correct": 0, "failed": 0, "mask_hits": 0}

    for x, y, didx, rot, zoom_factor, prompt, query in tqdm(val_data, desc="Validating", leave=False):
        img = x.to(device)
        pred, (inst_mask, sem_mask) = model.sample_ode(
            img, query, num_steps=num_steps, solver="euler", seed=42, return_masks=True
        )

        gt_pos = y[0].to(device)
        if gt_pos.ndim == 3:
            gt_pos = gt_pos.unsqueeze(1)
        combined_mask = torch.maximum(inst_mask, sem_mask)
        peak_idx = gt_pos.view(1, -1).argmax(dim=1)
        mask_at_gt = combined_mask.view(1, -1).gather(1, peak_idx.unsqueeze(1)).item()
        if mask_at_gt > 0.25:
            results["mask_hits"] += 1

        q_out, ang_out, w_out = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )
        success = evaluation.calculate_iou_match(
            q_out,
            ang_out,
            val_data.dataset.get_gtbb(int(didx.item()), float(rot.item()), float(zoom_factor.item())),
            no_grasps=1,
            grasp_width=w_out,
            threshold=iou_threshold,
        )
        if success:
            results["correct"] += 1
        else:
            results["failed"] += 1

    total = max(results["correct"] + results["failed"], 1)
    results["iou"] = results["correct"] / total
    results["mask_hit_rate"] = results["mask_hits"] / total
    return results


def train_one_epoch(epoch: int, model: RectifiedGraspFlow, device, train_data, optimizer, sam_cache=None):
    model.flow_net.train()
    total_loss = 0.0
    sub_losses = {}
    num_batches = 0

    pbar = tqdm(train_data, desc=f"Epoch {epoch:03d}", leave=False)
    for x, y, didx, rot, zoom_factor, prompt, query in pbar:
        img = x.to(device, non_blocking=True)
        yc = [yy.to(device, non_blocking=True) for yy in y]

        cached_sam = None
        if sam_cache is not None:
            cached_sam = fetch_aligned_cached_sam(sam_cache, didx, rot, zoom_factor, device)

        loss_dict = model.compute_loss(img, yc, query, cached_sam=cached_sam)
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.flow_net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict["losses"].items():
            sub_losses[k] = sub_losses.get(k, 0.0) + v.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "losses": {k: v / denom for k, v in sub_losses.items()},
    }


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    dt = datetime.datetime.now().strftime("%y%m%d_%H%M")
    run_name = f"{dt}_{'_'.join(args.description.split())}"
    save_folder = os.path.join(args.logdir, run_name)
    os.makedirs(save_folder, exist_ok=True)
    tb = tensorboardX.SummaryWriter(save_folder)

    with open(os.path.join(save_folder, "commandline_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(save_folder, "train.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dataset = get_dataset(args.dataset)
    max_samples = None if args.max_samples == 0 else args.max_samples

    train_dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=not args.no_augment,
        random_zoom=not args.no_augment,
        include_depth=0,
        include_rgb=1,
        seen=args.seen,
        add_file_path=args.add_file_path,
        max_samples=max_samples,
    )
    val_dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=False,
        random_zoom=False,
        include_depth=0,
        include_rgb=1,
        seen=args.seen,
        add_file_path=args.add_file_path,
        max_samples=max_samples,
    )

    indices = list(range(train_dataset.length))
    split = int(np.floor(args.split * train_dataset.length))
    if args.ds_shuffle:
        rng = np.random.RandomState(args.random_seed)
        rng.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info(f"Dataset size: {train_dataset.length} | Train: {len(train_indices)} | Val: {len(val_indices)}")

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=args.num_workers,
    )

    logging.info("Loading frozen SAM 3 and initializing FlowGraspUNet...")
    sam_grounder = Sam3Grounder(model_path=args.sam3_path, device=device)
    flow_net = FlowGraspUNet(base_ch=args.base_ch, sam_vec_dim=sam_grounder.sam_vec_dim).to(device)
    model = RectifiedGraspFlow(sam_grounder, flow_net).to(device)

    trainable_params = sum(p.numel() for p in flow_net.parameters() if p.requires_grad)
    logging.info(f"FlowGraspUNet trainable parameters: {trainable_params / 1e6:.2f}M")

    sam_cache = None
    if not args.no_cache_sam3:
        sam_cache = build_sam3_cache(sam_grounder, val_dataset, train_indices, min(args.batch_size, 8), device)
        sam_grounder.cpu()
        torch.cuda.empty_cache()
        logging.info(f"Cached SAM 3 features for {len(sam_cache)} training samples.")

    optimizer = AdamW(flow_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_iou = -1.0
    for epoch in range(args.epochs):
        train_res = train_one_epoch(epoch, model, device, train_data, optimizer, sam_cache=sam_cache)
        scheduler.step()

        tb.add_scalar("loss/train", train_res["loss"], epoch)
        for k, v in train_res["losses"].items():
            tb.add_scalar(f"train/{k}", v, epoch)

        if (epoch + 1) % args.epochs_per_eval == 0 or epoch == args.epochs - 1 or epoch == 0:
            sam_grounder.to(device)
            val_res = validate(model, device, val_data, args.iou_threshold, num_steps=args.num_steps)
            if sam_cache is not None:
                sam_grounder.cpu()
                torch.cuda.empty_cache()
            logging.info(
                f"Epoch {epoch:03d} | Train Loss: {train_res['loss']:.4f} | "
                f"Val IoU@{args.iou_threshold}: {val_res['correct']}/{val_res['correct'] + val_res['failed']} = {val_res['iou']:.4f} | "
                f"SAM3 Mask Hit: {val_res['mask_hit_rate']:.4f}"
            )
            tb.add_scalar("val/iou", val_res["iou"], epoch)
            tb.add_scalar("val/mask_hit_rate", val_res["mask_hit_rate"], epoch)

            if args.eval_train:
                train_eval_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,
                    sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                    num_workers=args.num_workers,
                )
                tr_eval = validate(model, device, train_eval_loader, args.iou_threshold, num_steps=args.num_steps)
                logging.info(f"Epoch {epoch:03d} | Train Set IoU@{args.iou_threshold}: {tr_eval['iou']:.4f}")
                tb.add_scalar("train/iou", tr_eval["iou"], epoch)

            if val_res["iou"] > best_iou:
                best_iou = val_res["iou"]
                ckpt_path = os.path.join(save_folder, f"best_epoch_{epoch:03d}_iou_{best_iou:.4f}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "iou": best_iou,
                        "base_ch": args.base_ch,
                        "sam_vec_dim": sam_grounder.sam_vec_dim,
                        "model_state_dict": flow_net.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                logging.info(f"Saved new best checkpoint (~35 MB): {ckpt_path}")


if __name__ == "__main__":
    main()
