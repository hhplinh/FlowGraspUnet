import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow.rectified_flow import RectifiedGraspFlow
from models.flow_grasp_net.flow_grasp_net import FlowGraspUNet
from models.flow_grasp_net.sam3_grounder import Sam3Grounder
from utils.data import get_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="data/grasp-anything++")
    parser.add_argument("--add-file-path", type=str, default="data/grasp-anything++")
    parser.add_argument("--sam3-path", type=str, default="/workspace/models/sam3")
    parser.add_argument("--seen", type=int, default=1)
    parser.add_argument("--num-vis", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        max_samples=max(args.num_vis, 16),
    )

    sam_grounder = Sam3Grounder(model_path=args.sam3_path, device=device)
    flow_net = FlowGraspUNet(base_ch=64).to(device)
    model = RectifiedGraspFlow(sam_grounder, flow_net).to(device)

    # 1. Test training forward + backward step
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    x, y, didx, rot, zoom_factor, prompt, query = next(iter(loader))
    x = x.to(device)
    yc = [yy.to(device) for yy in y]

    loss_dict = model.compute_loss(x, yc, query)
    loss_dict["loss"].backward()
    print(f"[PASS] Forward + backward OK | Loss: {loss_dict['loss'].item():.4f} | Details: {loss_dict['losses']}")

    # 2. Test 8-step Euler ODE sampling
    pred, (inst_masks, sem_masks) = model.sample_ode(x, query, num_steps=8, return_masks=True)
    print(
        f"[PASS] Euler ODE OK | pos: {tuple(pred['pos'].shape)}, "
        f"cos: {tuple(pred['cos'].shape)}, width: {tuple(pred['width'].shape)} | "
        f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    )

    # 3. Visualize SAM 3 grounding + GT grasp boxes on num_vis samples
    n = min(args.num_vis, len(ds))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        xi, yi, idx_i, rot_i, zoom_i, _, query_i = ds[i]
        xi_b = xi.unsqueeze(0).to(device)
        inst_m, sem_m, _ = sam_grounder(xi_b, [query_i])

        rgb_vis = sam_grounder._recover_rgb_01(xi_b)[0].permute(1, 2, 0).cpu().numpy()
        inst_np = inst_m[0, 0].cpu().numpy()
        sem_np = sem_m[0, 0].cpu().numpy()
        gt_pos = yi[0].squeeze().numpy()
        gtbbs = ds.get_gtbb(idx_i, rot_i, zoom_i)

        axes[i, 0].imshow(rgb_vis)
        axes[i, 0].set_title(f"RGB #{i}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb_vis)
        axes[i, 1].imshow(inst_np, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[i, 1].set_title(f"SAM3 Inst Mask\n'{str(query_i)[:38]}'", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(sem_np, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title(f"SAM3 Semantic (max={sem_np.max():.2f})", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(rgb_vis)
        axes[i, 3].imshow(gt_pos, cmap="viridis", alpha=0.35, vmin=0, vmax=1)
        for g in gtbbs:
            g.plot(axes[i, 3])
        axes[i, 3].set_title("GT Grasp Boxes + Pos", fontsize=9)
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig("smoke_test_sam3.png", dpi=150)
    print("[PASS] Saved visual verification to smoke_test_sam3.png")


if __name__ == "__main__":
    main()