# FlowGraspUnet

FlowGraspUnet is a language-driven grasp-generation method that decouples text-grounded spatial perception from grasp pose synthesis.

## Overview

The pipeline uses a frozen **SAM 3** model to ground a natural-language object description in the image and extract:

- An instance mask
- A dense semantic mask
- A 272-dimensional conditioning vector containing geometric summaries and a text embedding

A compact, trainable **FiLM-conditioned U-Net** predicts a four-channel grasp map containing:

- Position
- `cos(2*theta)`
- `sin(2*theta)`
- Gripper width

Training uses low-variance optimal-transport conditional flow matching. Inference integrates the learned velocity field with Euler or midpoint solvers.

## Main Features

- **Decoupled perception and generation:** SAM 3 is frozen; only the grasp flow U-Net is trained.
- **Compact generator:** about 9.04M trainable parameters.
- **Few-step sampling:** 16 Euler steps are used in the reported experiment.
- **Cached SAM 3 features:** per-sample cache files avoid recomputing grounding.
- **Four-channel grasp output:** position, double-angle cosine/sine, and width maps.

## Benchmark

Results use the cleaned **Grasp-Anything++ seen** split, `IoU >= 0.25`, and an angle error below 30 degrees.

| Method | Sampling | Top-1 IoU@0.25 |
|---|---:|---:|
| LGD | 50 DDIM steps | 4.00% |
| **FlowGraspUnet** | **16 Euler steps** | **21.50%** |

Additional diagnostics:

- SAM 3 mask hit rate: **37.5%**
- Mask-hit sample count: **43 / 75**
- Hardware: **NVIDIA RTX A4000**
- Training: **200 epochs**, approximately **6 hours**

The mask hit rate measures whether the ground-truth grasp peak lies inside the combined SAM 3 instance/semantic mask.

## Architecture

```text
RGB image + language prompt
          |
          v
    Frozen SAM 3
          |
          +--> instance mask
          +--> semantic mask
          +--> geometric + text vector
                         |
                         v
       Conditional FiLM U-Net
       224 -> 112 -> 56 -> 28
                         |
                         v
          4-channel grasp map
          [position, cos, sin, width]
```

The U-Net receives the noisy grasp state, RGB image, both SAM 3 masks, a masked image, coordinate maps, flow time, and SAM 3 conditioning. A bottleneck self-attention block operates at `28 x 28`.

## Installation

Python 3.9+ and a CUDA-enabled PyTorch environment are recommended.

```bash
# 1. Create clean Python 3.11 environment
conda create -n sam3_flow python=3.11 -y
conda activate sam3_flow

# 2. Install PyTorch + Transformers + vision utilities
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download SAM3
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('facebook/sam3', cache_dir='/workspace/models')
print('Saved to:', model_dir)
"

# 4. Symlink and verification
ln -sfn /workspace/models/models/facebook--sam3/snapshots/master /workspace/models/sam3
ls -lh /workspace/models/sam3

python -c "
import torch
from transformers import Sam3Model, Sam3Processor

local_path = '/workspace/models/sam3'
proc = Sam3Processor.from_pretrained(local_path, local_files_only=True)
model = Sam3Model.from_pretrained(local_path, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
print('SAM 3 loaded from local disk! VRAM (GB):', round(torch.cuda.memory_allocated() / 1e9, 2))
"
```

The dependencies in `requirements.txt` target CUDA 12.4. Install a PyTorch build compatible with your CUDA driver if needed.

## SAM 3 Checkpoint

The implementation expects a local SAM 3 model compatible with `Sam3Processor` and `Sam3Model`:

```text
/workspace/models/sam3
```

Override the path with `--sam3-path` when training or evaluating.

## Dataset

This implementation uses the cleaned **Grasp-Anything++ seen** split:

- 2,000 selected samples
- 1,800 training samples
- 200 held-out validation samples
- 224 x 224 RGB input
- Seen split
- Random seed `123`

Prepare Grasp-Anything++ in this layout:

```text
data/grasp-anything++/
├── image/
├── mask/
├── prompt/
├── positive_grasp/
└── grasp_instructions/
```

The seen split is stored in `split/grasp-anything++/test/seen.obj`. The dataset loader is implemented in `utils/data/grasp_anywhere_data.py`.

The official dataset download page is available [here](https://airvlab.github.io/grasp-anything/docs/download/).

## Training

Train the flow U-Net while caching frozen SAM 3 features:

```bash
python train_flow.py \
  --dataset grasp-anywhere \
  --dataset-path data/grasp-anything++ \
  --add-file-path data/grasp-anything++ \
  --sam3-path /workspace/models/sam3 \
  --sam3-cache-dir data/sam3_cache \
  --seen 1 \
  --split 0.9 \
  --input-size 224 \
  --base-ch 64 \
  --max-samples 2000 \
  --epochs 200 \
  --batch-size 16 \
  --grad-accum 2 \
  --num-steps 16 \
  --random-seed 123 \
  --description flowgrasp_seen
```

The default effective batch size is 32. Checkpoints are written to:

```text
logs_flow/<timestamp>_<description>/
```

The EMA model is saved as:

```text
best_epoch_*_iou_*.pt
```

To compute SAM 3 features every batch instead of using the cache, add `--no-cache-sam3`.

## Evaluation

Evaluate a FlowGraspUnet checkpoint:

```bash
python eval_flow.py \
  --checkpoint logs_flow/<run>/best_epoch_<epoch>_iou_<score>.pt \
  --sam3-path /workspace/models/sam3 \
  --dataset-path data/grasp-anything++ \
  --add-file-path data/grasp-anything++ \
  --seen 1 \
  --val-only \
  --split 0.9 \
  --random-seed 123 \
  --steps-list 1 2 4 8 16 \
  --solver euler \
  --best-of-k 5 \
  --iou-threshold 0.25
```

For the reported deterministic Top-1 result, use:

```text
--steps-list 16 --solver euler --best-of-k 1
```

## Visualization

Generate a qualitative PDF from the validation split:

```bash
python visualize_flow.py \
  --checkpoint logs_flow/<run>/best_epoch_<epoch>_iou_<score>.pt \
  --num-samples 3 \
  --only-pass \
  --save figs/qualitative.pdf
```

The script restores dataset and model settings from `commandline_args.json` in the checkpoint directory when available.

## Repository Layout

```text
train_flow.py                 FlowGraspUnet training entry point
eval_flow.py                  Few-step NFE and Top-1/Best-of-K evaluation
visualize_flow.py             Qualitative prediction visualization
flow/rectified_flow.py        OT conditional flow-matching wrapper
models/flow_grasp_net/        SAM 3 grounder and FiLM U-Net
utils/data/                   Grasp dataset loaders
split/grasp-anything++/       Seen and unseen split files
script/                       Existing LGD baseline evaluation scripts
trained-models/              Pretrained LGD baseline weights
```

## Baseline

The original LGD implementation and language-grasp baselines remain in this fork for comparison. Key baseline entry points include `train_network_diffusion.py` and `evaluate_diffusion.py`.

## Acknowledgments

This codebase builds on [Grasp-Anything](https://github.com/andvg3/Grasp-Anything) and the [Language-driven Grasp Detection repository](https://github.com/Fsoft-AIC/LGD).
