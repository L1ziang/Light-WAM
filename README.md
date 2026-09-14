# Light-WAM - CoRL 2026

[![arXiv](https://img.shields.io/badge/arXiv-2606.08242-b31b1b.svg)](https://arxiv.org/abs/2606.08242)

Codebase for **Light-WAM: Efficient World Action Models with State-Fusion Action Decoding**. This repository provides training and evaluation pipelines for **Light-WAM** on **LIBERO** and **RoboTwin2.0**.

## What Is Light-WAM?

<p align="center">
  <img src="./assets/overview.png" alt="Light-WAM overview" width="80%">
</p>

Light-WAM is a lightweight World Action Model for robot manipulation, centered on:

- Wan2.1-T2V-1.3B as a frozen video backbone
- lightweight adapters and LoRA updates
- future-video supervision in downsampled latent space
- learned-query pooling over adapted states
- StateFusionActionExpert for action-chunk decoding

## Repository Layout

```text
LightWAM/
├── configs/                 # Training and evaluation configs
├── scripts/                 # Main CLI entrypoints
├── experiments/             # LIBERO, RoboTwin2.0, and real-robot
├── src/lightwam/            # Model and dataset code
├── third_party/             # Simulation dependencies adapted from Fast-WAM
├── checkpoints/             # Wan weights and released checkpoints
├── data/                    # Datasets and caches
└── runs/                    # Training outputs
```

## Environment Setup

```bash
conda create -n lightwam python=3.10 -y
conda activate lightwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

#### FFmpeg Libraries for Precompute

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev
```

## Backbone Preparation

Set the Wan checkpoint directory first:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Hugging Face download command:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B   --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

## Data and Precompute

Raw datasets used by this repo come from Fast-WAM:

- LIBERO: [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
- RoboTwin2.0: [yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

Expected local layout:

```text
data/
├── libero_mujoco3.3.2/
│   ├── libero_10_no_noops_lerobot/
│   ├── libero_goal_no_noops_lerobot/
│   ├── libero_object_no_noops_lerobot/
│   └── libero_spatial_no_noops_lerobot/
└── robotwin2.0/
    └── robotwin2.0/
        ├── data/
        ├── meta/
        └── videos/
```

Offline cache release:

- offline cache repo: [l1ziang/lightwam-offline-cache](https://huggingface.co/datasets/l1ziang/lightwam-offline-cache)
- includes: LIBERO latent caches, RoboTwin2.0 latent caches, and LIBERO text cache

To restore locally:
```bash
cat robotwin_3cam384_sharded.tar.part-* | tar -xf -
```
Expected local layout:

```text
data/
├── latent_cache_Wan2.1-T2V-1.3B/
│   ├── libero_spatial_2cam224/
│   ├── libero_object_2cam224/
│   ├── libero_goal_2cam224/
│   ├── libero_10_2cam224/
│   └── robotwin_3cam384_sharded/
└── text_embeds_cache/
    ├── libero/
    └── robotwin/   # generate locally with the text-only cache command below
```

Precompute commands:

```bash
LIBERO_SUITE=spatial bash scripts/precompute_libero.sh
LIBERO_SUITE=object  bash scripts/precompute_libero.sh
LIBERO_SUITE=goal    bash scripts/precompute_libero.sh
LIBERO_SUITE=10      bash scripts/precompute_libero.sh
bash scripts/precompute_robotwin.sh
```
These commands generate text caches and offline future-video latent caches

Text-only cache commands:

```bash
LIBERO_SUITE=spatial RUN_TEXT=true RUN_VIDEO=false bash scripts/precompute_libero.sh
RUN_TEXT=true RUN_VIDEO=false bash scripts/precompute_robotwin.sh
```

## Training

```bash
bash scripts/train_libero_spatial.sh
bash scripts/train_libero_object.sh
bash scripts/train_libero_goal.sh
bash scripts/train_libero_10.sh
bash scripts/train_robotwin.sh
```

## Evaluation

Released checkpoints:

- checkpoint repo: [l1ziang/lightwam-checkpoints](https://huggingface.co/l1ziang/lightwam-checkpoints)
- includes: LIBERO and RoboTwin2.0 released checkpoints

Evaluation environment configs:

```bash
conda env create -f ./scripts/libero_env.full.yml
conda activate lightwam-libero-eval
pip install -e . --no-deps

conda env create -f ./scripts/robotwin_env.full.yml
conda activate lightwam-robotwin-eval
pip install -e . --no-deps
```

For LIBERO evaluation, install the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) package first, then set:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
```

Evaluation commands:

```bash
CKPT=/path/to/checkpoints/weights/xxxx.pt bash scripts/eval_libero.sh
CKPT=/path/to/checkpoints/weights/xxxx.pt bash scripts/eval_robotwin.sh
```

## Citation

```bibtex
@misc{li2026lightwam,
  title         = {Light-WAM: Efficient World Action Models with State-Fusion Action Decoding},
  author        = {Ziang Li and Dongzhou Cheng and Yibin Wang and Shiyue Wang and Xiaoyang Xu and Lingxuan Weng and Juan Wang and Jiaqi Wang},
  year          = {2026},
  eprint        = {2606.08242},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.08242}
}
```

## Acknowledgements

This codebase is primarily based on **Fast-WAM**. We thank the Fast-WAM project for open-sourcing a strong and practical foundation.
