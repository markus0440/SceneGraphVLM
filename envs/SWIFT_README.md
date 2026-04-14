# MSwift (Swift) environment for SFT

This folder holds a small **conda + pip** installer used across **SceneGraphVLM** dataset READMEs (PSG, AG, PVSG, …) for **supervised fine-tuning** with **Qwen 3.x** via [MSwift](https://github.com/modelscope/swift).

| Resource | Link |
|----------|------|
| **MSwift documentation** | [swift.readthedocs.io](https://swift.readthedocs.io/en/latest/) |
| **Qwen3 / Qwen3.5 SFT best practices** | [Qwen3.5 Best Practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html) |

## Prerequisites

Choose **one** of the two install methods below:

| Method | Requires |
|--------|----------|
| **Conda** (option A) | [conda](https://docs.conda.io/) on `PATH` with `conda init` applied |
| **Docker** (option B) | [Docker](https://docs.docker.com/get-docker/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

## Option A — Conda install

From the **SceneGraphVLM repository root**:

```bash
bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh
```

What it does:

- Creates a conda env named **`swift_qwen_3_5_sft`** (Python **3.11**) if it does not exist.
- Activates that env and runs **`pip install -r envs/requirements_files/requirements-swift-qwen-3-5-sft.txt`** (with `--no-build-isolation` as in the script).

Then activate whenever you work on SFT:

```bash
conda activate swift_qwen_3_5_sft
```

### Optional: custom env name

```bash
CONDA_ENV=my_swift_sft bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh
conda activate my_swift_sft
```

## Option B — Docker

Builds an **environment-only** image (CUDA 12.6 devel + Python 3.11 + all pip packages). No project code or data is baked in — the repo is bind-mounted at runtime.

### Build

```bash
bash envs/sh_scripts/build_docker_swift_qwen_3_5_sft.sh
```

Override CUDA / PyTorch version via env variables:

```bash
CUDA_TAG=12.8.1-devel-ubuntu24.04 TORCH_INDEX=cu128 \
  bash envs/sh_scripts/build_docker_swift_qwen_3_5_sft.sh
```

All build-arg overrides: `CUDA_TAG`, `TORCH_INDEX`, `MAX_JOBS`, `TORCH_CUDA_ARCH_LIST`, `IMAGE_TAG`.

### Run

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd)":/workspace \
  -it sgvlm-sft-env bash
```

Inside the container the full repo tree is available at `/workspace`, so training scripts work as usual (e.g. `bash sft/Qwen3.5/train_scripts/run_psg.sh`).

To pass Comet / other secrets, add `--env-file`:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd)":/workspace \
  --env-file sft/Qwen3.5/.comet_env \
  -it sgvlm-sft-env bash
```

## Other models

The scripts above are tailored for **Qwen 3.x SFT** in this repo. If you want to fine-tune **other model families**, do **not** rely on that requirements pin alone: follow the **[MSwift documentation](https://swift.readthedocs.io/en/latest/)** and install **MSwift the way upstream describes** (from the [Swift repo](https://github.com/modelscope/swift) / full `pip` install, extra deps, and CUDA bits for your stack). Use the official docs and best-practice pages for your target model instead of this minimal env file.

## Layout

```text
envs/
├── SWIFT_README.md                           # this file
├── sh_scripts/
│   ├── install_swift_qwen_3_5_sft.sh         # conda installer
│   ├── Dockerfile.swift_qwen_3_5_sft         # Docker image definition
│   └── build_docker_swift_qwen_3_5_sft.sh    # Docker build helper
└── requirements_files/
    └── requirements-swift-qwen-3-5-sft.txt
```

For training options, argument patterns, and Qwen3.5-specific tips, follow the **Best Practice** page linked above alongside the main Swift docs.

---

## Related documentation

- [SceneGraphVLM project README](../README.md)  
- [SFT training](../sft/SFT_README.md)  
- [Metrics & evaluation](../metrics/metrics.md)
