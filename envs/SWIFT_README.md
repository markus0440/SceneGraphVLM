# MSwift (Swift) conda environments

Conda and Docker installers for **SFT** and **speed-test** backends in SceneGraphVLM.

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

| Conda env | Backend | Install |
|-----------|---------|---------|
| `swift_qwen_3_5_sft` | **vLLM + HF** (Qwen, InternVL, DeepSeek/Ovis, SmolVLM) | `bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh` |
| `swift_qwen_sglang` | **SGLang** | `bash envs/sh_scripts/install_swift_qwen_sglang.sh` |
| `swift_qwen_lmdeploy` | **LMDeploy** (Qwen3-VL, Qwen3.5 native pipeline) | `bash envs/sh_scripts/install_swift_qwen_lmdeploy.sh` |

Why separate envs? **vLLM**, **SGLang**, and **LMDeploy** pin incompatible `torch` / `transformers` / backend versions. Do not merge into one env.

### 1. Primary: vLLM + HF (`swift_qwen_3_5_sft`)

From the **SceneGraphVLM repository root**:

```bash
bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh
conda activate swift_qwen_3_5_sft
```

Creates Python 3.11 env with ms-swift, vLLM, HF. Covers Qwen VL, InternVL, DeepSeek-VL2, Ovis, SmolVLM.

**RTX 5060 Ti / CUDA 12.8+:** if import fails, retry with:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
  bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh
```

Custom env name: `CONDA_ENV=my_swift_sft bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh`

### 2. Optional: SGLang (`swift_qwen_sglang`)

```bash
bash envs/sh_scripts/install_swift_qwen_sglang.sh
conda activate swift_qwen_sglang
```

**RTX 50xx (sm_120):** if `sglang-kernel` fails to load, build from source:

```bash
bash envs/sh_scripts/build_sglang_kernel_sm120.sh
conda activate swift_qwen_sglang
```

If smoke fails with **PyTorch/torchvision CUDA mismatch**, realign wheels inside the env (see install script output).

### 3. Optional: LMDeploy (`swift_qwen_lmdeploy`)

Native `lmdeploy.pipeline` for **Qwen3-VL** and **Qwen3.5** only. Qwen2-VL / InternVL on LMDeploy are not supported in this stack. Use **vLLM** from `swift_qwen_3_5_sft` instead.

```bash
bash envs/sh_scripts/install_swift_qwen_lmdeploy.sh
conda activate swift_qwen_lmdeploy
```

On **RTX 5060 Ti (sm_120)** the speed-test code applies SMEM patches automatically.

## Option B — Docker

Builds an **environment-only** image (CUDA 12.6 devel + Python 3.11 + all pip packages). No project code or data is baked in. The repo is bind-mounted at runtime.

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

The scripts above are tailored for **Qwen 3.x SFT** in this repo. For **other model families**, follow the **[MSwift documentation](https://swift.readthedocs.io/en/latest/)** and upstream install instructions instead of copying these pins blindly.

## Verify (speed test)

```bash
conda activate swift_qwen_3_5_sft
export CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024
python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py \
  --model Qwen/Qwen3.5-0.8B \
  --model-display-name Qwen3.5-0.8B \
  --infer-backend vllm \
  --force
```

Full backend matrices: [metrics/statistics/speed-test-infer/README.md](../metrics/statistics/speed-test-infer/README.md).

## Layout

```text
envs/
├── SWIFT_README.md
├── sh_scripts/
│   ├── _common.sh
│   ├── install_swift_qwen_3_5_sft.sh
│   ├── install_swift_qwen_sglang.sh
│   ├── install_swift_qwen_lmdeploy.sh
│   ├── build_sglang_kernel_sm120.sh      # optional, sm_120 only
│   ├── Dockerfile.swift_qwen_3_5_sft
│   └── build_docker_swift_qwen_3_5_sft.sh
└── requirements_files/
    ├── requirements-qwen-vllm.txt
    ├── requirements-qwen-sglang.txt
    └── requirements-qwen-lmdeploy.txt
```

## Troubleshooting

**InternVL / DeepSeek HF fails** (`configuration_*.py`, `no generate`): stale remote-code cache or transformers 5.x. Reinstall primary env. If needed:

```bash
conda activate swift_qwen_3_5_sft
pip install 'transformers>=4.56.2,<5.0' timm
rm -rf ~/.cache/huggingface/modules/transformers_modules
```

**HF-only N/A after fix:** `InternVL2.5-2B`, `Ovis2.5-2B` — use vLLM for those models.

## Related

- [SceneGraphVLM project README](../README.md)
- [SFT training](../sft/SFT_README.md)
- [Metrics & evaluation](../metrics/metrics.md)
