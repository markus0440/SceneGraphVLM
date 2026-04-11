# MSwift (Swift) environment for SFT

This folder holds a small **conda + pip** installer used across **SceneGraphVLM** dataset READMEs (PSG, AG, PVSG, …) for **supervised fine-tuning** with **Qwen 3.x** via [MSwift](https://github.com/modelscope/swift).

| Resource | Link |
|----------|------|
| **MSwift documentation** | [swift.readthedocs.io](https://swift.readthedocs.io/en/latest/) |
| **Qwen3 / Qwen3.5 SFT best practices** | [Qwen3.5 Best Practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html) |

## Prerequisites

- [**conda**](https://docs.conda.io/) (Miniconda or Anaconda) on your `PATH`, with `conda init` applied for your shell so `conda activate` works in scripts.

## Install (recommended)

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

## Other models

The script above is tailored for **Qwen 3.x SFT** in this repo. If you want to fine-tune **other model families**, do **not** rely on that requirements pin alone: follow the **[MSwift documentation](https://swift.readthedocs.io/en/latest/)** and install **MSwift the way upstream describes** (from the [Swift repo](https://github.com/modelscope/swift) / full `pip` install, extra deps, and CUDA bits for your stack). Use the official docs and best-practice pages for your target model instead of this minimal env file.

## Layout

```text
envs/
├── SWIFT_README.md                 # this file
├── sh_scripts/
│   └── install_swift_qwen_3_5_sft.sh
└── requirements_files/
    └── requirements-swift-qwen-3-5-sft.txt
```

For training options, argument patterns, and Qwen3.5-specific tips, follow the **Best Practice** page linked above alongside the main Swift docs.
