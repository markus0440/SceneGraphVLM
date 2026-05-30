# Dataset Preprocessing Environment

This folder contains the lightweight local environment for preparing SceneGraphVLM datasets and annotation variants. It is intentionally separate from the Swift / Qwen / GRPO training runtime, which is expected to run inside a Docker container mounted at `/workspace`.

## Prerequisites

- [conda](https://docs.conda.io/) on `PATH`
- `ffmpeg` on `PATH` for AG frame extraction and video utilities

## Install

From the SceneGraphVLM repository root:

```bash
bash envs/sh_scripts/install_dataset_preprocessing_env.sh
conda activate scenegraphvlm_data_prep
```

What it installs:

- core annotation dependencies: `tqdm`, `Pillow`, `numpy`
- PSG Hugging Face import support: `datasets`
- PVSG PSFR support: `opencv-python`, `matplotlib`
- PVSG MaxInfo support: `torch`, `transformers`, `maxvolpy`

It does not install `ms-swift`, `vllm`, `deepspeed`, FlashAttention, Qwen utilities, or Comet. Those belong to the training / evaluation container.

## Custom Env Name

```bash
CONDA_ENV=my_sgvlm_data_prep bash envs/sh_scripts/install_dataset_preprocessing_env.sh
conda activate my_sgvlm_data_prep
```

## Layout

```text
envs/
├── DATA_PREP_README.md
├── sh_scripts/
│   └── install_dataset_preprocessing_env.sh
└── requirements_files/
    └── requirements-dataset-preprocessing.txt
```

## Related Documentation

- [Annotation variants](../datasets/ANNOTATION_VARIANTS.md)
- [AG data](../datasets/annotations/AG_annot/AG_README.md)
- [PSG data](../datasets/annotations/PSG_annot/PSG_README.md)
- [PVSG data](../datasets/annotations/PVSG_annot/PVSG_README.md)
