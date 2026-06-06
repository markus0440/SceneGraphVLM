# SceneGraphVLM: Dynamic Scene Graph Generation from Video with Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.13667-b31b1b.svg)](https://arxiv.org/abs/2605.13667)
[![Weights](https://img.shields.io/badge/Weights-Zenodo-blue.svg)](https://zenodo.org/records/20511274?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc4ODcwYzg5LWRjM2UtNGVlMi1iMGNmLWY4NjcwMzJhNWE0NCIsImRhdGEiOnt9LCJyYW5kb20iOiI3OWFhY2ZhZDMwZWUwNTY1MGIyMzUyNGNkZTk1NTA4ZCJ9.CkJPxC6vjkmERWox2qEi8x9JtwDXs9McVy6G6m13wmwyUFe8orHPGP0KJvCAJWFA2FAZIkOZ-orIxf-X7eYhjw)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SceneGraphVLM is a compact end-to-end framework for generating scene graphs from images and videos with small vision-language models. Instead of using a multi-stage detector-relation pipeline, the model directly emits a structured graph in a token-efficient **TOON** text format.

The project combines:

- TOON-based scene graph serialization for shorter structured outputs;
- SFT for schema following and dataset adaptation;
- GRPO fine-tuning with hallucination-aware graph rewards;
- optional previous-frame graph prompting for video scene graph generation;
- vLLM-accelerated inference and evaluation utilities.

The current implementation supports experiments on **PSG**, **PVSG**, and **Action Genome**.

<div align="center">

![SceneGraphVLM inference over video with optional previous-frame TOON](docs/figures/fig1_pipeline.png)

</div>

<div align="center">

![SFT + GRPO learning scheme with graph-centric rewards](docs/figures/fig2_learning_scheme.png)

</div>

---

## Project status

The core preprocessing, training, reward, inference, evaluation, and visualization scripts are included, while some components are still being cleaned up for full one-command reproducibility.

| Component | Status |
|---|---|
| PSG / PVSG / AG annotation preprocessing | Available |
| TOON annotation conversion | Available |
| SFT training scripts | Available |
| GRPO reward plugins and launch scripts | Available |
| Qwen / vLLM inference scripts | Available |
| PSG / PVSG metric evaluation | Available |
| Visualization and video demo utilities | Available |
| Classical SGG benchmark adapter | Available for AG |
| Docker image | Available via Zenodo |
| Pretrained checkpoints | Available via Zenodo |
| Full environment lockfile | In process |
| One-command full paper reproduction | In process |
| CI tests / smoke tests | In process |

---

## Repository layout

```text
SceneGraphVLM/
├── datasets/                 # annotation docs, generated JSONL location, dataset tools
├── docs/                     # figures and paper-related assets
├── envs/                     # preprocessing environment requirements and install scripts
├── grpo/                     # GRPO reward plugins and launch scripts
├── metrics/                  # inference, evaluation, OpenRouter and Qwen judge utilities
├── sft/                      # supervised fine-tuning scripts
├── utils/                    # frame selection and annotation cleaning utilities
└── visualization/            # notebooks, demo rendering, field deployment scripts
```

The training scripts intentionally keep absolute `/workspace/...` paths because the expected workflow is a unified Docker/container setup with the repository mounted at `/workspace`.

Detailed documentation:

| Topic | Document |
|---|---|
| Metrics and evaluation | [`metrics/metrics.md`](metrics/metrics.md) |
| Docker image and checkpoints | [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) |
| Visualization | [`visualization/vis.md`](visualization/vis.md) |
| SFT training | [`sft/SFT_README.md`](sft/SFT_README.md) |
| GRPO training | [`grpo/README.md`](grpo/README.md) |
| Annotation variants | [`datasets/ANNOTATION_VARIANTS.md`](datasets/ANNOTATION_VARIANTS.md) |
| Dataset preprocessing environment | [`envs/DATA_PREP_README.md`](envs/DATA_PREP_README.md) |
| PVSG data and exports | [`datasets/annotations/PVSG_annot/PVSG_README.md`](datasets/annotations/PVSG_annot/PVSG_README.md) |
| PSG data | [`datasets/annotations/PSG_annot/PSG_README.md`](datasets/annotations/PSG_annot/PSG_README.md) |
| Action Genome data | [`datasets/annotations/AG_annot/AG_README.md`](datasets/annotations/AG_annot/AG_README.md) |
| BaseAnnot PVSG filtering | [`utils/BaseAnnot/BA.md`](utils/BaseAnnot/BA.md) |
| PSFR key-frame selection | [`utils/PSFR/PSFR.md`](utils/PSFR/PSFR.md) |
| MaxInfo key-frame selection | [`utils/MaxInfo/MI.md`](utils/MaxInfo/MI.md) |
| Annotation cleaning | [`utils/annotations_clean/CLEAN.md`](utils/annotations_clean/CLEAN.md) |

---

## Installation

The project uses two practical environments:

1. a lightweight local environment for dataset preprocessing;
2. a GPU training/inference environment with PyTorch, MSwift, vLLM, FlashAttention, and DeepSpeed.

### Dataset preprocessing environment

From the repository root:

```bash
bash envs/sh_scripts/install_dataset_preprocessing_env.sh
```

See [`envs/DATA_PREP_README.md`](envs/DATA_PREP_README.md) for details.

### Training / inference container

Download and load the released Docker image before running SFT, GRPO, or local
Qwen/vLLM inference. See [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).

SFT and GRPO are expected to run inside the project training container with the repository mounted at `/workspace`:

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm \
  --shm-size=16g \
  -v /path/to/SceneGraphVLM:/workspace \
  qwen-grpo:cu130
```

---

## Data preparation

The repository does not redistribute the original datasets. Download PSG, PVSG, and Action Genome from their official sources and place them according to the dataset-specific README files.

Expected high-level layout after data setup:

```text
datasets/
├── annotations/
│   ├── PSG_annot/
│   ├── PVSG_annot/
│   └── AG_annot/
├── frames/
└── data_playground/          # generated JSONL files; ignored by git
```

After the raw data is in place, build all annotation variants:

```bash
python datasets/tools/prepare_all_annotations.py --overwrite
```

This generates SFT, GRPO, evaluation, GT-prompt, GEN-prompt, and no-previous-graph variants under `datasets/data_playground/`.

---

## Supervised fine-tuning

SFT scripts are located in `sft/Qwen3.5/train_scripts/` and are designed for MSwift.

Run from inside the container:

```bash
cd /workspace

# PSG
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sft/Qwen3.5/train_scripts/run_psg.sh

# PVSG, BaseAnnot split
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sft/Qwen3.5/train_scripts/run_pvsg_base_annot.sh

# PVSG, PSFR split
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sft/Qwen3.5/train_scripts/run_pvsg_psfr.sh

# Action Genome
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sft/Qwen3.5/train_scripts/run_ag.sh
```

Outputs are written under:

```text
sft/Qwen3.5/work_dirs/
```

See [`sft/SFT_README.md`](sft/SFT_README.md) for the full list of scripts, arguments, and expected JSONL files.

---

## GRPO fine-tuning

GRPO scripts are located in `grpo/swift/scripts/`. Reward plugins are located in `grpo/swift/rewards/`.

Example runs:

```bash
cd /workspace

# PVSG
COMET_API_KEY=... bash grpo/swift/scripts/run_pvsg_grpo_qwen3_5.sh

# PSG
COMET_API_KEY=... bash grpo/swift/scripts/run_psg_grpo_qwen3_5.sh

# Action Genome
COMET_API_KEY=... bash grpo/swift/scripts/run_ag_grpo_qwen3_5.sh
```

To disable external logging:

```bash
REPORT_TO=none bash grpo/swift/scripts/run_pvsg_grpo_qwen3_5.sh
```

GRPO uses SFT checkpoints as warm starts. By default, scripts expect local checkpoints under:

```text
/workspace/grpo/models/work_dirs/
```

See [`grpo/README.md`](grpo/README.md) for colocated vLLM mode, server mode, reward plugins, resume options, and expected input files.

---

## Inference and Metrics

Released checkpoints can be used for local Qwen/vLLM inference after the dataset
JSONL files are prepared.

Example PVSG GEN-prompt inference on the regular prepared `test.jsonl`:

```bash
cd /workspace

CUDA_VISIBLE_DEVICES=0 python metrics/qwen-bench/infer/GEN-prompt/infer_swift_gen_prompt.py \
  --model checkpoints/PVSG \
  --test-jsonl datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --output-dir metrics/results/checkpoints-inference/released/PVSG-GEN-prompt \
  --run-name SceneGraphVLM-PVSG-GEN \
  --infer-backend vllm \
  --batch-size 64 \
  --max-new-tokens 2048 \
  --temperature 0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --response-prefix $'<answer>\n' \
  --prev-source model
```

Then compute PSG/PVSG scene-graph metrics.

```bash
CUDA_VISIBLE_DEVICES=0 python metrics/qwen-bench/eval/eval_sgg_metrics_with_qwen.py \
  --pred-jsonl metrics/results/checkpoints-inference/released/PVSG-GEN-prompt/SceneGraphVLM-PVSG-GEN.jsonl \
  --output-dir metrics/results/checkpoints-metrics/released/PVSG-GEN-prompt \
  --output-name SceneGraphVLM-PVSG-GEN-metrics.json \
  --iou-thr 0.5 \
  --strict-only
```

See [`metrics/metrics.md`](metrics/metrics.md) for the full quick start,
Action Genome metrics, OpenRouter evaluation, optional Qwen judge details,
inference options, and result layouts.

---

## Visualization

Visualization utilities are located in `visualization/video_demo_results/scripts/`.

Example ground-truth video rendering:

```bash
python visualization/video_demo_results/scripts/build_gt_video.py \
  --annotation-file datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --video-name 0004_11566980553 \
  --images-root . \
  --output-dir visualization/video_demo_results/videos_output/GT \
  --fps 5
```

Field/deployment-style GEN-prompt demo from a folder of ordered frames:

```bash
python visualization/video_demo_results/scripts/field_gen_deploy.py \
  --frames-dir /path/to/ordered_frames \
  --model /path/to/checkpoint \
  --infer-backend vllm \
  --fps 10 \
  --cuda-visible-devices 0
```

See [`visualization/vis.md`](visualization/vis.md) for video rendering, common demo layout, and notebook details.

---

## Third-party assets and datasets

This repository does not redistribute original PSG, PVSG, Action Genome data, pretrained foundation models, or third-party checkpoints. They are governed by their own licenses and terms of use.

Please refer to the original sources for:

- PSG;
- PVSG;
- Action Genome;
- Qwen / Qwen3.5;
- MSwift;
- vLLM;
- any other third-party models, datasets, and libraries used in experiments.

Generated datasets, checkpoints, model outputs, logs, media files, and local credentials should not be committed to the repository.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{makarov2026scenegraphvlm,
  title={SceneGraphVLM: Dynamic Scene Graph Generation from Video with Vision-Language Models},
  author={Makarov, Vladislav and Gizetdinov, Mark and Yudin, Dmitry},
  journal={arXiv preprint arXiv:2605.13667},
  year={2026}
}
```

## License

The code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).

The MIT License applies only to the code provided in this repository. Datasets, pretrained models, checkpoints, and third-party dependencies are subject to their own licenses and terms of use.
