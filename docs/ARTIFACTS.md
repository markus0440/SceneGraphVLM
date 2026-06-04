# Docker Image and Checkpoints

This document explains how to download the released SceneGraphVLM runtime
artifacts, load the Docker image, place the checkpoints, and run a quick
checkpoint sanity check.

The dataset preparation pipeline is documented separately in
[`datasets/ANNOTATION_VARIANTS.md`](../datasets/ANNOTATION_VARIANTS.md).

## Download

Download the artifact files from the Zenodo record:

<https://zenodo.org/records/20511274?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc4ODcwYzg5LWRjM2UtNGVlMi1iMGNmLWY4NjcwMzJhNWE0NCIsImRhdGEiOnt9LCJyYW5kb20iOiI3OWFhY2ZhZDMwZWUwNTY1MGIyMzUyNGNkZTk1NTA4ZCJ9.CkJPxC6vjkmERWox2qEi8x9JtwDXs9McVy6G6m13wmwyUFe8orHPGP0KJvCAJWFA2FAZIkOZ-orIxf-X7eYhjw>

The record provides:

| File | Purpose | MD5 |
|---|---|---|
| `qwen-grpo-cu130.tar.gz` | Docker image archive tagged as `qwen-grpo:cu130` | `4305c12c93847d5a7a8d55fbd730dec6` |
| `checkpoints.zip` | Released `AG`, `PSG`, and `PVSG` checkpoints | `9e3c8f6ccce2bf632894053d74c74fc6` |

## Install the Checkpoints

Place `checkpoints.zip` anywhere convenient, then unzip it from the repository
root:

```bash
cd /path/to/SceneGraphVLM
unzip /path/to/checkpoints.zip
```

After extraction, the expected layout is:

```text
SceneGraphVLM/
├── checkpoints/
│   ├── AG/
│   ├── PSG/
│   └── PVSG/
```

Each checkpoint directory is a Hugging Face style model directory with
`model.safetensors`, `config.json`, tokenizer files, and processor files.

## Load the Docker Image

Load the image archive:

```bash
docker load -i /path/to/qwen-grpo-cu130.tar.gz
```

Start an interactive container with the repository mounted at `/workspace`:

```bash
cd /path/to/SceneGraphVLM

docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm \
  --shm-size=16g \
  -v "$(pwd)":/workspace \
  qwen-grpo:cu130
```

The training and inference scripts intentionally use `/workspace/...` paths, so
keep the repository mounted at `/workspace` unless you also update those paths.

## Use the Released Checkpoints for Inference

Once the dataset JSONL files are prepared, pass a checkpoint directory to the
existing inference scripts. For example:

```bash
python metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py \
  --model checkpoints/PVSG \ #checkpoints/PSG
  --test-jsonl datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --output-dir metrics/results/checkpoints-inference/released/PVSG-GT-prompt \
  --run-name SceneGraphVLM-PVSG-GT \
  --infer-backend vllm \
  --batch-size 64 \
  --max-new-tokens 2048 \
  --temperature 0.0
```

