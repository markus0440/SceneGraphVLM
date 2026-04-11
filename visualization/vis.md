# Visualization

This directory contains **notebooks** and **offline tooling** to render scene graphs on images and to assemble **demo videos** for qualitative inspection (ground truth, model predictions, and side‑by‑side comparisons).

---

## Demo (PVSG)

The clip below is a **GIF** preview of the full demo MP4 (GitHub-friendly inline preview). Layout: **top** — ground-truth scene graph per frame; **bottom** — **GT-prompt** prediction (left) vs **GEN-prompt** prediction (right).

![PVSG demo: GT on top; GT-prompt vs GEN-prompt below](assets/pvsg_demo_common.gif)

---

## Directory layout

```text
visualization/
├── vis.md                          # this file
├── assets/                         # static media for documentation (e.g. demo GIF)
├── notebooks/
│   └── scene_graph_unified.ipynb   # exploratory / unified scene-graph visualization
└── video_demo_results/
    ├── scripts/                    # CLI tools (see below)
    ├── predicts_and_metrics/       # cached JSONL + per-frame metrics for demo videos
    └── videos_output/              # rendered MP4s (GT, GT_prompt, GEN_prompt, COMMON)
```

---

## `video_demo_results/scripts/`

All paths below are relative to the **repository root** unless stated otherwise. Several scripts expect **`ffmpeg`** on `PATH` for H.264 MP4 export.

### `build_gt_video.py`

**Role:** Renders **ground-truth** TOON onto each frame and encodes an MP4. Also hosts **shared helpers** (TOON parsing, drawing with Matplotlib, JSONL loading, PNG sequence → MP4 via ffmpeg) imported by the other scripts.

**Typical invocation:**

```bash
python visualization/video_demo_results/scripts/build_gt_video.py \
  --annotation-file datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --video-name 0004_11566980553 \
  --images-root . \
  --output-dir visualization/video_demo_results/videos_output/GT \
  --fps 5
```

**Outputs:** by default `visualization/video_demo_results/videos_output/GT/<video_name>/<video_name>_GT.mp4` (override with `--output-filename`, `--output-parent`).

**Inputs:** Swift-style JSONL (`messages` + `images`, GT in assistant) or sharegpt-style rows; `--video-name` is a **substring** of the frame path used to filter one video.

---

### `build_pred_video.py`

**Role:** End-to-end pipeline for **one PVSG video id**: (1) filter JSONL to that video, (2) run **`infer_swift_gt_prompt.py`** or **`infer_swift_gen_prompt.py`**, (3) run **`eval_sgg_metrics_with_qwen.py`** with **`--per-sample-jsonl`**, (4) render an MP4 with **predictions** and a **per-frame Qwen metric strip** (object/rel F1-style fields from eval). Output video frame rate is set by **`--fps`** (default `10.0`; forwarded to the PNG→MP4 encoder).

**Dependencies:** same stack as `metrics/qwen-bench` (ms-swift, vLLM or transformers for inference; scipy + vLLM judge for eval).

**Example (GT-prompt/GEN-prompt):**

```bash
python visualization/video_demo_results/scripts/build_pred_video.py \
  --prompt-mode GT_prompt \
  --annotation-file datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --video-name 0004_11566980553 \
  --model /path/to/checkpoint \
  --cuda-visible-devices 0 \
  --batch-size 8 \
  --gpu-memory-utilization 0.45 \
  --fps 10
```

**Artifacts (defaults):** under `visualization/video_demo_results/predicts_and_metrics/<GT_prompt|GEN_prompt>/<video_name>/`:

- `{run_name}.jsonl` — raw predictions  
- `frames_metrics.jsonl` — per-frame rows + `metrics` dict  
- `general_metrics.json` — aggregate eval summary  

Video: `visualization/video_demo_results/videos_output/<prompt_mode>/<video_name>/<video_name>_<prompt_mode>.mp4`.

Use **`--skip-infer`** / **`--skip-eval`** to reuse existing artifacts; **`--force-infer`** forwards `--force` to the infer script.

---

### `build_common_video.py`

**Role:** Merges **three** MP4s into a single **stacked** layout: **top** = GT clip; **bottom** = GT-prompt (left) + GEN-prompt (right). Synchronized by frame index; duration follows the **shortest** input stream.

**Convention-based invocation** (resolves paths under `videos_output/`):

```bash
cd visualization/video_demo_results/scripts
python build_common_video.py 0004_11566980553
```

Expects:

- `videos_output/GT/<name>/<name>_GT.mp4`  
- `videos_output/GT_prompt/<name>/<name>_GT_prompt.mp4`  
- `videos_output/GEN_prompt/<name>/<name>_GEN_prompt.mp4`  

**Output:** `videos_output/COMMON/<name>/<name>_COMMON.mp4`.

**Explicit mode:** supply `--top`, `--bottom-left`, `--bottom-right`, `-o` instead of `VIDEO_NAME`. Canvas size defaults: `--width 1920`, `--top-height 720`. Requires **ffmpeg**.

---

### `field_gen_deploy.py`

**Role:** **Field / deployment-style** demo: accepts a **raw input video** or an **ordered directory of frames**, builds PVSG **GEN-style** prompts (embedded template matching PVSG training), runs **temporal chaining** with ms-swift (**does not** import `infer_swift_gen_prompt.py`), writes JSONL and optionally an MP4 overlay.

**Outputs (defaults):**

- `visualization/video_demo_results/predicts_and_metrics/field_deploy/<stem>/<run_name>.jsonl`  
- `visualization/video_demo_results/videos_output/field_deploy/<stem>/<run_name>.mp4`  

**Example:**

```bash
python visualization/video_demo_results/scripts/field_gen_deploy.py \
  --video /path/to/clip.mp4 \
  --model /path/to/checkpoint \
  --infer-backend vllm \
  --fps 5 \
  --cuda-visible-devices 0
```

Use **`--frames-dir`** instead of **`--video`** for an image sequence. **`--no-video`** writes JSONL only. **`--results-root`** overrides the default demo root (`video_demo_results`).

---

## `notebooks/scene_graph_unified.ipynb`

Jupyter notebook for **interactive** exploration of scene-graph visualizations (unified workflow over samples in the notebook). Open with JupyterLab / VS Code; kernel must provide the same scientific stack as `build_gt_video.py` (e.g. `matplotlib`, `Pillow`, `numpy`).

---

## Dependencies (summary)

| Component | Purpose |
|-----------|---------|
| **ffmpeg** | MP4 encoding in `build_gt_video.py` and filter graph in `build_common_video.py` |
| **Python:** `numpy`, `Pillow`, `matplotlib` | Drawing boxes, arrows, labels; frame compositing |
| **ms-swift + torch + (vLLM or transformers)** | `build_pred_video.py`, `field_gen_deploy.py` inference |
| **scipy, vLLM (Qwen judge)** | `build_pred_video.py` evaluation step |

---

## Suggested workflow to reproduce the COMMON demo

1. Produce **GT** video: `build_gt_video.py` with the PVSG test JSONL and `--video-name`.  
2. Produce **GT_prompt** and **GEN_prompt** videos: `build_pred_video.py` twice (`--prompt-mode GT_prompt` and `GEN_prompt`) with the same `--video-name` and model.  
3. Merge: `build_common_video.py <video_name>`.
