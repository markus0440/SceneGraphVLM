# Panoptic Video Scene Graph (PVSG)

<p align="center">
  <img src="figures/teaser.png" alt="PVSG teaser (local copy for this README)" width="720" />
</p>

This folder documents **PVSG** ([Panoptic Video Scene Graph Generation](https://arxiv.org/pdf/2311.17058)) and how we prepare **TOON** labels and **Swift-style JSONL** inside **SceneGraphVLM** under `datasets/annotations/PVSG_annot/`.

## Quick Start

Use this path when preparing PVSG from a fresh clone.

1. Install the local preprocessing environment:

   ```bash
   bash envs/sh_scripts/install_dataset_preprocessing_env.sh
   conda activate scenegraphvlm_data_prep
   ```

2. Download and unpack the official PVSG data so the repo contains:

   ```text
   datasets/annotations/PVSG_annot/annotations/pvsg.json
   datasets/annotations/PVSG_annot/OpenPVSG/
     ego4d/
     epic_kitchen/
     vidor/
   ```

   Each source directory should contain its `frames/`, `masks/`, and `videos/` subfolders.

3. Run the repo-level pipeline from the SceneGraphVLM root:

   ```bash
   python datasets/tools/prepare_all_annotations.py \
     --datasets pvsg \
     --maxinfo-fp16 \
     --overwrite
   ```

   Omit `--maxinfo-fp16` if you want the MaxInfo CLIP pass to run in full precision.

The command builds dense PVSG TOON JSON, runs the configured filtering branches (`all`, `base_annot`, `maxinfo`, `psfr`), exports base SFT JSONL for each branch, cleans the scene graphs, rewrites image paths to `/workspace/datasets/frames/...`, and writes final SFT / GRPO / `noprevgraph` variants under `datasets/data_playground/PVSG_json/`.

The detailed sections below are for manual debugging, path overrides, and reference.

| Resource | Link |
|----------|------|
| **Paper (PVSG, CVPR 2023)** | [arXiv:2311.17058](https://arxiv.org/pdf/2311.17058) |
| **Project page** | [jingkang50.github.io/PVSG](https://jingkang50.github.io/PVSG/) |
| **Official code (OpenPVSG)** | [github.com/LilyDaytoy/OpenPVSG](https://github.com/LilyDaytoy/OpenPVSG.git) |
| **Dataset download** | [SharePoint — PVSGDataset](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EpHpnXP-ta9Nu1wD6FwkDWAB0LxY8oE9VNqsgv6ln-i8QQ?e=fURefF) |
| **Quick view** | [SharePoint — QuickView](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgvpTfCTMudLpxw-h0_BVdcBAHacUaAQD-u9OvkUlpaDBg?e=LXnqaX) |

The upstream release describes **400** videos (~76.5 s on average at **5 FPS**): **VidOR** (289), **EpicKitchen** (55), **Ego4D** (56). In `pvsg.json`, splits are **338 train** and **62 validation**. This repo maps PVSG **`val`** to **`test_*`** filenames (same idea as the OpenPVSG layout).

## Prerequisites

- **Python 3**
- **`numpy`**, **`Pillow`**, **`tqdm`** (and any extras required by optional thinning: **PyTorch** / **transformers** / **OpenCV**, etc.)

Install the local dataset-preprocessing environment:

```bash
# from the SceneGraphVLM repository root
bash envs/sh_scripts/install_dataset_preprocessing_env.sh
conda activate scenegraphvlm_data_prep
```

This environment is only for dataset preparation and annotation generation. Training and validation are expected to run inside the project Docker container mounted at `/workspace`.

## 1. Annotation JSON and unpacked media on disk

Place **`pvsg.json`** under:

```text
datasets/annotations/PVSG_annot/annotations/
  pvsg.json
```

Unzip the official archives from SharePoint (layout: **Ego4D**, **EpicKitchen**, **VidOR** plus `pvsg.json` at the share root). Follow the [OpenPVSG README](https://github.com/LilyDaytoy/OpenPVSG) (they recommend `unzip -j` to drop junk paths). 

<p align="center">
  <img src="figures/sharepoint_download_layout.png" alt="SharePoint layout: Ego4D, EpicKitchen, VidOR folders and pvsg.json" width="720" />
</p>

Point the preparation script at the unpacked RGB tree:

```text
datasets/annotations/PVSG_annot/OpenPVSG/
  ego4d/
    frames/
    masks/
    videos/
  epic_kitchen/
    frames/
    masks/
    videos/
  vidor/
    frames/
    masks/
    videos/
```

## 2. Build dense TOON

Run from the **repository root** (`SceneGraphVLM`).

- Reads `annotations/pvsg.json`.
- Loads RGB + masks from `OpenPVSG/{vidor,epic_kitchen,ego4d}/`.
- Resizes pixels to **640×480** and writes PNGs under `datasets/frames/PVSG_frames/{train_images,test_images}/`.
- Writes intermediate TOON JSON:

  `datasets/annotations/PVSG_annot/data_sft_original/train_annotations_toon_sft.json`  
  `datasets/annotations/PVSG_annot/data_sft_original/test_annotations_toon_sft.json`

```bash
cd /path/to/SceneGraphVLM

python datasets/annotations/PVSG_annot/tools/prepare_original_pvsg_sft.py
```

**Useful flags**

| Flag | Meaning |
|------|---------|
| `--repo_root` | SceneGraphVLM root (auto-detected if empty). |
| `--annotation` | Path to `pvsg.json` (default `datasets/annotations/PVSG_annot/annotations/pvsg.json`). |
| `--pvsg_data_root` | OpenPVSG root with per-source `frames/` and `masks/` (default `datasets/annotations/PVSG_annot/OpenPVSG`). |
| `--export_root` | Output dir for `*_annotations_toon_sft.json`. |
| `--images_out` | Root for PNGs (default `datasets/frames/PVSG_frames`). |
| `--sources` | Comma-separated sources (default `vidor,epic_kitchen,ego4d`). |
| `--splits` | Comma-separated splits (default `train,val`; `val` ==> `test_*.json`). |
| `--limit_videos` | Cap number of videos (`0` = all). |
| `--limit_frames` | Cap frames per video (`0` = all). |
| `--only_video` | Process a single video id. |
| `--num_workers` | Parallel workers (default: CPU count). |

## 3. Thin the TOON dataset (BaseAnnot, MaxInfo, PSFR)

PVSG is built from short clips sampled at fixed frame rate, but clips differ a lot in length and in how much each frame actually changes. Neighbouring frames are often almost identical, so treating every frame as an independent training example blows up the dataset with redundant pairs that share the same visual content and highly correlated scene graphs. That hurts VLM training: batches become dominated by near-duplicates, optimization is skewed, and the model gets less diverse supervision per real “scene change.” We therefore thin the TOON annotations (e.g. with BaseAnnot, MaxInfo, PSFR) to keep a smaller, less redundant set of frames while preserving useful variation, then export JSONL and optionally drop zero-relation rows so the supervision signal stays meaningful.

### 3.1 `utils/BaseAnnot/prepare_filtered_pvsg_sft.py`

**Defaults:** input `datasets/annotations/PVSG_annot/data_sft_original`, output `datasets/annotations/PVSG_annot/data_sft_base_annot`.

```bash
python utils/BaseAnnot/prepare_filtered_pvsg_sft.py
```

### 3.2 `utils/MaxInfo/pvsg_maxinfo_filter.py`

**Defaults:** input `datasets/annotations/PVSG_annot/data_sft_original` (full dense TOON), output `datasets/annotations/PVSG_annot/data_sft_maxinfo`. GPU recommended.

```bash
python utils/MaxInfo/pvsg_maxinfo_filter.py --fp16
```

### 3.3 `utils/PSFR/pvsg_psfr_filter.py`

**Defaults:** input `datasets/annotations/PVSG_annot/data_sft_original`, output `datasets/annotations/PVSG_annot/data_sft_psfr`, config `utils/PSFR/config_pvsg.json`.

```bash
python utils/PSFR/pvsg_psfr_filter.py
```

## 4. Export Swift JSONL (`sft_to_jsonl_pvsg.py`)

Run from the **repository root**. Point `--input_dir` at the TOON directory you produced after **Section 2** and optional **Section 3** (e.g. `data_sft_maxinfo` after MaxInfo on the full export, or `data_sft_base_annot` / `data_sft_psfr` if you use only one of those filters).

The **first** frame of each video uses **`PROMPT_NO_PREV`**. **Every later frame** uses **`PROMPT_WITH_PREV`**, with the previous frame’s `answer_toon` string substituted for `<<PREV_TOON>>` (full text in **Section 5**).

Default output directory:

- `datasets/data_playground/PVSG_json/pvsg_all_data_gt_prompt/train.jsonl`
- `datasets/data_playground/PVSG_json/pvsg_all_data_gt_prompt/test.jsonl`

```bash
cd /path/to/SceneGraphVLM

python datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py \
  --input_dir datasets/annotations/PVSG_annot/data_sft_maxinfo
```

**Path overrides:**

```bash
python datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py \
  --repo_root /path/to/SceneGraphVLM \
  --input_dir datasets/annotations/PVSG_annot/data_sft_maxinfo \
  --output_dir datasets/data_playground/PVSG_json/pvsg_maxinfo_gt_prompt
```

## 5. User prompts (full templates + expanded subsequent-frame example)

### 5.1 First frame of each video

```text
<image>
Generate a structured scene graph for an image of size (640 x 480) using the following text format.

Output Format:

<answer>
obj[N]{id,name,x1,y1,x2,y2}:
  id,name,x1,y1,x2,y2
  ...
rel[M]{subj,pred,obj}:
  subj,pred,obj
  ...
</answer>

Guidelines:
- Objects:
  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).
  - The name must be the object category name (e.g., person, umbrella).
  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.
  - Include all visible objects, even if they have no relationships.

- Relationships:
  - Represent interactions using integer object IDs in subj and obj.
  - pred is the relationship type (string), such as in-front-of, attached-to, beside.
  - Omit relationships for objects that do not participate in any interaction.

Example output:

<answer>
obj[7]{id,name,x1,y1,x2,y2}:
  1,person,281,272,524,438
  2,umbrella,273,123,640,434
  3,house,0,88,262,426
  4,window-other,163,262,195,294
  5,tree-merged,0,0,640,440
  6,sky-other-merged,0,0,459,123
  7,building-other-merged,537,164,640,291
rel[5]{subj,pred,obj}:
  1,in-front-of,5
  3,attached-to,4
  4,hanging-from,3
  5,beside,3
  6,over,5
</answer>

Now, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.
```

### 5.2 Subsequent frames

```text
<image>
Generate a structured scene graph for an image of size (640 x 480) using the following text format.

Output Format:

<answer>
obj[N]{id,name,x1,y1,x2,y2}:
  id,name,x1,y1,x2,y2
  ...
rel[M]{subj,pred,obj}:
  subj,pred,obj
  ...
</answer>

Guidelines:
- Objects:
  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).
  - The name must be the object category name (e.g., person, umbrella).
  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.
  - Include all visible objects, even if they have no relationships.

- Relationships:
  - Represent interactions using integer object IDs in subj and obj.
  - pred is the relationship type (string), such as in-front-of, attached-to, beside.
  - Omit relationships for objects that do not participate in any interaction.

Example output:

<answer>
obj[7]{id,name,x1,y1,x2,y2}:
  1,person,281,272,524,438
  2,umbrella,273,123,640,434
  3,house,0,88,262,426
  4,window-other,163,262,195,294
  5,tree-merged,0,0,640,440
  6,sky-other-merged,0,0,459,123
  7,building-other-merged,537,164,640,291
rel[5]{subj,pred,obj}:
  1,in-front-of,5
  3,attached-to,4
  4,hanging-from,3
  5,beside,3
  6,over,5
</answer>

You are also given the previous frame's ground-truth scene graph in TOON format.
Use it as temporal context, but rely primarily on the current image.
Important:
- Include all objects visible in the CURRENT image, even if they did not exist in the previous graph.
- Do NOT include objects that are NOT visible in the current image, even if they exist in the previous graph.
- Output ONLY the complete scene graph for the CURRENT image, using the TOON structure from the Output Format above, inside one <answer>...</answer> block.

Previous frame scene graph (TOON):

obj[7]{id,name,x1,y1,x2,y2}:
  1,person,284,275,511,439
  2,umbrella,234,133,640,411
  3,house,0,88,235,423
  4,window-other,111,245,178,294
  5,tree-merged,0,0,640,450
  6,sky-other-merged,0,0,451,121
  7,building-other-merged,511,178,640,222
rel[5]{subj,pred,obj}:
  1,in-front-of,5
  3,attached-to,4
  4,hanging-from,3
  5,beside,3

Now, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.
```

## 6. Final clean variants

The repo-level variant builder is the canonical final step for PVSG. Run it after `sft_to_jsonl_pvsg.py` has written `train.jsonl` / `test.jsonl` for each PVSG branch you need:

```bash
cd /path/to/SceneGraphVLM

python datasets/tools/build_annotation_variants.py --only pvsg --overwrite
```

It repairs malformed scene graphs where possible, drops rows with zero valid relations, rewrites image paths to `/workspace/datasets/frames/...`, and generates SFT / GRPO variants, including `noprevgraph` for video prompts. Use `--pvsg-variant <name>` to build one branch, or omit it to build all four configured branches: `pvsg_all_data_gt_prompt`, `pvsg_base_annot_gt_prompt`, `pvsg_maxinfo_gt_prompt`, and `pvsg_psfr_gt_prompt`.

Use `--emit-unclean` if you also need a debug copy of the unclean pre-filtered variants under `unclean/`; use `--no-clean` only when intentionally reproducing raw rows.

## 7. Manual pipeline (debug/reference)

```bash
cd /path/to/SceneGraphVLM

# 1) Full TOON + 640x480 frames
python datasets/annotations/PVSG_annot/tools/prepare_original_pvsg_sft.py

# 2) Thin TOON (optional — pick what you need; see Section 3)
#    BaseAnnot: original → base_annot
python utils/BaseAnnot/prepare_filtered_pvsg_sft.py
#    MaxInfo on FULL dense TOON (script defaults: input = data_sft_original)
python utils/MaxInfo/pvsg_maxinfo_filter.py --fp16
#    If you need MaxInfo on BaseAnnot output instead, run again, e.g.:
#    python utils/MaxInfo/pvsg_maxinfo_filter.py --fp16 \
#      --input_dir datasets/annotations/PVSG_annot/data_sft_base_annot \
#      --output_dir datasets/annotations/PVSG_annot/data_sft_maxinfo
#    PSFR: original → psfr (independent branch)
python utils/PSFR/pvsg_psfr_filter.py

# 3) JSONL from the TOON root you train on (example: MaxInfo output)
python datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py \
  --input_dir datasets/annotations/PVSG_annot/data_sft_maxinfo \
  --output_dir datasets/data_playground/PVSG_json/pvsg_maxinfo_gt_prompt

# 4) Build final clean variants for all PVSG branches that exist
python datasets/tools/build_annotation_variants.py --only pvsg --overwrite

# Or run the full dataset pipeline from scratch
python datasets/tools/prepare_all_annotations.py --datasets all --overwrite
```

For normal use, prefer the Quick Start command at the top of this README.

## 8. Directory tree (expected layout)

```text
SceneGraphVLM/
├── datasets/
│   ├── annotations/
│   │   └── PVSG_annot/
│   │       ├── PVSG_README.md
│   │       ├── figures/                 # demo.mp4, teaser.png, sharepoint_download_layout.png (this README)
│   │       ├── annotations/             # pvsg.json (often local only)
│   │       ├── OpenPVSG/                # ego4d | epic_kitchen | vidor (frames, masks, videos)
│   │       ├── data_sft_original/       # prepare_original_pvsg_sft.py
│   │       │   ├── train_annotations_toon_sft.json
│   │       │   └── test_annotations_toon_sft.json
│   │       ├── data_sft_base_annot/     # optional: BaseAnnot (Section 3)
│   │       ├── data_sft_maxinfo/        # optional: MaxInfo (Section 3)
│   │       ├── data_sft_psfr/           # optional: PSFR (Section 3)
│   │       └── tools/
│   │           ├── prepare_original_pvsg_sft.py
│   │           └── sft_to_jsonl_pvsg.py
│   ├── frames/
│   │   └── PVSG_frames/
│   │       ├── train_images/
│   │       └── test_images/
│   └── data_playground/
│       └── PVSG_json/
│           └── pvsg_all_data_gt_prompt/    # example sft_to_jsonl_pvsg.py output (Section 4)
│               ├── train.jsonl
│               ├── test.jsonl
│               ├── eval.jsonl
│               ├── train_noprevgraph.jsonl
│               ├── test_noprevgraph.jsonl
│               ├── eval_noprevgraph.jsonl
│               ├── annotation_variants_report.json
│               └── grpo/
│                   ├── train.jsonl
│                   ├── test.jsonl
│                   ├── eval.jsonl
│                   ├── train_noprevgraph.jsonl
│                   ├── test_noprevgraph.jsonl
│                   └── eval_noprevgraph.jsonl
```

## References

- **PVSG (CVPR 2023):** [arXiv:2311.17058](https://arxiv.org/pdf/2311.17058) · [Project page](https://jingkang50.github.io/PVSG/) · [OpenPVSG (GitHub)](https://github.com/LilyDaytoy/OpenPVSG.git)
- **Dataset (SharePoint):** [PVSGDataset](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EpHpnXP-ta9Nu1wD6FwkDWAB0LxY8oE9VNqsgv6ln-i8QQ?e=fURefF) · [QuickView](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgvpTfCTMudLpxw-h0_BVdcBAHacUaAQD-u9OvkUlpaDBg?e=LXnqaX)

---

## Related documentation

- [SceneGraphVLM project README](../../../README.md)  
- [Metrics & evaluation](../../../metrics/metrics.md)  
- [Visualization & demos](../../../visualization/vis.md)  
- [SFT training](../../../sft/SFT_README.md)  
- [PSG README](../PSG_annot/PSG_README.md) · [AG README](../AG_annot/AG_README.md)  
- [BaseAnnot filter](../../../utils/BaseAnnot/BA.md) · [PSFR](../../../utils/PSFR/PSFR.md) · [MaxInfo](../../../utils/MaxInfo/MI.md)
