# Action Genome (AG) — SceneGraphVLM layout

This folder mirrors the **Action Genome v1.0** workflow from the [official AG repository](https://github.com/JingweiJ/ActionGenome), but paths are aligned with the **SceneGraphVLM** tree under `datasets/annotations/AG_annot/`.

## Prerequisites

- **Python 3**
- **`ffmpeg`** on `PATH` (for frame dumping)
- Python packages used by the tools (e.g. `tqdm`, `Pillow`; same env as the rest of the repo)

## 1. Download videos and annotations

1. **Charades videos (480p)** — download from the [Charades / Prior page](https://prior.allenai.org/projects/charades).

2. Place the video files under:

   ```text
   datasets/annotations/AG_annot/Charades/videos/
   ```

   Each file should be named as expected by AG (e.g. `VIDEO_ID.mp4`), consistent with `annotations/frame_list.txt`.

3. **Action Genome annotations** — download from the [official Google Drive](https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys?usp=sharing).

4. Copy at least these files into:

   ```text
   datasets/annotations/AG_annot/annotations/
   ```

   - `object_bbox_and_relationship.pkl`
   - `person_bbox.pkl`
   - `frame_list.txt` 
   - `object_classes.txt`
   - `relationship_classes.txt`

This repository tracks the small text files under `annotations/`. The **`.pkl` files are not committed** — you must add them locally after download.

## 2. Dump frames from videos

The original release does not ship dumped frames. After all 480p videos are under `Charades/videos/`, extract frames into `Charades/frames/`.

**Working directory must be this folder** (`AG_annot`), because the script uses paths **relative to the current working directory**.

```bash
cd /path/to/SceneGraphVLM/datasets/annotations/AG_annot
```

**Annotated frames only** (same sampling as in the AG paper. Much smaller than full video — on the order of ~74 GB in the original release notes):

```bash
python tools/dump_frames.py
```

**All frames** from every video:

```bash
python tools/dump_frames.py --all_frames
```

**Overrides** (still relative to `cwd` = `AG_annot` unless you pass absolute paths):

```bash
python tools/dump_frames.py \
  --video_dir Charades/videos \
  --frame_dir Charades/frames \
  --annotation_dir annotations
```

Default layout after dumping:

```text
Charades/frames/<VIDEO_ID>/<FRAME>.png
```

## 3. Annotation structure (official AG)

`object_bbox_and_relationship.pkl` is a dictionary:

```text
{
  'VIDEO_ID/FRAME_ID': [
    {
      'class': 'book',
      'bbox': (x, y, w, h),
      'attention_relationship': ['looking_at'],
      'spatial_relationship': ['in_front_of'],
      'contacting_relationship': ['holding', 'touching'],
      'visible': True,
      'metadata': {
        'tag': 'VIDEO_ID/FRAME_ID',
        'set': 'train'
      }
    },
    ...
  ],
  ...
}
```

- **`visible`**: whether the interacted object is visible in the frame.

`person_bbox.pkl` holds per-frame person boxes (Faster R-CNN as in the paper).

Other files under `annotations/`:

| File | Role |
|------|------|
| `frame_list.txt` | All labeled frames |
| `object_classes.txt` | Object class names |
| `relationship_classes.txt` | Human–object relationship classes |

## 4. SceneGraphVLM tools (TOON + training JSONL)

These scripts are run from the **repository root** (`SceneGraphVLM`). Paths like `datasets/...` are relative to that root unless you pass absolute paths.

### 4.1 `prepare_original_ag_sft.py`

Reads AG pickles and **source frames** under `AG_annot/Charades/frames`, builds:

- **Intermediate JSON** (TOON labels + metadata):  
  `datasets/annotations/AG_annot/data_sft_original/train_annotations_toon_sft.json`  
  `datasets/annotations/AG_annot/data_sft_original/test_annotations_toon_sft.json`
- **Resized images (640×480)** for training/eval:  
  `datasets/frames/AG_frames/train_images/...`  
  `datasets/frames/AG_frames/test_images/...`  

Each sample’s `image_path` in JSON is **relative to the repo root** (POSIX `/`).

Default output locations (example if the repo root is `SceneGraphVLM`):

- `SceneGraphVLM/datasets/frames/AG_frames/` — e.g.  
  `/data/homes/makarov_vd/workspace/SceneGraphVLM/datasets/frames/AG_frames/`
- Intermediate JSON stays under `AG_annot/data_sft_original/` (see tree below).

```bash
cd /path/to/SceneGraphVLM

python datasets/annotations/AG_annot/tools/prepare_original_ag_sft.py
```

**Useful options:**

```bash
python datasets/annotations/AG_annot/tools/prepare_original_ag_sft.py \
  --repo_root /path/to/SceneGraphVLM \
  --ag_root datasets/annotations/AG_annot \
  --frames_root datasets/annotations/AG_annot/Charades/frames \
  --export_root datasets/annotations/AG_annot/data_sft_original \
  --images_out datasets/frames/AG_frames \
  --num_workers 32 \
  --limit 0
```

- `--keep_all` — do not filter to `visible=True` only (default is visible-only).
- `--limit N` — process at most `N` frames (`0` = all).
- `--num_workers` — `1` = sequential; `>1` = multiprocessing pool.

### 4.2 `sft_to_jsonl_ag.py`

Reads the intermediate JSON from step 4.1 and writes **Swift-style chat JSONL** (PVSG-like prompts with temporal context):

- `datasets/data_playground/AG_json/train.jsonl`
- `datasets/data_playground/AG_json/test.jsonl`

Example absolute path:  
`/data/homes/makarov_vd/workspace/SceneGraphVLM/datasets/data_playground/AG_json/`

```bash
cd /path/to/SceneGraphVLM

python datasets/annotations/AG_annot/tools/sft_to_jsonl_ag.py
```

**Overrides:**

```bash
python datasets/annotations/AG_annot/tools/sft_to_jsonl_ag.py \
  --repo_root /path/to/SceneGraphVLM \
  --export_root datasets/annotations/AG_annot/data_sft_original \
  --out_dir datasets/data_playground/AG_json
```

### Recommended pipeline (summary)

```bash
# 0) Download videos + pkl annotations into AG_annot (see above)

# 1) Dump frames (cwd = AG_annot)
cd datasets/annotations/AG_annot
python tools/dump_frames.py

# 2) Build TOON JSON + AG_frames (cwd = repo root)
cd /path/to/SceneGraphVLM
python datasets/annotations/AG_annot/tools/prepare_original_ag_sft.py --num_workers 32

# 3) JSONL for SFT / inference tooling
python datasets/annotations/AG_annot/tools/sft_to_jsonl_ag.py
```

## 5. Directory tree (expected layout)

Repository root = `SceneGraphVLM`.

```text
SceneGraphVLM/
├── datasets/
│   ├── annotations/
│   │   └── AG_annot/                    <== this README
│   │       ├── README.md
│   │       ├── annotations/
│   │       │   ├── frame_list.txt
│   │       │   ├── object_classes.txt
│   │       │   ├── relationship_classes.txt
│   │       │   ├── object_bbox_and_relationship.pkl   # from official download
│   │       │   └── person_bbox.pkl                   # from official download
│   │       ├── Charades/              # user-prepared (not in git)
│   │       │   ├── videos/
│   │       │   └── frames/
│   │       ├── data_sft_original/     # produced by prepare_original_ag_sft.py
│   │       │   ├── train_annotations_toon_sft.json
│   │       │   └── test_annotations_toon_sft.json
│   │       └── tools/
│   │           ├── dump_frames.py
│   │           ├── prepare_original_ag_sft.py
│   │           └── sft_to_jsonl_ag.py
│   ├── frames/
│   │   └── AG_frames/                 # produced by prepare_original_ag_sft.py
│   │       ├── train_images/
│   │       └── test_images/
│   └── data_playground/
│       └── AG_json/                   # produced by sft_to_jsonl_ag.py
│           ├── train.jsonl
│           └── test.jsonl
```

## References

- Action Genome paper: [CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.pdf)
- Original snippets / README: upstream Action Genome repository (frame dumping and pickle schema).
