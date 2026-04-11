# PSFR key-frame filter (`pvsg_psfr_filter.py`)

**PSFR** (*Patch-wise Sparse-Flow Retention*) is the name used in recent long-video work (e.g. **FocusGraph**, [arXiv:2603.04349](https://arxiv.org/abs/2603.04349)) for a **training-free** key-frame criterion based on **sparse optical flow** and **patch-wise** feature survival. In this repository, the **per-frame selection engine** comes from a colleague’s standalone project:

- Implementation / utilities: [strangecreator/key-frame-selection](https://github.com/strangecreator/key-frame-selection.git) (MIT).

The PVSG driver **`pvsg_psfr_filter.py`** calls `src/key_frame_selection.py` → **`select_keyframes_from_frames(...)`** with a JSON config (default **`config_pvsg.json`**), feeding an **explicit ordered list of frame paths** per video so no separate `frames_dir` export is required.

---

## Underlying pipeline (from the key-frame-selection design)

The method marks **keyframes** when **many spatial patches** simultaneously lose tracked corner features between consecutive frames.

### 1. Shi–Tomasi corners (per patch)

Corners are scored from the eigenvalues $(\lambda_1, \lambda_2)$ of the **structure tensor** (gradient covariance); a common response is the **minimum eigenvalue**:

$$
R = \min(\lambda_1, \lambda_2)
$$

Strong corners have large $R$. Frames are split into an $n_w \times n_h$ grid of **overlapping** patches (optional **centroidal** patches reduce boundary artifacts).

### 2. Lucas–Kanade tracking

Between consecutive frames, sparse flow minimizes the brightness constancy assumption in a window:

$$
I(x+u,\,y+v,\,t+1) \approx I(x,\,y,\,t)
$$

yielding displacement $(u,v)$ per tracked point (least-squares solve in small patches — standard OpenCV LK).

### 3. Patch drop and keyframe rule

For each patch, compare **how many** initial corners remain successfully tracked vs. the count after the step. A patch **drops** when the **retained fraction** falls below a threshold $\tau$:

$$
\frac{N_{\text{tracked}}}{N_{\text{initial}}} < \tau
$$

A frame becomes a **keyframe** when **at least $k$** patches drop **simultaneously** (config **`min_patches_k`**).

---

## `config_pvsg.json` (defaults in this repo)

| Section | Role |
|--------|------|
| **`preprocess`** | Resize each frame before detection (here **640×360**, linear interpolation). |
| **`patching`** | **`nw` / `nh`**: grid size; **`centroidal`**: extra center patches. |
| **`shi_tomasi`** | Corner budget, quality, min distance, block size, etc. |
| **`lucas_kanade`** | LK window, pyramid levels, termination criteria, **`max_error`**. |
| **`selection`** | **`retention_tau`** = $\tau$, **`min_patches_k`** = $k$, **`copy_keyframes`** (off for PVSG driver). |
| **`visualization`** | Off by default for batch PVSG. |

Edit **`utils/PSFR/config_pvsg.json`** to tune $\tau$, $k$, patch grid, or preprocessing.

---

## PVSG driver behaviour

1. Read `train_annotations_toon_sft.json` / `test_annotations_toon_sft.json` from **`--input_dir`**.
2. Group rows by **video id** from `image_path` (`…/frames/<video_id>/<frame>.png`).
3. For each video, sort frames, run **`select_keyframes_from_frames`** with **`frame_paths=…`** (no disk copy of keyframes unless you enable it in config and extend the wrapper).
4. Write filtered JSON to **`--output_dir`** with the **same basenames**; paths stay **repo-relative**.

If outputs **already exist**, the heavy pass is **skipped** and only stats are printed.

## Run

From the **SceneGraphVLM** repository root:

```bash
python utils/PSFR/pvsg_psfr_filter.py
```

Defaults:

- `--input_dir` → `datasets/annotations/PVSG_annot/data_sft_original`
- `--output_dir` → `datasets/annotations/PVSG_annot/data_sft_psfr`
- `--config` → `utils/PSFR/config_pvsg.json`

Explicit paths:

```bash
python utils/PSFR/pvsg_psfr_filter.py \
  --input_dir datasets/annotations/PVSG_annot/data_sft_original \
  --output_dir datasets/annotations/PVSG_annot/data_sft_psfr \
  --config utils/PSFR/config_pvsg.json
```

**Dependencies:** **OpenCV** (`cv2`), **numpy**, **tqdm** (see `utils/PSFR/src/` imports).

---

## Further reading

- **FocusGraph** (PSFR in long-video QA): [arXiv:2603.04349](https://arxiv.org/abs/2603.04349).
- **Standalone pipeline README** (logo, `config.json` field reference, video↔frames CLI): [key-frame-selection](https://github.com/strangecreator/key-frame-selection.git).

See `datasets/annotations/PVSG_annot/PVSG_README.md` for where PSFR sits in the PVSG export pipeline.
