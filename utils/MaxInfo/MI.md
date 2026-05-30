# MaxInfo filter (`pvsg_maxinfo_filter.py`)

This script implements a **training-free, embedding-space key-frame selector** for **PVSG** TOON JSON, following the **MaxInfo** idea: pick a subset of frames whose **CLIP image embeddings** are **diverse** in a low-dimensional subspace, using a **maximum-volume** (maxvol) pivot rule.

**References**

- Paper: [MaxInfo: A training-free key-frame selection method using maximum volume for enhanced video understanding](https://arxiv.org/pdf/2502.03183) (Li et al., arXiv:2502.03183).
- Upstream notebooks / context: [FusionBrainLab/MaxInfo](https://github.com/FusionBrainLab/MaxInfo.git).

Our code path: **`utils/MaxInfo/pvsg_maxinfo_filter.py`** (PVSG-specific driver; CLIP → SVD → `rect_maxvol` from **maxvolpy**).

---

## Mathematical outline

### 1. Frame embeddings (CLIP)

For a video with $n$ frames, load RGB crops from each row’s `image_path`, run **`openai/clip-vit-large-patch14-336`**, and take the **pooler** image vector for each frame:

$$
\mathbf{f}_i \in \mathbb{R}^{d}, \quad i = 1,\ldots,n
$$

Stack rows into a feature matrix:

$$
F \in \mathbb{R}^{n \times d}
$$

(Implementation uses float64 for the SVD stage.)

### 2. Dimensionality reduction (SVD)

Compute a thin SVD (economy form):

$$
F = U \Sigma V^\top, \quad U \in \mathbb{R}^{n \times \rho},\ \Sigma \in \mathbb{R}^{\rho \times \rho},\ V \in \mathbb{R}^{d \times \rho}
$$

with $\rho = \min(n-1,\, d,\, r_{\text{cfg}})$ and $r_{\text{cfg}}$ from CLI **`--r`** (default **8**). The script keeps the **leading left singular vectors** as a low-dimensional representation of frames:

$$
M = U_{:,1:\rho} \in \mathbb{R}^{n \times \rho}
$$

So each frame is a row $M_{i,:}$ in a $\rho$-dimensional subspace capturing most energy of the embedding sequence.

### 3. Maximum-volume subset (`rect_maxvol`)

**MaxInfo** selects a subset of row indices $I \subseteq \{1,\ldots,n\}$ so that the **parallelepiped** spanned by the chosen rows $\{M_{i,:} : i \in I\}$ has **large volume** in $\mathbb{R}^{\rho}$: informally, rows that are **not redundant** in the subspace (diverse directions / magnitudes).

We call **maxvolpy**’s **`rect_maxvol(M, tol=t)`** on $M$. The scalar **`tol`** trades off **how many** pivots (frames) are returned:

- **Larger `tol`** → **fewer** frames (more aggressive subsampling in this binding).
- **Smaller `tol`** → **more** frames.

Defaults in this repo are tuned for PVSG-scale JSON; see the module docstring in `pvsg_maxinfo_filter.py` for empirical retention vs. `tol`.

### 4. Post-processing

Selected indices are **sorted** by time. If the count is **odd** and $> 1$, the **last** index is dropped so the count is **even** (implementation detail for downstream batching).

---

## Pipeline in code (per split)

1. Load `train_annotations_toon_sft.json` / `test_annotations_toon_sft.json`.
2. Group samples by **video id** (parsed from `…/frames/<video_id>/<frame>.png` in `image_path`).
3. For each video: CLIP features → $M$ → `rect_maxvol` → keep only selected rows.
4. Write filtered arrays to `--output_dir` with the **same filenames**; `image_path` stays **repo-relative**.

If output JSON files **already exist**, the script **skips** CLIP and prints statistics only.

## Run

From the **SceneGraphVLM** repository root (GPU recommended):

```bash
python utils/MaxInfo/pvsg_maxinfo_filter.py --fp16
```

Defaults:

- `--input_dir` → `datasets/annotations/PVSG_annot/data_sft_original`
- `--output_dir` → `datasets/annotations/PVSG_annot/data_sft_maxinfo`

Full example:

```bash
python utils/MaxInfo/pvsg_maxinfo_filter.py \
  --input_dir datasets/annotations/PVSG_annot/data_sft_original \
  --output_dir datasets/annotations/PVSG_annot/data_sft_maxinfo \
  --repo_root /path/to/SceneGraphVLM \
  --r 8 \
  --tol 0.23 \
  --fp16 \
  --batch-size 64
```

Useful flags: `--train-json`, `--test-json`, `--train-only`. Dependencies: **PyTorch**, **transformers**, **Pillow**, **numpy**, **tqdm**, **maxvolpy**.

See `datasets/annotations/PVSG_annot/PVSG_README.md` for dataset layout.

---

## Related documentation

- [SceneGraphVLM project README](../../README.md) · [Metrics](../../metrics/metrics.md) · [PVSG README](../../datasets/annotations/PVSG_annot/PVSG_README.md)
