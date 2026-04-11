# BaseAnnot filter (`prepare_filtered_pvsg_sft.py`)

**BaseAnnot** is a **PVSG-only** post-processor on dense TOON JSON produced by `prepare_original_pvsg_sft.py`. It does **not** read pixels again: it **subsamples** frames per video using **label-change cues**, then **prunes** train/test so both splits share the **same object-name and relation-type vocabulary** (overlap / symmetry filter).

## Algorithm (high level)

Inputs: `train_annotations_toon_sft.json` and `test_annotations_toon_sft.json` in `--input_dir` (each row: `image_id`, `image_path`, `answer_toon`).

### Step A — per-video “keyframe” selection (annotation change)

1. Group rows by **video id** parsed from `image_id` (last `_`-separated token is the frame index; everything before is the video id).
2. Sort frames by frame index inside each video.
3. Parse each frame’s TOON into:
   - **Object set** — category names from the `obj[…]{id,name,x1,y1,x2,y2}:` block.
   - **Relation count** — the integer `N` in the header `rel[N]{subj,pred,obj}:` (declared count in the annotation).
4. **Always keep** the first frame of each video.
5. For each later frame, compare to the **last kept** frame:
   - If **object set** changed **or** **relation count** changed → **keep** this frame and update the “last kept” signature.
   - Otherwise → **drop** (redundant w.r.t. discrete graph structure used here).

Intuition: keep moments where the discrete **bag of categories** or **number of relation slots** changes; drop long runs where both stay identical.

#### Example A1 — same objects and same `rel[N]` count → drop middle frames

Assume one video `vid` and four consecutive frames (TOON abbreviated to **object set** + **`N` in `rel[N]`** only):

| Frame | Object set (category names) | `rel[N]` header |
|-------|-----------------------------|-----------------|
| 0 | `{person, table}` | `rel[2]` |
| 1 | `{person, table}` | `rel[2]` |
| 2 | `{person, table}` | `rel[2]` |
| 3 | `{person, table, cup}` | `rel[2]` |

Processing:

- **Keep 0** (always).
- **1 vs last kept (0):** same set, same `N=2` → **drop 1**.
- **2 vs last kept (0):** still same set and `N` → **drop 2**.
- **3 vs last kept (0):** object set gained **`cup`** → **keep 3**; last kept becomes frame 3.

**Output frames:** `{0, 3}` only.

#### Example A2 — same object set, different relation count → keep

| Frame | Object set | `rel[N]` |
|-------|------------|----------|
| 0 | `{adult, child}` | `rel[1]` |
| 1 | `{adult, child}` | `rel[3]` |

- **Keep 0**.
- **1 vs 0:** sets equal but **`1 → 3`** → relation **count** changed → **keep 1**.

So a jump only in **`N`** (even if the same category names appear) already forces a new keyframe.

### Step B — train / test vocabulary overlap (iterative)

Goal: avoid categories or predicate types that exist **only** in train or **only** in test (bad for a shared closed label space at eval).

Repeat until stable (max 100 iterations):

1. Collect **all object names** and **all relation predicate strings** appearing in train; same for test.
2. Compute sets **only in train** and **only in test** (set differences for objects and for relation types).
3. **Remove** from train every frame that contains **any** train-only object **or** train-only predicate type; **remove** from test every frame that contains **any** test-only object **or** test-only predicate type.
4. If nothing was removed, **stop**.

#### Example B — dropping train-only labels

Suppose after Step A you have:

- **Train** contains a frame whose TOON mentions object category **`zebra`** and predicate **`riding`**.
- **Test** never contains **`zebra`** anywhere (so **`zebra`** is **train-only**).

Then in the **first** overlap iteration, **every train frame** whose object list includes **`zebra`** is removed. If **`riding`** also appeared **only** in train, any train frame whose relation lines use **`riding`** would be removed too. The loop repeats with updated vocabularies until there are **no** train-only or test-only object names or predicate strings left (or until an iteration removes nothing).

So Step B is **not** “balance class counts”; it **strips** any sample that still carries a label restricted to one split, until the label inventories **match** between train and test.

After both steps, rows are rewritten to `--output_dir` with the **same JSON basenames**; `image_path` is normalized **relative to the repo root**.

## Run

From the **SceneGraphVLM** repository root:

```bash
python utils/BaseAnnot/prepare_filtered_pvsg_sft.py
```

Defaults:

- `--input_dir` → `datasets/annotations/PVSG_annot/data_sft_original`
- `--output_dir` → `datasets/annotations/PVSG_annot/data_sft_base_annot`

Override paths:

```bash
python utils/BaseAnnot/prepare_filtered_pvsg_sft.py \
  --repo_root /path/to/SceneGraphVLM \
  --input_dir datasets/annotations/PVSG_annot/data_sft_original \
  --output_dir datasets/annotations/PVSG_annot/data_sft_base_annot
```

See `datasets/annotations/PVSG_annot/PVSG_README.md` for how this fits the PVSG pipeline.

---

## Related documentation

- [SceneGraphVLM project README](../../README.md) · [Metrics](../../metrics/metrics.md) · [PVSG README](../../datasets/annotations/PVSG_annot/PVSG_README.md)
