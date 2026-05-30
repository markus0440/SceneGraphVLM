# Annotation variants

SceneGraphVLM trains in two modes:

- **SFT**: Swift chat JSONL with `messages = [user, assistant]`.
- **GRPO**: Swift RLHF JSONL with user-only `messages` and target `solution`.

For video datasets (`AG`, `PVSG`), the default prompt includes the previous frame's
ground-truth scene graph for every frame except the first frame of a video. The
`*_noprevgraph.jsonl` variants remove that temporal graph block from the prompt.

Panoptic masks are converted to bounding boxes during the dataset-specific
`prepare_original_*_sft.py` stage.

## End-to-end preparation

After cloning the repository, download the raw datasets described in:

- `datasets/annotations/AG_annot/AG_README.md`
- `datasets/annotations/PSG_annot/PSG_README.md`
- `datasets/annotations/PVSG_annot/PVSG_README.md`

Then run the top-level annotation pipeline:

```bash
python datasets/tools/prepare_all_annotations.py --overwrite
```

This command uses only scripts in this repository. It prepares intermediate TOON
JSON, exports base SFT `train.jsonl` / `test.jsonl`, and then builds all derived
annotation variants.

Final annotation files are **clean by default**: invalid boxes / dangling
relations are repaired where possible, samples with zero relations are removed,
and `images` paths are rewritten to `/workspace/datasets/frames/...` so training
and validation scripts can run inside the Docker container. Add
`--emit-unclean` to also save non-clean variants under `unclean/`, or `--no-clean`
to intentionally make the main outputs non-clean.

Use dry-run mode first to inspect the exact commands:

```bash
python datasets/tools/prepare_all_annotations.py --dry-run
```

Common options:

```bash
# If AG videos are downloaded but frames have not been extracted yet
python datasets/tools/prepare_all_annotations.py --datasets ag --ag-dump-frames

# If PSG annotations should be fetched from Hugging Face
python datasets/tools/prepare_all_annotations.py --datasets psg --psg-from-hf

# If base SFT train/test already exists and only variants are needed
python datasets/tools/prepare_all_annotations.py --skip-prepare --skip-sft --overwrite

# Also keep non-clean copies under each dataset's unclean/ directory
python datasets/tools/prepare_all_annotations.py --emit-unclean --overwrite
```

## Build derived variants only

First generate the base SFT `train.jsonl` / `test.jsonl` files with the dataset
READMEs under `datasets/annotations/*_annot/`.

Then run:

```bash
python datasets/tools/build_annotation_variants.py --overwrite
```

The script reads `datasets/data_playground` by default and writes final clean
annotation files in place.

Useful options:

```bash
# Only one dataset
python datasets/tools/build_annotation_variants.py --only ag --overwrite

# Only one PVSG filtering branch
python datasets/tools/build_annotation_variants.py \
  --only pvsg \
  --pvsg-variant pvsg_base_annot_gt_prompt \
  --overwrite

# Inspect required inputs without writing files
python datasets/tools/build_annotation_variants.py --dry-run
```

By default GRPO outputs rewrite image paths to `/workspace/datasets/frames/...`,
matching the Docker training scripts. The same path normalization is applied to
SFT outputs. Use `--image-path-mode local` for local absolute paths, `--no-clean`
for non-clean main outputs, or `--emit-unclean` to write both clean and non-clean
variants.

## Expected files

### AG

Under `datasets/data_playground/AG_json/`:

- SFT: `train.jsonl`, `test.jsonl`, `eval.jsonl`
- SFT no previous graph: `train_noprevgraph.jsonl`, `test_noprevgraph.jsonl`, `eval_noprevgraph.jsonl`
- GRPO: same six filenames under `grpo/`

Total: **12 annotation files**.

### PSG

Under `datasets/data_playground/PSG_json/`:

- SFT: `train.jsonl`, `test.jsonl`, `eval.jsonl`
- GRPO: `grpo/train.jsonl`, `grpo/test.jsonl`, `grpo/eval.jsonl`

Total: **6 annotation files**.

PSG has no temporal previous-frame graph. GRPO conversion canonicalizes PSG labels
to the strict hyphenated TOON form expected by the shared PSG/PVSG reward plugin.

### PVSG

The same AG-style 12-file layout is generated for each PVSG branch:

- `pvsg_all_data_gt_prompt`
- `pvsg_base_annot_gt_prompt`
- `pvsg_maxinfo_gt_prompt`
- `pvsg_psfr_gt_prompt`

Total: **48 annotation files**.

Default eval sizes are aligned with the historical training setup:

- AG: `148`
- PSG: `100`
- PVSG: `78`

Use `--ag-eval-size`, `--psg-eval-size`, and `--pvsg-eval-size` to override them.
