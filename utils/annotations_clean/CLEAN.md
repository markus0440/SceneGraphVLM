# Zero-relation cleanup (`clean_zero_rel_frames.py`)

Swift / MSwift JSONL for **PSG-style** scene graphs stores the assistant answer in TOON form with a block

`rel[N]{subj,pred,obj}:`

followed by relation lines. Some training rows have **no relations** (`N = 0` or an empty block). This script **drops those lines** so SFT does not waste steps on frames whose supervision has **zero predicates**.

## Behaviour

- Walks **`input_root`** recursively for `*.jsonl`.
- Skips stems ending in **`_clean`** (avoids re-processing outputs).
- For each `foo.jsonl`, writes **`foo_clean.jsonl`** next to it.
- Saves **`zero_rel_cleanup_report.json`** (or `--report-name`) under `input_root` with per-file and total counts.

Rows whose assistant text **does not** match the expected `rel[…]{subj,pred,obj}` header are **kept unchanged** and counted as `missing_rel_block` (e.g. AG `rel_pairs[…]` format).

## Run

From the **SceneGraphVLM** repository root:

```bash
python utils/annotations_clean/clean_zero_rel_frames.py datasets/data_playground/3RScan_json/3RScan_json_psfr_with_prev_gt
```

Another example (PVSG export tree):

```bash
python utils/annotations_clean/clean_zero_rel_frames.py \
  datasets/data_playground/PVSG_json/pvsg_maxinfo_gt_prompt \
  --report-name pvsg_zero_rel_cleanup_report.json
```

See also: dataset READMEs under `datasets/annotations/*`.

---

## Related documentation

- [SceneGraphVLM project README](../../README.md) · [Metrics](../../metrics/metrics.md) · [Visualization](../../visualization/vis.md)
