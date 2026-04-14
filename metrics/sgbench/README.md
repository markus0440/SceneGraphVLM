# Classical Scene Graph Metrics (`sgbench`)

This directory provides **classical** scene graph generation metrics — **Recall@K** and **Mean Recall@K** — without an LLM-based judge. Numbers produced here are directly comparable with results tables from OED, STTran, and other Action Genome baselines.

Reference implementation: `OED/models/evaluate_recall.py` (`BasicSceneGraphEvaluator`).

---

## Quick start

```bash
# From the repository root
python3 metrics/sgbench/eval_classical_metrics.py \
  --pred-jsonl metrics/results/checkpoints-inference/sft/AG/<inference-file>.jsonl \
  --output-dir metrics/results/checkpoints-metrics/sft/AG \
  --output-name <model>-classical-metrics.json \
  --mode sgdet \
  --iou-thr 0.5
```

**Input:** an inference JSONL produced by `metrics/qwen-bench/infer/` (or `metrics/open-router/`), where each line has `content` (GT TOON) and `predict` (model TOON).

**Output:** a JSON file with R@{10,20,50,100} and mR@{10,20,50,100} for both **with constraint** and **no constraint** evaluation protocols.

---

## `eval_classical_metrics.py`

### What it does

1. Parses TOON scene graphs from `content` (ground truth) and `predict` (model output).
2. Maps object names and predicate labels to the Action Genome vocabulary indices (37 object classes, 26 predicates: 3 attention + 6 spatial + 17 contacting).
3. Builds triplets following OED conventions:
   - Attention: `(person, object, predicate_idx)`
   - Spatial: `(object, person, predicate_idx)` — **reversed subject/object**
   - Contacting: `(person, object, predicate_idx)`
4. Constructs pseudo-distributions from VLM discrete labels (one-hot for attention, multi-hot for spatial/contacting) since VLMs produce labels rather than probability distributions.
5. Evaluates Recall@K using the same matching logic as OED: triplet identity matching + IoU ≥ threshold on both subject and object bounding boxes.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pred-jsonl` | *(required)* | Inference JSONL with `content` and `predict` fields |
| `--output-dir` | *(required)* | Directory for output JSON |
| `--output-name` | *(required)* | Output filename |
| `--mode` | `sgdet` | Evaluation mode: `sgdet`, `sgcls`, or `predcls` |
| `--iou-thr` | `0.5` | IoU threshold for box matching |
| `--limit` | `0` | Process only first N samples (0 = all) |

### Evaluation modes

| Mode | Uses from GT | Evaluates |
|------|-------------|-----------|
| `sgdet` | nothing | object detection + classification + predicates |
| `sgcls` | bounding boxes | object classification + predicates |
| `predcls` | bounding boxes + object classes | predicates only |

### Evaluation protocols

- **With constraint:** for each (subject, object) pair and each predicate type (attention / spatial / contacting), take the **argmax** predicate. Produces one triplet per type per pair. Standard in OED/STTran tables as **W/R@K**.
- **No constraint:** rank **all** candidate triplets by score, take top-100. Multi-label predictions expand into separate triplets. Standard as **N/R@K**.

### Dependencies

Only `numpy` and `torch` (CPU-only is sufficient). No heavy dependencies from OED (no `thop`, `cv2`, `pycocotools`).

---

## Metrics definitions

### Recall@K (R@K)

For each frame, the top-K predicted triplets are matched against all ground-truth triplets. A GT triplet is **hit** if there exists a predicted triplet with:
- identical `(subject_class, predicate, object_class)` tuple, **and**
- IoU(subject_box_pred, subject_box_gt) ≥ θ, **and**
- IoU(object_box_pred, object_box_gt) ≥ θ

$$R@K = \frac{|\text{hit GT triplets in top-}K|}{|\text{all GT triplets}|}$$

Reported as the mean over all frames.

### Mean Recall@K (mR@K)

Per-class variant: compute R@K separately for each of the 26 predicate classes, then average:

$$mR@K = \frac{1}{26} \sum_{c=1}^{26} R@K_c$$

This penalizes models that only predict frequent predicates (e.g. `not_contacting`, `looking_at`).

---

## Comparing VLM vs classical models

When comparing VLM results with OED/STTran numbers, keep in mind:

- **No confidence scores:** VLMs output a deterministic set of triplets without per-relation probabilities. The script assigns uniform scores (1.0). This means no triplet is "ranked out" of top-K, which can inflate R@K when the model predicts few triplets.
- **With constraint plateau:** R@20 ≈ R@50 ≈ R@100 in with_constraint mode is expected — VLMs typically produce ~3 triplets per pair (one per type), so all predictions fit within top-20.
- **Closed vocabulary prompt:** VLMs receive the full AG vocabulary in the prompt, while classical models learn it implicitly. This is a valid methodological difference, not an error.

For the fairest comparison, focus on **no_constraint R@50** and **mR@50** — these are the most commonly reported numbers in the SG literature and are least affected by the score-ranking asymmetry.

---

## Reference: OED repository

The `OED/` subdirectory contains the original OED implementation ("Towards One-stage End-to-End Dynamic Scene Graph Generation"). It is a full DETR-based training/evaluation codebase for Action Genome — **not** a metrics-only tool. See `OED/README.md` for its own dataset setup and checkpoints.

---

## Directory layout

```
metrics/sgbench/
├── README.md                      ← this file
├── eval_classical_metrics.py      ← R@K / mR@K evaluator for VLM outputs
└── OED/                           ← reference OED implementation (DETR-based SG model)
```
