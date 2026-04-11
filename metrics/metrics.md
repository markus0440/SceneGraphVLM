# Scene Graph Metrics and Inference

This document specifies **how evaluation metrics are computed**, where **inference scripts** reside, and **where artifacts are written**. Scene graphs are represented in **TOON** format (`obj[…]`, `rel[…]`). Parsing and matching are implemented in `metrics/qwen-bench/eval/eval_sgg_metrics_with_qwen.py`.

---

## 1. OpenRouter (`metrics/open-router`)

### Purpose

[OpenRouter](https://openrouter.ai/) is used to run **third-party multimodal models** (e.g. `openai/gpt-5.4-mini`, `google/gemini-3-flash-preview`) on the same **Swift-style JSONL** files as local Qwen runs: each line contains `messages` and `images[]`.

| Script | Mode |
|--------|------|
| `run_open_router_gt_prompt.py` | **GT-prompt:** user text is taken from the dataset **without** replacing the previous-frame block. |
| `run_open_router_generated_prompt.py` | **GEN-prompt (PVSG):** for frame index $t \ge 1$, the segment beginning with `Previous frame scene graph (TOON):` is filled with the **model’s prediction on the preceding frame** of the same video (temporal chaining). |

There is **no** standalone “metrics-only” script in this directory. After inference, metrics are computed with the **same** `eval_sgg_metrics_with_qwen.py` as for Swift-based runs (Section 2).

### API credentials

The environment variable **`OPEN_ROUTER_KEY`** must be set.

A convenient approach is a dotenv file **`metrics/open-router/env`** colocated with the scripts:

```env
OPEN_ROUTER_KEY=sk-or-v1-...your_key...
```

By default, scripts load `--env env`, i.e. `metrics/open-router/env`. The path may be overridden: `--env /path/to/.env`.

**Security:** do not commit real API keys to version control; keep `env` in `.gitignore` or export variables only in the shell session.

Dependencies: `requests`, `Pillow`, `tqdm`; optional `python-dotenv` for file-based configuration (without it, only the process environment is used).

### Execution (working directory `metrics/open-router`)

The current working directory matters: `--output-dir` and dataset paths are resolved **relative to the process cwd**, as in each script’s docstring.

**GT-prompt (PVSG example):**

```bash
cd metrics/open-router
python run_open_router_gt_prompt.py openai/gpt-5.4-mini \
  --data-test ../../datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --output-dir ../results/checkpoints-inference/open-router-models/PVSG-GT-prompt \
  --output-prefix psfr
```

**PSG (single-frame test):**

```bash
cd metrics/open-router
python run_open_router_gt_prompt.py openai/gpt-5.4-mini \
  --data-test ../../datasets/data_playground/PSG_json/test.jsonl \
  --output-dir ../results/checkpoints-inference/open-router-models/PSG \
  --output-prefix psg
```

**GEN-prompt (video-level chaining):**

```bash
cd metrics/open-router
python run_open_router_generated_prompt.py openai/gpt-5.4-mini \
  --data-test ../../datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --output-dir ../results/checkpoints-inference/open-router-models/PVSG-GEN-prompt \
  --output-prefix psfr
```

Additional flags include `--limit`, `--max-retries`, `--retry-delay`, `--dataset-suffix`, and `--env`.

### Output filename

$$
\texttt{\{output\_dir\}/\{model\_short\}-\{output\_prefix\}.jsonl}
$$

Here `model_short` denotes the portion of the model id after `/` (e.g. `gpt-5.4-mini`).

### JSONL record layout

Each line is a JSON object compatible with the evaluator: **`content`** holds ground-truth TOON from the dataset; **`predict`** holds the model response wrapped in `<answer>…</answer>`. Auxiliary fields record timing and billing where applicable.

Abbreviated example:

```json
{
  "messages": [{"role": "user", "content": "<image>\n..."}, {"role": "assistant", "content": "<answer>\nobj[...]\n...</answer>"}],
  "images": ["/abs/or/rel/path/to/frame.png"],
  "content": "<answer>... GT TOON ...</answer>",
  "predict": "<answer>... model TOON ...</answer>",
  "model_name": "gpt-5.4-mini",
  "dataset_suffix": "psfr",
  "gen_time_sec": 1.47,
  "cost": 0.00108
}
```

For selected model families, bounding boxes are mapped to image pixel coordinates inside the script (heuristics for GPT‑5 mini/nano, Gemini, etc.).

---

## 2. Qwen-bench / Swift (`metrics/qwen-bench`)

### Purpose

**Local inference** is performed via **ms-swift** (`swift.infer_engine`: **VllmEngine** or **TransformersEngine**). **Metric computation** is performed by **`eval/eval_sgg_metrics_with_qwen.py`**.

### Inference backend

In the **current** repository drivers, **`--infer-backend`** accepts only:

- **`vllm`** (default): batched server-style inference;
- **`transformers`**: in-process Hugging Face weights via `TransformersEngine`.

The broader **ms-swift** ecosystem may support additional engines (e.g. SGLang, LMDeploy). To use them here, **`build_swift_engine`** would need to be extended following [ms-swift documentation](https://github.com/modelscope/ms-swift), or the Swift CLI invoked separately. Only **`vllm`** and **`transformers`** are documented and exercised in this repository as shipped.

### Model argument

**`--model`** may be a **local path** (checkpoint directory after SFT/GRPO) or a **Hugging Face model id**, e.g. `Qwen/Qwen3.5-VL-7B-Instruct`.

### Dependencies

```bash
pip install 'ms-swift[llm]' -U
pip install vllm   # when using --infer-backend vllm
```

PyTorch and vision-language model dependencies apply as usual.

### GT-prompt: example command (repository root)

```bash
export CUDA_VISIBLE_DEVICES=0
export IMAGE_MAX_TOKEN_NUM=1024

python metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py \
  --model sft/Qwen3.5/work_dirs/your_exp/checkpoint-8844 \
  --test-jsonl datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test_clean.jsonl \
  --output-dir metrics/results/checkpoints-inference/sft/PVSG-GT-prompt \
  --run-name Qwen3.5-0.8B-SFT-maxinfo-checkpoint-8844-psfr-GT \
  --infer-backend vllm \
  --batch-size 64 \
  --max-new-tokens 2048 \
  --temperature 0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45 \
  --tensor-parallel-size 1 \
  --torch-dtype bfloat16
```

For zero-shot evaluation, use e.g. `--model Qwen/Qwen3.5-VL-7B-Instruct` with an appropriate `--run-name`.

Output path: **`{output_dir}/{run_name}.jsonl`**. If the file already exists, the run is **skipped** unless **`--force`** is supplied.

### GEN-prompt (temporal chaining): example command

```bash
export CUDA_VISIBLE_DEVICES=0
export IMAGE_MAX_TOKEN_NUM=1024

python metrics/qwen-bench/infer/GEN-prompt/infer_swift_gen_prompt.py \
  --model sft/Qwen3.5/work_dirs/your_exp/checkpoint-8844 \
  --test-jsonl datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test_clean.jsonl \
  --output-dir metrics/results/checkpoints-inference/sft/PVSG-GEN-prompt \
  --run-name Qwen3.5-0.8B-SFT-maxinfo-checkpoint-8844-psfr-GEN \
  --infer-backend vllm \
  --batch-size 64 \
  --prev-source model \
  --max-new-tokens 2048 \
  --temperature 0.0
```

- **`--prev-source model`:** the previous-frame block uses the model’s prediction at $t-1$ (video-consistent streaming).
- **`--prev-source gt`:** debugging mode; the previous-frame block uses ground truth.

Further options mirror the GT script (`--images-base`, `--template-type`, `--response-prefix`, `--enable-thinking`, `--auto-obj-prefix-fallback`, stop sequences, etc.); see `--help` on each script.

### Metric evaluation (`eval_sgg_metrics_with_qwen.py`)

Invoke **after** a JSONL file exists with **`content`** (ground truth) and **`predict`** (model output):

```bash
python metrics/qwen-bench/eval/eval_sgg_metrics_with_qwen.py \
  --pred-jsonl metrics/results/checkpoints-inference/sft/PVSG-GT-prompt/Qwen3.5-0.8B-SFT-maxinfo-checkpoint-8844-psfr-GT.jsonl \
  --output-dir metrics/results/checkpoints-metrics/sft/PVSG-GT-prompt \
  --output-name Qwen3.5-0.8B-SFT-maxinfo-checkpoint-8844-psfr-GT-metrics.json \
  --iou-thr 0.5 \
  --batch-size-qwen 32 \
  --gpu-memory-utilization 0.40 \
  --qwen-model-path Qwen/Qwen3-4B-Instruct-2507
```

The environment variable **`QWEN_MODEL_PATH`** selects the judge when **`--qwen-model-path`** is empty. The judge is loaded through **vLLM** (`vllm.LLM`), independently of the Swift inference stack.

Optional **`--per-sample-jsonl path.jsonl`:** each line is the original sample augmented with a per-frame **`metrics`** dictionary (useful for debugging and visualization).

---

### Metric definitions (as implemented)

**Parsing.** TOON is extracted from `content` / `predict` after normalization of `</think>` and `<answer>` blocks. Objects carry `id`, category name (normalized: lower case; `_` and `-` mapped to spaces), and bounding boxes. Relations are triples (subject id, predicate string, object id).

#### Bounding-box IoU

For axis-aligned boxes $A = [x_1^a,y_1^a,x_2^a,y_2^a]$ and $B$:

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

If the union area is zero, $\text{IoU}(A,B) = 0$.

#### Object matching

An IoU matrix $\text{IoU}_{ij}$ is formed over all ground-truth and predicted objects. **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) minimizes total cost $-\text{IoU}$ (equivalently, maximizes the sum of paired IoUs). A pair is retained only if

$$
\text{IoU}_{ij} \ge \theta,\quad \theta = \texttt{--iou-thr}\quad (\text{default: } 0.5).
$$

The mean IoU over accepted pairs is stored per sample as **`bbox_mean_iou_matched`**.

#### Objects: TP, FP, FN and precision, recall, F1

After pairs are fixed by IoU, **category names** are compared:

- exact string match after normalization counts as agreement;
- otherwise the pair is submitted to the **Qwen judge** (below): output **1** denotes synonymy / equivalent labeling; **0** denotes disagreement.

Let $N_{\text{gt}}$ and $N_{\text{pred}}$ denote object counts, and let $T$ denote the number of matched pairs (IoU $\ge \theta$) whose labels are deemed correct (exact match or judge output 1). The implementation uses:

$$
TP = T,\quad FP = N_{\text{pred}} - TP,\quad FN = N_{\text{gt}} - TP.
$$

$$
P_{\text{obj}} = \frac{TP}{TP+FP},\quad R_{\text{obj}} = \frac{TP}{TP+FN},\quad F1_{\text{obj}} = \frac{2 P_{\text{obj}} R_{\text{obj}}}{P_{\text{obj}} + R_{\text{obj}}}
$$

(with $F1_{\text{obj}} = 0$ if $P_{\text{obj}} + R_{\text{obj}} = 0$.)

In the aggregated JSON, **`Obj_AP@50_*`**, **`Obj_Recall_*`**, and **`Obj_F1_*`** denote the **arithmetic mean** of the corresponding per-sample quantities (macro-averaging over frames). Suffixes **`strict`** (all disputed pairs treated as mismatch; judge treated as 0) and **`Qwen`** (judge responses applied) distinguish the two regimes.

#### Relations: candidates and a second Hungarian step

A candidate is a pair $(r^{\text{gt}}, r^{\text{pred}})$ such that, **after mapping predicted object ids to ground-truth ids** via the bbox match, the **subject and object** refer to the same ground-truth entities. Predicate strings either match exactly or are adjudicated by the **triplet judge** (Qwen), with subject/object names aligned to ground-truth when object-level synonymy has already been established.

A binary cost matrix of size $N^{\text{rel}}_{\text{gt}} \times N^{\text{rel}}_{\text{pred}}$ is built: cost $0$ indicates an admissible edge (predicate agreement or positive judge verdict), cost $1$ otherwise. **Minimum-cost assignment** yields $TP$ as the count of zero-cost assignments; $FP$, $FN$, $P$, $R$, and $F1$ for relations follow the same formulas as for objects.

#### Aggregate scene-graph score

Per sample:

$$
\text{SGG\_score} = \frac{F1_{\text{obj}} + F1_{\text{rel}}}{2}.
$$

In the summary JSON, **`SGG_Score_strict`** and **`SGG_Score_qwen`** report the dataset-level mean.

### Qwen judge: invocation and outputs

- **Default model:** `Qwen/Qwen3-4B-Instruct-2507`, or a local directory via **`--qwen-model-path`** / **`QWEN_MODEL_PATH`**.
- **Objects:** batched prompts include textual summaries of the full ground-truth and predicted scenes; the system message requires a **single digit** `1` or `0`. Responses are parsed with simple token rules (`_parse_synonym_answer`).
- **Relations:** for non-identical predicates, prompts include full graphs and the specific triplet pair (after subject/object name normalization).

The judge is **not** applied to the entire graph indiscriminately: deterministic IoU matching and exact string matches are resolved first; Qwen is invoked only for **disputed** labels and predicates, reducing call volume.

### Example metrics summary (single JSON file)

Typical path: `metrics/results/checkpoints-metrics/sft/PVSG-GT-prompt/...-metrics.json`:

```json
{
  "predictions_file": "metrics/results/checkpoints-inference/sft/PVSG-GT-prompt/Qwen3.5-0.8B-SFT-base_annot-checkpoint-7860-psfr-GT.jsonl",
  "num_samples": 3039,
  "num_valid_toon": 3038,
  "num_invalid_toon": 1,
  "invalid_rate_pct": 0.0329,
  "iou_thr": 0.5,
  "time_sec": { "N": 3039, "mean": 0.037, "sigma": 0.015, "median": 0.034, "phys": "(0.037 ± 0.015) s" },
  "Obj_AP@50_strict": 0.7386,
  "Obj_AP@50_Qwen": 0.7412,
  "Obj_Recall_strict": 0.7328,
  "Obj_Recall_Qwen": 0.7357,
  "Obj_F1_strict": 0.7294,
  "Obj_F1_qwen": 0.7321,
  "Rel_AP@50_strict": 0.6312,
  "Rel_AP@50_Qwen": 0.6331,
  "Rel_Recall_strict": 0.6234,
  "Rel_Recall_Qwen": 0.6248,
  "Rel_F1_strict": 0.6139,
  "Rel_F1_qwen": 0.6154,
  "SGG_Score_strict": 0.6716,
  "SGG_Score_qwen": 0.6738
}
```

Invalid or unparseable predicted TOON yields zero metrics for that row and contributes to **`invalid_rate_pct`**.

---

## 3. Result artifacts (`metrics/results`)

The following layout is used in this repository:

```text
metrics/results/
├── checkpoints-inference/     # raw model runs (JSONL)
│   ├── sft/
│   │   ├── PVSG-GT-prompt/
│   │   ├── PVSG-GEN-prompt/
│   │   └── PSG/
│   ├── grpo/
│   │   └── PVSG-GT-prompt/ …
│   └── open-router-models/
│       ├── PVSG-GT-prompt/
│       ├── PVSG-GEN-prompt/
│       └── PSG/
└── checkpoints-metrics/       # one aggregated JSON per run
    ├── sft/
    ├── grpo/
    └── …
```

- **Inference JSONL:** one line per frame or sample; fields include **`content`**, **`predict`**, **`images`**, **`gen_time_sec`**; OpenRouter rows additionally include **`cost`**, **`model_name`**, **`dataset_suffix`** where applicable.
- **Metrics JSON:** one file per prediction run; fields as in the example above.

Filenames are chosen via **`--run-name`** (Swift) or **`{model_short}-{output_prefix}`** (OpenRouter) so that dataset, prompt type, and frame subsampling variant (e.g. `psfr`, `psg`) remain identifiable.

---

## 4. Classical metrics (`metrics/sgbench`)

The **`metrics/sgbench/`** directory is reserved for **classical** scene graph generation metrics (e.g. mean recall at K, recall at K, predicate-set conventions from the literature) **without** an LLM-based judge.

**Status:** not yet implemented. Once code is added, input formats (JSON/JSONL) and TOON alignment should be documented in this file.

---

## End-to-end pipeline

1. Obtain Swift-format test JSONL under `datasets/data_playground/...`.
2. Run inference with either an OpenRouter script or `infer_swift_*.py` → `metrics/results/checkpoints-inference/.../*.jsonl`.
3. Run **`eval_sgg_metrics_with_qwen.py`** with **`--pred-jsonl`** pointing to that file → `metrics/results/checkpoints-metrics/.../*-metrics.json`.

---

## Related documentation

- [SceneGraphVLM project README](../README.md)  
- [Visualization & demo videos](../visualization/vis.md)  
- [SFT training](../sft/SFT_README.md)  
- [PVSG dataset & exports](../datasets/annotations/PVSG_annot/PVSG_README.md)
