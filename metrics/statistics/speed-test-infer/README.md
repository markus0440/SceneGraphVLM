# Scene graph speed test (portable pipeline)

Benchmark **PSG GT-prompt** scene-graph generation at **batch size 1**, with warmup and JSON artifacts portable across GPUs (e.g. RTX 5060 Ti vs A100).

## Layout

```text
metrics/statistics/speed-test-infer/
  py_scripts/
    speed_test_infer.py       # main benchmark entry point
    aggregate_speed_results.py
    swift_engine.py
    model_patches.py
    lmdeploy_native_engine.py
    smolvlm_engine.py
    speed_test_utils.py
  sh_scripts/
    run_all_speed_tests.sh    # vLLM + HF matrix
    run_lmdeploy_benchmarks.sh
    run_sglang_benchmarks.sh
  configs/
    models_hf.yaml
  README.md

metrics/results/speed_test_infer/
  vLLM/ | HF/ | LMDeploy/ | SGLang/
  summary.csv
  pivot_tokens_per_second.json
```

## 0. Install environments

```bash
cd /path/to/SceneGraphVLM
bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh
conda activate swift_qwen_3_5_sft
```

Details: [envs/SWIFT_README.md](../../../envs/SWIFT_README.md).

## Single run

```bash
export CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024

python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py \
  --model Qwen/Qwen3.5-0.8B \
  --model-display-name Qwen3.5-0.8B \
  --infer-backend vllm \
  --manifest datasets/data_playground/PSG_json/test.jsonl \
  --warmup-runs 2 \
  --force
```

Backends: `vllm`, `transformers` (HF), `lmdeploy`, `sglang`.

**HF model ids:** see [configs/models_hf.yaml](configs/models_hf.yaml).  
Qwen3.5 multimodal models are `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` (not `Qwen3.5-VL-*-Instruct`).  
Older VL line: `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`.

**SmolVLM2** (`HuggingFaceTB/SmolVLM2-500M-Instruct`) is **not** in ms-swift. The script uses a native vLLM/HF/SGLang path (`smolvlm_engine.py`). Requires `pip install num2words`. vLLM SmolVLM does not report TTFT (batch API). HF does.

## Batch runs (by backend)

From repo root:

```bash
# vLLM + HF (all model families)
bash metrics/statistics/speed-test-infer/sh_scripts/run_all_speed_tests.sh

# LMDeploy (legacy + native envs)
bash metrics/statistics/speed-test-infer/sh_scripts/run_lmdeploy_benchmarks.sh

# SGLang
bash metrics/statistics/speed-test-infer/sh_scripts/run_sglang_benchmarks.sh
```

Each script skips models that already have `status=ok` with 10 measured samples. Set `FORCE=1` to re-run everything.

## Aggregate results

```bash
python metrics/statistics/speed-test-infer/py_scripts/aggregate_speed_results.py \
  --results-dir metrics/results/speed_test_infer
```

Prints an ASCII comparison table and writes `summary.csv` + `pivot_tokens_per_second.json`.

## Output JSON

| Key | Meaning |
|-----|---------|
| `gpu` | GPU name |
| `model` / `model_display_name` | HF id / label |
| `accelerator` | vLLM, HF, LMDeploy, SGLang |
| `samples[].tokens_per_second` | main metric |
| `aggregate.*` | `mean`, `median`, `std` (σ), `phys` = `mean ± σ` |
| `input_tokens_total` | prompt size (`usage.prompt_tokens` when available) |
| `text_prompt_tokens_total` | `input_tokens_total − visual_tokens` |
| `visual_tokens` | vision grid tokens (`image_grid_thw // merge_size²`). Not the single `<\|image_pad\|>` left in vLLM encode |
| `system_prompt_tokens` | explicit system in JSONL or `template.default_system` (Qwen3.5: usually **0**) |
| `template_prefix_tokens` | chat-template assistant prefix (e.g. empty-thinking block for Qwen3.5) |
| `user_prompt_tokens` | text prompt minus system and template prefix |
| `summary_table` / `summary_table_text` | aggregated stats + ASCII table |
| `environment` | versions, driver, conda env |

Token counts use Swift `template.encode` plus image-processor grid math for vLLM/LMDeploy/SGLang (those backends expand vision at runtime, so counting `image_token_id` in `input_ids` would be wrong).

## Environments

| Conda env | Backend | Install script |
|-----------|---------|------------------|
| `swift_qwen_3_5_sft` | vLLM + HF (all model families) | `envs/sh_scripts/install_swift_qwen_3_5_sft.sh` |
| `swift_qwen_sglang` | SGLang | `envs/sh_scripts/install_swift_qwen_sglang.sh` |
| `swift_qwen_lmdeploy` | LMDeploy (Qwen3-VL, Qwen3.5) | `envs/sh_scripts/install_swift_qwen_lmdeploy.sh` |

## Related

- [metrics/metrics.md](../../metrics.md)
- [envs/SWIFT_README.md](../../../envs/SWIFT_README.md)
