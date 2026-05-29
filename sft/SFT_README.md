# Supervised fine-tuning (SFT) in SceneGraphVLM

Training scripts target **[MSwift](https://github.com/modelscope/swift)** (`swift sft`) and follow the same practical defaults as the upstream **[Qwen3.5 SFT best-practice guide](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html)** (e.g. **`--add_non_thinking_prefix true`**, **`--loss_scale ignore_empty_think`**, **`torch_dtype bfloat16`**, flash-style attention path, etc.). For **other models or recipes**, use whatever MSwift documents for that stack instead of copying these flags blindly.

**Environment:** install the conda env used by the dataset READMEs from [`envs/SWIFT_README.md`](../envs/SWIFT_README.md) (`bash envs/sh_scripts/install_swift_qwen_3_5_sft.sh`), then `conda activate swift_qwen_3_5_sft`.

---

## Layout: `sft/Qwen3.5/`

```text
sft/Qwen3.5/
├── train_scripts/           # thin wrappers → run_sft.sh
│   ├── run_sft.sh           # core driver: swift sft + DeepSpeed + logging
│   ├── run_psg.sh
│   ├── run_ag.sh
│   ├── run_pvsg_base_annot.sh
│   ├── run_pvsg_maxinfo.sh
│   ├── run_pvsg_psfr.sh
│   └── run_pvsg_all.sh
├── work_dirs/               # outputs: checkpoints, Swift run folders, logging.jsonl
├── logs/                    # ignored by git if present (legacy). Primary logs are under work_dirs/.../logs/
├── .hf_cache/               # local HF cache (HF_HOME / HF_DATASETS_CACHE)
└── .comet_env               # optional: COMET_API_KEY (git-ignored — create locally)
```

- **`train_scripts/`** — all dataset-specific entrypoints **`exec` into `run_sft.sh`**, which builds the final `swift sft` command.
- **`work_dirs/`** — default **`WORK_DIRS`** (overridable via env). Each run uses  
  **`work_dirs/<exp_name>_<model_tag>/`** (e.g. `psg_close_Qwen3.5-0.8B`) as **`--output_dir`**. MSwift then creates a versioned subdirectory (e.g. `v2-20260409-191817/`) with **`logging.jsonl`**, **`args.json`**, **`checkpoint-*`**, etc.
- **`<exp_name>`** comes from **`--exp_name`** in the wrapper, or from the train JSONL path if omitted (see `run_sft.sh`).
- **`<model_tag>`** is derived from **`--model`** (default **`Qwen/Qwen3.5-0.8B`** → `Qwen3.5-0.8B`).

---

## How to run

From the **SceneGraphVLM repository root** (recommended):

```bash
conda activate swift_qwen_3_5_sft
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sft/Qwen3.5/train_scripts/run_psg.sh
```

Or `cd sft/Qwen3.5/train_scripts` and call a script with a **relative path to JSONL** (see comments at the top of `run_sft.sh`).

| Script | Train JSONL | Validation JSONL |
|--------|-------------|------------------|
| `run_psg.sh` | `datasets/data_playground/PSG_json/train.jsonl` | `.../PSG_json/test.jsonl` | `qwen_3_5_psg` |
| `run_ag.sh` | `.../AG_json/train.jsonl` | `.../AG_json/test.jsonl` |
| `run_pvsg_base_annot.sh` | `.../PVSG_json/pvsg_base_annot_gt_prompt/train_clean.jsonl` | `.../pvsg_psfr_gt_prompt/test_clean.jsonl` |
| `run_pvsg_maxinfo.sh` | `.../pvsg_maxinfo_gt_prompt/train_clean.jsonl` | `.../pvsg_psfr_gt_prompt/test_clean.jsonl` |
| `run_pvsg_psfr.sh` | `.../pvsg_psfr_gt_prompt/train_clean.jsonl` | `.../pvsg_psfr_gt_prompt/test_clean.jsonl` |
| `run_pvsg_all.sh` | `.../pvsg_all_data_gt_prompt/train.jsonl` | `.../pvsg_psfr_gt_prompt/test.jsonl` |

Override data root: **`DATA_BASE=/path/to/data_playground`** (wrappers default to `../../../datasets/data_playground` relative to `train_scripts/`).

Extra CLI flags are passed through to **`run_sft.sh`** (e.g. `./run_psg.sh --batch_size 2 --grad_accum 4`).

---

## Training hyperparameters (defaults in `run_sft.sh`)

Unless overridden by flags or env:

| Setting | Default |
|---------|---------|
| Model | `Qwen/Qwen3.5-0.8B` (`--model` / `MODEL`) |
| Tuner | **Full** finetune (`--tuner_type full`. LoRA supported via flags) |
| Epochs | **5** (`--epochs`. Wrappers also honor `EPOCHS=`) |
| Learning rate | **1e-5** |
| Max length | **8192** |
| Batch | **`BATCH_SIZE` per device** default **4** (AG: **12** if exactly **1** GPU) |
| Gradient accumulation | **2** (`GRAD_ACCUM`) |
| Warmup | **`warmup_ratio` 0.05** |
| Precision | **`bfloat16`** |
| Attention | **`flash_attention_2`** (or `flash_attn` when `--packing true`) |
| Padding-free | **`true`** (`PADDING_FREE`) |
| Liger | **`--use_liger_kernel true`** |
| DeepSpeed | **`zero2`** (`--deepspeed` / `DEEPSPEED`. `zero3` optional) |
| Group by length | **`true`** |
| Packing | **`false`** |
| Dataset | **`--split_dataset_ratio 0`**. Val set is explicit `--val_dataset` |
| Eval / save | **Every half epoch** in steps (`eval_steps` / `save_steps` derived from JSONL length) |
| Checkpoints kept | **`save_total_limit` 4**. **`load_best_model_at_end`** on **`eval_loss`** |
| Dataloader workers | **16** |
| Logging interval | **`logging_steps` 10** |

Qwen3.5-specific flags wired in **`run_sft.sh`**: **`--add_non_thinking_prefix true`**, **`--loss_scale ignore_empty_think`** (aligned with the best-practice doc for non-reasoning chat data).

**GPUs:** if **`CUDA_VISIBLE_DEVICES`** is set, **`NPROC_PER_NODE`** = number of listed IDs. Otherwise **`--num_gpus`** (default **4**) sets `CUDA_VISIBLE_DEVICES=0,1,…`.

---

## Logs, checkpoints, and experiment tracking

- **Console + file log:** `run_sft.sh` uses **`tee`** to append a timestamped file under  
  **`work_dirs/<exp_name>_<model_tag>/logs/train_YYYYMMDD_HHMMSS.log`**.
- **MSwift / Hugging Face Trainer:** metrics and events also land in the run subdirectory as **`logging.jsonl`** (next to `checkpoint-*`).
- **Default `--report_to`:** **`comet_ml`** (unless you pass **`--report_to none`** or e.g. **`tensorboard,comet_ml`**).

### Comet ML (default)

1. Create a file **`sft/Qwen3.5/.comet_env`** (or `train_scripts/.comet_env`. The driver prefers the parent path first) containing:
   ```bash
   export COMET_API_KEY="YOUR_COMET_API_KEY"
   ```
2. Optionally set **`COMET_PROJECT_NAME`** / **`COMET_EXPERIMENT_NAME`** in the shell. Dataset wrappers already export sensible **`COMET_PROJECT_NAME`** defaults (see table above) and set an experiment **title** string.

**`*.comet_env` is git-ignored** — do not commit API keys.

To **disable** cloud logging:  
`./run_psg.sh --report_to none` (or export and pass the same through wrappers).

---

## Resume and overrides

- **`--resume /path/to/checkpoint-dir`** → passed as **`--resume_from_checkpoint`**. If **`--extra_epochs`** is omitted, **`--epochs`** is reused as the **total** epoch count for the resumed run (see `run_sft.sh` comments).
- **`WORK_DIRS`**, **`HF_CACHE_ROOT`**, **`MASTER_PORT`**, **`COMET_*`**, **`BATCH_SIZE`**, **`GRAD_ACCUM`**, **`DEEPSPEED`**, VL **`MAX_PIXELS`** / **`VIDEO_*`** — see the long header comment in **`run_sft.sh`** for the full list.

---

## References

- [MSwift documentation](https://swift.readthedocs.io/en/latest/)
- [Qwen3.5 SFT best practices](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html)
- Local env installer: [`envs/SWIFT_README.md`](../envs/SWIFT_README.md)

---

## Related documentation

- [SceneGraphVLM project README](../README.md)  
- [Metrics & evaluation](../metrics/metrics.md)  
- [Visualization & demos](../visualization/vis.md)  
- [PVSG dataset & exports](../datasets/annotations/PVSG_annot/PVSG_README.md)
