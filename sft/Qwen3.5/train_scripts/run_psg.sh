#!/usr/bin/env bash
# PSG (PSG_json): closed-vocabulary scene graph SFT, full finetune.
# Data: repo datasets/data_playground/PSG_json (train.jsonl / test.jsonl).
# Qwen3.5 template: run_sft.sh uses --add_non_thinking_prefix and ignore_empty_think;
# use the same template at inference to avoid train/infer mismatch.
#
# Run (from this directory):
#   CUDA_VISIBLE_DEVICES=0 ./run_psg.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_psg.sh
#   EPOCHS=10 ./run_psg.sh
# OOM: ./run_psg.sh --batch_size 2 --grad_accum 4
#
# Env:
#   DATA_BASE   Override JSONL root (default: ../../../datasets/data_playground).
#   EPOCHS      Default 5 if not passed as --epochs.
#   NUM_GPUS    When CUDA_VISIBLE_DEVICES unset (default 4).
#   COMET_PROJECT_NAME, COMET_EXPERIMENT_NAME  See run_sft.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-$SCRIPT_DIR/../../../datasets/data_playground}"

EPOCHS="${EPOCHS:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen_3_5_psg}"
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-Qwen3.5-0.8B | PSG | full SFT | epochs=${EPOCHS}}"

NUM_EXTRA=()
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && NUM_EXTRA+=(--num_gpus "${NUM_GPUS:-4}")

exec "$SCRIPT_DIR/run_sft.sh" \
  --train "$DATA_BASE/PSG_json/train_clean.jsonl" \
  --test "$DATA_BASE/PSG_json/test_clean.jsonl" \
  --exp_name psg_close \
  --tuner_type full \
  --epochs "$EPOCHS" \
  "${NUM_EXTRA[@]}" \
  "$@"
