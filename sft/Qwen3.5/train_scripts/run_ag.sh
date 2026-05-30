#!/usr/bin/env bash
# Action Genome (AG_json): closed-vocabulary scene graph SFT, full finetune.
# Data: repo datasets/data_playground/AG_json (train.jsonl / test.jsonl).
#
# Run (from this directory):
#   CUDA_VISIBLE_DEVICES=0 ./run_ag.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_ag.sh
#   EPOCHS=10 ./run_ag.sh
#   ./run_ag.sh --epochs 10
# OOM / long context: ./run_ag.sh --batch_size 2 --grad_accum 4
#
# Env:
#   DATA_BASE   Override JSONL root (default: ../../../datasets/data_playground).
#   EPOCHS      Default 5 if not passed as --epochs.
#   NUM_GPUS    Used only when CUDA_VISIBLE_DEVICES is unset (default 4).
#   BATCH_SIZE  Per-GPU micro-batch; default 12 if 1 GPU, else 4 (AG-specific).
#   GRAD_ACCUM  Gradient accumulation steps (default 2).
#   COMET_PROJECT_NAME, COMET_EXPERIMENT_NAME  Logging (see run_sft.sh, .comet_env).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-$SCRIPT_DIR/../../../datasets/data_playground}"

EPOCHS="${EPOCHS:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen_3_5_ag}"
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-Qwen3.5-0.8B AG full SFT epochs=${EPOCHS}}"

NUM_EXTRA=()
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && NUM_EXTRA+=(--num_gpus "${NUM_GPUS:-4}")

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _csv="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
  IFS=',' read -r -a _ag_cuda <<< "$_csv"
  AG_NPROC="${#_ag_cuda[@]}"
else
  AG_NPROC="${NUM_GPUS:-4}"
fi
if [[ "$AG_NPROC" -eq 1 ]]; then
  export BATCH_SIZE="${BATCH_SIZE:-12}"
else
  export BATCH_SIZE="${BATCH_SIZE:-4}"
fi
export GRAD_ACCUM="${GRAD_ACCUM:-2}"

exec "$SCRIPT_DIR/run_sft.sh" \
  --train "$DATA_BASE/AG_json/train.jsonl" \
  --test "$DATA_BASE/AG_json/test.jsonl" \
  --exp_name ag_close \
  --tuner_type full \
  --epochs "$EPOCHS" \
  "${NUM_EXTRA[@]}" \
  "$@"
