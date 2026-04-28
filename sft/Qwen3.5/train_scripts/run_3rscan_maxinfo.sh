#!/usr/bin/env bash
# 3RScan maxinfo (with_prev_gt): train on MaxInfo, validate on PSFR.
#
# Run (from this directory):
#   CUDA_VISIBLE_DEVICES=0 ./run_3rscan_maxinfo.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_3rscan_maxinfo.sh
#   EPOCHS=10 ./run_3rscan_maxinfo.sh
# Defaults are conservative for full-dataset runs to avoid OOM.
#
# Env:
#   DATA_BASE   Override JSONL root (default: ../../../datasets/data_playground).
#   EPOCHS      Default 5.
#   NUM_GPUS    Used only when CUDA_VISIBLE_DEVICES is unset (default 4).
#   BATCH_SIZE  Per-GPU micro-batch (default 2).
#   GRAD_ACCUM  Gradient accumulation steps (default 4).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-$SCRIPT_DIR/../../../datasets/data_playground}"
R3SCAN="$DATA_BASE/3RScan_json"
SUBSET="$R3SCAN/3RScan_json_maxinfo_with_prev_gt"
VAL_SUBSET="$R3SCAN/3RScan_json_psfr_with_prev_gt"

EPOCHS="${EPOCHS:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen_3_5_3rscan}"
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-Qwen3.5-0.8B | 3RScan maxinfo with_prev_gt clean | epochs=${EPOCHS}}"

NUM_EXTRA=()
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && NUM_EXTRA+=(--num_gpus "${NUM_GPUS:-4}")

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _csv="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
  IFS=',' read -r -a _r3rs_cuda <<< "$_csv"
  R3RS_NPROC="${#_r3rs_cuda[@]}"
else
  R3RS_NPROC="${NUM_GPUS:-4}"
fi
if [[ "$R3RS_NPROC" -ge 1 ]]; then
  export BATCH_SIZE="${BATCH_SIZE:-2}"
fi
export GRAD_ACCUM="${GRAD_ACCUM:-4}"

exec "$SCRIPT_DIR/run_sft.sh" \
  --train "$SUBSET/train_clean.jsonl" \
  --test "$VAL_SUBSET/test_clean.jsonl" \
  --exp_name 3rscan_maxinfo_with_prev_gt \
  --tuner_type full \
  --epochs "$EPOCHS" \
  "${NUM_EXTRA[@]}" \
  "$@"
