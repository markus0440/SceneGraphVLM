#!/usr/bin/env bash
# 3RScan (3RScan_json): closed-vocabulary scene graph SFT with extended TOON
# (color + material per object), full finetune.
# Data: repo datasets/data_playground/3RScan_json_${TEMPORAL_MODE} (train.jsonl / test.jsonl).
#
# Run (from this directory):
#   CUDA_VISIBLE_DEVICES=0 ./run_3rscan.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_3rscan.sh
#   EPOCHS=10 ./run_3rscan.sh
#   ./run_3rscan.sh --epochs 10
# Defaults are conservative for full-dataset runs to avoid OOM.
# Faster (if memory allows): ./run_3rscan.sh --batch_size 4 --grad_accum 2
#
# Env:
#   DATA_BASE      Override JSONL root (default: ../../../datasets/data_playground).
#   EPOCHS         Default 5 if not passed as --epochs.
#   TEMPORAL_MODE  no_prev_gt (default) | with_prev_gt. Selects the
#                  3RScan_json_${TEMPORAL_MODE} subset and propagates to
#                  --exp_name and COMET_EXPERIMENT_NAME.
#   NUM_GPUS       Used only when CUDA_VISIBLE_DEVICES is unset (default 4).
#   BATCH_SIZE     Per-GPU micro-batch (default 2).
#   GRAD_ACCUM     Gradient accumulation steps (default 4).
#   COMET_PROJECT_NAME, COMET_EXPERIMENT_NAME  Logging (see run_sft.sh, .comet_env).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-$SCRIPT_DIR/../../../datasets/data_playground}"

EPOCHS="${EPOCHS:-5}"
TEMPORAL_MODE="${TEMPORAL_MODE:-with_prev_gt}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen_3_5_3rscan}"
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-Qwen3.5-0.8B | 3RScan full SFT | epochs=${EPOCHS} | temporal_mode=${TEMPORAL_MODE}}"

case "$TEMPORAL_MODE" in
  with_prev_gt|no_prev_gt) ;;
  *) echo "ERROR: TEMPORAL_MODE must be 'with_prev_gt' or 'no_prev_gt' (got: $TEMPORAL_MODE)" >&2; exit 1 ;;
esac

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
  --train "$DATA_BASE/3RScan_json_${TEMPORAL_MODE}/train_clean.jsonl" \
  --test "$DATA_BASE/3RScan_json_${TEMPORAL_MODE}/test_clean.jsonl" \
  --exp_name 3rscan_close_${TEMPORAL_MODE} \
  --tuner_type full \
  --epochs "$EPOCHS" \
  "${NUM_EXTRA[@]}" \
  "$@"
