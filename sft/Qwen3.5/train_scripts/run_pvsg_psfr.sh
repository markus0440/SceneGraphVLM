#!/usr/bin/env bash
# PVSG PSFR (pvsg_psfr_gt_prompt): train and validate on the PSFR JSONL split.
#
# Run (from this directory):
#   CUDA_VISIBLE_DEVICES=0 ./run_pvsg_psfr.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_pvsg_psfr.sh
#   EPOCHS=10 ./run_pvsg_psfr.sh
#
# Env:
#   DATA_BASE   Override JSONL root (default: ../../../datasets/data_playground).
#   EPOCHS      Default 5.
#   NUM_GPUS    When CUDA_VISIBLE_DEVICES unset (default 4).
#   COMET_PROJECT_NAME, COMET_EXPERIMENT_NAME  See run_sft.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-$SCRIPT_DIR/../../../datasets/data_playground}"
PVSG="$DATA_BASE/PVSG_json"
PSFR="$PVSG/pvsg_psfr_gt_prompt"

EPOCHS="${EPOCHS:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen_3_5_pvsg}"
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-Qwen3.5-0.8B | PVSG psfr full SFT clean | epochs=${EPOCHS}}"

NUM_EXTRA=()
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && NUM_EXTRA+=(--num_gpus "${NUM_GPUS:-4}")

exec "$SCRIPT_DIR/run_sft.sh" \
  --train "$PSFR/train_clean.jsonl" \
  --test "$PSFR/test_clean.jsonl" \
  --exp_name pvsg_psfr \
  --tuner_type full \
  --epochs "$EPOCHS" \
  "${NUM_EXTRA[@]}" \
  "$@"
