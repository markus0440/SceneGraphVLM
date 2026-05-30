#!/usr/bin/env bash
# PVSG base annotation (pvsg_base_annot_gt_prompt): train on base annot split;
# validation on PSFR test (same held-out split as other PVSG experiments).
#
# Run (from this directory):
#   CUDA_VISIBLE_DEVICES=0 ./run_pvsg_base_annot.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_pvsg_base_annot.sh
#   EPOCHS=10 ./run_pvsg_base_annot.sh
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

EPOCHS="${EPOCHS:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen_3_5_pvsg}"
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-Qwen3.5-0.8B | PVSG base_annot train | psfr val | epochs=${EPOCHS}}"

NUM_EXTRA=()
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && NUM_EXTRA+=(--num_gpus "${NUM_GPUS:-4}")

exec "$SCRIPT_DIR/run_sft.sh" \
  --train "$PVSG/pvsg_base_annot_gt_prompt/train.jsonl" \
  --test "$PVSG/pvsg_psfr_gt_prompt/test.jsonl" \
  --exp_name pvsg_base_annot \
  --tuner_type full \
  --epochs "$EPOCHS" \
  "${NUM_EXTRA[@]}" \
  "$@"
