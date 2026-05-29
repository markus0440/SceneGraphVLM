#!/usr/bin/env bash
# LMDeploy speed tests — one model at a time, skip status=ok (10 samples).
# Native lmdeploy>=0.10 pipeline: Qwen3-VL, Qwen3.5 (env swift_qwen_lmdeploy).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$STI_ROOT/../../.." && pwd)"
STI_PY="$STI_ROOT/py_scripts/speed_test_infer.py"

cleanup_infer_procs () {
  # Do NOT use bare "lmdeploy" — matches conda env paths, tee log names, --infer-backend.
  pkill -f "[s]peed_test_infer.py" 2>/dev/null || true
  pkill -f "[l]mdeploy serve" 2>/dev/null || true
  pkill -f "vllm.entrypoints" 2>/dev/null || true
}

cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/metrics/results/speed_test_infer/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/lmdeploy_matrix_$(date +%Y%m%d_%H%M%S).log"
RESULT_DIR="$REPO_ROOT/metrics/results/speed_test_infer"
FORCE="${FORCE:-0}"
ENV_NAME="${LM_DEPLOY_ENV:-swift_qwen_lmdeploy}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export USE_HF=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$(conda info --base)/etc/profile.d/conda.sh"

already_ok () {
  local NAME="$1"
  python3 - "$NAME" "$RESULT_DIR" <<'PY'
import json, sys
from pathlib import Path
name, root = sys.argv[1:3]
base = Path(root) / "LMDeploy"
if not base.is_dir():
    sys.exit(1)
for f in base.glob("*.json"):
    d = json.load(open(f))
    if d.get("model_display_name") == name and d.get("infer_backend") == "lmdeploy":
        if d.get("status") == "ok" and (d.get("num_measured_samples") or 0) >= 10:
            sys.exit(0)
sys.exit(1)
PY
}

run_one () {
  local MODEL="$1"
  local NAME="$2"
  shift 2
  local EXTRA=("$@")

  if [[ "$FORCE" != "1" ]] && already_ok "$NAME"; then
    echo "[skip ok] $NAME lmdeploy" | tee -a "$LOG_FILE"
    return 0
  fi

  echo "======================================================================" | tee -a "$LOG_FILE"
  echo "[$(date -Iseconds)] $ENV_NAME | lmdeploy | $NAME | $MODEL" | tee -a "$LOG_FILE"
  echo "======================================================================" | tee -a "$LOG_FILE"

  conda activate "$ENV_NAME"
  export PATH="$(conda info --base)/envs/$ENV_NAME/bin:$PATH"
  cleanup_infer_procs
  sleep 5

  if python "$STI_PY" \
    --model "$MODEL" \
    --model-display-name "$NAME" \
    --infer-backend lmdeploy \
    --warmup-runs 2 \
    --force \
    "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    echo "[OK] $NAME lmdeploy" | tee -a "$LOG_FILE"
  else
    echo "[FAIL] $NAME lmdeploy" | tee -a "$LOG_FILE"
  fi
  cleanup_infer_procs
  sleep 8
}

echo "Log: $LOG_FILE  FORCE=$FORCE  ENV=$ENV_NAME" | tee -a "$LOG_FILE"

STD=(--gpu-memory-utilization 0.85 --max-model-len 8192)
QWEN4B=(--gpu-memory-utilization 0.85 --max-model-len 4096)

run_one Qwen/Qwen3.5-0.8B Qwen3.5-0.8B "${STD[@]}"
run_one Qwen/Qwen3.5-2B Qwen3.5-2B "${STD[@]}"
run_one Qwen/Qwen3.5-4B Qwen3.5-4B "${QWEN4B[@]}"
run_one Qwen/Qwen3-VL-2B-Instruct Qwen3-VL-2B "${STD[@]}"
run_one Qwen/Qwen3-VL-4B-Instruct Qwen3-VL-4B "${QWEN4B[@]}"

echo "[$(date -Iseconds)] FINISHED" | tee -a "$LOG_FILE"
