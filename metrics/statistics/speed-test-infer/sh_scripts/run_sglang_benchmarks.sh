#!/usr/bin/env bash
# SGLang speed tests — one model at a time, skip status=ok (10 samples).
# Env: swift_qwen_sglang (Qwen2/2.5/3/3.5-VL via ms-swift SglangEngine).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$STI_ROOT/../../.." && pwd)"
STI_PY="$STI_ROOT/py_scripts/speed_test_infer.py"

cleanup_infer_procs () {
  # Do NOT use bare "sglang" — matches conda env paths, script/tee names, --infer-backend.
  pkill -f "[s]peed_test_infer.py" 2>/dev/null || true
  pkill -f "sglang::scheduler" 2>/dev/null || true
  pkill -f "sglang::detokenizer" 2>/dev/null || true
  pkill -f "vllm.entrypoints" 2>/dev/null || true
}

cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/metrics/results/speed_test_infer/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sglang_matrix_$(date +%Y%m%d_%H%M%S).log"
RESULT_DIR="$REPO_ROOT/metrics/results/speed_test_infer"
FORCE="${FORCE:-0}"
ENV_NAME="${SGLANG_ENV:-swift_qwen_sglang}"

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
base = Path(root) / "SGLang"
if not base.is_dir():
    sys.exit(1)
for f in base.glob("*.json"):
    d = json.load(open(f))
    if d.get("model_display_name") == name and d.get("infer_backend") == "sglang":
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
    echo "[skip ok] $NAME sglang" | tee -a "$LOG_FILE"
    return 0
  fi

  echo "======================================================================" | tee -a "$LOG_FILE"
  echo "[$(date -Iseconds)] $ENV_NAME | sglang | $NAME | $MODEL" | tee -a "$LOG_FILE"
  echo "======================================================================" | tee -a "$LOG_FILE"

  conda activate "$ENV_NAME"
  export PATH="$(conda info --base)/envs/$ENV_NAME/bin:$PATH"
  cleanup_infer_procs
  sleep 5

  if python "$STI_PY" \
    --model "$MODEL" \
    --model-display-name "$NAME" \
    --infer-backend sglang \
    --warmup-runs 2 \
    --no-stream \
    --force \
    "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    echo "[OK] $NAME sglang" | tee -a "$LOG_FILE"
  else
    echo "[FAIL] $NAME sglang" | tee -a "$LOG_FILE"
  fi
  cleanup_infer_procs
  sleep 8
}

echo "Log: $LOG_FILE  FORCE=$FORCE  ENV=$ENV_NAME" | tee -a "$LOG_FILE"

STD=(--gpu-memory-utilization 0.85 --max-model-len 8192)
QWEN4B=(--gpu-memory-utilization 0.85 --max-model-len 4096)

# --- Qwen (swift_qwen_sglang; ms-swift SglangEngine) ---
run_one Qwen/Qwen3.5-0.8B Qwen3.5-0.8B "${STD[@]}"
run_one Qwen/Qwen3.5-2B Qwen3.5-2B "${STD[@]}"
run_one Qwen/Qwen3.5-4B Qwen3.5-4B "${QWEN4B[@]}"
run_one Qwen/Qwen3-VL-2B-Instruct Qwen3-VL-2B "${STD[@]}"
run_one Qwen/Qwen3-VL-4B-Instruct Qwen3-VL-4B "${QWEN4B[@]}"
run_one Qwen/Qwen2-VL-2B-Instruct Qwen2-VL-2B "${STD[@]}"
run_one Qwen/Qwen2.5-VL-3B-Instruct Qwen2.5-VL-3B "${STD[@]}"
# 7B: skip on RTX 5060 Ti 16GB (OOM on vLLM/LMDeploy too)
# run_one Qwen/Qwen2.5-VL-7B-Instruct Qwen2.5-VL-7B --gpu-memory-utilization 0.88 --max-model-len 4096
# run_one Qwen/Qwen2-VL-7B-Instruct Qwen2-VL-7B --gpu-memory-utilization 0.88 --max-model-len 4096

# --- InternVL (ms-swift SglangEngine + processor_output patch) ---
run_one OpenGVLab/InternVL2_5-1B InternVL2.5-1B "${STD[@]}"
run_one OpenGVLab/InternVL2_5-2B InternVL2.5-2B "${STD[@]}"
run_one OpenGVLab/InternVL3-1B InternVL3-1B "${STD[@]}"
run_one OpenGVLab/InternVL3-2B InternVL3-2B "${STD[@]}"
run_one OpenGVLab/InternVL3_5-1B InternVL3.5-1B "${STD[@]}"
run_one OpenGVLab/InternVL3_5-2B InternVL3.5-2B "${STD[@]}"

# --- SmolVLM (native sglang.Engine; ms-swift has no smolvlm model_type) ---
run_one HuggingFaceTB/SmolVLM2-500M-Instruct SmolVLM2-500M "${STD[@]}"
run_one HuggingFaceTB/SmolVLM-500M-Instruct SmolVLM-500M "${STD[@]}"

echo "[$(date -Iseconds)] FINISHED" | tee -a "$LOG_FILE"
