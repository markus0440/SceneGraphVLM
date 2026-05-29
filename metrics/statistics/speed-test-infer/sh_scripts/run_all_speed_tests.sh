#!/usr/bin/env bash
# Full speed-test matrix: vLLM + HF for all models that CAN run on RTX 5060 Ti 16GB.
# Skips runs that already have status=ok (10 samples). Set FORCE=1 to re-run everything.
#
# Single env swift_qwen_3_5_sft (Qwen, InternVL, DeepSeek/Ovis, SmolVLM).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$STI_ROOT/../../.." && pwd)"
STI_PY="$STI_ROOT/py_scripts/speed_test_infer.py"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/metrics/results/speed_test_infer/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_matrix_$(date +%Y%m%d_%H%M%S).log"
RESULT_DIR="$REPO_ROOT/metrics/results/speed_test_infer"
FORCE="${FORCE:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export USE_HF=1
export HF_XET_HIGH_PERFORMANCE=1

source "$(conda info --base)/etc/profile.d/conda.sh"

already_ok () {
  local NAME="$1" BACKEND="$2"
  python3 - "$NAME" "$BACKEND" "$RESULT_DIR" <<'PY'
import json, sys
from pathlib import Path

name, backend, root = sys.argv[1:4]
subdir = {"vllm": "vLLM", "transformers": "HF"}.get(backend, backend)
root = Path(root)
candidates = [root / subdir, root]
for base in candidates:
    if not base.is_dir():
        continue
    for f in base.glob("*.json"):
        d = json.load(open(f))
        if d.get("model_display_name") == name and d.get("infer_backend") == backend:
            if d.get("status") == "ok" and (d.get("num_measured_samples") or 0) >= 10:
                sys.exit(0)
sys.exit(1)
PY
}

run_job () {
  local ENV="$1" BACKEND="$2" MODEL="$3" NAME="$4"
  shift 4
  local EXTRA=("$@")

  if [[ "$FORCE" != "1" ]] && already_ok "$NAME" "$BACKEND"; then
    echo "[skip ok] $NAME $BACKEND" | tee -a "$LOG_FILE"
    return 0
  fi

  echo "======================================================================" | tee -a "$LOG_FILE"
  echo "[$(date -Iseconds)] $ENV | $BACKEND | $NAME | $MODEL" | tee -a "$LOG_FILE"
  echo "======================================================================" | tee -a "$LOG_FILE"

  conda activate "$ENV" >> "$LOG_FILE" 2>&1
  pkill -f "vllm" 2>/dev/null || true
  sleep 3

  if python "$STI_PY" \
    --model "$MODEL" \
    --model-display-name "$NAME" \
    --infer-backend "$BACKEND" \
    --warmup-runs 2 \
    --force \
    "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    echo "[OK] $NAME $BACKEND" | tee -a "$LOG_FILE"
  else
    echo "[FAIL] $NAME $BACKEND" | tee -a "$LOG_FILE"
  fi
}

run_pair () {
  local ENV="$1" NAME="$2" VLLM_MODEL="$3" HF_MODEL="$4"
  shift 4
  local EXTRA=("$@")
  run_job "$ENV" vllm "$VLLM_MODEL" "$NAME" "${EXTRA[@]}"
  run_job "$ENV" transformers "$HF_MODEL" "$NAME" "${EXTRA[@]}"
}

echo "Log: $LOG_FILE  FORCE=$FORCE" | tee -a "$LOG_FILE"

LARGE=(--gpu-memory-utilization 0.90 --max-model-len 6144)
QWEN4B_VLLM=(--gpu-memory-utilization 0.85 --max-model-len 4096)

# --- swift_qwen_3_5_sft ---
QWEN=swift_qwen_3_5_sft
run_pair "$QWEN" Qwen3.5-0.8B Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-0.8B
run_pair "$QWEN" Qwen3.5-2B Qwen/Qwen3.5-2B Qwen/Qwen3.5-2B
run_job "$QWEN" vllm Qwen/Qwen3.5-4B Qwen3.5-4B "${QWEN4B_VLLM[@]}"
run_job "$QWEN" transformers Qwen/Qwen3.5-4B Qwen3.5-4B
run_pair "$QWEN" Qwen3-VL-2B Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-2B-Instruct "${LARGE[@]}"
run_pair "$QWEN" Qwen3-VL-4B Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Instruct "${LARGE[@]}"
run_pair "$QWEN" Qwen2-VL-2B Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen2-VL-2B-Instruct "${LARGE[@]}"
run_pair "$QWEN" Qwen2.5-VL-3B Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-3B-Instruct "${LARGE[@]}"
# 7B: tight VRAM (5060 Ti 16GB) — same as on A100 protocol, lower KV budget
SEVENB=(--gpu-memory-utilization 0.92 --max-model-len 4096)
run_pair "$QWEN" Qwen2.5-VL-7B Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-7B-Instruct "${SEVENB[@]}"
run_pair "$QWEN" Qwen2-VL-7B Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen2-VL-7B-Instruct "${SEVENB[@]}"
run_pair "$QWEN" SmolVLM2-500M HuggingFaceTB/SmolVLM2-500M-Instruct HuggingFaceTB/SmolVLM2-500M-Instruct
run_pair "$QWEN" SmolVLM-500M HuggingFaceTB/SmolVLM-500M-Instruct HuggingFaceTB/SmolVLM-500M-Instruct

# --- InternVL (swift_qwen_3_5_sft) ---
IV=swift_qwen_3_5_sft
run_pair "$IV" InternVL2.5-1B OpenGVLab/InternVL2_5-1B OpenGVLab/InternVL2_5-1B
run_pair "$IV" InternVL2.5-2B OpenGVLab/InternVL2_5-2B OpenGVLab/InternVL2_5-2B
run_pair "$IV" InternVL3-1B OpenGVLab/InternVL3-1B OpenGVLab/InternVL3-1B
run_pair "$IV" InternVL3-2B OpenGVLab/InternVL3-2B OpenGVLab/InternVL3-2B
run_pair "$IV" InternVL3.5-1B OpenGVLab/InternVL3_5-1B OpenGVLab/InternVL3_5-1B
run_pair "$IV" InternVL3.5-2B OpenGVLab/InternVL3_5-2B OpenGVLab/InternVL3_5-2B

# --- DeepSeek / Ovis (swift_qwen_3_5_sft) ---
DS=swift_qwen_3_5_sft
run_pair "$DS" DeepSeek-VL2-tiny-3B deepseek-ai/deepseek-vl2-tiny deepseek-ai/deepseek-vl2-tiny "${LARGE[@]}"
OVIS_VLLM=(--gpu-memory-utilization 0.88 --max-model-len 4096)
run_job "$DS" vllm AIDC-AI/Ovis2.5-2B Ovis2.5-2B "${OVIS_VLLM[@]}"
run_job "$DS" transformers AIDC-AI/Ovis2.5-2B Ovis2.5-2B "${LARGE[@]}"

echo "======================================================================" | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] FINISHED" | tee -a "$LOG_FILE"
python3 - <<'PY' | tee -a "$LOG_FILE"
import json
from pathlib import Path
root = Path("metrics/results/speed_test_infer")
ok = fail = 0
for f in sorted(root.rglob("*.json")):
    if "logs" in f.parts:
        continue
    d = json.load(open(f))
    if d.get("status") == "ok" and (d.get("num_measured_samples") or 0) >= 10:
        ok += 1
    else:
        fail += 1
        print("FAIL", d.get("model_display_name"), d.get("infer_backend"), "|", (d.get("error") or d.get("status"))[:100])
print(f"\nOK (10 samples): {ok}  FAIL/partial: {fail}  total json: {ok+fail}")
print("\nNote: 7B on 16GB may FAIL with OOM — re-run alone after pkill -f vllm if needed.")
PY
