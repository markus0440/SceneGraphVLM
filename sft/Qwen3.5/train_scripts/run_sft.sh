#!/usr/bin/env bash
# ms-swift SFT driver (`swift sft`). Wrappers pass --train / --test JSONL paths.
#
# From repo root (paths relative to cwd):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash sft/Qwen3.5/train_scripts/run_sft.sh \
#     --train datasets/data_playground/PSG_json/train.jsonl \
#     --test  datasets/data_playground/PSG_json/test.jsonl
# From this directory:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_sft.sh \
#     --train ../../../datasets/data_playground/PSG_json/train.jsonl \
#     --test  ../../../datasets/data_playground/PSG_json/test.jsonl
#
# GPUs:
#   If CUDA_VISIBLE_DEVICES is set, its length defines NPROC_PER_NODE; --num_gpus is ignored.
#   Otherwise --num_gpus N (default 4) sets CUDA_VISIBLE_DEVICES=0,1,...,N-1.
#
# Required:
#   --train PATH   Training JSONL (ms-swift dataset file)
#   --test PATH    Validation JSONL (passed as --val_dataset; split_dataset_ratio 0)
#
# Optional CLI (all also overridable via env where noted):
#   --epochs N              Training epochs (default 5). Env: same via wrappers.
#   --resume PATH           Checkpoint dir for --resume_from_checkpoint. If set and
#                           --extra_epochs omitted, training length uses --epochs as extra epochs.
#   --extra_epochs N        Override num_train_epochs when resuming.
#   --num_gpus N            GPU count when CUDA_VISIBLE_DEVICES unset (default 4).
#   --model ID              Hugging Face model id (default Qwen/Qwen3.5-0.8B). Env: MODEL.
#   --exp_name NAME         Prefix for output dir under work_dirs: {exp_name}_{model_tag}
#                           (e.g. ag_close_Qwen3.5-0.8B). Default from train path if omitted.
#   --tuner_type full|lora  Full finetune vs LoRA (default full).
#   --lora_rank N           LoRA rank (default 8).
#   --lora_alpha N          LoRA alpha (default 32).
#   --target_modules STR    LoRA target modules (default all-linear).
#   --batch_size N          per_device_train/eval_batch_size. Env: BATCH_SIZE (default 4).
#   --grad_accum N          gradient_accumulation_steps. Env: GRAD_ACCUM (default 2).
#   --logging_steps N       Logging interval. Env: LOGGING_STEPS (default 10).
#   --max_length N          Max sequence length (default 8192).
#   --learning_rate F         Adam LR (default 1e-5).
#   --dataloader_workers N  dataloader_num_workers. Env: DATALOADER_NUM_WORKERS (default 16).
#   --dataset_num_proc N    --dataset_num_proc for swift. Env: DATASET_NUM_PROC (optional).
#   --group_by_length BOOL  group_by_length true|false. Env: GROUP_BY_LENGTH (default true).
#   --packing BOOL          Sequence packing. Env: PACKING if flag omitted (default false).
#   --deepspeed zero2|zero3 DeepSpeed preset. Env: DEEPSPEED (default zero2).
#   --save_only_model BOOL  Smaller checkpoints; resume may miss optimizer. Env: implicit.
#   --save_total_limit N    Max checkpoints to keep. Env: SAVE_TOTAL_LIMIT (default 4).
#   --load_best_model_at_end BOOL  Env: LOAD_BEST_MODEL_AT_END (default true).
#   --metric_for_best_model NAME   Env: METRIC_FOR_BEST_MODEL (default eval_loss).
#   --greater_is_better BOOL       Env: GREATER_IS_BETTER (default false).
#   --report_to LIST        e.g. tensorboard,comet_ml or none. Default: comet_ml.
#   --comet_project NAME    Sets COMET_PROJECT_NAME.
#   --model_author STR      swift metadata (default swift).
#   --model_name STR        swift metadata (default swift-robot).
#
# Env (not duplicated as flags): WORK_DIRS (default ../work_dirs from this script), HF_CACHE_ROOT,
#   MASTER_PORT, COMET_*, ATTN_IMPL,
# PADDING_FREE, TRITON_CACHE_DIR, VL limits MAX_PIXELS, VIDEO_MAX_PIXELS, FPS_MAX_FRAMES,
# IMAGE_MAX_TOKEN_NUM, VIDEO_MAX_TOKEN_NUM.
#
# Comet: load .comet_env from train_scripts/ or parent sft/Qwen3.5/ (preferred).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMET_ENV_CANDIDATES=(
  "$SCRIPT_DIR/../.comet_env"
  "$SCRIPT_DIR/.comet_env"
)
for comet_env in "${COMET_ENV_CANDIDATES[@]}"; do
  if [[ -f "$comet_env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$comet_env"
    set +a
    break
  fi
done

# Default: sft/Qwen3.5/work_dirs (sibling of train_scripts/)
WORK_DIRS="${WORK_DIRS:-$SCRIPT_DIR/../work_dirs}"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
PADDING_FREE="${PADDING_FREE:-true}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

HF_CACHE_ROOT="${HF_CACHE_ROOT:-$SCRIPT_DIR/.hf_cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

if [[ -z "${MASTER_PORT:-}" ]]; then
  if MASTER_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)"; then
    export MASTER_PORT
  else
    export MASTER_PORT="29500"
  fi
else
  export MASTER_PORT
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN=""
TEST=""
COMET_PROJECT=""
EPOCHS=5
RESUME=""
EXTRA_EPOCHS=""
NUM_GPUS=4
EXP_NAME=""
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_LENGTH=8192
LEARNING_RATE="1e-5"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-}"
TUNER_TYPE="full"
LORA_RANK=8
LORA_ALPHA=32
TARGET_MODULES="all-linear"
MODEL_AUTHOR="swift"
MODEL_NAME="swift-robot"
REPORT_TO=""
PACKING_CLI=""
DEEPSPEED_CFG="${DEEPSPEED:-zero2}"
SAVE_ONLY_MODEL="false"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-true}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-false}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-true}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN="$2"; shift 2 ;;
    --test) TEST="$2"; shift 2 ;;
    --comet_project) COMET_PROJECT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --extra_epochs) EXTRA_EPOCHS="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --logging_steps) LOGGING_STEPS="$2"; shift 2 ;;
    --grad_accum) GRAD_ACCUM="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --dataloader_workers) DATALOADER_NUM_WORKERS="$2"; shift 2 ;;
    --dataset_num_proc) DATASET_NUM_PROC="$2"; shift 2 ;;
    --tuner_type) TUNER_TYPE="$2"; shift 2 ;;
    --lora_rank) LORA_RANK="$2"; shift 2 ;;
    --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
    --target_modules) TARGET_MODULES="$2"; shift 2 ;;
    --model_author) MODEL_AUTHOR="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --report_to) REPORT_TO="$2"; shift 2 ;;
    --packing) PACKING_CLI="$2"; shift 2 ;;
    --deepspeed) DEEPSPEED_CFG="$2"; shift 2 ;;
    --save_only_model) SAVE_ONLY_MODEL="$2"; shift 2 ;;
    --save_total_limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --load_best_model_at_end) LOAD_BEST_MODEL_AT_END="$2"; shift 2 ;;
    --metric_for_best_model) METRIC_FOR_BEST_MODEL="$2"; shift 2 ;;
    --greater_is_better) GREATER_IS_BETTER="$2"; shift 2 ;;
    --group_by_length) GROUP_BY_LENGTH="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TRAIN" || -z "$TEST" ]]; then
  echo "Usage: $0 --train TRAIN.jsonl --test TEST.jsonl [options]" >&2
  exit 1
fi

[[ -n "$COMET_PROJECT" ]] && export COMET_PROJECT_NAME="$COMET_PROJECT"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _csv="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
  IFS=',' read -r -a _cuda_ids <<< "$_csv"
  NPROC_PER_NODE="${#_cuda_ids[@]}"
  [[ "$NPROC_PER_NODE" -ge 1 ]] || { echo "ERROR: CUDA_VISIBLE_DEVICES is empty or invalid" >&2; exit 1; }
else
  [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --num_gpus must be a positive integer" >&2; exit 1; }
  NPROC_PER_NODE="$NUM_GPUS"
  _csv=""
  for ((i = 0; i < NUM_GPUS; i++)); do
    [[ -n "$_csv" ]] && _csv+=","
    _csv+="$i"
  done
  export CUDA_VISIBLE_DEVICES="$_csv"
fi
export NPROC_PER_NODE

[[ -n "$RESUME" && -z "$EXTRA_EPOCHS" ]] && EXTRA_EPOCHS="$EPOCHS"

if [[ -z "$EXP_NAME" ]]; then
  _train_dir="$(dirname "$TRAIN")"
  _train_parent="$(dirname "$_train_dir")"
  EXP_NAME="$(basename "$_train_dir")_$(basename "$_train_parent")"
fi
export COMET_EXPERIMENT_NAME="${COMET_EXPERIMENT_NAME:-$EXP_NAME}"

MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG//[^a-zA-Z0-9._-]/_}"

if [[ -n "$RESUME" ]]; then
  OUTPUT_DIR="$(cd "$(dirname "$RESUME")" && pwd)"
else
  OUTPUT_DIR="$WORK_DIRS/${EXP_NAME}_${MODEL_TAG}"
fi

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR" "$WORK_DIRS"

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="$(mktemp -d /tmp/swift_sft.XXXXXX)"
elif [[ ${#TMPDIR} -gt 72 ]]; then
  export TMPDIR="$(mktemp -d /tmp/swift_sft.XXXXXX)"
fi
mkdir -p "$TMPDIR"

LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

TRAIN_SAMPLES=$(wc -l <"$TRAIN")
EFFECTIVE_BATCH=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
STEPS_PER_EPOCH=$(( (TRAIN_SAMPLES + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))
EVAL_STEPS=$(( STEPS_PER_EPOCH / 2 ))
[[ "$EVAL_STEPS" -lt 1 ]] && EVAL_STEPS=1
SAVE_STEPS="$EVAL_STEPS"

PACKING="${PACKING_CLI:-${PACKING:-false}}"
ATTN_EFFECTIVE="$ATTN_IMPL"
[[ "$PACKING" == "true" ]] && ATTN_EFFECTIVE="flash_attn"

REPORT_ARGS=(--report_to)
if [[ -n "$REPORT_TO" ]]; then
  if [[ "$REPORT_TO" == "none" ]]; then
    REPORT_ARGS+=(none)
  else
    _IFS=$IFS
    IFS=',' read -r -a _rparts <<< "${REPORT_TO//[[:space:]]/}"
    IFS=$_IFS
    for x in "${_rparts[@]}"; do
      [[ -n "$x" ]] && REPORT_ARGS+=("$x")
    done
  fi
else
  REPORT_ARGS+=(comet_ml)
fi

case "$TUNER_TYPE" in
  full) LORA_ARGS=() ;;
  lora) LORA_ARGS=(--lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --target_modules "$TARGET_MODULES") ;;
  *) echo "ERROR: --tuner_type must be full or lora" >&2; exit 1 ;;
esac

RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

DATASET_NUM_PROC_ARGS=()
[[ -n "$DATASET_NUM_PROC" ]] && DATASET_NUM_PROC_ARGS=(--dataset_num_proc "$DATASET_NUM_PROC")

if [[ "$LOAD_BEST_MODEL_AT_END" == "true" ]]; then
  BEST_CKPT_ARGS=(
    --load_best_model_at_end true
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
    --greater_is_better "$GREATER_IS_BETTER"
  )
else
  BEST_CKPT_ARGS=(--load_best_model_at_end false)
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_ds_$$}"
mkdir -p "$TRITON_CACHE_DIR"

[[ -n "${MAX_PIXELS:-}" ]] && export MAX_PIXELS
[[ -n "${VIDEO_MAX_PIXELS:-}" ]] && export VIDEO_MAX_PIXELS
[[ -n "${FPS_MAX_FRAMES:-}" ]] && export FPS_MAX_FRAMES
[[ -n "${IMAGE_MAX_TOKEN_NUM:-}" ]] && export IMAGE_MAX_TOKEN_NUM
[[ -n "${VIDEO_MAX_TOKEN_NUM:-}" ]] && export VIDEO_MAX_TOKEN_NUM

[[ "$GROUP_BY_LENGTH" == "true" || "$GROUP_BY_LENGTH" == "false" ]] || {
  echo "ERROR: --group_by_length must be true or false (got: $GROUP_BY_LENGTH)" >&2
  exit 1
}

exec > >(tee -a "$LOG_FILE") 2>&1

# Qwen3.5: add_non_thinking_prefix + ignore_empty_think match official SFT recipe for non-reasoning data.
swift sft \
  --model "$MODEL" \
  --run_name "$COMET_EXPERIMENT_NAME" \
  --tuner_type "$TUNER_TYPE" \
  "${LORA_ARGS[@]}" \
  --dataset "$TRAIN" \
  --val_dataset "$TEST" \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --split_dataset_ratio 0 \
  --val_dataset_shuffle false \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EXTRA_EPOCHS:-$EPOCHS}" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --group_by_length "$GROUP_BY_LENGTH" \
  --packing "$PACKING" \
  --output_dir "$OUTPUT_DIR" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  "${BEST_CKPT_ARGS[@]}" \
  --save_only_model "$SAVE_ONLY_MODEL" \
  --logging_steps "$LOGGING_STEPS" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy delete \
  --warmup_ratio 0.05 \
  "${DATASET_NUM_PROC_ARGS[@]}" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --use_liger_kernel true \
  --attn_impl "$ATTN_EFFECTIVE" \
  --padding_free "$PADDING_FREE" \
  --deepspeed "$DEEPSPEED_CFG" \
  --model_author "$MODEL_AUTHOR" \
  --model_name "$MODEL_NAME" \
  "${REPORT_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
