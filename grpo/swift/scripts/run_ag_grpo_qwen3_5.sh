#!/usr/bin/env bash
set -euo pipefail

# Absolute paths are kept intentionally for the unified Docker image.
GRPO_ROOT="${GRPO_ROOT:-/workspace/grpo}"
DATA_ROOT="${DATA_ROOT:-/workspace/datasets/data_playground}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/grpo/models}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,6}"

export COMET_START_ONLINE="${COMET_START_ONLINE:-1}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-scenegraphvlm}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-qwen3_5_1b_grpo_ag_nozerorel}"
REPORT_TO="${REPORT_TO:-comet_ml}"
if [[ "${REPORT_TO}" == "comet_ml" ]]; then
  : "${COMET_API_KEY:?COMET_API_KEY is required when REPORT_TO=comet_ml. Set REPORT_TO=none to disable external logging.}"
  export COMET_API_KEY
fi

export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-60}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_ROOT}/work_dirs/AG_without_GT_prompt/checkpoint-13149" \
  --model_type qwen3_5 \
  --template qwen3_5 \
  --enable_thinking false \
  --stop_words "</answer>" "<|endoftext|>" "<|im_end|>" "<|end|>" \
  --tuner_type full \
  --freeze_vit true --freeze_aligner true --freeze_llm false \
  --dataset "${DATA_ROOT}/AG_json/grpo/train_noprevgraph.jsonl" \
  --val_dataset "${DATA_ROOT}/AG_json/grpo/eval_noprevgraph.jsonl" \
  --split_dataset_ratio 0 \
  --val_dataset_shuffle false \
  --eval_strategy steps \
  --eval_steps 100 \
  --per_device_eval_batch_size 24 \
  --metric_for_best_model reward \
  --greater_is_better true \
  --create_checkpoint_symlink true \
  --external_plugins "${GRPO_ROOT}/swift/rewards/ag_toon_reward.py" \
  --reward_funcs \
  sgg_format_reward sgg_node_acc_reward sgg_node_box_reward sgg_edge_hard_reward sgg_edge_precision_reward sgg_edge_f1_reward sgg_obj_hallucination_penalty sgg_edge_hallucination_penalty \
  sgg_diag_frac_no_rel sgg_diag_num_pred_objs sgg_diag_num_pred_rels sgg_diag_frac_invalid_rel sgg_diag_has_answer_tags \
  --reward_weights \
  1 1 1 1 1 1 1 1 \
  0 0 0 0 0 \
  --use_vllm true \
  --vllm_mode colocate \
  --system "${GRPO_ROOT}/swift/prompts/system_prompt.txt" \
  --temperature 0.9 \
  --vllm_gpu_memory_utilization 0.30 \
  --vllm_max_model_len 7296 \
  --sleep_level 0 \
  --vllm_tensor_parallel_size 1 \
  --deepspeed zero2 \
  --torch_dtype bfloat16 \
  --max_length 6144 \
  --truncation_strategy delete \
  --max_completion_length 1024 \
  --per_device_train_batch_size 24 \
  --generation_batch_size 48 \
  --gradient_accumulation_steps 1 \
  --num_generations 12 \
  --num_iterations 1 \
  --learning_rate 6e-7 \
  --beta 0.04 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  --scale_rewards none \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 16 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 30 \
  --output_dir "${MODEL_ROOT}/qwen3_5_1b_grpo_ag_noprevgraph" \
  --log_completions true \
  --attn_impl flash_attention_2 \
  --padding_free false \
  --report_to "${REPORT_TO}" \
  "${EXTRA_ARGS[@]}"
