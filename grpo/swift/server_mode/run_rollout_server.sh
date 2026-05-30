#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

swift rollout \
  --model "${ROLLOUT_MODEL:-Qwen/Qwen3.5-0.8B}" \
  --model_type qwen3_5 \
  --template qwen3_5 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attention_2 \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_data_parallel_size "${VLLM_DATA_PARALLEL_SIZE:-1}"
