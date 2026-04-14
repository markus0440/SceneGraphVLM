#!/usr/bin/env bash
# Build the swift_qwen_3_5_sft environment Docker image.
#
# Usage (from any directory):
#   bash envs/sh_scripts/build_docker_swift_qwen_3_5_sft.sh
#   IMAGE_TAG=my-tag bash envs/sh_scripts/build_docker_swift_qwen_3_5_sft.sh
#
# Build-arg overrides (via env):
#   CUDA_TAG          e.g. 12.6.2-devel-ubuntu24.04
#   TORCH_INDEX       e.g. cu126
#   MAX_JOBS          parallel compilation jobs (default 4)
#   TORCH_CUDA_ARCH_LIST  e.g. "8.0;8.6;8.9;9.0a"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-sgvlm-sft-env}"

BUILD_ARGS=()
[[ -n "${CUDA_TAG:-}" ]]              && BUILD_ARGS+=(--build-arg "CUDA_TAG=${CUDA_TAG}")
[[ -n "${TORCH_INDEX:-}" ]]           && BUILD_ARGS+=(--build-arg "TORCH_INDEX=${TORCH_INDEX}")
[[ -n "${MAX_JOBS:-}" ]]              && BUILD_ARGS+=(--build-arg "MAX_JOBS=${MAX_JOBS}")
[[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]  && BUILD_ARGS+=(--build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")

echo "=== Building image: ${IMAGE_TAG} ==="
echo "    Dockerfile : ${SCRIPT_DIR}/Dockerfile.swift_qwen_3_5_sft"
echo "    Context    : ${ENVS_DIR}/"

docker build \
  -f "${SCRIPT_DIR}/Dockerfile.swift_qwen_3_5_sft" \
  -t "${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}" \
  "${ENVS_DIR}/"
