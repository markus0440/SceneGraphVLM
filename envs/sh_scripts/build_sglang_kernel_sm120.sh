#!/usr/bin/env bash
# Build sglang-kernel from source for Blackwell (sm_120) + torch 2.11 (sglang 0.5.12).
# Run from repo root. Requires: nvcc (CUDA 12.8+), ~30–90 min, several GB disk/RAM.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-swift_qwen_sglang}"
SGLANG_TAG="${SGLANG_TAG:-v0.5.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
BUILD_DIR="${BUILD_DIR:-/tmp/sglang-src}"
MAX_JOBS="${MAX_JOBS:-4}"
COMPILE_THREADS="${COMPILE_THREADS:-2}"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ensure_conda

if ! pkg-config --exists numa 2>/dev/null && [[ ! -f /usr/include/numa.h ]]; then
  echo "[deps] libnuma-dev missing (required by mscclpp in sgl-kernel build)"
  if command -v apt-get >/dev/null 2>&1; then
    echo "  Run: sudo apt-get install -y libnuma-dev libibverbs-dev"
    if [[ "${INSTALL_BUILD_DEPS:-}" == "1" ]] && command -v sudo >/dev/null 2>&1; then
      sudo apt-get install -y libnuma-dev libibverbs-dev
    else
      echo "[error] Install system deps and retry, or: INSTALL_BUILD_DEPS=1 bash $0" >&2
      exit 1
    fi
  else
    echo "[error] Install numa development headers for your distro, then retry." >&2
    exit 1
  fi
fi

echo "[1/5] Align torch with sglang 0.5.12 (torch==2.11.0)"
conda run -n "$CONDA_ENV" pip install -U pip wheel setuptools scikit-build-core cmake ninja uv
conda run -n "$CONDA_ENV" pip install "torch==2.11.0" "torchvision==0.26.0" --index-url "$TORCH_INDEX"

echo "[2/5] Clone sglang ${SGLANG_TAG} → ${BUILD_DIR}"
if [[ -d "$BUILD_DIR/.git" ]]; then
  git -C "$BUILD_DIR" fetch --tags --depth 1 origin "refs/tags/${SGLANG_TAG}:refs/tags/${SGLANG_TAG}" 2>/dev/null || true
  git -C "$BUILD_DIR" checkout -f "$SGLANG_TAG"
  git -C "$BUILD_DIR" submodule update --init --recursive
else
  rm -rf "$BUILD_DIR"
  git clone --depth 1 --branch "$SGLANG_TAG" --recursive https://github.com/sgl-project/sglang.git "$BUILD_DIR"
fi

echo "[3/5] Build sgl-kernel wheel (MAX_JOBS=${MAX_JOBS}, compile threads=${COMPILE_THREADS})"
cd "$BUILD_DIR/sgl-kernel"
conda run -n "$CONDA_ENV" env MAX_JOBS="$MAX_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" \
  CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=${COMPILE_THREADS}" \
  make build

echo "[4/5] Smoke test sgl_kernel + CUDA"
conda run -n "$CONDA_ENV" python -c "
import torch
import sgl_kernel
x = torch.zeros(1, device='cuda')
print('sgl_kernel ok', torch.__version__, 'cc', torch.cuda.get_device_capability())
"

echo "[5/5] Done. Run speed test:"
echo "  conda activate $CONDA_ENV"
echo "  export CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024"
echo "  python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py --model Qwen/Qwen3.5-0.8B --model-display-name Qwen3.5-0.8B --infer-backend sglang --force"
