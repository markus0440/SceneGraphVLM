#!/usr/bin/env bash
# Qwen2/3/3.5-VL: ms-swift + vLLM + HF (speed-test primary env).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-swift_qwen_3_5_sft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# cu124 wheel index; override for RTX 50xx if needed: TORCH_INDEX=https://download.pytorch.org/whl/cu128
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ensure_conda
create_conda_env "$CONDA_ENV" "$PYTHON_VERSION"

echo "[pip] torch from $TORCH_INDEX"
conda run -n "$CONDA_ENV" pip install -U pip wheel setuptools
conda run -n "$CONDA_ENV" pip install torch torchvision --index-url "$TORCH_INDEX"
pip_install_requirements "$CONDA_ENV" "$REQ_DIR/requirements-qwen-vllm.txt"

smoke_import "$CONDA_ENV" "import swift, vllm, torch; print('ok', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo ""
echo "[done] conda activate $CONDA_ENV"
echo "  export CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024"
echo "  python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py --model Qwen/Qwen3.5-0.8B --model-display-name Qwen3.5-0.8B --infer-backend vllm --force"
