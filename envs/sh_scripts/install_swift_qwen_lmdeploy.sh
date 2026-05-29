#!/usr/bin/env bash
# Qwen3-VL + Qwen3.5 via native lmdeploy pipeline (lmdeploy>=0.10).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-swift_qwen_lmdeploy}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ensure_conda
create_conda_env "$CONDA_ENV" "$PYTHON_VERSION"

conda run -n "$CONDA_ENV" pip install -U pip wheel "setuptools>=70,<82"
conda run -n "$CONDA_ENV" pip install torch torchvision --index-url "$TORCH_INDEX"
pip_install_requirements "$CONDA_ENV" "$REQ_DIR/requirements-qwen-lmdeploy.txt"

smoke_import "$CONDA_ENV" "
import sys
sys.path.insert(0, '$REPO_ROOT/metrics/statistics/speed-test-infer/py_scripts')
from model_patches import patch_lmdeploy_consumer_gpu
patch_lmdeploy_consumer_gpu()
import lmdeploy
from lmdeploy_native_engine import needs_native_lmdeploy
assert needs_native_lmdeploy('Qwen/Qwen3.5-0.8B')
print('  lmdeploy', lmdeploy.__version__)
print('  native path OK')
"

echo "[done] conda activate $CONDA_ENV"
echo "  python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py --model Qwen/Qwen3.5-0.8B --model-display-name Qwen3.5-0.8B --infer-backend lmdeploy --force"
