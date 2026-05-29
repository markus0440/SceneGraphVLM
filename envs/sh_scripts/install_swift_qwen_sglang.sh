#!/usr/bin/env bash
# Optional: Qwen + SGLang only (separate from vLLM env). May require matching CUDA toolkit for sglang-kernel.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-swift_qwen_sglang}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# RTX 50xx (sm_120): match torch+torchvision from one index (same as swift_qwen_3_5_sft uses cu130).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ensure_conda
create_conda_env "$CONDA_ENV" "$PYTHON_VERSION"

conda run -n "$CONDA_ENV" pip install -U pip wheel setuptools
pip_install_requirements "$CONDA_ENV" "$REQ_DIR/requirements-qwen-sglang.txt"
# sglang pins torch==2.11.0; requirements may pull mismatched torchvision — align both on cu130.
echo "[pip] reconcile torch==2.11.0 + torchvision from $TORCH_INDEX"
conda run -n "$CONDA_ENV" pip install "torch==2.11.0" "torchvision==0.26.0" --index-url "$TORCH_INDEX"

smoke_import "$CONDA_ENV" "import qwen_vl_utils, decord; import torch; torch.zeros(1, device='cuda'); from swift.infer_engine import SglangEngine; print('ok', torch.__version__)"

echo "[done] conda activate $CONDA_ENV"
echo "  export CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024"
echo "  python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py --model Qwen/Qwen3.5-0.8B --model-display-name Qwen3.5-0.8B --infer-backend sglang --force"
