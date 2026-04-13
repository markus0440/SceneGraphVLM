#!/usr/bin/env bash
set -e

CONDA_ENV="${CONDA_ENV:-swift_qwen_3_5_sft}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REQUIREMENTS="${ENV_DIR}/requirements_files/requirements-swift-qwen-3-5-sft.txt"

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  conda create -n "${CONDA_ENV}" python=3.11 -y
fi

conda activate "${CONDA_ENV}"

python -m pip install -U pip
python -m pip install --no-build-isolation -r "${REQUIREMENTS}"
