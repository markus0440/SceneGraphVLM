#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-scenegraphvlm_data_prep}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS="${ENV_DIR}/requirements_files/requirements-dataset-preprocessing.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${CONDA_ENV}"

python -m pip install -U pip wheel
python -m pip install "setuptools<60"
python -m pip install "Cython<3"
python -m pip install -r "${REQUIREMENTS}"

MAXVOLPY_VERSION="0.3.8"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

curl -L \
  "https://files.pythonhosted.org/packages/source/m/maxvolpy/maxvolpy-${MAXVOLPY_VERSION}.tar.gz" \
  -o "${TMP_DIR}/maxvolpy-${MAXVOLPY_VERSION}.tar.gz"

tar -xzf "${TMP_DIR}/maxvolpy-${MAXVOLPY_VERSION}.tar.gz" -C "${TMP_DIR}"

sed -i 's/extra_info=npconf\.lapack_opt_info/extra_info={}/' \
  "${TMP_DIR}/maxvolpy-${MAXVOLPY_VERSION}/maxvolpy/setup.py"

python -m pip install \
  --no-build-isolation \
  --no-deps \
  "${TMP_DIR}/maxvolpy-${MAXVOLPY_VERSION}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "Warning: ffmpeg was not found on PATH."
  echo "AG frame extraction and some video utilities require ffmpeg."
fi

echo
echo "Dataset preprocessing environment is ready:"
echo "  conda activate ${CONDA_ENV}"
