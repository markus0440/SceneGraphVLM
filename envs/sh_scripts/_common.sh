# Shared helpers for conda env installers (SFT + speed-test).
# shellcheck shell=bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_DIR="$(cd "$INSTALL_DIR/../requirements_files" && pwd)"
REPO_ROOT="$(cd "$INSTALL_DIR/../.." && pwd)"

conda_exists() {
  command -v conda >/dev/null 2>&1
}

ensure_conda() {
  if ! conda_exists; then
    echo "[error] conda not found. Install Miniconda and run: conda init bash" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

create_conda_env() {
  local name="$1"
  local pyver="${2:-3.11}"
  if env_exists "$name"; then
    echo "[skip] conda env already exists: $name"
  else
    echo "[create] conda env $name (python=$pyver)"
    conda create -y -n "$name" "python=${pyver}"
  fi
}

pip_install_requirements() {
  local env_name="$1"
  local req_file="$2"
  echo "[pip] $env_name ← $req_file"
  conda run -n "$env_name" pip install -U pip wheel setuptools
  conda run -n "$env_name" pip install -r "$req_file"
}

smoke_import() {
  local env_name="$1"
  shift
  echo "[smoke] $*"
  conda run -n "$env_name" python -c "$*"
}
