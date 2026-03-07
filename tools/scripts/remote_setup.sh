#!/usr/bin/env bash
# Remote GPU machine setup script for CTCF experiments.
# Usage: bash tools/scripts/remote_setup.sh [--data-dir /path/to/data]
#
# This script:
#   1. Installs system packages
#   2. Installs Miniconda (if missing)
#   3. Clones/updates the repo
#   4. Creates/updates conda env from environment.yml
#   5. Reinstalls pinned CUDA PyTorch wheels (deterministic stack)
#   6. Verifies GPU + torch
#   7. Saves dataset root for tools/scripts/run_experiments.sh
set -euo pipefail

REPO_URL="https://github.com/Palllladium/CTCF.git"
BRANCH="ctcf_legacy"
WORK_DIR="$HOME/CTCF"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="oasis-ctcf"
TORCH_VER="2.9.0"
TORCHVISION_VER="0.24.0"
TORCHAUDIO_VER="2.9.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

DATA_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== [1/7] System packages ==="
if command -v apt-get &>/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq git wget tmux curl
elif command -v yum &>/dev/null; then
  sudo yum install -y git wget tmux curl
else
  echo "WARNING: unknown package manager, assuming git/wget/tmux are available"
fi

echo "=== [2/7] Miniconda ==="
if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
  echo "Miniconda installed at $CONDA_DIR"
else
  echo "Miniconda already installed at $CONDA_DIR"
fi
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

echo "=== [3/7] Clone repo ==="
if [ -d "$WORK_DIR" ]; then
  echo "Repo already exists at $WORK_DIR, pulling latest..."
  cd "$WORK_DIR" && git fetch origin && git checkout "$BRANCH" && git pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

echo "=== [4/7] Conda environment ==="
if conda env list | grep -q "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists, updating..."
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"

echo "=== [5/7] Pin CUDA PyTorch stack ==="
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python -m pip install --index-url "$TORCH_INDEX_URL" \
  "torch==${TORCH_VER}" \
  "torchvision==${TORCHVISION_VER}" \
  "torchaudio==${TORCHAUDIO_VER}"

echo "=== [6/7] Verify GPU + PyTorch ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

echo "=== [7/7] Dataset ==="
if [ -n "$DATA_DIR" ]; then
  echo "Using data directory: $DATA_DIR"
  if [ ! -d "$DATA_DIR" ]; then
    echo "WARNING: $DATA_DIR does not exist yet. Upload your data before running experiments."
    mkdir -p "$DATA_DIR"
  fi
  echo "DATA_DIR=$DATA_DIR" > "$WORK_DIR/.env_data"
  echo "Data path saved to $WORK_DIR/.env_data"
else
  echo "No --data-dir specified."
  echo "You can still run with default /data or pass --data-dir later."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Repo:  $WORK_DIR"
echo "  Env:   conda activate $ENV_NAME"
echo "  Torch: torch==$TORCH_VER torchvision==$TORCHVISION_VER torchaudio==$TORCHAUDIO_VER (cu128)"
echo "  Next:  bash tools/scripts/run_experiments.sh --data-dir /path/to/data"
echo "============================================"
