#!/usr/bin/env bash
# tools/install_mamba.sh
#
# Install mamba_ssm and causal-conv1d in the active conda environment for
# MambaMorph and VMambaMorph training. Requires:
#   - Linux (Mamba kernels are not officially supported on Windows)
#   - NVIDIA GPU + CUDA Toolkit (>= 11.6) installed system-wide (`nvcc` in PATH)
#   - Active conda env with PyTorch+CUDA already installed
#
# Tested target: advisor's machine (Ubuntu, conda env "ctcf", CUDA Toolkit 13.2).
# On CUDA 13.x the install will likely build from source (~10-15 min).
#
# Usage:
#   conda activate ctcf
#   bash tools/install_mamba.sh

set -e

echo "── Step 1/5: Verifying environment ──"
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found in PATH. Install full CUDA Toolkit (>=11.6) first."
    echo "  Download: https://developer.nvidia.com/cuda-toolkit-archive"
    echo "  Or via apt: sudo apt-get install nvidia-cuda-toolkit"
    exit 1
fi
nvcc --version

echo
echo "── Step 1/3: Upgrading pip toolchain ──"
pip install --upgrade pip wheel setuptools

echo
echo "── Step 2/3: Installing causal-conv1d (~5-10 min compile) ──"
pip install "causal-conv1d>=1.4.0" --no-build-isolation

echo
echo "── Step 3/3: Installing mamba-ssm (~10-15 min compile) ──"
pip install mamba-ssm --no-build-isolation

echo
echo "── Verification ──"
python -c "
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch, einops
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 100, 64).cuda()
y = m(x); print(f'mamba_ssm.Mamba OK: {y.shape}')
print('selective_scan_fn imported OK')
print(f'einops {einops.__version__} OK')
"

echo
echo "✓ Installation successful. mamba_ssm ready for MambaMorph and VMambaMorph training."
