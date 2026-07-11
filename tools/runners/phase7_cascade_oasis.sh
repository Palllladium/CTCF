#!/usr/bin/env bash
# Phase 7 cascade runs on OASIS.
#
# Main Paper 2 cascade matrix:
#   - LKU-8      lightweight large-kernel CNN backbone
#   - LKU-32     high-capacity LKU width under our unsupervised protocol
#   - MambaMorph diffeomorphic SSM backbone + SVF L3
#   - VMambaMorph diffeomorphic cross-scan SSM backbone + SVF L3
#
# Usage:
#   conda activate ctcf
#   bash tools/phase7_cascade_oasis.sh
#
# Override defaults:
#   MAX_EPOCH=100 GPU=0 PATHS_PROFILE=--2 bash tools/phase7_cascade_oasis.sh
#
# Fast smoke / staged execution:
#   SKIP_LKU32=1 SKIP_MAMBA=1 SKIP_VMAMBA=1 MAX_EPOCH=5 bash tools/phase7_cascade_oasis.sh

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_LKU8="${SKIP_LKU8:-0}"
SKIP_LKU32="${SKIP_LKU32:-0}"
SKIP_MAMBA="${SKIP_MAMBA:-0}"
SKIP_VMAMBA="${SKIP_VMAMBA:-0}"

COMMON="--ds OASIS ${PATHS_PROFILE} --gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"
CTCF_COMMON="${COMMON} --w_ncc 1.0 --w_reg 1.0 --w_icon 0.05 --w_jac 0.005"

run_ctcf() {
    local exp_name="$1"; shift
    echo "> ${exp_name}"
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}" ${CTCF_COMMON}
}

if [ "${SKIP_LKU8}" != "1" ]; then
    run_ctcf "P7_CASC_LKU8_OASIS" --config CTCF-CascadeA-LKU8
fi

if [ "${SKIP_LKU32}" != "1" ]; then
    run_ctcf "P7_CASC_LKU32_OASIS" --config CTCF-CascadeA-LKU32
fi

if [ "${SKIP_MAMBA}" != "1" ]; then
    run_ctcf "P7_CASC_MAMBA_SVF_OASIS" --config CTCF-CascadeA-Mamba --l3_svf 1
fi

if [ "${SKIP_VMAMBA}" != "1" ]; then
    run_ctcf "P7_CASC_VMAMBA_SVF_OASIS" --config CTCF-CascadeA-VMamba --l3_svf 1
fi

echo "Phase 7 cascade runs complete."
echo "Results in logs/P7_*/logfile.log and experiments/P7_*/ckpt/best.pth"
