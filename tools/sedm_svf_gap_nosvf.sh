#!/usr/bin/env bash
# SEDM SVF gap closure: train and infer only the missing NoSVF runs.
#
# Purpose:
#   Close the fair L3-SVF OFF/ON comparison matrix for SEDM without rerunning
#   already completed SVF checkpoints.
#
# Runs:
#   SEDM_CASC_LKU8_NOSVF_IXI
#   SEDM_CASC_VXM_NOSVF_OASIS
#   SEDM_CASC_VXM_NOSVF_IXI
#   SEDM_CASC_MAMBA_NOSVF_OASIS
#   SEDM_CASC_MAMBA_NOSVF_IXI
#
# Usage:
#   conda activate ctcf
#   bash tools/sedm_svf_gap_nosvf.sh
#
# Common overrides:
#   GPU=0 MAX_EPOCH=100 PATHS_PROFILE=--2 bash tools/sedm_svf_gap_nosvf.sh
#   RUN_TRAIN=0 bash tools/sedm_svf_gap_nosvf.sh
#   RUN_INFERENCE=0 bash tools/sedm_svf_gap_nosvf.sh
#   SKIP_MAMBA_OASIS=1 SKIP_MAMBA_IXI=1 bash tools/sedm_svf_gap_nosvf.sh
#   DRY_RUN=1 bash tools/sedm_svf_gap_nosvf.sh

set -euo pipefail

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"

SKIP_LKU8_IXI="${SKIP_LKU8_IXI:-0}"
SKIP_VXM_OASIS="${SKIP_VXM_OASIS:-0}"
SKIP_VXM_IXI="${SKIP_VXM_IXI:-0}"
SKIP_MAMBA_OASIS="${SKIP_MAMBA_OASIS:-0}"
SKIP_MAMBA_IXI="${SKIP_MAMBA_IXI:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

COMMON="--gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"
CTCF_BASE="${COMMON} --w_ncc 1.0 --w_icon 0.05 --w_jac 0.005"
CTCF_OASIS="--ds OASIS ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 1.0"
CTCF_IXI="--ds IXI ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 4.0"

run_cmd() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "$*"
        return
    fi
    "$@"
}

train_ctcf() {
    local exp_name="$1"; shift
    echo "==================================================================="
    echo "> TRAIN ${exp_name}"
    echo "==================================================================="
    run_cmd "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}"
}

infer_ctcf() {
    local exp_name="$1"
    local ds="$2"
    local ctcf_config="$3"
    local ctcf_l3_svf="$4"
    local ckpt="results/${exp_name}/ckpt/best.pth"
    local out_dir="${OUT}/inference/${exp_name}"
    local use_test_flag=""

    [ "${ds}" = "IXI" ] && use_test_flag="--use_test"

    if [ "${DRY_RUN}" != "1" ] && [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${exp_name} - no checkpoint at ${ckpt}"
        return
    fi

    echo "==================================================================="
    echo "> INFER ${exp_name}"
    echo "==================================================================="
    run_cmd "${PYBIN}" -m experiments.inference \
        --ds "${ds}" ${PATHS_PROFILE} --gpu "${GPU}" \
        --model ctcf \
        --ckpt "${ckpt}" \
        --strict_ckpt 0 \
        --hd95 \
        ${use_test_flag} \
        --ctcf_config "${ctcf_config}" \
        --ctcf_l3_svf "${ctcf_l3_svf}" \
        --out_dir "${out_dir}"
}

if [ "${RUN_TRAIN}" = "1" ]; then
    if [ "${SKIP_LKU8_IXI}" != "1" ]; then
        train_ctcf "SEDM_CASC_LKU8_NOSVF_IXI" \
            --config CTCF-CascadeA-LKU8 \
            --l3_svf 0 \
            ${CTCF_IXI}
    fi

    if [ "${SKIP_VXM_OASIS}" != "1" ]; then
        train_ctcf "SEDM_CASC_VXM_NOSVF_OASIS" \
            --config CTCF-CascadeA-VM \
            --l3_svf 0 \
            ${CTCF_OASIS}
    fi

    if [ "${SKIP_VXM_IXI}" != "1" ]; then
        train_ctcf "SEDM_CASC_VXM_NOSVF_IXI" \
            --config CTCF-CascadeA-VM \
            --l3_svf 0 \
            ${CTCF_IXI}
    fi

    if [ "${SKIP_MAMBA_OASIS}" != "1" ]; then
        train_ctcf "SEDM_CASC_MAMBA_NOSVF_OASIS" \
            --config CTCF-CascadeA-Mamba \
            --l3_svf 0 \
            ${CTCF_OASIS}
    fi

    if [ "${SKIP_MAMBA_IXI}" != "1" ]; then
        train_ctcf "SEDM_CASC_MAMBA_NOSVF_IXI" \
            --config CTCF-CascadeA-Mamba \
            --l3_svf 0 \
            ${CTCF_IXI}
    fi
fi

if [ "${RUN_INFERENCE}" = "1" ]; then
    if [ "${SKIP_LKU8_IXI}" != "1" ]; then
        infer_ctcf "SEDM_CASC_LKU8_NOSVF_IXI" IXI CTCF-CascadeA-LKU8 0
    fi

    if [ "${SKIP_VXM_OASIS}" != "1" ]; then
        infer_ctcf "SEDM_CASC_VXM_NOSVF_OASIS" OASIS CTCF-CascadeA-VM 0
    fi

    if [ "${SKIP_VXM_IXI}" != "1" ]; then
        infer_ctcf "SEDM_CASC_VXM_NOSVF_IXI" IXI CTCF-CascadeA-VM 0
    fi

    if [ "${SKIP_MAMBA_OASIS}" != "1" ]; then
        infer_ctcf "SEDM_CASC_MAMBA_NOSVF_OASIS" OASIS CTCF-CascadeA-Mamba 0
    fi

    if [ "${SKIP_MAMBA_IXI}" != "1" ]; then
        infer_ctcf "SEDM_CASC_MAMBA_NOSVF_IXI" IXI CTCF-CascadeA-Mamba 0
    fi
fi

echo "==================================================================="
echo "SEDM SVF gap closure complete."
echo "New checkpoints: results/SEDM_CASC_*/ckpt/best.pth"
echo "New inference:   ${OUT}/inference/SEDM_CASC_*/per_case.csv"
echo "==================================================================="
