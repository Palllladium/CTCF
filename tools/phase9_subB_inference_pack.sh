#!/usr/bin/env bash
# Phase 9 Sub-B repair inference pack.
#
# Purpose:
#   Re-run only the CTCF inference jobs whose previous metrics were suspect
#   because experiments.inference did not expose the Level-3 SVF flag.
#
# Why this exists:
#   Training scripts used explicit --l3_svf values, but inference previously
#   rebuilt CTCF models from config defaults. That mismatched these checkpoints:
#     - VoxelMorph/LKU SVF checkpoints were inferred as NoSVF.
#     - Mamba NoSVF checkpoints were inferred as SVF.
#
# Usage on the lab machine:
#   conda activate ctcf
#   GPU=0 PATHS_PROFILE=--2 bash tools/phase9_subB_inference_pack.sh
#
# Output:
#   results/SEDM_l3svf_recheck/inference/<exp_name>/per_case.csv
#   results/SEDM_l3svf_recheck/inference/<exp_name>/summary.{csv,json}
#
# Common overrides:
#   OUT=results/SEDM bash tools/phase9_subB_inference_pack.sh
#   SKIP_VXM=1 bash tools/phase9_subB_inference_pack.sh
#   SKIP_LKU8=1 bash tools/phase9_subB_inference_pack.sh
#   SKIP_LKU32=1 bash tools/phase9_subB_inference_pack.sh
#   SKIP_MAMBA_NOSVF=1 bash tools/phase9_subB_inference_pack.sh

set -euo pipefail

GPU="${GPU:-0}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM_l3svf_recheck}"

SKIP_VXM="${SKIP_VXM:-0}"
SKIP_LKU8="${SKIP_LKU8:-0}"
SKIP_LKU32="${SKIP_LKU32:-0}"
SKIP_MAMBA_NOSVF="${SKIP_MAMBA_NOSVF:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

mkdir -p "${OUT}/inference"

run_inf() {
    local exp_name="$1"
    local ds="$2"
    local ctcf_config="$3"
    local ctcf_l3_svf="$4"

    local ckpt="results/${exp_name}/ckpt/best.pth"
    local out_dir="${OUT}/inference/${exp_name}"
    local use_test_flag=""

    [ "${ds}" = "IXI" ] && use_test_flag="--use_test"

    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${exp_name} - no checkpoint at ${ckpt}"
        return
    fi

    echo "==================================================================="
    echo "> INFER ${exp_name}"
    echo "  ds=${ds}, config=${ctcf_config}, ctcf_l3_svf=${ctcf_l3_svf}"
    echo "  out=${out_dir}"
    echo "==================================================================="

    "${PYBIN}" -m experiments.inference \
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

if [ "${SKIP_VXM}" != "1" ]; then
    run_inf P9_CASC_VXM_SVF_OASIS OASIS CTCF-CascadeA-VM 1
    run_inf P9_CASC_VXM_SVF_IXI   IXI   CTCF-CascadeA-VM 1
fi

if [ "${SKIP_LKU8}" != "1" ]; then
    run_inf P9_CASC_LKU8_SVF_OASIS OASIS CTCF-CascadeA-LKU8 1
    run_inf P9_CASC_LKU8_SVF_IXI   IXI   CTCF-CascadeA-LKU8 1
fi

if [ "${SKIP_LKU32}" != "1" ]; then
    run_inf P8_CASC_LKU32_SVF_OASIS OASIS CTCF-CascadeA-LKU32 1
    run_inf P8_CASC_LKU32_SVF_IXI   IXI   CTCF-CascadeA-LKU32 1
fi

if [ "${SKIP_MAMBA_NOSVF}" != "1" ]; then
    run_inf P8_CASC_MAMBA_NOSVF_OASIS OASIS CTCF-CascadeA-Mamba 0
    run_inf P9_CASC_MAMBA_NOSVF_IXI   IXI   CTCF-CascadeA-Mamba 0
fi

echo "==================================================================="
echo "Phase 9 Sub-B repair inference complete."
echo "Output: ${OUT}/inference/<exp_name>/"
echo "Send back this output directory plus the console log."
echo "==================================================================="
