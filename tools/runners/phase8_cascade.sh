#!/usr/bin/env bash
# Phase 8 cascade runs (Paper 2 finalization).
#
# REVISION HISTORY:
#   v1: 6 runs (LKU8_FIXSCHED, LKU32_OASIS, LKU32_SVF, MAMBA_NOSVF, MAMBA_IXI, LKU32_SVF_IXI).
#   v2: 4 runs after LKU-32 NaN at ep14:
#       - DONE: P8_CASC_LKU8_FIXSCHED_OASIS (0.8263, sanity passed).
#       - DROPPED: P8_CASC_LKU32_OASIS (deterministic NaN ep14, partial 0.8202@ep11 documented).
#       - Remaining 4 runs reordered: stable Mamba first, LKU-32 SVF after.
#
# Goals (remaining):
#   1. Isolate L3-SVF contribution for Mamba (1 control run, OASIS).
#   2. Test SVF-on-L3 hypothesis as the LKU-32 cascade stabilizer (best LKU vs Mamba).
#   3. Cross-dataset generalization on IXI for the two best cascades.
#
# Usage:
#   conda activate ctcf
#   bash tools/phase8_cascade.sh
#
# Override defaults:
#   MAX_EPOCH=100 GPU=0 PATHS_PROFILE=--2 bash tools/phase8_cascade.sh
#
# Staged execution (recommended — set -e aborts on any failure):
#   # Batch 1: stable Mamba runs (independent of LKU-32 outcome)
#   SKIP_LKU32_SVF=1 SKIP_LKU32_SVF_IXI=1 bash tools/phase8_cascade.sh
#   # Batch 2: LKU-32 SVF test on OASIS
#   SKIP_MAMBA_NOSVF=1 SKIP_MAMBA_IXI=1 SKIP_LKU32_SVF_IXI=1 bash tools/phase8_cascade.sh
#   # Batch 3: LKU-32 SVF IXI (only if Batch 2 succeeded)
#   SKIP_MAMBA_NOSVF=1 SKIP_MAMBA_IXI=1 SKIP_LKU32_SVF=1 bash tools/phase8_cascade.sh

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_MAMBA_NOSVF="${SKIP_MAMBA_NOSVF:-0}"
SKIP_MAMBA_IXI="${SKIP_MAMBA_IXI:-0}"
SKIP_LKU32_SVF="${SKIP_LKU32_SVF:-0}"
SKIP_LKU32_SVF_IXI="${SKIP_LKU32_SVF_IXI:-0}"

COMMON="--gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"
CTCF_BASE="${COMMON} --w_ncc 1.0 --w_icon 0.05 --w_jac 0.005"
CTCF_OASIS="--ds OASIS ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 1.0"
CTCF_IXI="--ds IXI ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 4.0"

run_ctcf() {
    local exp_name="$1"; shift
    echo "==================================================================="
    echo "> ${exp_name}"
    echo "==================================================================="
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}"
}

# 1. Mamba cascade WITHOUT SVF on L3.
#    Mamba's L2 already integrates VecInt internally; this run tells us whether
#    L3-SVF is necessary for 0% folds or whether L2 integration alone suffices.
#    Most stable run; goes first to ensure Mamba data is collected even if LKU-32 SVF fails.
if [ "${SKIP_MAMBA_NOSVF}" != "1" ]; then
    run_ctcf "P8_CASC_MAMBA_NOSVF_OASIS" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 0 \
        ${CTCF_OASIS}
fi

# 2. Mamba SVF cascade on IXI. Generalization check for the headline backbone.
#    Independent of LKU-32 outcome; provides cross-dataset evidence for Paper 2.
if [ "${SKIP_MAMBA_IXI}" != "1" ]; then
    run_ctcf "P8_CASC_MAMBA_SVF_IXI" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 1 \
        ${CTCF_IXI}
fi

# 3. LKU-32 cascade with SVF on L3 + --l1_from_start 1.
#    Tests two hypotheses simultaneously:
#      (a) --l1_from_start 1 fixes the ep6 distribution-shift NaN (P7_CASC_LKU32 mode).
#      (b) SVF on L3 fixes the ep14 ICON-spike NaN (P8_CASC_LKU32 mode).
#    Best-effort LKU representative for head-to-head against Mamba SVF cascade.
#    If this crashes too: drop LKU-32 from Paper 2, keep LKU-8 as single LKU point.
if [ "${SKIP_LKU32_SVF}" != "1" ]; then
    run_ctcf "P8_CASC_LKU32_SVF_OASIS" \
        --config CTCF-CascadeA-LKU32 \
        --l1_from_start 1 \
        --l3_svf 1 \
        ${CTCF_OASIS}
fi

# 4. LKU-32 SVF cascade on IXI. Generalization check for best LKU.
#    Only meaningful if (3) succeeded on OASIS.
if [ "${SKIP_LKU32_SVF_IXI}" != "1" ]; then
    run_ctcf "P8_CASC_LKU32_SVF_IXI" \
        --config CTCF-CascadeA-LKU32 \
        --l1_from_start 1 \
        --l3_svf 1 \
        ${CTCF_IXI}
fi

echo "==================================================================="
echo "Phase 8 cascade runs complete."
echo "Results in logs/P8_*/logfile.log and experiments/P8_*/ckpt/best.pth"
echo "==================================================================="
