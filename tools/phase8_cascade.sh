#!/usr/bin/env bash
# Phase 8 cascade runs (Paper 2 finalization).
#
# Goals:
#   1. Validate --l1_from_start 1 as the cascade-warmup fix on a known-stable
#      LKU-8 baseline (sanity).
#   2. Add LKU-32 (33.35M) cascade points for the params spectrum.
#   3. Head-to-head LKU-32 SVF vs Mamba SVF at comparable diffeomorphic protocol.
#   4. Isolate SVF-on-L3 contribution for Mamba (one no-SVF control).
#   5. Cross-dataset generalization on IXI for the two best cascades.
#
# Run order is intentional:
#   - LKU-8 fixed-schedule first (cheap, ~1.5h, gates the schedule fix)
#   - LKU-32 no-SVF before LKU-32 SVF (isolates the SVF effect for the bigger LKU)
#   - Mamba no-SVF after LKU is settled (cheaper to schedule late)
#   - IXI runs last (only fire if OASIS counterparts succeeded)
#
# Usage:
#   conda activate ctcf
#   bash tools/phase8_cascade.sh
#
# Override defaults:
#   MAX_EPOCH=100 GPU=0 PATHS_PROFILE=--2 bash tools/phase8_cascade.sh
#
# Staged execution:
#   SKIP_LKU8_SCHED=1 SKIP_LKU32=1 ... bash tools/phase8_cascade.sh

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_LKU8_SCHED="${SKIP_LKU8_SCHED:-0}"
SKIP_LKU32="${SKIP_LKU32:-0}"
SKIP_LKU32_SVF="${SKIP_LKU32_SVF:-0}"
SKIP_MAMBA_NOSVF="${SKIP_MAMBA_NOSVF:-0}"
SKIP_MAMBA_IXI="${SKIP_MAMBA_IXI:-0}"
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

# 1. Sanity: --l1_from_start 1 on known-good LKU-8 cascade.
#    Expected: Dice within +/-0.005 of P7_CASC_LKU8_OASIS (0.8248), no crash.
if [ "${SKIP_LKU8_SCHED}" != "1" ]; then
    run_ctcf "P8_CASC_LKU8_FIXSCHED_OASIS" \
        --config CTCF-CascadeA-LKU8 \
        --l1_from_start 1 \
        ${CTCF_OASIS}
fi

# 2. LKU-32 cascade, no SVF. Tests that the schedule fix unblocks the
#    capacity-33M backbone that NaN'd in P7_CASC_LKU32_OASIS at ep6.
if [ "${SKIP_LKU32}" != "1" ]; then
    run_ctcf "P8_CASC_LKU32_OASIS" \
        --config CTCF-CascadeA-LKU32 \
        --l1_from_start 1 \
        ${CTCF_OASIS}
fi

# 3. LKU-32 cascade with SVF on L3. Best-effort LKU representative for the
#    head-to-head against Mamba SVF cascade.
if [ "${SKIP_LKU32_SVF}" != "1" ]; then
    run_ctcf "P8_CASC_LKU32_SVF_OASIS" \
        --config CTCF-CascadeA-LKU32 \
        --l1_from_start 1 \
        --l3_svf 1 \
        ${CTCF_OASIS}
fi

# 4. Mamba cascade WITHOUT SVF on L3. Mamba's L2 already integrates VecInt
#    internally; this run tells us whether L3-SVF is necessary for 0% folds
#    or whether L2 integration alone suffices.
if [ "${SKIP_MAMBA_NOSVF}" != "1" ]; then
    run_ctcf "P8_CASC_MAMBA_NOSVF_OASIS" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 0 \
        ${CTCF_OASIS}
fi

# 5. Mamba SVF cascade on IXI. Generalization check for the headline backbone.
if [ "${SKIP_MAMBA_IXI}" != "1" ]; then
    run_ctcf "P8_CASC_MAMBA_SVF_IXI" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 1 \
        ${CTCF_IXI}
fi

# 6. LKU-32 SVF cascade on IXI. Generalization check for best LKU.
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
