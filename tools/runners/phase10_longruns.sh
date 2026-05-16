#!/usr/bin/env bash
# Phase 10 — Paper 2 finalization longruns (500 epochs).
#
# Strategy:
#   - Mamba SVF on both datasets = headline (best Pareto from Phase 9)
#   - LKU-8 SVF on both datasets = CNN-family convergence baseline
#   - VxM unified SVF on both datasets = lightweight CNN, unified protocol (NEW)
#   - Mamba NoSVF OASIS = ablation point (pure-Dice ceiling)
#
# Excluded vs earlier plan:
#   - LKU-32 (Δ_cascade ≈ 0 on OASIS, no significant Fold gain on IXI — confirmed F1)
#   - VMamba (Pareto-dominated by Mamba)
#   - Mamba NoSVF IXI (Fold WORSE than Paper 1 on IXI per stat-tests; not viable)
#   - EfficientMorph (deferred to full Paper 2 backbone matrix, 100ep only)
#
# Resume strategy:
#   All Phase 9 ckpts live in results/<EXP_NAME>/ckpt/last.pth.
#   Phase 10 runs use --resume to continue from 100ep ckpts (saves ~80% compute).
#   ONLY VxM unified runs from scratch (new config, no existing ckpt).
#
# VxM Unified note: new config (full-res L2, L3 base_ch=32) goes directly to longrun
# without intermediate 100ep verification (VxM compact; cheaper to commit to 500ep).
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase10_longruns.sh
#
# Skip flags for staged execution (long-running, GPU may need reservation slots).

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-500}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_MAMBA_SVF_OASIS="${SKIP_MAMBA_SVF_OASIS:-0}"
SKIP_MAMBA_SVF_IXI="${SKIP_MAMBA_SVF_IXI:-0}"
SKIP_MAMBA_NOSVF_OASIS="${SKIP_MAMBA_NOSVF_OASIS:-0}"
SKIP_LKU8_SVF_OASIS="${SKIP_LKU8_SVF_OASIS:-0}"
SKIP_LKU8_SVF_IXI="${SKIP_LKU8_SVF_IXI:-0}"
SKIP_VXM_UNIFIED_SVF_OASIS="${SKIP_VXM_UNIFIED_SVF_OASIS:-0}"
SKIP_VXM_UNIFIED_SVF_IXI="${SKIP_VXM_UNIFIED_SVF_IXI:-0}"

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

# ============================================================
# Phase 10 longruns — 500 epochs
# ============================================================

# 1. Mamba SVF OASIS — headline, resume from P7_CASC_MAMBA_SVF_OASIS
if [ "${SKIP_MAMBA_SVF_OASIS}" != "1" ]; then
    run_ctcf "P10_LONGRUN_MAMBA_SVF_OASIS" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 1 \
        --resume results/P7_CASC_MAMBA_SVF_OASIS/ckpt/last.pth \
        ${CTCF_OASIS}
fi

# 2. Mamba SVF IXI — headline cross-dataset, resume from P8_CASC_MAMBA_SVF_IXI
if [ "${SKIP_MAMBA_SVF_IXI}" != "1" ]; then
    run_ctcf "P10_LONGRUN_MAMBA_SVF_IXI" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 1 \
        --resume results/P8_CASC_MAMBA_SVF_IXI/ckpt/last.pth \
        ${CTCF_IXI}
fi

# 3. Mamba NoSVF OASIS — ablation, resume from SEDM_CASC_MAMBA_NOSVF_OASIS
if [ "${SKIP_MAMBA_NOSVF_OASIS}" != "1" ]; then
    run_ctcf "P10_LONGRUN_MAMBA_NOSVF_OASIS" \
        --config CTCF-CascadeA-Mamba \
        --l3_svf 0 \
        --resume results/SEDM_CASC_MAMBA_NOSVF_OASIS/ckpt/last.pth \
        ${CTCF_OASIS}
fi

# 4. LKU-8 SVF OASIS — CNN convergence, resume from P9_CASC_LKU8_SVF_OASIS
if [ "${SKIP_LKU8_SVF_OASIS}" != "1" ]; then
    run_ctcf "P10_LONGRUN_LKU8_SVF_OASIS" \
        --config CTCF-CascadeA-LKU8 \
        --l3_svf 1 \
        --resume results/P9_CASC_LKU8_SVF_OASIS/ckpt/last.pth \
        ${CTCF_OASIS}
fi

# 5. LKU-8 SVF IXI — CNN cross-dataset, resume from P9_CASC_LKU8_SVF_IXI
if [ "${SKIP_LKU8_SVF_IXI}" != "1" ]; then
    run_ctcf "P10_LONGRUN_LKU8_SVF_IXI" \
        --config CTCF-CascadeA-LKU8 \
        --l3_svf 1 \
        --resume results/P9_CASC_LKU8_SVF_IXI/ckpt/last.pth \
        ${CTCF_IXI}
fi

# 6. VxM Unified SVF OASIS — from scratch (new config)
if [ "${SKIP_VXM_UNIFIED_SVF_OASIS}" != "1" ]; then
    run_ctcf "P10_LONGRUN_VXM_UNIFIED_SVF_OASIS" \
        --config CTCF-CascadeA-VM-Unified \
        --l3_svf 1 \
        ${CTCF_OASIS}
fi

# 7. VxM Unified SVF IXI — from scratch (new config)
if [ "${SKIP_VXM_UNIFIED_SVF_IXI}" != "1" ]; then
    run_ctcf "P10_LONGRUN_VXM_UNIFIED_SVF_IXI" \
        --config CTCF-CascadeA-VM-Unified \
        --l3_svf 1 \
        ${CTCF_IXI}
fi

echo ""
echo "==================================================================="
echo "Phase 10 longruns complete."
echo "Results in logs/P10_*/logfile.log and results/P10_*/ckpt/best.pth"
echo ""
echo "Next steps:"
echo "  - Extend tools/runners/phase9_matrix_and_inference.sh with P10 configs"
echo "    (or write phase10 equivalent) to run inference on Phase 10 ckpts."
echo "  - Run tools/analysis/compute_stats.py sedm_vs_paper1 with P10 cascades added."
echo "  - Update results/SEDM/summary/ + memory/phase10_results.md"
echo "==================================================================="
