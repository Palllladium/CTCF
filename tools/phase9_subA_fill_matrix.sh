#!/usr/bin/env bash
# Phase 9 Sub-A — fill 100ep matrix gaps for Paper 2 finalization.
#
# Goals:
#   1. VxM under matched CTCF protocol on OASIS+IXI (L2-only + cascade with SVF).
#   2. LKU-8 cascade with SVF on both datasets (P7 had no SVF — replace).
#   3. LKU-8/LKU-32/Mamba/VMamba L2-only baselines on IXI (P7_CTRL had OASIS only).
#   4. Mamba NoSVF on IXI (close SVF-redundancy claim across both datasets).
#   5. VMamba cascade on IXI (matrix completeness; not in longruns later).
#
# Uniform SVF protocol across Paper 2:
#   - CNN/transformer cascades:  --l3_svf 1 (no internal velocity integration → SVF helps folds)
#   - SSM cascades:              --l3_svf 0 (internal VecInt makes L3-SVF redundant; Phase 8 finding)
#
# Order: small/fast first (VxM), heavy last (Mamba/VMamba IXI cascades).
# This way smallest data points come back early; if compute breaks late, we've still got the
# cheap L2-only and VxM rows in the matrix.
#
# Usage:
#   conda activate ctcf
#   bash tools/phase9_subA_fill_matrix.sh
#
# Override defaults:
#   MAX_EPOCH=100 GPU=0 PATHS_PROFILE=--2 bash tools/phase9_subA_fill_matrix.sh
#
# Staged execution (12 SKIP flags, one per run):
#   SKIP_VXM_L2_OASIS=1 SKIP_VXM_CASC_OASIS=1 ... bash tools/phase9_subA_fill_matrix.sh

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_VXM_L2_OASIS="${SKIP_VXM_L2_OASIS:-0}"
SKIP_VXM_CASC_OASIS="${SKIP_VXM_CASC_OASIS:-0}"
SKIP_VXM_L2_IXI="${SKIP_VXM_L2_IXI:-0}"
SKIP_VXM_CASC_IXI="${SKIP_VXM_CASC_IXI:-0}"
SKIP_LKU8_L2_IXI="${SKIP_LKU8_L2_IXI:-0}"
SKIP_LKU8_CASC_OASIS="${SKIP_LKU8_CASC_OASIS:-0}"
SKIP_LKU8_CASC_IXI="${SKIP_LKU8_CASC_IXI:-0}"
SKIP_LKU32_L2_IXI="${SKIP_LKU32_L2_IXI:-0}"
SKIP_MAMBA_L2_IXI="${SKIP_MAMBA_L2_IXI:-0}"
SKIP_MAMBA_NOSVF_IXI="${SKIP_MAMBA_NOSVF_IXI:-0}"
SKIP_VMAMBA_L2_IXI="${SKIP_VMAMBA_L2_IXI:-0}"
SKIP_VMAMBA_CASC_IXI="${SKIP_VMAMBA_CASC_IXI:-0}"

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
# 1-4. VxM (small, fast — comes first as smoke test for the matrix)
# ============================================================

# 1. VxM L2-only OASIS — matched CTCF protocol baseline (replaces P1 native VxM solo).
if [ "${SKIP_VXM_L2_OASIS}" != "1" ]; then
    run_ctcf "P9_CTRL_VXM_L2ONLY_OASIS" --config CTCF-VM-solo ${CTCF_OASIS}
fi

# 2. VxM cascade OASIS with SVF — replaces P2_13 under uniform protocol.
if [ "${SKIP_VXM_CASC_OASIS}" != "1" ]; then
    run_ctcf "P9_CASC_VXM_SVF_OASIS" --config CTCF-CascadeA-VM --l3_svf 1 ${CTCF_OASIS}
fi

# 3. VxM L2-only IXI.
if [ "${SKIP_VXM_L2_IXI}" != "1" ]; then
    run_ctcf "P9_CTRL_VXM_L2ONLY_IXI" --config CTCF-VM-solo ${CTCF_IXI}
fi

# 4. VxM cascade IXI with SVF.
if [ "${SKIP_VXM_CASC_IXI}" != "1" ]; then
    run_ctcf "P9_CASC_VXM_SVF_IXI" --config CTCF-CascadeA-VM --l3_svf 1 ${CTCF_IXI}
fi

# ============================================================
# 5-7. LKU-8 (medium-fast)
# ============================================================

# 5. LKU-8 L2-only IXI.
if [ "${SKIP_LKU8_L2_IXI}" != "1" ]; then
    run_ctcf "P9_CTRL_LKU8_L2ONLY_IXI" --config CTCF-LKU8-solo ${CTCF_IXI}
fi

# 6. LKU-8 cascade OASIS with SVF — replaces P7_CASC_LKU8_OASIS (which had no SVF).
#    This makes LKU-8 cascade protocol consistent with VxM/LKU-32/Swin-DCA across Paper 2.
if [ "${SKIP_LKU8_CASC_OASIS}" != "1" ]; then
    run_ctcf "P9_CASC_LKU8_SVF_OASIS" --config CTCF-CascadeA-LKU8 --l3_svf 1 ${CTCF_OASIS}
fi

# 7. LKU-8 cascade IXI with SVF.
if [ "${SKIP_LKU8_CASC_IXI}" != "1" ]; then
    run_ctcf "P9_CASC_LKU8_SVF_IXI" --config CTCF-CascadeA-LKU8 --l3_svf 1 ${CTCF_IXI}
fi

# ============================================================
# 8. LKU-32 L2-only IXI (cascade IXI already done in P8)
# ============================================================

if [ "${SKIP_LKU32_L2_IXI}" != "1" ]; then
    run_ctcf "P9_CTRL_LKU32_L2ONLY_IXI" --config CTCF-LKU32-solo ${CTCF_IXI}
fi

# ============================================================
# 9-10. Mamba (heavy)
# ============================================================

# 9. Mamba L2-only IXI.
if [ "${SKIP_MAMBA_L2_IXI}" != "1" ]; then
    run_ctcf "P9_CTRL_MAMBA_L2ONLY_IXI" --config CTCF-Mamba-solo ${CTCF_IXI}
fi

# 10. Mamba NoSVF cascade IXI — closes the SVF-redundancy claim across both datasets.
if [ "${SKIP_MAMBA_NOSVF_IXI}" != "1" ]; then
    run_ctcf "P9_CASC_MAMBA_NOSVF_IXI" --config CTCF-CascadeA-Mamba --l3_svf 0 ${CTCF_IXI}
fi

# ============================================================
# 11-12. VMamba (heaviest; matrix completeness, not in longruns)
# ============================================================

# 11. VMamba L2-only IXI.
if [ "${SKIP_VMAMBA_L2_IXI}" != "1" ]; then
    run_ctcf "P9_CTRL_VMAMBA_L2ONLY_IXI" --config CTCF-VMamba-solo ${CTCF_IXI}
fi

# 12. VMamba cascade IXI with SVF — matches OASIS config (P7_CASC_VMAMBA_SVF).
if [ "${SKIP_VMAMBA_CASC_IXI}" != "1" ]; then
    run_ctcf "P9_CASC_VMAMBA_SVF_IXI" --config CTCF-CascadeA-VMamba --l3_svf 1 ${CTCF_IXI}
fi

echo "==================================================================="
echo "Phase 9 Sub-A complete."
echo "Results in logs/P9_*/logfile.log and experiments/P9_*/ckpt/best.pth"
echo ""
echo "Next: Sub-B (cascade complexity table + per-pair stat extraction)"
echo "Then: Sub-C (500ep longruns of one backbone per family on both datasets)"
echo "==================================================================="
