#!/usr/bin/env bash
# Phase 11 — first batch: ablation sweep over 4 architectural / training-time
# improvements on the Mamba SVF cascade base (currently the Paper 2 headline).
#
# Four experiments:
#   A. L3 capacity scaling      (--l3_base_ch 64)
#   B. M1 Multi-head L3         (--l3_num_heads K, K ∈ {2,4,8})
#   C. M2 EMA self-distillation (--ema_decay α --ema_lambda λ, 3 grid points)
#   D. M3 Cascade-aware reg     (--w_reg_l1, --w_reg_l3 grid, 4 points incl. baseline)
#
# Each run is fresh (no --resume): architectural changes (A, B) break ckpt
# compatibility; training-time changes (C, D) require fresh comparison to
# isolate the contribution of the new mechanism.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase11_batch0.sh
#
# Override defaults (env vars):
#   GPU=0 MAX_EPOCH=100 PATHS_PROFILE=--2 bash tools/runners/phase11_batch0.sh
#
# Skip flags for staged execution:
#   SKIP_A=1 / SKIP_B=1 / SKIP_C=1 / SKIP_D=1 — skip an entire experiment block
#   SKIP_K2=1 / SKIP_K4=1 / SKIP_K8=1 — skip individual K values in B
#   SKIP_EMA_<ID>=1 — skip individual EMA grid point (see C section)
#   SKIP_REG_<ID>=1 — skip individual cascade-reg grid point (see D section)

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_A="${SKIP_A:-0}"
SKIP_B="${SKIP_B:-0}"
SKIP_C="${SKIP_C:-0}"
SKIP_D="${SKIP_D:-0}"

# B (multi-head) individual skips
SKIP_K2="${SKIP_K2:-0}"
SKIP_K4="${SKIP_K4:-0}"
SKIP_K8="${SKIP_K8:-0}"

# C (EMA) individual skips
SKIP_EMA_AGGRESSIVE="${SKIP_EMA_AGGRESSIVE:-0}"    # alpha=0.99,  lambda=0.5
SKIP_EMA_BALANCED="${SKIP_EMA_BALANCED:-0}"        # alpha=0.999, lambda=0.5
SKIP_EMA_SOFT="${SKIP_EMA_SOFT:-0}"                # alpha=0.999, lambda=0.1

# D (cascade-aware reg) individual skips
SKIP_REG_FLAT="${SKIP_REG_FLAT:-0}"                # w_l1=1,   w_l2=1, w_l3=1   (sanity baseline)
SKIP_REG_STRONG_L1="${SKIP_REG_STRONG_L1:-0}"      # w_l1=3,   w_l2=1, w_l3=0.3
SKIP_REG_VERY_STRONG="${SKIP_REG_VERY_STRONG:-0}"  # w_l1=5,   w_l2=1, w_l3=0.1
SKIP_REG_MID="${SKIP_REG_MID:-0}"                  # w_l1=2,   w_l2=1, w_l3=0.5

# Common args
COMMON="--gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"
CTCF_BASE="${COMMON} --w_ncc 1.0 --w_icon 0.05 --w_jac 0.005"
CTCF_OASIS="--ds OASIS ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 1.0"

# Mamba SVF base config — applied to ALL Phase 11 runs in this batch
MAMBA_SVF="--config CTCF-CascadeA-Mamba --l3_svf 1"

run_ctcf() {
    local exp_name="$1"; shift
    echo "> ${exp_name}"
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}"
}


# A. L3 capacity scaling — l3_base_ch=64 (no new code; flag passthrough)
#    Expected: +0.005-0.012 Dice (R5 prior was +0.016 on Swin-DCA, but full-res
#    transition was confounded there; on Mamba this isolates pure capacity).
#    VRAM forecast: 45-50GB (Mamba SVF baseline 31.9GB + ~10-15GB L3 widening).
if [ "${SKIP_A}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_L3CH64_OASIS" \
        ${MAMBA_SVF} \
        --l3_base_ch 64 \
        ${CTCF_OASIS}
fi


# B. M1 — Multi-head L3 with learned routing (K parallel flow heads).
#    Sweep K ∈ {2, 4, 8}. K=1 is the existing Mamba SVF (Phase 10) and is NOT
#    re-run here — its number is the comparison baseline.
#    Heads + routing zero-init → starts identical to single-head zero-init L3.
#    VRAM forecast: ≈ baseline (added params are tiny relative to U-Net body).
if [ "${SKIP_B}" != "1" ]; then
    if [ "${SKIP_K2}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_MH2_OASIS" \
            ${MAMBA_SVF} \
            --l3_num_heads 2 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_K4}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_MH4_OASIS" \
            ${MAMBA_SVF} \
            --l3_num_heads 4 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_K8}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_MH8_OASIS" \
            ${MAMBA_SVF} \
            --l3_num_heads 8 \
            ${CTCF_OASIS}
    fi
fi


# C. M2 — EMA self-distillation (mean-teacher with flow L1 consistency).
#    Grid: (decay, lambda) ∈ {(0.99, 0.5), (0.999, 0.5), (0.999, 0.1)}.
#      AGGRESSIVE: fast teacher tracking, strong consistency.
#      BALANCED:   slow teacher tracking (canonical mean-teacher), strong cons.
#      SOFT:       slow teacher tracking, mild consistency.
#    EMA teacher forward adds compute but not parameters; VRAM ≈ +30-40GB
#    (no_grad teacher activations; should fit on 96GB).
if [ "${SKIP_C}" != "1" ]; then
    if [ "${SKIP_EMA_AGGRESSIVE}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_EMA_AGGR_OASIS" \
            ${MAMBA_SVF} \
            --ema_decay 0.99 --ema_lambda 0.5 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_EMA_BALANCED}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_EMA_BAL_OASIS" \
            ${MAMBA_SVF} \
            --ema_decay 0.999 --ema_lambda 0.5 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_EMA_SOFT}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_EMA_SOFT_OASIS" \
            ${MAMBA_SVF} \
            --ema_decay 0.999 --ema_lambda 0.1 \
            ${CTCF_OASIS}
    fi
fi


# D. M3 — Cascade-aware regularization (per-level diffusion weights).
#    L_reg = w_reg * [w_l1 · L_diff(phi_L1) + w_l2 · L_diff(phi_L2) + w_l3 · L_diff(delta_L3)]
#    Grid:
#      FLAT       : (1,1,1)   = sanity (per-level decomposition with neutral weights;
#                               check that the new code path agrees with baseline within noise)
#      STRONG_L1  : (3,1,0.3) = strong L1 smoothness + relax L3 (allow sharp local refinement)
#      VERY_STRONG: (5,1,0.1) = very strong L1 smoothness + very relaxed L3
#      MID        : (2,1,0.5) = mild redistribution
#    Baseline reference: Mamba SVF Phase 10 OASIS (Dice 0.8314 with uniform w_reg=1).
if [ "${SKIP_D}" != "1" ]; then
    if [ "${SKIP_REG_FLAT}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_REG_FLAT_OASIS" \
            ${MAMBA_SVF} \
            --w_reg_l1 1.0 --w_reg_l2 1.0 --w_reg_l3 1.0 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_REG_STRONG_L1}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_REG_STRONG_OASIS" \
            ${MAMBA_SVF} \
            --w_reg_l1 3.0 --w_reg_l2 1.0 --w_reg_l3 0.3 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_REG_VERY_STRONG}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_REG_VSTRONG_OASIS" \
            ${MAMBA_SVF} \
            --w_reg_l1 5.0 --w_reg_l2 1.0 --w_reg_l3 0.1 \
            ${CTCF_OASIS}
    fi
    if [ "${SKIP_REG_MID}" != "1" ]; then
        run_ctcf "P11_MAMBA_SVF_REG_MID_OASIS" \
            ${MAMBA_SVF} \
            --w_reg_l1 2.0 --w_reg_l2 1.0 --w_reg_l3 0.5 \
            ${CTCF_OASIS}
    fi
fi


echo ""
echo "Phase 11 batch 0 complete (or skipped per env flags)."
echo ""
echo "Outputs:"
echo "  - logs/P11_*/logfile.log         training logs"
echo "  - results/P11_*/ckpt/best.pth    best-val ckpts"
echo ""
echo "Next steps:"
echo "  1. Inference on each P11_* ckpt (re-use tools/runners/phase10_inference.sh"
echo "     pattern; add P11_* entries)."
echo "  2. Aggregate to results/SEDM/summary/ — extend aggregate_results.py CONFIGS."
echo "  3. Rank improvements by Dice gain vs baseline (Mamba SVF Phase 10 OASIS 0.8314)."
echo "  4. Lock the best stack -> Phase 11.1 = stack longruns 500ep."
