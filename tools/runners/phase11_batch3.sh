#!/usr/bin/env bash
# Phase 11 — batch 3: formally close the mixing axis (2 CTCF runs, OASIS 100ep).
#
# Context (all OASIS 100ep, anchor = 0.8274 = P7 Mamba SVF K=1, old code path):
#   The mixing screening is otherwise done. Two positive levers survive, both in-noise & non-novel:
#     HADAMARD correspondence  +0.0014 (~0 params; CORR27 == HADAMARD, cost-volume capacity is moot)
#     L3 base_ch=64            +0.0022 (brute capacity, +11GB VRAM, +30% time)
#   Dead: M1 multi-head, M3 cascade-aware reg (all variants < anchor), M2 EMA (never run).
#
# This batch answers the last two open questions before the axis is declared closed:
#   1. STACK  — do the two positive levers ADD or SATURATE? Hadamard + L3 base_ch=64 together.
#               Expected <= 0.831 (i.e. still under the 0.832-0.833 capacity+epochs ceiling);
#               either way it gives a defensible "even combined, mixing does not break the ceiling".
#   2. CTRL   — clean code-version control: Mamba SVF on the CURRENT code with corr_mode=none.
#               The anchor 0.8274 is an OLD Phase-7 ckpt; the only honest positive mechanism
#               (HADAMARD +0.0014) is currently compared across code versions. corr_mode=none on
#               today's code SHOULD reproduce ~0.8274 (memory: 6-ch, delta=0 init, byte-compatible);
#               this run removes the confound so the Hadamard A/B is clean.
#
# NB no longruns here (project rule: 500ep only on the single locked config). Competitor 500ep
#   for Track V fairness is deliberately NOT in this batch — decided at the strategy line.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase11_batch3.sh
# Override (env vars):
#   GPU=0 MAX_EPOCH=100 PATHS_PROFILE=--2 bash tools/runners/phase11_batch3.sh
# Staged skips:
#   SKIP_STACK=1 SKIP_CTRL=1

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_STACK="${SKIP_STACK:-0}"
SKIP_CTRL="${SKIP_CTRL:-0}"

COMMON="--gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"

# CTCF base (Mamba SVF cascade, OASIS, loss identical to batch1/batch2)
CTCF_BASE="${COMMON} --w_ncc 1.0 --w_icon 0.05 --w_jac 0.005"
CTCF_OASIS="--ds OASIS ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 1.0"
MAMBA_SVF="--config CTCF-CascadeA-Mamba --l3_svf 1"

run_ctcf() {
    local exp_name="$1"; shift
    echo "> ${exp_name}"
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}"
}


# 1. STACK — Hadamard correspondence + L3 base_ch=64 on the Mamba SVF base.
#    Compare vs anchor 0.8274 and vs each lever alone (HADAMARD 0.8288, L3CH64 0.8296):
#      additive  -> ~0.831 (recovers both deltas)
#      saturated -> ~0.829-0.830 (levers overlap; the more likely outcome)
if [ "${SKIP_STACK}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_STACK_OASIS" \
        ${MAMBA_SVF} --l3_corr_mode hadamard --l3_base_ch 64 \
        ${CTCF_OASIS}
fi


# 2. CTRL — Mamba SVF, current code, corr_mode=none (default base_ch). Matched-code anchor.
#    Should reproduce ~0.8274; if it does, the Hadamard +0.0014 is a clean same-code A/B and the
#    old Phase-7 anchor stands. If it drifts, report THIS as the 100ep anchor instead.
if [ "${SKIP_CTRL}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_CTRL_NONE_OASIS" \
        ${MAMBA_SVF} --l3_corr_mode none \
        ${CTCF_OASIS}
fi


echo ""
echo "Phase 11 batch 3 complete (or skipped per env flags)."
echo ""
echo "Next:"
echo "  1. STACK vs 0.8274 / 0.8288 / 0.8296 — additive or saturated? Either closes the mixing axis."
echo "  2. CTRL_NONE should ~= 0.8274; confirms the Hadamard A/B is same-code. Use CTRL as the"
echo "     100ep anchor in the stats run if it differs from the old Phase-7 0.8274."
echo "  3. Run tools/runners/phase11_infer_stats.sh to emit per_case.csv + paired Wilcoxon + FDR."
