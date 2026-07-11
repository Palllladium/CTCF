#!/usr/bin/env bash
# Phase 11 — inference + paired-Wilcoxon/FDR stats closure (reusable).
#
# Emits per_case.csv for every Phase 11 ckpt + the matched anchor, then runs the BH-FDR-corrected
# paired Wilcoxon (tools/analysis/compute_stats.py phase11). The per_case.csv files are tiny (KB) —
# transfer them back; the stats step can then be re-run anywhere.
#
# Covers two families (anchor for OASIS = P11_MAMBA_SVF_CTRL_NONE_OASIS, matched-code 100ep):
#   Mixing axis (CTCF, OASIS, model=ctcf): CTRL_NONE, HADAMARD, CORR27, L3CH64, MH2, M3 grid, STACK.
#     M3 runs differ from the anchor only in the training loss (per-level reg) — the model is a plain
#     Mamba SVF cascade, so they infer with the base CTCF args (no overrides).
#   Track V baselines (model=corrmlp / sacb): CorrMLP + SACB on OASIS (N=19) and IXI (N=115 test).
#     Requires the corrmlp/sacb inference adapters (added 2026-06-24 to model_adapters.py).
#
# IXI anchor for Track V = P10_LONGRUN_MAMBA_SVF_IXI (N=115 test, already inferred in Phase 10).
# IXI is plateau-saturated past 100ep, so comparing 100ep competitors to it is fair (note in paper).
#
# Ckpts must live at results/<EXP_NAME>/ckpt/best.pth. Existing per_case.csv are overwritten.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase11_infer_stats.sh
# Override (env vars):
#   GPU=0 PATHS_PROFILE=--2 OUT=results/SEDM bash tools/runners/phase11_infer_stats.sh
# Skip flags:
#   SKIP_CTCF=1     — skip the CTCF mixing-axis inferences
#   SKIP_TRACKV=1   — skip the CorrMLP/SACB baseline inferences
#   SKIP_STATS=1    — skip the compute_stats phase11 step

set -e

GPU="${GPU:-0}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM}"

SKIP_CTCF="${SKIP_CTCF:-0}"
SKIP_TRACKV="${SKIP_TRACKV:-0}"
SKIP_STATS="${SKIP_STATS:-0}"

# Required for tools/analysis/* imports of experiments.core.*
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

mkdir -p "${OUT}/inference" "${OUT}/summary"

CTCF_BASE_INF="--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"

# run_inf <exp> <model> <ds> <extra_args> <use_test_flag>
run_inf() {
    local exp="$1"; local model="$2"; local ds="$3"; local extra="$4"; local use_test_flag="$5"
    local ckpt="results/${exp}/ckpt/best.pth"
    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${exp} — no ckpt at ${ckpt}"
        return
    fi
    echo "==========================================="
    echo ">> ${exp} (${model}, ${ds})"
    echo "==========================================="
    "${PYBIN}" -m experiments.inference \
        --ds "${ds}" ${PATHS_PROFILE} --gpu "${GPU}" \
        --model "${model}" --ckpt "${ckpt}" --strict_ckpt 0 --hd95 \
        ${use_test_flag} ${extra} \
        --out_dir "${OUT}/inference/${exp}"
}


# ============================================================
# 1. CTCF mixing-axis inferences (OASIS, val = test N=19)
# ============================================================
if [ "${SKIP_CTCF}" != "1" ]; then
    run_inf P11_MAMBA_SVF_CTRL_NONE_OASIS  ctcf OASIS "${CTCF_BASE_INF} --ctcf_l3_corr_mode none"                  ""
    run_inf P11_MAMBA_SVF_HADAMARD_OASIS   ctcf OASIS "${CTCF_BASE_INF} --ctcf_l3_corr_mode hadamard"              ""
    run_inf P11_MAMBA_SVF_CORR27_OASIS     ctcf OASIS "${CTCF_BASE_INF} --ctcf_l3_corr_mode corr"                  ""
    run_inf P11_MAMBA_SVF_L3CH64_OASIS     ctcf OASIS "${CTCF_BASE_INF} --ctcf_l3_base_ch 64"                      ""
    run_inf P11_MAMBA_SVF_MH2_OASIS        ctcf OASIS "${CTCF_BASE_INF} --ctcf_l3_num_heads 2"                     ""
    # M3 grid: loss-only change, model is a plain Mamba SVF cascade -> base args, no overrides.
    run_inf P11_MAMBA_SVF_REG_FLAT_OASIS    ctcf OASIS "${CTCF_BASE_INF}" ""
    run_inf P11_MAMBA_SVF_REG_MID_OASIS     ctcf OASIS "${CTCF_BASE_INF}" ""
    run_inf P11_MAMBA_SVF_REG_STRONG_OASIS  ctcf OASIS "${CTCF_BASE_INF}" ""
    run_inf P11_MAMBA_SVF_REG_VSTRONG_OASIS ctcf OASIS "${CTCF_BASE_INF}" ""
    run_inf P11_MAMBA_SVF_STACK_OASIS      ctcf OASIS "${CTCF_BASE_INF} --ctcf_l3_corr_mode hadamard --ctcf_l3_base_ch 64" ""
fi


# ============================================================
# 2. Track V baselines (CorrMLP / SACB) — OASIS N=19 + IXI N=115 test
# ============================================================
if [ "${SKIP_TRACKV}" != "1" ]; then
    run_inf P11_CORRMLP_OASIS corrmlp OASIS "" ""
    run_inf P11_SACB_OASIS    sacb    OASIS "" ""
    run_inf P11_CORRMLP_IXI   corrmlp IXI   "" "--use_test"
    run_inf P11_SACB_IXI      sacb    IXI   "" "--use_test"
fi


# ============================================================
# 3. Paired Wilcoxon + Hodges-Lehmann + BH-FDR (mixing axis + Track V)
# ============================================================
if [ "${SKIP_STATS}" != "1" ]; then
    echo ""
    echo "==========================================="
    echo ">> Phase 11 paired Wilcoxon + BH-FDR"
    echo "==========================================="
    "${PYBIN}" tools/analysis/compute_stats.py phase11 \
        --sedm-root "${OUT}/inference" \
        --out "${OUT}/summary/phase11_stats.csv" \
        2>&1 | tee "${OUT}/summary/phase11_stats.txt"
fi

echo ""
echo "==================================================================="
echo "Phase 11 inference + stats complete."
echo ""
echo "Files to send back to user (all small):"
echo "  - ${OUT}/inference/P11_*/per_case.csv      (CTCF mechanisms + Track V)"
echo "  - ${OUT}/summary/phase11_stats.csv"
echo "  - ${OUT}/summary/phase11_stats.txt"
echo "==================================================================="
