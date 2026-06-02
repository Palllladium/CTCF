#!/usr/bin/env bash
# Phase 10 — Inference + cross-dataset + stats for Paper 2 finalization.
#
# Runs:
#   1. Inference on 7 Phase 10 longrun ckpts (native dataset for each ckpt)
#   2. Cross-dataset zero-shot on Mamba SVF headline ckpts (2 runs)
#   3. Re-runs aggregator (picks up Phase 10 entries via updated CONFIGS)
#   4. Re-runs paired Wilcoxon stat-tests vs Paper 1 CTCF Swin-DCA cascade
#
# Phase 10 ckpts must live at results/<EXP_NAME>/ckpt/best.pth.
# Existing Phase 9 inference results in results/SEDM/inference/<exp>/ are NOT touched.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase10_inference.sh
#
# Override defaults:
#   GPU=0 PATHS_PROFILE=--2 OUT=results/SEDM bash tools/runners/phase10_inference.sh
#
# Skip flags:
#   SKIP_NATIVE=1     — skip 7 native-dataset inferences
#   SKIP_CROSS=1      — skip 2 cross-dataset zero-shot inferences
#   SKIP_AGGREGATE=1  — skip aggregator re-run
#   SKIP_STATS=1      — skip Wilcoxon stat-tests

set -e

GPU="${GPU:-0}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM}"

SKIP_NATIVE="${SKIP_NATIVE:-0}"
SKIP_CROSS="${SKIP_CROSS:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
SKIP_STATS="${SKIP_STATS:-0}"

# Required for tools/analysis/* imports of experiments.core.*
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

mkdir -p "${OUT}/inference" "${OUT}/summary"

# Native inference helper
run_inf() {
    local exp_name="$1"; local ds="$2"; local config_arg="$3"
    local ckpt="results/${exp_name}/ckpt/best.pth"
    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${exp_name} — no ckpt at ${ckpt}"
        return
    fi
    local use_test=""; [ "${ds}" = "IXI" ] && use_test="--use_test"
    local out_dir="${OUT}/inference/${exp_name}"

    echo "==========================================="
    echo ">> ${exp_name} (${ds}, native)"
    echo "==========================================="
    "${PYBIN}" -m experiments.inference \
        --ds "${ds}" ${PATHS_PROFILE} --gpu "${GPU}" \
        --model ctcf \
        --ckpt "${ckpt}" \
        --strict_ckpt 0 \
        --hd95 \
        ${use_test} \
        ${config_arg} \
        --out_dir "${out_dir}"
}

# Cross-dataset inference helper: source ckpt evaluated on TARGET test set
run_inf_cross() {
    local src_exp="$1"; local target_ds="$2"; local config_arg="$3"; local cross_name="$4"
    local ckpt="results/${src_exp}/ckpt/best.pth"
    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] cross ${src_exp} → ${target_ds}: no ckpt"
        return
    fi
    local use_test=""; [ "${target_ds}" = "IXI" ] && use_test="--use_test"
    local out_dir="${OUT}/inference/${cross_name}"

    echo "==========================================="
    echo ">> ${cross_name} (${src_exp} ckpt → ${target_ds} test set)"
    echo "==========================================="
    "${PYBIN}" -m experiments.inference \
        --ds "${target_ds}" ${PATHS_PROFILE} --gpu "${GPU}" \
        --model ctcf \
        --ckpt "${ckpt}" \
        --strict_ckpt 0 \
        --hd95 \
        ${use_test} \
        ${config_arg} \
        --out_dir "${out_dir}"
}

# ============================================================
# 1. Native inference (7 runs)
# ============================================================
if [ "${SKIP_NATIVE}" != "1" ]; then
    run_inf P10_LONGRUN_MAMBA_SVF_OASIS       OASIS "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"
    run_inf P10_LONGRUN_MAMBA_SVF_IXI         IXI   "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"
    run_inf P10_LONGRUN_MAMBA_NOSVF_OASIS     OASIS "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0"
    run_inf P10_LONGRUN_LKU8_SVF_OASIS        OASIS "--ctcf_config CTCF-CascadeA-LKU8 --ctcf_l3_svf 1"
    run_inf P10_LONGRUN_LKU8_SVF_IXI          IXI   "--ctcf_config CTCF-CascadeA-LKU8 --ctcf_l3_svf 1"
    run_inf P10_LONGRUN_VXM_UNIFIED_SVF_OASIS OASIS "--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
    run_inf P10_LONGRUN_VXM_UNIFIED_SVF_IXI   IXI   "--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
fi

# ============================================================
# 2. Cross-dataset zero-shot (Mamba SVF headline)
# ============================================================
if [ "${SKIP_CROSS}" != "1" ]; then
    # OASIS-trained Mamba SVF ckpt evaluated on IXI test set
    run_inf_cross P10_LONGRUN_MAMBA_SVF_OASIS IXI   "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1" \
                  P10_CROSS_MAMBA_SVF_OASIS_TO_IXI
    # IXI-trained Mamba SVF ckpt evaluated on OASIS test set
    run_inf_cross P10_LONGRUN_MAMBA_SVF_IXI   OASIS "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1" \
                  P10_CROSS_MAMBA_SVF_IXI_TO_OASIS
fi

# ============================================================
# 3. Aggregate (Phase 9 + Phase 10 in one summary)
# ============================================================
if [ "${SKIP_AGGREGATE}" != "1" ]; then
    echo ""
    echo "==========================================="
    echo ">> Aggregating with Phase 10 data"
    echo "==========================================="
    "${PYBIN}" tools/analysis/aggregate_sedm_results.py \
        --inference-dir "${OUT}/inference" \
        --complexity "${OUT}/complexity.csv" \
        --output-dir "${OUT}/summary"
fi

# ============================================================
# 4. Paired Wilcoxon stat-tests vs Paper 1 (Phase 9 + Phase 10 entries)
# ============================================================
if [ "${SKIP_STATS}" != "1" ]; then
    echo ""
    echo "==========================================="
    echo ">> Paired Wilcoxon vs Paper 1"
    echo "==========================================="
    "${PYBIN}" tools/analysis/compute_stats.py sedm_vs_paper1 \
        --infer-root results/infer \
        --sedm-root "${OUT}/inference" \
        2>&1 | tee "${OUT}/summary/stat_tests_vs_paper1.txt"
fi

echo ""
echo "==================================================================="
echo "Phase 10 inference + aggregation complete."
echo ""
echo "Files to send back to user:"
echo "  - ${OUT}/inference/P10_LONGRUN_*/per_case.csv     (7 files)"
echo "  - ${OUT}/inference/P10_CROSS_*/per_case.csv       (2 files)"
echo "  - ${OUT}/summary/aggregated.csv"
echo "  - ${OUT}/summary/main_oasis.{md,tex}"
echo "  - ${OUT}/summary/main_ixi.{md,tex}"
echo "  - ${OUT}/summary/cascade_delta.{md,tex}"
echo "  - ${OUT}/summary/stat_tests.md"
echo "  - ${OUT}/summary/stat_tests_vs_paper1.txt"
echo "==================================================================="
