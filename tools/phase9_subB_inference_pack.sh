#!/usr/bin/env bash
# Phase 9 Sub-B — inference + complexity + aggregation for SEDM paper.
#
# Runs inference.py on every available cascade/L2-only checkpoint from P7/P8/P9
# (skips missing ones gracefully), computes cascade-level computational complexity,
# and aggregates per-pair metrics into paste-ready Markdown + LaTeX tables.
#
# Output structure:
#   results/SEDM/
#     inference/<exp_name>/best/per_case.csv      (default inference.py output dir)
#     inference/<exp_name>/best/summary.csv
#     complexity.csv                              (params, GFLOPs, VRAM, throughput)
#     summary/main_oasis.md                       (paste-ready Markdown)
#     summary/main_oasis.tex                      (paste-ready LaTeX tabularx)
#     summary/main_ixi.md
#     summary/main_ixi.tex
#     summary/cascade_delta.md                    (L2-only vs cascade Δ)
#     summary/cascade_delta.tex
#     summary/stat_tests.md                       (paired Wilcoxon vs Paper 1 baselines)
#     summary/README.md                           (overview, what's where, how to paste)
#
# Usage:
#   conda activate ctcf
#   bash tools/phase9_subB_inference_pack.sh
#
# Override defaults:
#   GPU=0 PATHS_PROFILE=--2 OUT=results/SEDM bash tools/phase9_subB_inference_pack.sh
#
# Skip flags (для частичных перезапусков):
#   SKIP_INFERENCE=1 — пропустить inference, только перегнерировать summary tables
#   SKIP_COMPLEXITY=1 — пропустить cascade complexity
#   SKIP_AGGREGATE=1 — пропустить aggregation

set -e

GPU="${GPU:-0}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM}"

# Ensure project root is on PYTHONPATH so 'from experiments.core...' imports work
# when calling python tools/model_complexity.py directly (it's not a package).
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

mkdir -p "${OUT}/summary"

# Helper: run inference.py for a given experiment.
# Gracefully skips if checkpoint is absent (so the script tolerates partial Sub-A completion).
run_inf() {
    local exp_name="$1"
    local ds="$2"
    local model="$3"
    local config_arg="$4"
    local extra_args="$5"

    # Checkpoints are saved by make_exp_dirs() to results/<exp_name>/ckpt/,
    # NOT to experiments/<exp_name>/ckpt/. See utils/runtime.py:95.
    local ckpt="results/${exp_name}/ckpt/best.pth"
    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${exp_name} — no ckpt at ${ckpt}"
        return
    fi

    local use_test_flag=""
    [ "${ds}" = "IXI" ] && use_test_flag="--use_test"

    local out_dir="${OUT}/inference/${exp_name}"

    echo "==========================================="
    echo ">> ${exp_name} (${ds}, model=${model})"
    echo "==========================================="

    "${PYBIN}" -m experiments.inference \
        --ds "${ds}" ${PATHS_PROFILE} --gpu "${GPU}" \
        --model "${model}" \
        --ckpt "${ckpt}" \
        --strict_ckpt 0 \
        --hd95 \
        ${use_test_flag} \
        ${config_arg} \
        ${extra_args} \
        --out_dir "${out_dir}"
}

if [ "${SKIP_INFERENCE}" != "1" ]; then

    # ===== Cascade runs OASIS =====
    run_inf P7_CASC_LKU8_OASIS          OASIS ctcf "--ctcf_config CTCF-CascadeA-LKU8"  ""
    run_inf P8_CASC_LKU8_FIXSCHED_OASIS OASIS ctcf "--ctcf_config CTCF-CascadeA-LKU8"  ""
    run_inf P8_CASC_LKU32_SVF_OASIS     OASIS ctcf "--ctcf_config CTCF-CascadeA-LKU32" ""
    run_inf P7_CASC_MAMBA_SVF_OASIS     OASIS ctcf "--ctcf_config CTCF-CascadeA-Mamba" ""
    run_inf P8_CASC_MAMBA_NOSVF_OASIS   OASIS ctcf "--ctcf_config CTCF-CascadeA-Mamba" ""
    run_inf P7_CASC_VMAMBA_SVF_OASIS    OASIS ctcf "--ctcf_config CTCF-CascadeA-VMamba" ""
    run_inf P9_CASC_VXM_SVF_OASIS       OASIS ctcf "--ctcf_config CTCF-CascadeA-VM" ""
    run_inf P9_CASC_LKU8_SVF_OASIS      OASIS ctcf "--ctcf_config CTCF-CascadeA-LKU8" ""

    # ===== Cascade runs IXI =====
    run_inf P8_CASC_MAMBA_SVF_IXI       IXI ctcf "--ctcf_config CTCF-CascadeA-Mamba"  ""
    run_inf P8_CASC_LKU32_SVF_IXI       IXI ctcf "--ctcf_config CTCF-CascadeA-LKU32" ""
    run_inf P9_CASC_VXM_SVF_IXI         IXI ctcf "--ctcf_config CTCF-CascadeA-VM" ""
    run_inf P9_CASC_LKU8_SVF_IXI        IXI ctcf "--ctcf_config CTCF-CascadeA-LKU8" ""
    run_inf P9_CASC_MAMBA_NOSVF_IXI     IXI ctcf "--ctcf_config CTCF-CascadeA-Mamba" ""
    run_inf P9_CASC_VMAMBA_SVF_IXI      IXI ctcf "--ctcf_config CTCF-CascadeA-VMamba" ""

    # ===== L2-only controls OASIS =====
    run_inf P7_CTRL_LKU8_L2ONLY_OASIS   OASIS ctcf "--ctcf_config CTCF-LKU8-solo" ""
    run_inf P7_CTRL_LKU32_L2ONLY_OASIS  OASIS ctcf "--ctcf_config CTCF-LKU32-solo" ""
    run_inf P7_CTRL_MAMBA_L2ONLY_OASIS  OASIS ctcf "--ctcf_config CTCF-Mamba-solo" ""
    run_inf P7_CTRL_VMAMBA_L2ONLY_OASIS OASIS ctcf "--ctcf_config CTCF-VMamba-solo" ""
    run_inf P9_CTRL_VXM_L2ONLY_OASIS    OASIS ctcf "--ctcf_config CTCF-VM-solo" ""

    # ===== L2-only controls IXI =====
    run_inf P9_CTRL_LKU8_L2ONLY_IXI     IXI ctcf "--ctcf_config CTCF-LKU8-solo" ""
    run_inf P9_CTRL_LKU32_L2ONLY_IXI    IXI ctcf "--ctcf_config CTCF-LKU32-solo" ""
    run_inf P9_CTRL_MAMBA_L2ONLY_IXI    IXI ctcf "--ctcf_config CTCF-Mamba-solo" ""
    run_inf P9_CTRL_VMAMBA_L2ONLY_IXI   IXI ctcf "--ctcf_config CTCF-VMamba-solo" ""
    run_inf P9_CTRL_VXM_L2ONLY_IXI      IXI ctcf "--ctcf_config CTCF-VM-solo" ""

fi

# ===== Cascade complexity (params, GFLOPs, VRAM, throughput) =====
if [ "${SKIP_COMPLEXITY}" != "1" ]; then
    echo ""
    echo "==========================================="
    echo ">> Cascade complexity"
    echo "==========================================="
    "${PYBIN}" tools/model_complexity.py --gpu "${GPU}" --out "${OUT}/complexity.csv"
fi

# ===== Aggregate per_case.csv into paste-ready tables =====
if [ "${SKIP_AGGREGATE}" != "1" ]; then
    echo ""
    echo "==========================================="
    echo ">> Aggregating into paste-ready tables"
    echo "==========================================="
    "${PYBIN}" tools/aggregate_sedm_results.py \
        --inference-dir "${OUT}/inference" \
        --complexity "${OUT}/complexity.csv" \
        --output-dir "${OUT}/summary"
fi

echo ""
echo "==========================================="
echo "SEDM packaging complete."
echo "Output: ${OUT}/"
echo "  inference/<exp>/per_case.csv       — per-pair metrics"
echo "  complexity.csv                     — params, GFLOPs, VRAM, throughput"
echo "  summary/main_oasis.{md,tex}        — paste-ready main OASIS table"
echo "  summary/main_ixi.{md,tex}          — paste-ready main IXI table"
echo "  summary/cascade_delta.{md,tex}     — L2-only vs cascade Δ"
echo "  summary/stat_tests.md              — paired Wilcoxon vs Paper 1"
echo "  summary/README.md                  — overview"
echo "==========================================="
