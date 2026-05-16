#!/usr/bin/env bash
# Phase 9 — Paper 2 matrix completion + inference packaging (merged Sub-A + Sub-B).
#
# Two stages, controlled via skip flags. Stage 1 trains 12 cascade/L2-only
# configurations at 100 epochs to close the OASIS+IXI backbone matrix; Stage 2
# runs inference on all available checkpoints (including legacy P7/P8 runs) and
# aggregates per-pair metrics into paste-ready summary tables.
#
# Stage 1 — Training (formerly phase9_subA_fill_matrix.sh):
#   12 runs, ~55h GPU. Skip with SKIP_STAGE1=1 if matrix already filled.
#
# Stage 2 — Inference + complexity + aggregation (formerly phase9_subB_inference_pack.sh):
#   24 inference runs (gracefully skips missing ckpts), ~1.5–2.5h GPU.
#   Updates results/SEDM/inference/ and results/SEDM/summary/.
#
# Usage:
#   conda activate ctcf
#   bash tools/phase9_matrix_and_inference.sh                  # both stages
#   SKIP_STAGE1=1 bash tools/phase9_matrix_and_inference.sh    # inference only
#   SKIP_STAGE2=1 bash tools/phase9_matrix_and_inference.sh    # training only

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"

# Stage 1 (training) per-run skip flags
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

# Stage 2 (inference) skip flags
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

# Required for tools/analysis/model_complexity.py imports
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

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
# Stage 1 — Training (12 runs to fill 100ep matrix gaps)
# ============================================================
if [ "${SKIP_STAGE1}" != "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "STAGE 1: training to fill 100ep matrix"
    echo "════════════════════════════════════════════════════════════════════"

    # VxM (small, fast) — note: uses legacy CTCF-CascadeA-VM config. For new
    # work under unified protocol see CTCF-CascadeA-VM-Unified in configs.py.
    [ "${SKIP_VXM_L2_OASIS}" != "1" ]    && run_ctcf "P9_CTRL_VXM_L2ONLY_OASIS" --config CTCF-VM-solo ${CTCF_OASIS}
    [ "${SKIP_VXM_CASC_OASIS}" != "1" ]  && run_ctcf "P9_CASC_VXM_SVF_OASIS" --config CTCF-CascadeA-VM --l3_svf 1 ${CTCF_OASIS}
    [ "${SKIP_VXM_L2_IXI}" != "1" ]      && run_ctcf "P9_CTRL_VXM_L2ONLY_IXI" --config CTCF-VM-solo ${CTCF_IXI}
    [ "${SKIP_VXM_CASC_IXI}" != "1" ]    && run_ctcf "P9_CASC_VXM_SVF_IXI" --config CTCF-CascadeA-VM --l3_svf 1 ${CTCF_IXI}

    # LKU-8
    [ "${SKIP_LKU8_L2_IXI}" != "1" ]     && run_ctcf "P9_CTRL_LKU8_L2ONLY_IXI" --config CTCF-LKU8-solo ${CTCF_IXI}
    [ "${SKIP_LKU8_CASC_OASIS}" != "1" ] && run_ctcf "P9_CASC_LKU8_SVF_OASIS" --config CTCF-CascadeA-LKU8 --l3_svf 1 ${CTCF_OASIS}
    [ "${SKIP_LKU8_CASC_IXI}" != "1" ]   && run_ctcf "P9_CASC_LKU8_SVF_IXI" --config CTCF-CascadeA-LKU8 --l3_svf 1 ${CTCF_IXI}

    # LKU-32 L2-only IXI (cascade IXI already in P8)
    [ "${SKIP_LKU32_L2_IXI}" != "1" ]    && run_ctcf "P9_CTRL_LKU32_L2ONLY_IXI" --config CTCF-LKU32-solo ${CTCF_IXI}

    # Mamba (heavy)
    [ "${SKIP_MAMBA_L2_IXI}" != "1" ]    && run_ctcf "P9_CTRL_MAMBA_L2ONLY_IXI" --config CTCF-Mamba-solo ${CTCF_IXI}
    [ "${SKIP_MAMBA_NOSVF_IXI}" != "1" ] && run_ctcf "P9_CASC_MAMBA_NOSVF_IXI" --config CTCF-CascadeA-Mamba --l3_svf 0 ${CTCF_IXI}

    # VMamba (heaviest; matrix completeness only, not in longruns)
    [ "${SKIP_VMAMBA_L2_IXI}" != "1" ]   && run_ctcf "P9_CTRL_VMAMBA_L2ONLY_IXI" --config CTCF-VMamba-solo ${CTCF_IXI}
    [ "${SKIP_VMAMBA_CASC_IXI}" != "1" ] && run_ctcf "P9_CASC_VMAMBA_SVF_IXI" --config CTCF-CascadeA-VMamba --l3_svf 1 ${CTCF_IXI}
fi

# ============================================================
# Stage 2 — Inference + complexity + aggregation
# ============================================================
if [ "${SKIP_STAGE2}" != "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "STAGE 2: inference + complexity + aggregation"
    echo "════════════════════════════════════════════════════════════════════"

    mkdir -p "${OUT}/summary"

    # Helper: run inference.py for a given experiment; skip if no ckpt.
    run_inf() {
        local exp_name="$1"; local ds="$2"; local config_arg="$3"; local extra="$4"
        local ckpt="results/${exp_name}/ckpt/best.pth"

        if [ ! -f "${ckpt}" ]; then
            echo "[SKIP] ${exp_name} — no ckpt at ${ckpt}"
            return
        fi

        local use_test=""; [ "${ds}" = "IXI" ] && use_test="--use_test"
        local out_dir="${OUT}/inference/${exp_name}"

        echo "==========================================="
        echo ">> ${exp_name} (${ds})"
        echo "==========================================="

        "${PYBIN}" -m experiments.inference \
            --ds "${ds}" ${PATHS_PROFILE} --gpu "${GPU}" \
            --model ctcf \
            --ckpt "${ckpt}" \
            --strict_ckpt 0 \
            --hd95 \
            ${use_test} \
            ${config_arg} \
            ${extra} \
            --out_dir "${out_dir}"
    }

    if [ "${SKIP_INFERENCE}" != "1" ]; then
        # OASIS cascades
        run_inf P7_CASC_LKU8_OASIS          OASIS "--ctcf_config CTCF-CascadeA-LKU8"  ""
        run_inf P8_CASC_LKU8_FIXSCHED_OASIS OASIS "--ctcf_config CTCF-CascadeA-LKU8"  ""
        run_inf P8_CASC_LKU32_SVF_OASIS     OASIS "--ctcf_config CTCF-CascadeA-LKU32" ""
        run_inf P7_CASC_MAMBA_SVF_OASIS     OASIS "--ctcf_config CTCF-CascadeA-Mamba" ""
        run_inf P8_CASC_MAMBA_NOSVF_OASIS   OASIS "--ctcf_config CTCF-CascadeA-Mamba" ""
        run_inf P7_CASC_VMAMBA_SVF_OASIS    OASIS "--ctcf_config CTCF-CascadeA-VMamba" ""
        run_inf P9_CASC_VXM_SVF_OASIS       OASIS "--ctcf_config CTCF-CascadeA-VM" ""
        run_inf P9_CASC_LKU8_SVF_OASIS      OASIS "--ctcf_config CTCF-CascadeA-LKU8" ""

        # IXI cascades
        run_inf P8_CASC_MAMBA_SVF_IXI       IXI "--ctcf_config CTCF-CascadeA-Mamba"  ""
        run_inf P8_CASC_LKU32_SVF_IXI       IXI "--ctcf_config CTCF-CascadeA-LKU32" ""
        run_inf P9_CASC_VXM_SVF_IXI         IXI "--ctcf_config CTCF-CascadeA-VM" ""
        run_inf P9_CASC_LKU8_SVF_IXI        IXI "--ctcf_config CTCF-CascadeA-LKU8" ""
        run_inf P9_CASC_MAMBA_NOSVF_IXI     IXI "--ctcf_config CTCF-CascadeA-Mamba" ""
        run_inf P9_CASC_VMAMBA_SVF_IXI      IXI "--ctcf_config CTCF-CascadeA-VMamba" ""

        # L2-only OASIS
        run_inf P7_CTRL_LKU8_L2ONLY_OASIS   OASIS "--ctcf_config CTCF-LKU8-solo" ""
        run_inf P7_CTRL_LKU32_L2ONLY_OASIS  OASIS "--ctcf_config CTCF-LKU32-solo" ""
        run_inf P7_CTRL_MAMBA_L2ONLY_OASIS  OASIS "--ctcf_config CTCF-Mamba-solo" ""
        run_inf P7_CTRL_VMAMBA_L2ONLY_OASIS OASIS "--ctcf_config CTCF-VMamba-solo" ""
        run_inf P9_CTRL_VXM_L2ONLY_OASIS    OASIS "--ctcf_config CTCF-VM-solo" ""

        # L2-only IXI
        run_inf P9_CTRL_LKU8_L2ONLY_IXI     IXI "--ctcf_config CTCF-LKU8-solo" ""
        run_inf P9_CTRL_LKU32_L2ONLY_IXI    IXI "--ctcf_config CTCF-LKU32-solo" ""
        run_inf P9_CTRL_MAMBA_L2ONLY_IXI    IXI "--ctcf_config CTCF-Mamba-solo" ""
        run_inf P9_CTRL_VMAMBA_L2ONLY_IXI   IXI "--ctcf_config CTCF-VMamba-solo" ""
        run_inf P9_CTRL_VXM_L2ONLY_IXI      IXI "--ctcf_config CTCF-VM-solo" ""
    fi

    if [ "${SKIP_COMPLEXITY}" != "1" ]; then
        echo ""
        echo ">> Cascade complexity"
        "${PYBIN}" tools/analysis/model_complexity.py --gpu "${GPU}" --out "${OUT}/complexity.csv"
    fi

    if [ "${SKIP_AGGREGATE}" != "1" ]; then
        echo ""
        echo ">> Aggregating into paste-ready tables"
        "${PYBIN}" tools/analysis/aggregate_sedm_results.py \
            --inference-dir "${OUT}/inference" \
            --complexity "${OUT}/complexity.csv" \
            --output-dir "${OUT}/summary"
    fi
fi

echo ""
echo "==================================================================="
echo "Phase 9 (matrix + inference) complete."
echo "Training: results/<exp>/ckpt/best.pth (Stage 1)"
echo "Inference: ${OUT}/inference/<exp>/per_case.csv (Stage 2)"
echo "Summary: ${OUT}/summary/*.{md,tex}"
echo "==================================================================="
