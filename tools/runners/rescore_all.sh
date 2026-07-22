#!/usr/bin/env bash
# Re-score every published configuration under the corrected fold metrics.
#
# WHY: inference now emits three fold counts per case instead of one —
#   j_leq0_central_percent   central differences  (the community default; VoxelMorph/TransMorph lineage)
#   j_leq0_corners_percent   8 one-sided schemes  (what we published before 2026-07-22)
#   j_leq0_percent           all 10 determinants  (Liu et al., IJCV 2024 — the full criterion)
#   ndv_percent              non-diffeomorphic volume
# They differ by orders of magnitude on the SAME field, and per_case.csv does not store the
# deformation field, so the only way to obtain them consistently is one inference pass that emits
# all four. No training, no weight changes — checkpoints are read, never written.
#
# This orchestrates the existing runners rather than restating their config lists, so a config
# cannot be silently dropped by a list that drifted out of sync.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/rescore_all.sh
#
# Skip flags (each stage is independent and resumable):
#   SKIP_P10=1 SKIP_P9=1 SKIP_P11=1 SKIP_PAPER1=1
#
# Runtime: ~4-6 h total on one GPU.

set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT="${OUT:-results/SEDM}"

SKIP_P10="${SKIP_P10:-0}"
SKIP_P9="${SKIP_P9:-0}"
SKIP_P11="${SKIP_P11:-0}"
SKIP_PAPER1="${SKIP_PAPER1:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

echo "############################################################"
echo "# Re-scoring under the corrected fold metrics"
echo "# Output: ${OUT}/   (back up the previous one first)"
echo "############################################################"

# 1. Phase 10 — 7 longruns (the headline table) + 2 cross-dataset + aggregation.   ~1-2 h
if [ "${SKIP_P10}" != "1" ]; then
    echo ""
    echo "########## 1/4  Phase 10 — headline 500ep + cross-dataset ##########"
    GPU="${GPU}" PATHS_PROFILE="${PROFILE}" OUT="${OUT}" \
        bash tools/runners/phase10_inference.sh
fi

# 2. Phase 9 Stage 2 — the 24-config extended table. Training is skipped.         ~1.5-2.5 h
if [ "${SKIP_P9}" != "1" ]; then
    echo ""
    echo "########## 2/4  Phase 9 Stage 2 — extended table (inference only) ##########"
    SKIP_STAGE1=1 GPU="${GPU}" PATHS_PROFILE="${PROFILE}" \
        bash tools/runners/phase9_matrix_and_inference.sh
fi

# 3. Phase 11 — 15 ablation configs + the CorrMLP/SACB competitor baselines.      ~1 h
if [ "${SKIP_P11}" != "1" ]; then
    echo ""
    echo "########## 3/4  Phase 11 — ablations + competitor baselines ##########"
    GPU="${GPU}" PATHS_PROFILE="${PROFILE}" \
        bash tools/runners/phase11_infer_stats.sh
fi

# 4. Paper-1 baseline (Swin-DCA). Not covered by any inference runner: its checkpoints predate the
#    SEDM layout. Every "vs Paper 1" statement needs it on the same metric as everything else.  ~15 min
if [ "${SKIP_PAPER1}" != "1" ]; then
    echo ""
    echo "########## 4/4  Paper-1 Swin-DCA baseline ##########"
    BLOCK=RS GPU="${GPU}" PROFILE="${PROFILE}" \
        bash tools/runners/tto_phase2.sh
fi

echo ""
echo "############################################################"
echo "Re-scoring complete."
echo ""
echo "Send back:"
echo "  ${OUT}/          — all per_case.csv (the leaf data; every table rebuilds from these)"
echo "  results/tto2/RS_*  — Paper-1 baseline + the guard probe"
echo ""
echo "NOT re-scored on purpose: SEDM_CASC_*_NOSVF (5 configs, 100ep)."
echo "  SVF-vs-NoSVF is carried by the 500ep rows where the effect is far larger;"
echo "  these were already slated to be dropped from the extended table."
echo "############################################################"
