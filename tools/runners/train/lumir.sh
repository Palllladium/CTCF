#!/usr/bin/env bash
# LUMIR 2024 training of one CTCF configuration on one GPU, then export of the explicitly selected validation
# displacements from the last checkpoint. Validation Dice uses the public coarse labels of the LUMIR 2025 T1 pairs.
#   CONFIG (default CTCF-CascadeA-Mamba), MODEL (architecture overrides shared by training and export, default
#   "--l3_svf 0"), TRAIN (further training flags), ITERS (iterations per epoch, default 1000), EPOCHS (default 40),
#   GPU (default 0), PAIRS=lumir24|lumir25 and EXPECTED_GIT_HEAD are required. PROFILE=1 runs 200 updates.
#   EXP fixes the run name; RESUME=1 continues its last.pth with the same total EPOCHS and training recipe.
#   Put --disable_l1/--disable_l3 in MODEL so training and export use the same active levels.
#   TIME_STEPS (default 6) sets the same integration count for training and export.
#   Layout under $CTCF_DATA_DIR/LUMIR: imagesTr/, imagesVal/ and LUMIR_dataset.json of the 2024 archive;
#   LUMIR25/{imagesVal, simplelabelsVal, LUMIR25_dataset.json} of the public 2025 validation download.
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${EXPECTED_GIT_HEAD:?Set EXPECTED_GIT_HEAD to the reviewed commit}"
[ "$(git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD" ] || { echo "[lumir] unexpected HEAD"; exit 1; }
case "${PAIRS:-}" in lumir24|lumir25) ;; *) echo "[lumir] set PAIRS=lumir24 or lumir25 explicitly"; exit 1 ;; esac
git diff --quiet HEAD -- || { echo "[lumir] tracked files differ from HEAD: commit or restore them"; exit 1; }

PY="${PYBIN:-${PY:-/data/mooncake/P/envs/ctcf/bin/python}}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
CONFIG="${CONFIG:-CTCF-CascadeA-Mamba}"
MODEL="${MODEL:---l3_svf 0}"
TRAIN="${TRAIN:-}"
TIME_STEPS="${TIME_STEPS:-6}"
case "$TIME_STEPS" in ''|*[!0-9]*) echo "[lumir] TIME_STEPS must be a non-negative integer"; exit 1 ;; esac
for arg in $MODEL $TRAIN; do
    case "$arg" in --time_steps|--time_steps=*) echo "[lumir] use TIME_STEPS instead of --time_steps"; exit 1 ;; esac
done
for arg in $TRAIN; do
    case "$arg" in --disable_l1|--disable_l1=*|--disable_l3|--disable_l3=*)
        echo "[lumir] put --disable_l1/--disable_l3 in MODEL, not TRAIN"; exit 1 ;;
    esac
done
ITERS="${ITERS:-1000}"
EPOCHS="${EPOCHS:-40}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESUME="${RESUME:-0}"
case "$RESUME" in 0|1) ;; *) echo "[lumir] RESUME must be 0 or 1"; exit 1 ;; esac
if [ "$RESUME" = 1 ]; then
    : "${EXP:?Set the original EXP when resuming}"
fi
EXP="${EXP:-LUMIR_$STAMP}"
if [ "${PROFILE:-0}" = 1 ]; then
    [ "$RESUME" = 0 ] || { echo "[lumir] profiling cannot resume training"; exit 1; }
    ITERS=200
    EPOCHS=1
    EXP="LUMIR_PROFILE_$STAMP"
fi
RUN="results/lumir/$EXP"
resume_args=()
if [ "$RESUME" = 1 ]; then
    [ -f "results/$EXP/ckpt/last.pth" ] || { echo "[lumir] last.pth missing for $EXP"; exit 1; }
    resume_args=(--resume "results/$EXP/ckpt/last.pth")
elif [ -e "$RUN" ] || [ -e "results/$EXP" ] || [ -e "logs/$EXP" ]; then
    echo "[lumir] EXP already exists; choose another EXP or RESUME=1"; exit 1
fi
ATTEMPT="$RUN/attempts/${STAMP}_$$"

mkdir -p "$ATTEMPT"
echo "$(date -u +%FT%TZ) HEAD $(git rev-parse HEAD) CONFIG=$CONFIG MODEL=$MODEL TRAIN=$TRAIN TIME_STEPS=$TIME_STEPS ITERS=$ITERS EPOCHS=$EPOCHS" \
    > "$ATTEMPT/run_info.txt"
echo "training=lumir24 coarse_monitor=lumir25 official_export=$PAIRS RESUME=$RESUME" | tee -a "$ATTEMPT/run_info.txt"
"$PY" -m pytest -q -p no:cacheprovider tools/analysis/tests/test_lumir.py > "$ATTEMPT/pre_tests.txt" 2>&1 \
    || { echo "[lumir] unit tests failed: $ATTEMPT/pre_tests.txt"; exit 1; }

# shellcheck disable=SC2086
"$PY" -m experiments.train_CTCF --ds LUMIR --paths 3 --gpu 0 --exp "$EXP" --config "$CONFIG" $MODEL $TRAIN \
    --time_steps "$TIME_STEPS" "${resume_args[@]}" --max_epoch "$EPOCHS" --max_train_iters "$ITERS" --use_tb 0 --quiet 1 \
    > "$ATTEMPT/train.txt" 2>&1
cp -r "logs/$EXP" "$ATTEMPT/training_logs"
if [ "${PROFILE:-0}" = 1 ]; then
    grep "\[perf" "$ATTEMPT/train.txt" | tail -n 1
    exit 0
fi

# shellcheck disable=SC2086
"$PY" -m experiments.export_lumir --paths 3 --gpu 0 --config "$CONFIG" $MODEL --ckpt "results/$EXP/ckpt/last.pth" \
    --time_steps "$TIME_STEPS" --pairs "$PAIRS" --out "$ATTEMPT/export_$PAIRS" --name "$EXP" \
    > "$ATTEMPT/export_$PAIRS.txt" 2>&1

mkdir -p results/exports
ARCHIVE="results/exports/${EXP}__${STAMP}_$$.zip"
"$PY" - "$RUN" "$ARCHIVE" << 'EOF'
import sys
import zipfile
from pathlib import Path

run, archive = Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(run.rglob("*")):
        if path.is_file() and not path.name.endswith((".pth", ".pt", ".nii.gz", ".zip")):
            z.write(path, Path(Path(archive).stem) / path.relative_to(run))
EOF
echo "[lumir] done: $ARCHIVE (manifests and logs; displacements stay in $RUN)"
