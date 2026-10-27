#!/usr/bin/env bash
# LUMIR 2024 training of one CTCF configuration on one GPU, then export of the explicitly selected validation
# displacements from the last checkpoint. Validation Dice uses the public coarse labels of the LUMIR 2025 T1 pairs.
#   CONFIG (any models/CTCF/configs.py preset), MODEL (shared architecture overrides), TRAIN (loss/optimizer flags),
#   PRECISION=fp16|bf16|fp32|tf32 is required and shared by training, validation and export.
#   STAGES=ramp|full (default ramp); profiling always activates all configured stages, including backward.
#   ITERS (default 1000, 0 visits all moving images), EPOCHS (default 40), BATCH_SIZE (default 1),
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
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$EXPECTED_GIT_HEAD" ] || { echo "[lumir] unexpected HEAD"; exit 1; }
case "${PAIRS:-}" in lumir24|lumir25) ;; *) echo "[lumir] set PAIRS=lumir24 or lumir25 explicitly"; exit 1 ;; esac
git diff --quiet HEAD -- || { echo "[lumir] tracked files differ from HEAD: commit or restore them"; exit 1; }

PY="${PYBIN:-${PY:-/data/mooncake/P/envs/ctcf/bin/python}}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
GPU="${GPU:-0}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "[lumir] GPU must name one device; this runner does not implement DDP"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
CONFIG="${CONFIG:-CTCF-CascadeA-Mamba}"
MODEL="${MODEL:-}"
TRAIN="${TRAIN:-}"
read -r -a model_args <<< "${MODEL//$'\n'/ }"
read -r -a training_args <<< "${TRAIN//$'\n'/ }"
: "${PRECISION:?Set PRECISION to fp16, bf16, fp32 or tf32}"
case "$PRECISION" in fp16|bf16|fp32|tf32) ;; *) echo "[lumir] invalid PRECISION=$PRECISION"; exit 1 ;; esac
STAGES="${STAGES:-ramp}"
case "$STAGES" in ramp|full) ;; *) echo "[lumir] STAGES must be ramp or full"; exit 1 ;; esac
BATCH_SIZE="${BATCH_SIZE:-1}"
TIME_STEPS="${TIME_STEPS:-6}"
case "$TIME_STEPS" in ''|*[!0-9]*) echo "[lumir] TIME_STEPS must be a non-negative integer"; exit 1 ;; esac
for arg in "${model_args[@]}" "${training_args[@]}"; do
    case "$arg" in --time_steps|--time_steps=*) echo "[lumir] use TIME_STEPS instead of --time_steps"; exit 1 ;; esac
    case "$arg" in --config|--config=*) echo "[lumir] use CONFIG instead of --config"; exit 1 ;; esac
    case "$arg" in --precision|--precision=*|--finetune|--finetune=*|--batch_size|--batch_size=*)
        echo "[lumir] use PRECISION, STAGES and BATCH_SIZE instead of overriding their CLI flags"; exit 1 ;;
    esac
done
for arg in "${training_args[@]}"; do
    case "$arg" in --l1_from_start|--l1_from_start=*) ;;
    --disable_l1|--disable_l1=*|--disable_l3|--disable_l3=*|--l1_*|--l2_*|--l3_*|--use_checkpoint|--use_checkpoint=*)
        echo "[lumir] put architecture overrides in MODEL so export uses the same configuration"; exit 1 ;;
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
    STAGES=full
fi
stage_args=(--finetune 0)
[ "$STAGES" = ramp ] || stage_args=(--finetune 1)
RUN="results/lumir/$EXP"
resume_args=()
if [ "$RESUME" = 1 ]; then
    [ -f "results/$EXP/ckpt/last.pth" ] || { echo "[lumir] last.pth missing for $EXP"; exit 1; }
    resume_args=(--resume "results/$EXP/ckpt/last.pth")
elif [ -e "$RUN" ] || [ -e "results/$EXP" ] || [ -e "logs/$EXP" ]; then
    echo "[lumir] EXP already exists; choose another EXP or RESUME=1"; exit 1
fi
common_args=(--ds LUMIR --paths 3 --gpu 0 --config "$CONFIG" "${model_args[@]}" --time_steps "$TIME_STEPS" --precision "$PRECISION")
train_args=("${common_args[@]}" --exp "$EXP" "${training_args[@]}" --max_epoch "$EPOCHS" --max_train_iters "$ITERS" \
    --use_tb 0 --quiet 1 --batch_size "$BATCH_SIZE" "${stage_args[@]}")
printf -v recipe '%q ' "CTCF_DATA_DIR=$CTCF_DATA_DIR" "${train_args[@]}" "PAIRS=$PAIRS"
if [ "$RESUME" = 1 ]; then
    [ -f "$RUN/recipe.txt" ] && [ "$(cat "$RUN/recipe.txt")" = "$recipe" ] \
        || { echo "[lumir] resume recipe differs or is missing; start a new EXP for a changed recipe"; exit 1; }
else
    mkdir -p "$RUN"
    echo "$recipe" > "$RUN/recipe.txt"
fi
ATTEMPT="$RUN/attempts/${STAMP}_$$"
mkdir -p "$ATTEMPT/exit_codes"
echo "$(date -u +%FT%TZ) HEAD=$HEAD GPU=$GPU PROFILE=${PROFILE:-0} RESUME=$RESUME" > "$ATTEMPT/run_info.txt"
echo "$recipe" >> "$ATTEMPT/run_info.txt"
CHILD=""
required=(pretests train)
[ "${PROFILE:-0}" = 1 ] || required+=(export)
finish() {
    local code=$?
    trap - EXIT INT TERM
    if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" 2>/dev/null || true; fi
    if [ -d "logs/$EXP" ]; then cp -r "logs/$EXP" "$ATTEMPT/training_logs" || { [ "$code" != 0 ] || code=1; }; fi
    "$PY" -m tools.analysis.run_artifacts package --run-root "$ATTEMPT" --prefix "results/exports/${EXP}__${STAMP}_$$" \
        --exit-code "$code" --git-head "$HEAD" --label LUMIR --required-steps "${required[@]}" || { [ "$code" != 0 ] || code=1; }
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
step() {
    local name=$1 code=0; shift
    echo "[lumir] START $name: $ATTEMPT/$name.log"
    PYTHONUNBUFFERED=1 "$@" > "$ATTEMPT/$name.log" 2>&1 & CHILD=$!
    wait "$CHILD" || code=$?; CHILD=""
    echo "$code" > "$ATTEMPT/exit_codes/$name.txt"
    if [ "$code" = 0 ]; then echo "[lumir] PASS $name"; else echo "[lumir] ERROR $name (exit $code)"; fi
    return "$code"
}
echo "[lumir] EXP=$EXP PID=$$ ATTEMPT=$ATTEMPT"
step pretests env CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider tools/analysis/tests/test_lumir.py tools/analysis/tests/test_precision.py
started=$SECONDS
step train "$PY" -m experiments.train_CTCF "${train_args[@]}" "${resume_args[@]}"
echo "wall_seconds=$((SECONDS - started)) includes=model_setup,training,validation,checkpoint_io" > "$ATTEMPT/train_timing.txt"
if [ "${PROFILE:-0}" = 1 ]; then
    grep "\[perf" "$ATTEMPT/train.log" | tail -n 1
    cat "$ATTEMPT/train_timing.txt"
else
    step export "$PY" -m experiments.export_lumir "${common_args[@]}" --ckpt "results/$EXP/ckpt/last.pth" \
        --pairs "$PAIRS" --out "$ATTEMPT/export_$PAIRS" --name "$EXP"
fi
