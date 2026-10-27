#!/usr/bin/env bash
# LUMIR 2024 training of one CTCF configuration via the shared trainer, then export of the selected validation
# displacements from the last checkpoint. Validation Dice uses the public coarse labels of the LUMIR 2025 T1 pairs.
#   CONFIG (any models/CTCF/configs.py preset), MODEL (shared architecture overrides), TRAIN (loss/optimizer flags),
#   PRECISION=fp16|bf16|fp32|tf32 is required and shared by training, validation and export.
#   STAGES=ramp|full (default ramp); profiling always activates all configured stages, including backward.
#   ITERS is optimizer updates per epoch (0 visits all moving images, padded to GLOBAL_BATCH).
#   ITERS and EPOCHS are explicit for training; BATCH_SIZE defaults to 1.
#   GPUS (space/comma-separated, or GPU for one device), GLOBAL_BATCH, PAIRS and EXPECTED_GIT_HEAD are required.
#   BATCH_SIZE is per GPU; accumulation preserves GLOBAL_BATCH. PROFILE=1 measures PROFILE_UPDATES (default 200)
#   after PERF_WARMUP_UPDATES (profile default 5, training default 0). Warm-up updates still train the model.
#   EXP fixes the run name; RESUME=1 continues its last.pth with the same total EPOCHS and training recipe.
#   Put --disable_l1/--disable_l3 in MODEL so training and export use the same active levels.
#   TIME_STEPS (default 6) sets the same integration count for training and export.
#   PROFILE disables checkpoint writes; PRETESTS=0 explicitly records skipped tests.
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
GPU_SPEC="${GPUS:-${GPU_LIST:-${GPU:-0}}}"
read -r -a GPU <<< "${GPU_SPEC//,/ }"
[ "${#GPU[@]}" -gt 0 ] || { echo "[lumir] GPUS is empty"; exit 1; }
declare -A seen=()
for gpu in "${GPU[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && [ -z "${seen[$gpu]:-}" ] || { echo "[lumir] invalid/duplicate GPU: $gpu"; exit 1; }
    seen[$gpu]=1
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU[*]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
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
: "${GLOBAL_BATCH:?Set GLOBAL_BATCH explicitly, including for a single-GPU comparison}"
NUM_WORKERS="${NUM_WORKERS:-2}"
[[ "$GLOBAL_BATCH" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || { echo "[lumir] GLOBAL_BATCH and BATCH_SIZE must be positive integers"; exit 1; }
(( GLOBAL_BATCH % (BATCH_SIZE * ${#GPU[@]}) == 0 )) \
    || { echo "[lumir] GLOBAL_BATCH must be divisible by BATCH_SIZE * number of GPUs"; exit 1; }
TIME_STEPS="${TIME_STEPS:-6}"
PRETESTS="${PRETESTS:-1}"
case "$PRETESTS" in 0|1) ;; *) echo "[lumir] PRETESTS must be 0 or 1"; exit 1 ;; esac
case "$TIME_STEPS" in ''|*[!0-9]*) echo "[lumir] TIME_STEPS must be a non-negative integer"; exit 1 ;; esac
for arg in "${model_args[@]}" "${training_args[@]}"; do
    case "$arg" in --time_steps|--time_steps=*) echo "[lumir] use TIME_STEPS instead of --time_steps"; exit 1 ;; esac
    case "$arg" in --config|--config=*) echo "[lumir] use CONFIG instead of --config"; exit 1 ;; esac
    case "$arg" in --precision|--precision=*|--finetune|--finetune=*|--batch_size|--batch_size=*|--global_batch_size|--global_batch_size=*|--num_workers|--num_workers=*|--perf_warmup_updates|--perf_warmup_updates=*)
        echo "[lumir] use PRECISION, STAGES, BATCH_SIZE, GLOBAL_BATCH and NUM_WORKERS for their CLI flags"; exit 1 ;;
    esac
done
for arg in "${training_args[@]}"; do
    case "$arg" in --l1_from_start|--l1_from_start=*) ;;
    --disable_l1|--disable_l1=*|--disable_l3|--disable_l3=*|--l1_*|--l2_*|--l3_*|--use_checkpoint|--use_checkpoint=*)
        echo "[lumir] put architecture overrides in MODEL so export uses the same configuration"; exit 1 ;;
    esac
done
if [ "${PROFILE:-0}" != 1 ]; then
    : "${ITERS:?Set ITERS to the agreed optimizer updates per epoch (0 for a full pass)}"
    : "${EPOCHS:?Set EPOCHS to the agreed total epoch count}"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESUME="${RESUME:-0}"
case "$RESUME" in 0|1) ;; *) echo "[lumir] RESUME must be 0 or 1"; exit 1 ;; esac
if [ "$RESUME" = 1 ]; then
    : "${EXP:?Set the original EXP when resuming}"
fi
EXP="${EXP:-LUMIR_$STAMP}"
save_ckpt=1
if [ "${PROFILE:-0}" = 1 ]; then
    [ "$RESUME" = 0 ] || { echo "[lumir] profiling cannot resume training"; exit 1; }
    PERF_WARMUP_UPDATES="${PERF_WARMUP_UPDATES:-5}"
    ITERS=$(( ${PROFILE_UPDATES:-200} + PERF_WARMUP_UPDATES ))
    EPOCHS=1
    EXP="LUMIR_PROFILE_$STAMP"
    STAGES=full
    save_ckpt=0
fi
PERF_WARMUP_UPDATES="${PERF_WARMUP_UPDATES:-0}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$ITERS" =~ ^[0-9]+$ ]] \
    || { echo "[lumir] EPOCHS must be positive and ITERS non-negative"; exit 1; }
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
    --use_tb 0 --quiet 1 --save_ckpt "$save_ckpt" --batch_size "$BATCH_SIZE" --global_batch_size "$GLOBAL_BATCH" --num_workers "$NUM_WORKERS" \
    --perf_warmup_updates "$PERF_WARMUP_UPDATES" "${stage_args[@]}")
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
echo "$(date -u +%FT%TZ) HEAD=$HEAD GPUS=${GPU[*]} PROFILE=${PROFILE:-0} RESUME=$RESUME PRETESTS=${PRETESTS:-1}" > "$ATTEMPT/run_info.txt"
echo "$recipe" >> "$ATTEMPT/run_info.txt"
CHILD=""
required=(preflight train)
[ "$PRETESTS" = 0 ] || required+=(pretests)
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
if command -v nvidia-smi >/dev/null; then nvidia-smi topo -m > "$ATTEMPT/gpu_topology.txt" 2>&1 || true; fi
if command -v nvidia-smi >/dev/null; then nvidia-smi > "$ATTEMPT/gpu_state.txt" 2>&1 || true; fi
if [ "$PRETESTS" = 1 ]; then
    step pretests env CUDA_VISIBLE_DEVICES="-1" "$PY" -m pytest -q -p no:cacheprovider \
        tools/analysis/tests/test_lumir.py tools/analysis/tests/test_precision.py tools/analysis/tests/test_distributed.py
else
    echo "[lumir] SKIPPED pretests: PRETESTS=0" | tee "$ATTEMPT/pretests.log"
fi
step preflight "$PY" -m tools.analysis.check_lumir --paths 3 --config "$CONFIG" --precision "$PRECISION" \
    --pairs "$PAIRS" --world-size "${#GPU[@]}" --out "$ATTEMPT/preflight.json"
launch=("$PY" -m experiments.train_CTCF)
if [ "${#GPU[@]}" -gt 1 ]; then
    launch=("$PY" -m torch.distributed.run --standalone --nproc-per-node="${#GPU[@]}" --max-restarts=0 --module experiments.train_CTCF)
fi
started=$SECONDS
step train "${launch[@]}" "${train_args[@]}" "${resume_args[@]}"
echo "wall_seconds=$((SECONDS - started)) includes=model_setup,training,validation,checkpoint_io" > "$ATTEMPT/train_timing.txt"
if [ "${PROFILE:-0}" = 1 ]; then
    grep -E "\[perf|\[throughput" "$ATTEMPT/train.log" | tail -n 2
    cat "$ATTEMPT/train_timing.txt"
else
    step export env CUDA_VISIBLE_DEVICES="${GPU[0]}" "$PY" -m experiments.export_lumir "${common_args[@]}" --ckpt "results/$EXP/ckpt/last.pth" \
        --pairs "$PAIRS" --out "$ATTEMPT/export_$PAIRS" --name "$EXP"
fi
