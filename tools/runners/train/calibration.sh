#!/usr/bin/env bash
# Expanded A calibration, a fresh OASIS Mamba and a readiness-gated LUMIR control.
# Allocate disjoint A_GPUS, MAMBA_GPU, LUMIR_GPUS; never include another user's devices.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${EXPECTED_GIT_HEAD:?Set EXPECTED_GIT_HEAD to the reviewed commit}"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$EXPECTED_GIT_HEAD" ] && git diff --quiet HEAD -- || { echo "[calibration] unexpected or dirty HEAD"; exit 1; }
PY="${PYBIN:-/data/mooncake/P/envs/ctcf/bin/python}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-results/P10_LONGRUN_MAMBA_NOSVF_OASIS/ckpt/best.pth}"
[ -f "$PARENT_CHECKPOINT" ] || { echo "[calibration] missing P10 checkpoint: $PARENT_CHECKPOINT"; exit 1; }
command -v flock > /dev/null || { echo "[calibration] flock is required on the training node"; exit 1; }
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}" NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
read -r -a A_GPU <<< "${A_GPUS:-1 2}"
read -r -a L_GPU <<< "${LUMIR_GPUS:-4 5 6 7}"
MAMBA_GPU="${MAMBA_GPU:-3}"
[ "${#A_GPU[@]}" -gt 0 ] && [ "${#L_GPU[@]}" = 4 ] || { echo "[calibration] need A GPUs and four LUMIR GPUs"; exit 1; }
declare -A seen=()
for gpu in "${A_GPU[@]}" "$MAMBA_GPU" "${L_GPU[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && [ -z "${seen[$gpu]:-}" ] || { echo "[calibration] invalid/overlapping GPU: $gpu"; exit 1; }
    seen[$gpu]=1
done
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-CALIBRATION_${STAMP}_${HEAD:0:12}}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "[calibration] invalid RUN_ID"; exit 1; }
RUN="results/calibration/$RUN_ID"
ATTEMPT="${STAMP}_$$"
case "${1:-run}" in
    start|resume)
        mode=$1
        if [ "$mode" = start ] && [ -d "$RUN" ]; then echo "[calibration] RUN_ID exists; use resume <RUN_ID>"; exit 1; fi
        if [ "$mode" = resume ]; then
            [ -n "${2:-}" ] || { echo "[calibration] resume requires the original RUN_ID"; exit 1; }
            RUN_ID=$2; RUN="results/calibration/$RUN_ID"
            [[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] && [ -f "$RUN/head.txt" ] && [ "$(cat "$RUN/head.txt")" = "$HEAD" ] \
                || { echo "[calibration] unknown RUN_ID or changed HEAD"; exit 1; }
            flock -n "$RUN/.lock" true || { echo "[calibration] RUN_ID is already active: $RUN_ID"; exit 1; }
        fi
        "$PY" -c 'import nibabel, scipy, pytest' || { echo "[calibration] missing Python dependency; check the existing environment"; exit 1; }
        LOG="/tmp/${RUN_ID}_${ATTEMPT}.log"
        nohup env RUN_ID="$RUN_ID" bash "$0" >"$LOG" 2>&1 < /dev/null &
        echo "PID=$! RUN_ID=$RUN_ID LOG=$LOG"
        exit 0 ;;
    run) ;;
    *) echo "[calibration] usage: calibration.sh [start | resume RUN_ID]"; exit 1 ;;
esac
mkdir -p "$RUN"
exec 9>"$RUN/.lock"
flock -n 9 || { echo "[calibration] RUN_ID is already active: $RUN_ID"; exit 1; }
mkdir -p "$RUN/logs/$ATTEMPT" "$RUN/exit_codes" "$RUN/.pids/$ATTEMPT" results/exports
if [ -f "$RUN/head.txt" ]; then
    [ "$(cat "$RUN/head.txt")" = "$HEAD" ] || { echo "[calibration] RUN_ID belongs to another HEAD"; exit 1; }
else
    echo "$HEAD" > "$RUN/head.txt"
fi
MAMBA_EXP="${RUN_ID}_mamba294"
LUMIR_EXP="${RUN_ID}_lumir_control"
LANES=()
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/status.txt"; }
owned_worker() {
    [ -r "/proc/$1/environ" ] && tr '\0' '\n' < "/proc/$1/environ" | grep -Fx "CTCF_CALIBRATION_OWNER=$RUN_ID:$ATTEMPT" > /dev/null
}
package() {
    local suffix=$1 code=$2 file; shift 2
    local -a roots=(--include-root "logs/$MAMBA_EXP")
    for file in results/lumir/"${RUN_ID}"*; do [ ! -d "$file" ] || roots+=(--include-root "$file"); done
    "$PY" -m tools.analysis.run_artifacts package --run-root "$RUN" --prefix "results/exports/${RUN_ID}__${ATTEMPT}${suffix}" \
        --exit-code "$code" --git-head "$HEAD" --label "CALIBRATION${suffix}" --required-steps "$@" "${roots[@]}"
}
finish() {
    local code=$? file pid waited
    trap - EXIT INT TERM
    for pid in "${LANES[@]}"; do
        if [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$$" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    for file in "$RUN/.pids/$ATTEMPT/"*; do
        [ -f "$file" ] || continue
        read -r pid < "$file"
        if owned_worker "$pid"; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    for file in "$RUN/.pids/$ATTEMPT/"*; do
        [ -f "$file" ] || continue
        read -r pid < "$file"; waited=0
        while owned_worker "$pid" && [ "$waited" -lt 60 ]; do sleep 1; waited=$((waited+1)); done
        if owned_worker "$pid"; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
    done
    wait 2>/dev/null || true
    package "" "$code" pretests prepare readiness lumir_control mamba294 a_calibration || { [ "$code" != 0 ] || code=1; }
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
step() {
    local name=$1 devices=$2 proof=$3 child code=0; shift 3
    if [ "$(cat "$RUN/exit_codes/$name.txt" 2>/dev/null || true)" = 0 ] && [ -f "$proof" ]; then log "SKIP $name (completed)"; return 0; fi
    log "START $name GPU=$devices LOG=$RUN/logs/$ATTEMPT/$name.log"
    setsid env CUDA_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1 CTCF_CALIBRATION_OWNER="$RUN_ID:$ATTEMPT" "$@" > "$RUN/logs/$ATTEMPT/$name.log" 2>&1 &
    child=$!; echo "$child" > "$RUN/.pids/$ATTEMPT/$name"
    wait "$child" || code=$?
    rm -f "$RUN/.pids/$ATTEMPT/$name"
    echo "$code" > "$RUN/exit_codes/$name.txt"
    if [ "$code" = 0 ]; then log "PASS $name"; else log "ERROR $name (exit $code)"; fi
    return "$code"
}
log "RUN_ID=$RUN_ID HEAD=$HEAD A_GPUS=${A_GPU[*]} MAMBA_GPU=$MAMBA_GPU LUMIR_GPUS=${L_GPU[*]} PID=$$"
df -h "$RUN" | tee "$RUN/logs/$ATTEMPT/disk.log"
"$PY" -c 'import os,shutil; free=shutil.disk_usage("results").free/2**30; required=float(os.environ.get("MIN_FREE_GIB",200)); print(f"free={free:.1f} GiB required={required:g} GiB (configurable initial reserve)"); raise SystemExit(0 if free>=required>=0 else 1)' \
    | tee -a "$RUN/logs/$ATTEMPT/disk.log"
if [ "${PRETESTS:-1}" = 1 ]; then
    step pretests -1 "$RUN/pretests.done" "$PY" -m pytest -q -p no:cacheprovider tools/analysis/tests/population \
        tools/analysis/tests/test_calibration.py tools/analysis/tests/test_distributed.py tools/analysis/tests/test_lumir.py tools/analysis/tests/test_precision.py
    touch "$RUN/pretests.done"
else
    echo "SKIPPED: PRETESTS=0" > "$RUN/logs/$ATTEMPT/pretests.log"
    echo 0 > "$RUN/exit_codes/pretests.txt"
fi
step prepare -1 "" "$PY" -m experiments.population.calibrate --task prepare --run "$RUN/a" --data "$CTCF_DATA_DIR/OASIS_L2R_2021_task03/All"
for variant in one four_batch8 four_batch4 four_micro2 four_tf32; do
    devices="${L_GPU[*]}"; micro=1; global=8; precision=bf16
    case "$variant" in one) devices="${L_GPU[0]}";; four_batch4) global=4;; four_micro2) micro=2;; four_tf32) precision=tf32;; esac
    exp="${RUN_ID}_profile_${variant}_$ATTEMPT"
    if ! step "profile_$variant" "${devices// /,}" "$RUN/profile_$variant.done" env GPUS="$devices" PROFILE_EXP="$exp" \
        PROFILE=1 RESUME=0 PROFILE_UPDATES=20 PERF_WARMUP_UPDATES=5 GLOBAL_BATCH="$global" BATCH_SIZE="$micro" \
        PRECISION="$precision" CONFIG=CTCF-CascadeA-Mamba MODEL="--l3_svf 0" TRAIN="" TIME_STEPS=6 PAIRS=lumir25 \
        PYBIN="$PY" PRETESTS=0 bash tools/runners/train/lumir.sh; then
        log "OPTIONAL PROFILE FAILED: $variant; the production recipe is checked separately"
    else touch "$RUN/profile_$variant.done"; fi
done
lumir_lane() {
    trap - EXIT INT TERM
    step readiness "$(IFS=,; echo "${L_GPU[*]}")" "$RUN/readiness/report.json" "$PY" -m tools.analysis.check_lumir_training --run "$RUN/readiness" --gpus "${L_GPU[*]}" || return 1
    [ ! -f "$RUN/lumir_exp.txt" ] || LUMIR_EXP="$(cat "$RUN/lumir_exp.txt")"
    if [ ! -f "$RUN/lumir_control.done" ] && [ -d "results/lumir/$LUMIR_EXP" ] && [ ! -f "results/$LUMIR_EXP/ckpt/last.pth" ]; then
        log "LUMIR: preserving incomplete $LUMIR_EXP; restarting under a fresh EXP"
        LUMIR_EXP="${RUN_ID}_lumir_control_${ATTEMPT}"
    fi
    echo "$LUMIR_EXP" > "$RUN/lumir_exp.txt"
    resume=0; [ ! -f "results/$LUMIR_EXP/ckpt/last.pth" ] || resume=1
    step lumir_control "$(IFS=,; echo "${L_GPU[*]}")" "$RUN/lumir_control.done" env GPUS="${L_GPU[*]}" EXP="$LUMIR_EXP" \
        RESUME="$resume" PROFILE=0 GLOBAL_BATCH=4 BATCH_SIZE=1 PRECISION=bf16 CONFIG=CTCF-CascadeA-Mamba MODEL="--l3_svf 0" \
        TRAIN="--lr 0.0001" STAGES=ramp EPOCHS=40 ITERS=1000 TIME_STEPS=6 PAIRS=lumir25 PYBIN="$PY" PRETESTS=0 \
        bash tools/runners/train/lumir.sh || return 1
    touch "$RUN/lumir_control.done"
}
mamba_lane() {
    trap - EXIT INT TERM
    args=()
    [ ! -f "results/$MAMBA_EXP/ckpt/last.pth" ] || args=(--resume "results/$MAMBA_EXP/ckpt/last.pth")
    step mamba294 "$MAMBA_GPU" "$RUN/mamba294.done" "$PY" -m experiments.train_CTCF --ds OASIS --paths 3 --gpu 0 \
        --config CTCF-CascadeA-Mamba --l3_svf 0 --precision bf16 --exp "$MAMBA_EXP" --max_epoch 500 --batch_size 1 \
        --lr 0.0001 --time_steps 6 --oasis_split "$RUN/a/split.json" --quiet 1 --use_tb 0 --save_ckpt 1 "${args[@]}" || return 1
    touch "$RUN/mamba294.done"
}
a_lane() {
    trap - EXIT INT TERM
    local i pid code=0 next=0 running=0 name
    local -a pids=() jobs=(lin64_s0 a1_64_s0 lin64_s1 a1_64_s1 lin278 a1_278 a1_w16)
    for ((i=0; i<${#A_GPU[@]}; i++)); do
        step "labels_${#A_GPU[@]}_$i" "${A_GPU[$i]}" "$RUN/a/labels_${#A_GPU[@]}_$i.done" \
            "$PY" -m experiments.population.calibrate --task labels --run "$RUN/a" --shard "$i" --shards "${#A_GPU[@]}" --device cuda:0 & pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || code=1; done
    local ridge_pid=""
    if [ "$code" = 0 ]; then
        jobs+=(A0 A_lin A1)
        step ridge -1 "$RUN/a/ridge/n_277.json" "$PY" -m experiments.population.calibrate --task ridge --run "$RUN/a" & ridge_pid=$!
    else
        log "SKIP ridge/local K2: label fits failed; all seven image-only A fits will still run"
        for name in ridge local_A0 local_A_lin local_A1; do echo 125 > "$RUN/exit_codes/$name.txt"; done
    fi
    pids=()
    while [ "$next" -lt "${#jobs[@]}" ] || [ "$running" -gt 0 ]; do
        for ((i=0; i<${#A_GPU[@]}; i++)); do
            pid="${pids[$i]:-}"
            if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then wait "$pid" || code=1; unset 'pids[i]'; running=$((running-1)); pid=""; fi
            if [ -z "$pid" ] && [ "$next" -lt "${#jobs[@]}" ]; then
                name="${jobs[$next]}"
                if [[ "$name" == A* ]]; then
                    step "local_$name" "${A_GPU[$i]}" "$RUN/a/local/$name/report.json" "$PY" -m experiments.population.calibrate --task local --arm "$name" --run "$RUN/a" --device cuda:0 &
                else
                    step "fit_$name" "${A_GPU[$i]}" "$RUN/a/arms/$name/report.json" "$PY" -m experiments.population.calibrate --task fit --arm "$name" --run "$RUN/a" --device cuda:0 &
                fi
                pids[$i]=$!; next=$((next+1)); running=$((running+1))
            fi
        done
        [ "$running" = 0 ] || sleep 1
    done
    if [ -n "$ridge_pid" ]; then wait "$ridge_pid" || code=1; fi
    step parent "${A_GPU[0]}" "$RUN/a/parent/report.json" "$PY" -m experiments.population.calibrate --task parent --run "$RUN/a" \
        --parent-checkpoint "$PARENT_CHECKPOINT" --device cuda:0 || code=1
    step a_summary -1 "" "$PY" -m experiments.population.calibrate --task summary --run "$RUN/a" || code=1
    return "$code"
}
lumir_lane & lumir_pid=$!
mamba_lane & mamba_pid=$!
a_lane & a_pid=$!
LANES=("$lumir_pid" "$mamba_pid" "$a_pid")
failed=0
wait "$lumir_pid" || failed=1
LANES=("$mamba_pid" "$a_pid")
code=0; wait "$a_pid" || code=$?
echo "$code" > "$RUN/exit_codes/a_calibration.txt"
[ "$code" = 0 ] || failed=1
LANES=("$mamba_pid")
package "__A_LUMIR" "$failed" pretests prepare readiness lumir_control a_calibration || failed=1
wait "$mamba_pid" || failed=1
LANES=()
log "FINISHED exit=$failed"
exit "$failed"
