#!/usr/bin/env bash
# Expanded A calibration, a fresh OASIS Mamba and a readiness-gated LUMIR control.
# Allocate disjoint A_GPUS, MAMBA_GPU, LUMIR_GPUS; never include another user's devices.
# GPUs that finish their long job join the A queue; A jobs are restartable, so a later commit may continue
# a run with ALLOW_NEW_HEAD=1, and heads.txt records which HEAD each attempt ran.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${EXPECTED_GIT_HEAD:?Set EXPECTED_GIT_HEAD to the reviewed commit}"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$EXPECTED_GIT_HEAD" ] && git diff --quiet HEAD -- || { echo "[calibration] unexpected or dirty HEAD"; exit 1; }
PY="${PYBIN:-/data/mooncake/P/envs/ctcf/bin/python}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-results/P10_LONGRUN_MAMBA_NOSVF_OASIS/ckpt/best.pth}"
[ -f "$PARENT_CHECKPOINT" ] || { echo "[calibration] missing P10 checkpoint: $PARENT_CHECKPOINT"; exit 1; }
command -v flock > /dev/null || { echo "[calibration] flock is required on the training node"; exit 1; }
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}" NCCL_NVLS_ENABLE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TMPDIR="$PWD/results/tmp"
mkdir -p "$TMPDIR"
read -r -a A_GPU <<< "${A_GPUS:-2}"
read -r -a L_GPU <<< "${LUMIR_GPUS:-4 5 6 7}"
MAMBA_GPU="${MAMBA_GPU:-3}"
A_SLOTS="${A_SLOTS:-3}"
KILL_RETRIES="${KILL_RETRIES:-2}"
KILL_RETRY_SECONDS="${KILL_RETRY_SECONDS:-120}"
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
            [[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] && [ -f "$RUN/head.txt" ] || { echo "[calibration] unknown RUN_ID"; exit 1; }
            [ "$(cat "$RUN/head.txt")" = "$HEAD" ] || [ "${ALLOW_NEW_HEAD:-0}" = 1 ] \
                || { echo "[calibration] changed HEAD; set ALLOW_NEW_HEAD=1 to continue under this commit"; exit 1; }
            flock -n "$RUN/.lock" true || { echo "[calibration] RUN_ID is already active: $RUN_ID"; exit 1; }
        fi
        "$PY" -c 'import nibabel, scipy, pytest' || { echo "[calibration] missing Python dependency; check the existing environment"; exit 1; }
        mkdir -p results/calibration
        LOG="results/calibration/${RUN_ID}_${ATTEMPT}.log"
        nohup env RUN_ID="$RUN_ID" bash "$0" >"$LOG" 2>&1 < /dev/null &
        echo "PID=$! RUN_ID=$RUN_ID LOG=$LOG"
        exit 0 ;;
    run) ;;
    *) echo "[calibration] usage: calibration.sh [start | resume RUN_ID]"; exit 1 ;;
esac
mkdir -p "$RUN"
exec 9>"$RUN/.lock"
flock -n 9 || { echo "[calibration] RUN_ID is already active: $RUN_ID"; exit 1; }
mkdir -p "$RUN/logs/$ATTEMPT" "$RUN/exit_codes" "$RUN/.pids/$ATTEMPT" "$RUN/.released/$ATTEMPT" results/exports
if [ -f "$RUN/head.txt" ]; then
    [ "$(cat "$RUN/head.txt")" = "$HEAD" ] || [ "${ALLOW_NEW_HEAD:-0}" = 1 ] \
        || { echo "[calibration] RUN_ID belongs to another HEAD; set ALLOW_NEW_HEAD=1 to continue"; exit 1; }
else
    echo "$HEAD" > "$RUN/head.txt"
fi
echo "$ATTEMPT $HEAD" >> "$RUN/heads.txt"
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
    setsid env CUDA_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1 CTCF_CALIBRATION_OWNER="$RUN_ID:$ATTEMPT" "$@" >> "$RUN/logs/$ATTEMPT/$name.log" 2>&1 &
    child=$!; echo "$child" > "$RUN/.pids/$ATTEMPT/$name"
    wait "$child" || code=$?
    rm -f "$RUN/.pids/$ATTEMPT/$name"
    echo "$code" > "$RUN/exit_codes/$name.txt"
    if [ "$code" = 0 ]; then log "PASS $name"; else log "ERROR $name (exit $code)"; fi
    return "$code"
}
# A bounded restart after an external SIGKILL, from the job's own checkpoints. Stopping this wrapper kills the lanes,
# so an intentional stop never restarts anything.
with_retries() {
    local tries=0 code
    while :; do
        code=0; "$@" || code=$?
        [ "$code" = 137 ] && [ "$tries" -lt "$KILL_RETRIES" ] || return "$code"
        tries=$((tries+1))
        log "RETRY after SIGKILL ($tries/$KILL_RETRIES): $* "
        sleep "$KILL_RETRY_SECONDS"
    done
}
release_gpus() {
    local gpu
    for gpu in "$@"; do touch "$RUN/.released/$ATTEMPT/$gpu"; done
    log "GPUs joining the A queue: $*"
}
log "RUN_ID=$RUN_ID HEAD=$HEAD A_GPUS=${A_GPU[*]} MAMBA_GPU=$MAMBA_GPU LUMIR_GPUS=${L_GPU[*]} PID=$$"
df -h "$RUN" / | tee "$RUN/logs/$ATTEMPT/disk.log"
"$PY" -c 'import os,shutil; free=shutil.disk_usage("results").free/2**30; required=float(os.environ.get("MIN_FREE_GIB",200)); print(f"free={free:.1f} GiB required={required:g} GiB (configurable initial reserve)"); raise SystemExit(0 if free>=required>=0 else 1)' \
    | tee -a "$RUN/logs/$ATTEMPT/disk.log"
if [ "${PRETESTS:-1}" = 1 ]; then
    step pretests -1 "$RUN/pretests_${HEAD:0:12}.done" "$PY" -m pytest -q -p no:cacheprovider tools/analysis/tests/population \
        tools/analysis/tests/test_calibration.py tools/analysis/tests/test_distributed.py tools/analysis/tests/test_lumir.py tools/analysis/tests/test_precision.py
    touch "$RUN/pretests_${HEAD:0:12}.done"
else
    echo "SKIPPED: PRETESTS=0" >> "$RUN/logs/$ATTEMPT/pretests.log"
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
lumir_attempt() {
    [ ! -f "$RUN/lumir_exp.txt" ] || LUMIR_EXP="$(cat "$RUN/lumir_exp.txt")"
    if [ ! -f "$RUN/lumir_control.done" ] && [ -d "results/lumir/$LUMIR_EXP" ] && [ ! -f "results/$LUMIR_EXP/ckpt/last.pth" ]; then
        log "LUMIR: preserving incomplete $LUMIR_EXP; restarting under a fresh EXP"
        LUMIR_EXP="${RUN_ID}_lumir_control_${ATTEMPT}_$SECONDS"
    fi
    echo "$LUMIR_EXP" > "$RUN/lumir_exp.txt"
    local resume=0
    [ ! -f "results/$LUMIR_EXP/ckpt/last.pth" ] || resume=1
    step lumir_control "$(IFS=,; echo "${L_GPU[*]}")" "$RUN/lumir_control.done" env GPUS="${L_GPU[*]}" EXP="$LUMIR_EXP" \
        RESUME="$resume" PROFILE=0 GLOBAL_BATCH=4 BATCH_SIZE=1 PRECISION=bf16 CONFIG=CTCF-CascadeA-Mamba MODEL="--l3_svf 0" \
        TRAIN="--lr 0.0001" STAGES=ramp EPOCHS=40 ITERS=1000 TIME_STEPS=6 PAIRS=lumir25 PYBIN="$PY" PRETESTS=0 \
        bash tools/runners/train/lumir.sh
}
certified_export() {
    local name=$1 experiment=$2 out=$3; shift 3
    with_retries step "$name" "${L_GPU[0]}" "$out/manifest.json" "$PY" -m experiments.export_lumir \
        --ds LUMIR --paths 3 --gpu 0 --config CTCF-CascadeA-Mamba --l3_svf 0 --time_steps 6 --precision bf16 \
        --ckpt "results/$experiment/ckpt/last.pth" --pairs lumir25 --out "$out" --certified 1 "$@"
}
lumir_work() {
    local tries=0 code ready certified=1
    with_retries step readiness "$(IFS=,; echo "${L_GPU[*]}")" "$RUN/readiness/report.json" "$PY" -m tools.analysis.check_lumir_training --run "$RUN/readiness" --gpus "${L_GPU[*]}" || return 1
    # One field of the readiness checkpoint through the certified export, before the long control. A failure holds
    # only that export: the control and its raw ZIP do not depend on it, and its checkpoint stays for a later attempt.
    ready="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiments"][-1])' "$RUN/readiness/report.json")"
    certified_export lumir_certified_check "$ready" "$RUN/readiness/certified_check" --name certified_check --limit_pairs 1 || certified=0
    # DDP reports a killed rank through its launcher, so any failure of the long control may continue from last.pth.
    while :; do
        code=0; lumir_attempt || code=$?
        [ "$code" != 0 ] && [ "$tries" -lt "$KILL_RETRIES" ] || break
        tries=$((tries+1))
        log "RETRY lumir_control after exit $code ($tries/$KILL_RETRIES)"
        sleep "$KILL_RETRY_SECONDS"
    done
    [ "$code" = 0 ] || return "$code"
    touch "$RUN/lumir_control.done"
    LUMIR_EXP="$(cat "$RUN/lumir_exp.txt")"
    package "__LUMIR" 0 pretests prepare readiness lumir_control || true
    release_gpus "${L_GPU[@]:1}"
    if [ "$certified" = 0 ]; then
        log "HELD lumir_certified: its one-pair check failed"
        echo 125 > "$RUN/exit_codes/lumir_certified.txt"
        return 1
    fi
    certified_export lumir_certified "$LUMIR_EXP" "results/lumir/$LUMIR_EXP/certified_lumir25" --name "${LUMIR_EXP}_certified"
}
lumir_lane() {
    trap - EXIT INT TERM
    local code=0
    lumir_work || code=$?
    release_gpus "${L_GPU[@]}"
    return "$code"
}
mamba_attempt() {
    local -a args=()
    [ ! -f "results/$MAMBA_EXP/ckpt/last.pth" ] || args=(--resume "results/$MAMBA_EXP/ckpt/last.pth")
    step mamba294 "$MAMBA_GPU" "$RUN/mamba294.done" "$PY" -m experiments.train_CTCF --ds OASIS --paths 3 --gpu 0 \
        --config CTCF-CascadeA-Mamba --l3_svf 0 --precision bf16 --exp "$MAMBA_EXP" --max_epoch 250 --schedule_max_epoch 500 --batch_size 1 \
        --lr 0.0001 --time_steps 6 --oasis_split "$RUN/a/split.json" --quiet 1 --use_tb 0 --save_ckpt 1 "${args[@]}"
}
mamba_work() {
    local cohort code=0
    with_retries mamba_attempt || return 1
    touch "$RUN/mamba294.done"
    for cohort in dev heldout oasis19; do
        with_retries step "reference_mamba294_$cohort" "$MAMBA_GPU" "$RUN/a/references/mamba294/$cohort/report.json" "$PY" -m experiments.population.calibrate \
            --task reference --label mamba294 --cohort "$cohort" --parent-checkpoint "results/$MAMBA_EXP/ckpt/last.pth" --run "$RUN/a" --device cuda:0 || code=1
    done
    return "$code"
}
mamba_lane() {
    trap - EXIT INT TERM
    local code=0
    mamba_work || code=$?
    release_gpus "$MAMBA_GPU"
    return "$code"
}
# One queue on free slots of the A GPUs; each stage waits for the previous one, and a failure stays visible without
# cancelling anything else. Entries are "name|proof|command" whose command contains no spaces inside an argument.
enqueue() { queue+=("$1|$2|$3"); }
# The pair check reports only when the raw chain held. Without that, evaluation would fail alike on every pair, so it
# is held while training goes on. A failed repair or certificate leaves raw scores valid, so evaluation still runs.
evaluate() {
    if [ -f "$RUN/a/arms/lin64_s0/last/pair_check.json" ]; then enqueue "$@"; return; fi
    log "HELD $1: the raw chain failed its pair check"
    echo 125 > "$RUN/exit_codes/$1.txt"
    code=1
}
assess() {
    local arm=$1 point=$2 label=${2//\//_}; shift 2
    evaluate "assess_${arm}_$label" "$RUN/a/arms/$arm/$label/report.json" "$CALIBRATE --task assess --arm $arm --point $point $*"
}
grow_slots() {
    local file gpu s
    for file in "$RUN/.released/$ATTEMPT"/*; do
        [ -e "$file" ] || continue
        gpu=${file##*/}
        [ -z "${added[$gpu]:-}" ] || continue
        added[$gpu]=1
        for ((s=0; s<A_SLOTS; s++)); do slots+=("$gpu"); done
    done
}
drain() {
    local i pid next=0 running=0 name proof cmd
    local -a pids=()
    while [ "$next" -lt "${#queue[@]}" ] || [ "$running" -gt 0 ]; do
        grow_slots
        for ((i=0; i<${#slots[@]}; i++)); do
            pid="${pids[$i]:-}"
            if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then wait "$pid" || code=1; unset 'pids[i]'; running=$((running-1)); pid=""; fi
            if [ -z "$pid" ] && [ "$next" -lt "${#queue[@]}" ]; then
                IFS='|' read -r name proof cmd <<< "${queue[$next]}"
                # shellcheck disable=SC2086
                with_retries step "$name" "${slots[$i]}" "$proof" $cmd &
                pids[$i]=$!; next=$((next+1)); running=$((running+1))
            fi
        done
        [ "$running" = 0 ] || sleep "${QUEUE_POLL_SECONDS:-1}"
    done
    queue=()
}
a_lane() {
    trap - EXIT INT TERM
    local i s pid arm code=0 ridge_pid=""
    local CALIBRATE="$PY -m experiments.population.calibrate --run $RUN/a --device cuda:0"
    local -a queue=() slots=() label_pids=()
    local -A added=()
    for i in "${A_GPU[@]}"; do for ((s=0; s<A_SLOTS; s++)); do slots+=("$i"); done; done
    enqueue fit_lin64_s0 "$RUN/a/arms/lin64_s0/report.json" "$CALIBRATE --task fit --arm lin64_s0"
    drain
    # The whole chain on one real full-size pair, before any other evaluation.
    enqueue pair_check "$RUN/a/arms/lin64_s0/last/pair_check.json" "$CALIBRATE --task assess --arm lin64_s0 --point last --limit-pairs 1"
    drain
    for ((i=0; i<${#A_GPU[@]}; i++)); do
        with_retries step "labels_${#A_GPU[@]}_$i" "${A_GPU[$i]}" "$RUN/a/labels_${#A_GPU[@]}_$i.done" \
            "$PY" -m experiments.population.calibrate --task labels --run "$RUN/a" --shard "$i" --shards "${#A_GPU[@]}" --device cuda:0 & label_pids+=($!)
    done
    for pid in "${label_pids[@]}"; do wait "$pid" || code=1; done
    if [ "$code" = 0 ]; then
        with_retries step ridge -1 "$RUN/a/ridge/n_277.json" "$PY" -m experiments.population.calibrate --task ridge --run "$RUN/a" & ridge_pid=$!
        for arm in A0 A_lin A1; do
            enqueue "local_$arm" "$RUN/a/local/$arm/report.json" "$CALIBRATE --task local --arm $arm"
        done
    else
        log "SKIP ridge/local K2: label fits failed; all image-only A fits will still run"
        for arm in ridge local_A0 local_A_lin local_A1; do echo 125 > "$RUN/exit_codes/$arm.txt"; done
    fi
    enqueue parent "$RUN/a/parent/report.json" "$CALIBRATE --task parent --parent-checkpoint $PARENT_CHECKPOINT"
    for arm in a1_64_s0 lin64_s1 a1_64_s1 a1_w16; do
        enqueue "fit_$arm" "$RUN/a/arms/$arm/report.json" "$CALIBRATE --task fit --arm $arm"
    done
    drain
    assess lin64_s0 last --predictive 1
    assess a1_64_s0 last --predictive 1
    assess lin64_s1 last
    assess a1_64_s1 last
    assess a1_w16 last
    for arm in dev oasis19; do
        enqueue "reference_p10_$arm" "$RUN/a/references/p10/$arm/report.json" "$CALIBRATE --task reference --label p10 --cohort $arm --parent-checkpoint $PARENT_CHECKPOINT"
    done
    drain
    package "__A64" "$code" pretests prepare pair_check assess_lin64_s0_last assess_a1_64_s0_last assess_lin64_s1_last assess_a1_64_s1_last assess_a1_w16_last || true
    for arm in lin278 a1_278 a0_64_s0; do
        enqueue "fit_$arm" "$RUN/a/arms/$arm/report.json" "$CALIBRATE --task fit --arm $arm"
    done
    drain
    for arm in lin278 a1_278; do
        assess "$arm" update_6400
        assess "$arm" last
        assess "$arm" update_19200 --certify 0
    done
    assess a0_64_s0 last
    for arm in lin64_s0 a1_64_s0; do assess "$arm" update_3200 --certify 0; done
    for arm in lin64_s0 a1_64_s0 lin64_s1 a1_64_s1 a0_64_s0; do
        enqueue "extend_$arm" "$RUN/a/arms/$arm/ext/report.json" "$CALIBRATE --task extend --arm $arm"
    done
    for arm in lin64_s0 a1_64_s0 a1_w16 lin278 a1_278; do
        evaluate "train_probe_$arm" "$RUN/a/arms/$arm/last_train/report.json" "$CALIBRATE --task assess --arm $arm --point last --cohort train --repair 0 --certify 0"
    done
    drain
    assess lin64_s0 ext/last --predictive 1
    assess a1_64_s0 ext/last --predictive 1
    for arm in lin64_s1 a1_64_s1 a0_64_s0; do assess "$arm" ext/last; done
    for arm in lin64_s0 a1_64_s0; do assess "$arm" ext/epoch_200 --certify 0; done
    for arm in lin278 a1_278; do
        for s in 0 1; do
            evaluate "dev_fit_${arm}_$s" "$RUN/a/arms/$arm/last_dev/fit_${s}_2.done" "$CALIBRATE --task assess --arm $arm --point last --cohort dev --stage fit --shard $s --shards 2"
        done
        evaluate "oasis19_$arm" "$RUN/a/arms/$arm/last_oasis19/REP/summary.json" "$CALIBRATE --task oasis19 --arm $arm --point last"
    done
    drain
    for arm in lin278 a1_278; do
        for s in 0 1 2 3 4 5; do
            evaluate "dev_score_${arm}_$s" "$RUN/a/arms/$arm/last_dev/score_${s}_6.done" "$CALIBRATE --task assess --arm $arm --point last --cohort dev --stage score --shard $s --shards 6"
        done
    done
    drain
    for arm in lin278 a1_278; do
        evaluate "dev_summary_$arm" "$RUN/a/arms/$arm/last_dev/report.json" "$CALIBRATE --task assess --arm $arm --point last --cohort dev --stage summary"
    done
    drain
    if [ -n "$ridge_pid" ]; then wait "$ridge_pid" || code=1; fi
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
