#!/usr/bin/env bash
# Paired IXI contrasts, frozen consensus and bounded follow-up diagnostics.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-start}"
RUN_ID="${2:-B_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID"; exit 1; }
RUN="results/contrasts/$RUN_ID"
WORKFLOW="${WORKFLOW:-$(cat "$RUN/workflow.txt" 2>/dev/null || echo main)}"
[[ "$WORKFLOW" = main || "$WORKFLOW" = followup ]] || { echo "Unknown WORKFLOW"; exit 1; }
if [ "$MODE" = stop ]; then
    [ -f "$RUN/runner.pid" ] || { echo "No matching live runner; no process was stopped."; exit 0; }
    read -r pid < "$RUN/runner.pid"
    if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "contrasts.sh _work $RUN_ID" &&
       [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ]; then
        kill -TERM -- "-$pid"
        echo "Stop requested; wait for the final package before resuming."
    else
        echo "No matching live runner; no process was stopped."
    fi
    exit 0
fi
: "${EXPECTED_GIT_HEAD:?Set the reviewed commit SHA}"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$EXPECTED_GIT_HEAD" ] && git diff --quiet HEAD -- || { echo "Unexpected/modified HEAD"; exit 1; }
if [ "$MODE" != _work ]; then
    case "$MODE" in
        start) [ ! -e "$RUN" ] || { echo "RUN_ID exists; use resume"; exit 1; } ;;
        resume) [ -f "$RUN/head.txt" ] || { echo "Unknown RUN_ID"; exit 1; } ;;
        *) echo "Usage: contrasts.sh [start|resume|stop] [RUN_ID]"; exit 1 ;;
    esac
    mkdir -p "$RUN"
    if [ -f "$RUN/workflow.txt" ]; then
        [ "$(cat "$RUN/workflow.txt")" = "$WORKFLOW" ] || { echo "Changed workflow"; exit 1; }
    else echo "$WORKFLOW" > "$RUN/workflow.txt"; fi
    LOG="$RUN/runner_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    nohup setsid bash tools/runners/train/contrasts.sh _work "$RUN_ID" > "$LOG" 2>&1 < /dev/null &
    echo "PID=$! RUN_ID=$RUN_ID LOG=$LOG"
    exit 0
fi
exec 9> "$RUN/.lock"
flock -n 9 || { echo "RUN_ID already active"; exit 1; }
echo "$$" > "$RUN/runner.pid"
if [ -f "$RUN/head.txt" ]; then
    [ "$(cat "$RUN/head.txt")" = "$HEAD" ] || { echo "RUN_ID belongs to another HEAD"; exit 1; }
else
    echo "$HEAD" > "$RUN/head.txt"
fi
PY="${PYBIN:-/data/mooncake/P/envs/ctcf/bin/python}"
PARENT="${PARENT_CHECKPOINT:-results/CALIBRATION_20261029T035516Z_78301dd6d80c_mamba294/ckpt/last.pth}"
DATA="${OASIS_ALL_ROOT:-/data/mooncake/P/OASIS_L2R_2021_task03/All}"
MANIFEST="${IXI_MANIFEST:-}"
read -r -a GPU <<< "${GPUS:-0 1 2 3}"
declare -A seen=()
for gpu in "${GPU[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && [ -z "${seen[$gpu]:-}" ] || { echo "Invalid/duplicate GPU: $gpu"; exit 1; }
    seen[$gpu]=1
done
[ "${#GPU[@]}" -gt 0 ] && [ -f "$PARENT" ] && [ -d "$DATA" ] || { echo "Missing GPU, parent or OASIS data"; exit 1; }
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOGS="$RUN/logs/$ATTEMPT"
mkdir -p "$LOGS" "$RUN/exit_codes"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/status.txt"; }
bind() {
    local digest
    digest="$(sha256sum "$2" | cut -d ' ' -f 1)" || return 1
    if [ -f "$RUN/$1.sha256" ]; then
        [ "$(cat "$RUN/$1.sha256")" = "$digest" ] || { echo "Changed $1"; return 1; }
    else echo "$digest" > "$RUN/$1.sha256"; fi
}
bind parent "$PARENT"
PIDS=()
REQUIRED=(pretests check parent b0 b1 b2 summary consensus)
if [ "$WORKFLOW" = followup ]; then REQUIRED=(pretests prepare profile controls anchor repair smooth summary repair_summary); fi
finish() {
    local code=$? includes=() arm
    trap - EXIT INT
    trap '' TERM
    if [ "$(ps -o pgid= -p $$ | tr -d ' ')" = "$$" ]; then kill -TERM -- "-$$" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    for arm in "logs/${RUN_ID}"_* "results/${RUN_ID}"_*; do [ ! -d "$arm" ] || includes+=(--include-root "$arm"); done
    "$PY" -m tools.analysis.run_artifacts package --run-root "$RUN" --prefix "results/exports/${RUN_ID}__$ATTEMPT" \
        --git-head "$HEAD" --exit-code "$code" --label "CONTRASTS" \
        --required-steps "${REQUIRED[@]}" "${includes[@]}" || { [ "$code" != 0 ] || code=1; }
    rm -f "$RUN/runner.pid"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "RUN_ID=$RUN_ID HEAD=$HEAD GPUS=${GPU[*]} PID=$$ MANIFEST=$MANIFEST"
FREE_DEFAULT=200
if [ "$WORKFLOW" = followup ]; then FREE_DEFAULT=80; fi
"$PY" -c 'import shutil,sys; sys.exit(shutil.disk_usage("results").free < int(sys.argv[1])*1024**3)' "${MIN_FREE_GIB:-$FREE_DEFAULT}"
job() {
    local task=$1 gpu=$2 code=0; shift 2
    if [ "$task" != prepare ] && [ "$(cat "$RUN/exit_codes/$task.txt" 2>/dev/null || true)" = 0 ]; then log "SKIP $task (completed)"; return 0; fi
    log "START $task GPU=$gpu LOG=$LOGS/$task.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@" > "$LOGS/$task.log" 2>&1 || code=$?
    echo "$code" > "$RUN/exit_codes/$task.txt"
    if [ "$code" = 0 ]; then log "PASS $task"; else log "ERROR $task (exit $code)"; fi
    return "$code"
}
if [ "${PRETESTS:-1}" = 1 ]; then
    TESTS=(tools/analysis/tests/test_contrasts.py tools/analysis/tests/test_contrasts_runner.py tools/analysis/tests/test_precision.py)
    if [ "$WORKFLOW" = followup ]; then TESTS+=(tools/analysis/tests/test_contrast_checks.py tools/analysis/tests/test_repair_checks.py); fi
    job pretests "" "$PY" -m pytest -q "${TESTS[@]}" \
        tools/analysis/tests/test_run_artifacts.py::RunArtifactsTest::test_failed_run_can_be_packaged_without_site_packages
else echo 0 > "$RUN/exit_codes/pretests.txt"; log "SKIPPED pretests: PRETESTS=0"; fi
followup() {
    local source="${SOURCE_RUN:-results/contrasts/B_20261108T180120Z}" status=0 repair_pid slots slot name gpu repeat workers benchmark controls first pid
    local -a common=(--source "$source" --manifest "$MANIFEST" --checkpoint "$PARENT")
    local -a repair=(--source "$source" --manifest "$MANIFEST" --out "$RUN" --oasis-root "$DATA")
    [ -f "$MANIFEST" ] && [ -f "$source/head.txt" ] || { echo "Missing source run or manifest"; return 1; }
    bind manifest "$MANIFEST" || return 1
    bind source "$source/parent/recipe.json" || return 1
    job prepare "" "$PY" -m experiments.contrast_diagnostics prepare "${common[@]}" --out "$RUN" || return 1
    for workers in 0 8; do
        for benchmark in 0 1; do
            name="profile_${workers}_${benchmark}"
            job "$name" "${GPU[0]}" "$PY" -m experiments.contrast_diagnostics profile "${common[@]}" \
                --out "$RUN/$name" --exp "${RUN_ID}_${name}_$ATTEMPT" --workers "$workers" --benchmark "$benchmark" || status=1
        done
    done
    echo "$status" > "$RUN/exit_codes/profile.txt"
    short_control() {
        local name=$1 gpu=$2 arm=$3 reg=$4 seed=${5:-0}
        job "$name" "$gpu" "$PY" -m experiments.contrast_diagnostics train "${common[@]}" \
            --out "$RUN/$name" --exp "${RUN_ID}_$name" --arm "$arm" --w_reg "$reg" --seed "$seed" || return 1
        job "cert_$name" "$gpu" "$PY" -m experiments.repair_diagnostics short "${repair[@]}" --arm-dir "$RUN/$name"
    }
    if [ "${#GPU[@]}" -gt 1 ]; then
        job repair "${GPU[-1]}" "$PY" -m experiments.repair_diagnostics saved "${repair[@]}" & repair_pid=$!
        slots=$((${#GPU[@]}-1))
    else
        job repair "${GPU[0]}" "$PY" -m experiments.repair_diagnostics saved "${repair[@]}" || status=1
        slots=1; repair_pid=""
    fi
    PIDS=()
    for ((slot=0; slot<slots; slot++)); do
        (
            code=0; names=(b0_w4_s0 b0_w1_s0 b1_w1_s0 b2_w1_s0)
            for ((index=slot; index<4; index+=slots)); do
                name=${names[$index]}; arm=${name%%_*}; reg=${name:4:1}
                short_control "$name" "${GPU[$slot]}" "$arm" "$reg" || code=1
            done
            exit "$code"
        ) & PIDS+=($!)
    done
    controls=0
    for pid in "${PIDS[@]}"; do wait "$pid" || controls=1; done
    echo "$controls" > "$RUN/exit_codes/controls.txt"
    [ "$controls" = 0 ] || status=1
    job anchor "" "$PY" -m experiments.contrast_diagnostics anchor "${common[@]}" --out "$RUN" --exp "${RUN_ID}_b0_w4_s0" || status=1
    if [ -f "$RUN/anchor.json" ] && "$PY" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["add_b1_w4"])' "$RUN/anchor.json"; then
        short_control b1_w4_s0 "${GPU[0]}" b1 4 || status=1
    fi
    job decision "" "$PY" -m experiments.contrast_diagnostics decide "${common[@]}" --out "$RUN" || status=1
    if [ -f "$RUN/decision.json" ]; then
        repeat=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["repeat_arm"] or "")' "$RUN/decision.json")
        if [ -n "$repeat" ]; then
            short_control b0_w1_s1 "${GPU[0]}" b0 1 1 & first=$!
            if [ "$slots" -gt 1 ]; then gpu=${GPU[1]}; else wait "$first" || status=1; first=""; gpu=${GPU[0]}; fi
            short_control "${repeat}_w1_s1" "$gpu" "$repeat" 1 1 || status=1
            if [ -n "$first" ]; then wait "$first" || status=1; fi
        fi
    fi
    job smooth "${GPU[0]}" "$PY" -m experiments.contrast_diagnostics smooth "${common[@]}" --out "$RUN/smoothing" || status=1
    job summary "" "$PY" -m experiments.contrast_diagnostics summary "${common[@]}" --out "$RUN" || status=1
    if [ -n "$repair_pid" ]; then wait "$repair_pid" || status=1; fi
    job repair_summary "" "$PY" -m experiments.repair_diagnostics finish "${repair[@]}" || status=1
    return "$status"
}
if [ "$WORKFLOW" = followup ]; then
    failed=0
    followup || failed=1
    log "FINISHED exit=$failed"
    exit "$failed"
fi
consensus() {
    job consensus "$1" "$PY" -m experiments.dispatch1.eval_consensus --checkpoint "$PARENT" --data "$DATA" --out "$RUN/consensus" --budget_hours "${CONSENSUS_HOURS:-8}"
}
failed=0
CONSENSUS_PID=""
if [ "${#GPU[@]}" -gt 1 ]; then
    consensus "${GPU[-1]}" & CONSENSUS_PID=$!
    slots=$((${#GPU[@]}-1))
else
    consensus "${GPU[0]}" || failed=1
    slots=1
fi
arm() {
    local name=$1 exp="${RUN_ID}_$1" weights="results/${RUN_ID}_$1/ckpt/last.pth" resume=()
    [ ! -f "$weights" ] || resume=(--resume "$weights")
    "$PY" -m experiments.contrasts train --exp "$exp" --manifest "$MANIFEST" --arm "$name" \
        --checkpoint "$PARENT" --out "$RUN/$name" "${resume[@]}" &&
    "$PY" -m experiments.contrasts evaluate --manifest "$MANIFEST" --checkpoint "$weights" --arm "$name" --out "$RUN/$name"
}
if [ "${RUN_B:-1}" = 1 ] && [ -f "$MANIFEST" ]; then
    bind manifest "$MANIFEST"
    if job check "${GPU[0]}" "$PY" -m experiments.contrasts check --manifest "$MANIFEST" --checkpoint "$PARENT" --out "$RUN/check" \
        --budget_hours "${BUDGET_HOURS:-72}" --consensus_reserve_hours "${CONSENSUS_HOURS:-8}"; then
        for ((slot=0; slot<slots; slot++)); do
            (
                status=0
                names=(b0 b1 b2)
                for ((index=slot; index<3; index+=slots)); do job "${names[$index]}" "${GPU[$slot]}" arm "${names[$index]}" || status=1; done
                exit "$status"
            ) & PIDS+=($!)
        done
    else failed=1; fi
else
    log "HELD B: supply IXI_MANIFEST after image review; consensus is independent"
    failed=1
fi
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
if [ "$(cat "$RUN/exit_codes/check.txt" 2>/dev/null || true)" = 0 ]; then
    job parent "${GPU[0]}" "$PY" -m experiments.contrasts evaluate --manifest "$MANIFEST" --checkpoint "$PARENT" --out "$RUN/parent" || failed=1
fi
job summary "" "$PY" -m experiments.contrasts summary --out "$RUN" || failed=1
if [ -n "$CONSENSUS_PID" ]; then wait "$CONSENSUS_PID" || failed=1; fi
log "FINISHED exit=$failed"
exit "$failed"
