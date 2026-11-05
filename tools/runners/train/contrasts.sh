#!/usr/bin/env bash
# Paired IXI contrast arms and an independent frozen OASIS consensus evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-start}"
RUN_ID="${2:-B_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID"; exit 1; }
RUN="results/contrasts/$RUN_ID"
if [ "$MODE" = stop ]; then
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
    digest="$(sha256sum "$2" | cut -d ' ' -f 1)"
    if [ -f "$RUN/$1.sha256" ]; then
        [ "$(cat "$RUN/$1.sha256")" = "$digest" ] || { echo "Changed $1"; return 1; }
    else echo "$digest" > "$RUN/$1.sha256"; fi
}
bind parent "$PARENT"
PIDS=()
finish() {
    local code=$? includes=() arm
    trap - EXIT INT
    trap '' TERM
    if [ "$(ps -o pgid= -p $$ | tr -d ' ')" = "$$" ]; then kill -TERM -- "-$$" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    for arm in b0 b1 b2; do includes+=(--include-root "logs/${RUN_ID}_$arm" --include-root "results/${RUN_ID}_$arm"); done
    "$PY" -m tools.analysis.run_artifacts package --run-root "$RUN" --prefix "results/exports/${RUN_ID}__$ATTEMPT" \
        --git-head "$HEAD" --exit-code "$code" --label "CONTRASTS" \
        --required-steps pretests check parent b0 b1 b2 summary consensus "${includes[@]}" || { [ "$code" != 0 ] || code=1; }
    rm -f "$RUN/runner.pid"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "RUN_ID=$RUN_ID HEAD=$HEAD GPUS=${GPU[*]} PID=$$ MANIFEST=$MANIFEST"
"$PY" -c 'import shutil,sys; sys.exit(shutil.disk_usage("results").free < int(sys.argv[1])*1024**3)' "${MIN_FREE_GIB:-200}"
job() {
    local task=$1 gpu=$2 code=0; shift 2
    if [ "$(cat "$RUN/exit_codes/$task.txt" 2>/dev/null || true)" = 0 ]; then log "SKIP $task (completed)"; return 0; fi
    log "START $task GPU=$gpu LOG=$LOGS/$task.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@" > "$LOGS/$task.log" 2>&1 || code=$?
    echo "$code" > "$RUN/exit_codes/$task.txt"
    if [ "$code" = 0 ]; then log "PASS $task"; else log "ERROR $task (exit $code)"; fi
    return "$code"
}
if [ "${PRETESTS:-1}" = 1 ]; then
    job pretests "" "$PY" -m pytest -q tools/analysis/tests/test_contrasts.py tools/analysis/tests/test_contrasts_runner.py tools/analysis/tests/test_precision.py \
        tools/analysis/tests/test_run_artifacts.py::RunArtifactsTest::test_failed_run_can_be_packaged_without_site_packages
else echo 0 > "$RUN/exit_codes/pretests.txt"; log "SKIPPED pretests: PRETESTS=0"; fi
consensus() {
    job consensus "$1" "$PY" -m experiments.dispatch1.eval_consensus --checkpoint "$PARENT" --data "$DATA" --out "$RUN/consensus"
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
    if job check "${GPU[0]}" "$PY" -m experiments.contrasts check --manifest "$MANIFEST" --checkpoint "$PARENT" --out "$RUN/check"; then
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
