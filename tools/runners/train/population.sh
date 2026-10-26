#!/usr/bin/env bash
# Preliminary population-model diagnostics, not the main A0/A_lin/A1 training experiment.
# GPUS accepts any free device IDs (spaces or commas). RUN_ID resumes completed tasks at the same HEAD.
# Do not run the same RUN_ID concurrently. Payload arrays/checkpoints stay on the server, outside the compact ZIP.
# Diagnostic PROFILE_WIDTH/LEVELS/PROFILE_EPOCHS/PROBE_STEPS/MAP_STEPS are frozen by prepare in config.json.
# PY defaults to the node's ctcf environment; the population unit tests run first (PRETESTS=0 skips them).
# Real OASIS/IXI Psi boundary checks run concurrently on CPU; CERTIFICATE_ROOT defaults to results.
#
#   EXPECTED_GIT_HEAD=<sha> GPUS="0 1 2 3" nohup bash tools/runners/train/population.sh > population.out 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${EXPECTED_GIT_HEAD:?Set EXPECTED_GIT_HEAD to the reviewed commit}"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$EXPECTED_GIT_HEAD" ] || { echo "[population] unexpected HEAD"; exit 1; }
git diff --quiet HEAD -- || { echo "[population] tracked files differ from HEAD"; exit 1; }
PY="${PYBIN:-${PY:-/data/mooncake/P/envs/ctcf/bin/python}}"
DATA="${OASIS_ALL_ROOT:-${CTCF_DATA_DIR:-/data/mooncake/P}/OASIS_L2R_2021_task03/All}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-results/P10_LONGRUN_MAMBA_NOSVF_OASIS/ckpt/best.pth}"
GPU_SPEC="${GPUS:-${GPU_LIST:-0 1 2 3}}"
read -r -a GPU <<< "${GPU_SPEC//,/ }"
[ "${#GPU[@]}" -gt 0 ] || { echo "[population] GPUS is empty"; exit 1; }
declare -A seen=()
for gpu in "${GPU[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && [ -z "${seen[$gpu]:-}" ] || { echo "[population] invalid/duplicate GPU: $gpu"; exit 1; }
    seen[$gpu]=1
done
[ -d "$DATA" ] || { echo "[population] missing data: $DATA"; exit 1; }
[ -f "$PARENT_CHECKPOINT" ] || { echo "[population] missing parent checkpoint: $PARENT_CHECKPOINT"; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-POPULATION_PROFILE_${STAMP}_${HEAD:0:12}_$$}"
RUN="results/population/$RUN_ID"
ATTEMPT="${STAMP}_$$"
mkdir -p "$RUN/logs/$ATTEMPT" "$RUN/exit_codes" "$RUN/.pids/$ATTEMPT" results/exports
if [ -f "$RUN/head.txt" ]; then
    [ "$(cat "$RUN/head.txt")" = "$HEAD" ] || { echo "[population] RUN_ID belongs to another HEAD"; exit 1; }
else
    echo "$HEAD" > "$RUN/head.txt"
fi
echo "$(date -u +%FT%TZ) HEAD=$HEAD GPUS=${GPU[*]} DATA=$DATA PY=$PY" > "$RUN/logs/$ATTEMPT/run_info.txt"
if command -v nvidia-smi >/dev/null; then nvidia-smi topo -m > "$RUN/logs/$ATTEMPT/gpu_topology.txt" 2>&1 || true; fi
PIDS=()
CERT_PID=""
TASKS=(a0 linear a1 probe_ref probe_train0 probe_train1 probe_train2 probe_val0 probe_val1 k3)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/status.txt"; }
finish() {
    local code=$? pid_file pid
    trap - EXIT INT TERM
    [ -z "$CERT_PID" ] || kill -TERM "$CERT_PID" 2>/dev/null || true
    for pid_file in "$RUN/.pids/$ATTEMPT/"*; do
        [ -f "$pid_file" ] || continue
        read -r pid < "$pid_file"
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    "$PY" - "$RUN" "results/exports/${RUN_ID}__${ATTEMPT}" "$code" "${TASKS[@]}" << 'EOF'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

run, prefix, exit_code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
codes = {path.stem: int(path.read_text().strip()) for path in (run / "exit_codes").glob("*.txt")}
complete = exit_code == 0 and all(codes.get(task) == 0 for task in ("pretests", "prepare", "certificate", *sys.argv[4:], "summary"))
status = "COMPLETE" if complete else "PARTIAL" if exit_code in (130, 143) else "FAILED"
archive = Path(f"{prefix}__{status}.zip")
manifest = {"status": status, "head": (run / "head.txt").read_text().strip(), "exit_codes": codes, "runner_exit_code": exit_code,
            "payload": "arrays and checkpoints retained in the run directory on the server"}
(run / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
files = [path for path in sorted(run.rglob("*")) if path.is_file() and path.suffix in (".json", ".csv", ".txt", ".log", ".py", ".md")]
root = Path(archive.stem)
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
    sums = []
    for path in files:
        name = path.relative_to(run)
        data = path.read_bytes()
        handle.writestr(str(root / name).replace("\\", "/"), data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {name.as_posix()}")
    handle.writestr(str(root / "SHA256SUMS").replace("\\", "/"), "\n".join(sums) + "\n")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
sidecar = Path(str(archive) + ".sha256")
sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
print(f"[POPULATION PROFILE PACKAGE] {archive.resolve()}")
print(f"[POPULATION PROFILE SIDECAR] {sidecar.resolve()}")
print(f"{digest}  {archive.name}")
print(f"[POPULATION PROFILE HEAVY RETAINED] {run}")
EOF
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "RUN_ID=$RUN_ID HEAD=$HEAD GPUS=${GPU[*]}"
if [ "${PRETESTS:-1}" = 1 ]; then
    if CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider tools/analysis/tests/population \
        tools/analysis/tests/test_cert_boundary.py tools/analysis/tests/test_certify_psi.py tools/dev/check_certificate_soundness.py \
        tools/analysis/tests/test_precision.py tools/analysis/tests/test_lumir.py \
        --ignore=tools/analysis/tests/population/test_runner.py > "$RUN/logs/$ATTEMPT/pre_tests.txt" 2>&1; then
        echo 0 > "$RUN/exit_codes/pretests.txt"
    else
        code=$?; echo "$code" > "$RUN/exit_codes/pretests.txt"; log "ERROR pretests"; exit "$code"
    fi
else
    echo 0 > "$RUN/exit_codes/pretests.txt"
fi
worker() { "$PY" -m experiments.population.profile --task "$1" --run "$RUN" --data-root "$DATA" --parent-checkpoint "$PARENT_CHECKPOINT" "${@:2}"; }
if worker prepare > "$RUN/logs/$ATTEMPT/prepare.log" 2>&1; then
    echo 0 > "$RUN/exit_codes/prepare.txt"
else
    code=$?; echo "$code" > "$RUN/exit_codes/prepare.txt"; log "ERROR prepare"; exit "$code"
fi
if [ -f "$RUN/certificate.done" ] && [ -s "$RUN/certificate/report.json" ]; then
    log "SKIP certificate (completed)"
else
    log "START certificate on CPU"
    CUDA_VISIBLE_DEVICES="" "$PY" -m tools.analysis.certify_psi --root "${CERTIFICATE_ROOT:-results}" \
        --out "$RUN/certificate" > "$RUN/logs/$ATTEMPT/certificate.log" 2>&1 &
    CERT_PID=$!
fi
job() {
    local task=$1 gpu=$2 child code=0
    if [ -f "$RUN/$task.done" ] && [ -s "$RUN/$task/report.json" ]; then log "SKIP $task (completed)"; return 0; fi
    log "START $task on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" -m experiments.population.profile \
        --task "$task" --run "$RUN" --data-root "$DATA" --parent-checkpoint "$PARENT_CHECKPOINT" \
        --device cuda:0 > "$RUN/logs/$ATTEMPT/$task.log" 2>&1 &
    child=$!; echo "$child" > "$RUN/.pids/$ATTEMPT/$task"
    wait "$child" || code=$?
    rm -f "$RUN/.pids/$ATTEMPT/$task"
    echo "$code" > "$RUN/exit_codes/$task.txt"
    if [ "$code" = 0 ]; then touch "$RUN/$task.done"; log "PASS $task"; else log "ERROR $task (exit $code)"; fi
    return "$code"
}
failed=0; next=0; running=0
while [ "$next" -lt "${#TASKS[@]}" ] || [ "$running" -gt 0 ]; do
    for ((slot=0; slot<${#GPU[@]}; slot++)); do
        pid="${PIDS[$slot]:-}"
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || failed=1
            unset 'PIDS[slot]'; running=$((running - 1)); pid=""
        fi
        if [ -z "$pid" ] && [ "$next" -lt "${#TASKS[@]}" ]; then
            job "${TASKS[$next]}" "${GPU[$slot]}" &
            PIDS[$slot]=$!; next=$((next + 1)); running=$((running + 1))
        fi
    done
    [ "$running" = 0 ] || sleep 1
done
PIDS=()
if [ -n "$CERT_PID" ]; then
    code=0; wait "$CERT_PID" || code=$?; CERT_PID=""
    echo "$code" > "$RUN/exit_codes/certificate.txt"
    if [ "$code" = 0 ]; then touch "$RUN/certificate.done"; log "PASS certificate"; else failed=1; log "ERROR certificate (exit $code)"; fi
fi
if worker summary > "$RUN/logs/$ATTEMPT/summary.log" 2>&1; then
    echo 0 > "$RUN/exit_codes/summary.txt"
else
    code=$?; echo "$code" > "$RUN/exit_codes/summary.txt"; failed=1
fi
log "finished: runner_exit_code=$failed RUN_ID=$RUN_ID"
exit "$failed"
