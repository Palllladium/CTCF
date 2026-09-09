#!/usr/bin/env bash
# Read-only numerical replay and FP32 comparison. Never resumes production training.
set -Eeuo pipefail
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"
readonly PYBIN="${PYBIN:-python}"
readonly EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:?Set the diagnostic commit SHA}"
readonly DATA_CONTRACT="results/stage5_data/manifests/data_contract.json"
readonly IMAGE_ROOT="results/stage5_data/image_only"
readonly GPU_LIST="${GPU_LIST:-0,1,2,3}"
readonly ATTEMPT="S5_PRECISIONDIAG_$(date -u +%Y%m%dT%H%M%SZ)_$$_${EXPECTED_GIT_HEAD:0:12}"
readonly OUTPUT="results/stage5_diagnostics/$ATTEMPT"
readonly CAPTURE_ROOT="results/stage5_heavy/$ATTEMPT"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

[[ "$EXPECTED_GIT_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo "[FAIL] Invalid expected SHA"; exit 2; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_GIT_HEAD" ]] || { echo "[FAIL] Wrong HEAD"; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "[FAIL] Dirty tree"; exit 2; }
command -v "$PYBIN" >/dev/null || { echo "[FAIL] Python executable unavailable: $PYBIN"; exit 2; }
contract_values="$("$PYBIN" -m tools.analysis.stage5.precision_contract --runner-values)" || {
  echo "[FAIL] Cannot load diagnostic contract"; exit 2;
}
mapfile -t contract_lines <<< "$contract_values"
readonly SOURCE_RUN="${contract_lines[0]}"
jobs=("${contract_lines[@]:1}")
[[ -n "$SOURCE_RUN" && ${#jobs[@]} -gt 0 && ${#GPUS[@]} -gt 0 ]] || {
  echo "[FAIL] At least one GPU and one diagnostic job are required"; exit 2;
}
[[ "$GPU_LIST" != *, ]] || { echo "[FAIL] Empty GPU index"; exit 2; }
readonly SOURCE_ROOT="results/stage5/$SOURCE_RUN"
readonly CHECKPOINT_ROOT="results/stage5_heavy/$SOURCE_RUN/checkpoints"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ && -z "${SEEN[$gpu]:-}" ]] || {
    echo "[FAIL] GPU indices must be distinct non-negative integers"; exit 2;
  }
  SEEN[$gpu]=1
done
command -v flock >/dev/null || { echo "[FAIL] flock is required"; exit 2; }
# Opening read-only avoids modifying the source run's lock file.
exec 9<"$SOURCE_ROOT/stage5.lock"
flock -sn 9 || { echo "[FAIL] Source run is still active"; exit 3; }
mkdir -p results/stage5_diagnostics results/stage5_heavy results/exports
mkdir "$OUTPUT"
declare -A ACTIVE=()
declare -A PID_GPU=()
finish() {
  local result=$? pid tick remaining package_result
  trap - EXIT
  trap '' INT TERM
  # Only the processes launched by this shell are signalled. Wait for all writers
  # before hashing and packaging, including when a worker failed or we were stopped.
  if (( ${#ACTIVE[@]} )); then
    for pid in "${!ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for tick in {1..10}; do
      remaining=0
      for pid in "${!ACTIVE[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then remaining=1; fi
      done
      if (( remaining == 0 )); then break; fi
      sleep 1
    done
    for pid in "${!ACTIVE[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    for pid in "${!ACTIVE[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
  printf '%s\n' "$result" > "$OUTPUT/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT/finished_at.txt"
  set +e
  "$PYBIN" -m tools.analysis.stage5.precision_report \
    --output-root "$OUTPUT" --export-root results/exports \
    --expected-git-head "$EXPECTED_GIT_HEAD" --source-run "$SOURCE_RUN" \
    --runner-exit-code "$result"
  package_result=$?
  if (( package_result != 0 )); then
    echo "[PRECISION DIAG ATTENTION] Finalization needs review; compact files: $OUTPUT"
    if (( result == 0 )); then result=1; fi
  fi
  echo "[PRECISION DIAG HEAVY RETAINED] $CAPTURE_ROOT"
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir "$CAPTURE_ROOT"
git rev-parse HEAD > "$OUTPUT/git_head.txt"
git status --porcelain=v1 --untracked-files=all > "$OUTPUT/git_status.txt"
printf '%s\n' "$SOURCE_RUN" > "$OUTPUT/source_run.txt"
printf '%s\n' "$CAPTURE_ROOT" > "$OUTPUT/capture_root.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT/started_at.txt"
printf 'PYBIN=%q GPU_LIST=%q EXPECTED_GIT_HEAD=%q bash tools/runners/train/stage5_precision_diagnostic.sh\n' \
  "$PYBIN" "$GPU_LIST" "$EXPECTED_GIT_HEAD" > "$OUTPUT/command.sh"
nvidia-smi > "$OUTPUT/nvidia-smi.txt" 2>&1
printf 'job\tgpu\tpid\n' > "$OUTPUT/jobs.tsv"
printf 'job\tpid\texit_code\n' > "$OUTPUT/job_exit_codes.tsv"
launch_job() {
  local job="$1" gpu="$2" pid
  local -a command_args
  command_args=(
    "$PYBIN" -u -m tools.analysis.diagnose_stage5_precision
    --repo-root "$REPO_ROOT" --expected-git-head "$EXPECTED_GIT_HEAD"
    --source-root "$SOURCE_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"
    --data-contract "$DATA_CONTRACT" --image-root "$IMAGE_ROOT"
    --job "$job" --output "$OUTPUT/$job.json" --capture-root "$CAPTURE_ROOT"
  )
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu" > "$OUTPUT/command_$job.sh"
  printf '%q ' "${command_args[@]}" >> "$OUTPUT/command_$job.sh"
  printf '\n' >> "$OUTPUT/command_$job.sh"
  CUDA_VISIBLE_DEVICES="$gpu" "${command_args[@]}" > "$OUTPUT/$job.log" 2>&1 &
  pid=$!
  ACTIVE[$pid]="$job"
  PID_GPU[$pid]="$gpu"
  printf '%s\t%s\t%s\n' "$job" "$gpu" "$pid" >> "$OUTPUT/jobs.tsv"
  echo "[PRECISION DIAG START] $job GPU=$gpu PID=$pid $OUTPUT/$job.log"
}

# Physical indices need not start at zero. Each worker sees its assigned GPU
# as cuda:0 through CUDA_VISIBLE_DEVICES. Never share a GPU between live jobs.
free_gpus=("${GPUS[@]}")
next_job=0
failed=0
while (( next_job < ${#jobs[@]} || ${#ACTIVE[@]} )); do
  while (( next_job < ${#jobs[@]} && ${#free_gpus[@]} )); do
    launch_job "${jobs[$next_job]}" "${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    next_job=$((next_job + 1))
  done
  for pid in "${!ACTIVE[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then continue; fi
    code=0
    wait "$pid" || code=$?
    printf '%s\t%s\t%s\n' "${ACTIVE[$pid]}" "$pid" "$code" >> "$OUTPUT/job_exit_codes.tsv"
    echo "[PRECISION DIAG FINISH] ${ACTIVE[$pid]} GPU=${PID_GPU[$pid]} EXIT=$code"
    free_gpus+=("${PID_GPU[$pid]}")
    unset 'ACTIVE[$pid]' 'PID_GPU[$pid]'
    if (( code != 0 )); then failed=1; fi
  done
  # Polling avoids requiring Bash 5.1 wait -n -p on the server.
  if (( ${#ACTIVE[@]} )); then sleep 1; fi
done
for job in "${jobs[@]}"; do tail -n 8 "$OUTPUT/$job.log"; done
exit "$failed"
