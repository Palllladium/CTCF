#!/usr/bin/env bash
# Isolated replay of the reviewed failed run. Never calls the production runner.
set -Eeuo pipefail
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"
readonly PYBIN="${PYBIN:-/data/mooncake/P/envs/ctcf/bin/python}"
readonly EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:?Set the diagnostic commit SHA}"
readonly SOURCE_RUN="S5_DEVELOPMENT_20260907T175008Z_458489f77fc6"
readonly SOURCE_ROOT="results/stage5/$SOURCE_RUN"
readonly CHECKPOINT_ROOT="results/stage5_heavy/$SOURCE_RUN/checkpoints"
readonly DIAGNOSTIC_MODE="${DIAGNOSTIC_MODE:-amp}"
case "$DIAGNOSTIC_MODE" in
  amp) PREFIX="S5_AMPDIAG" ;;
  ncc) PREFIX="S5_NCCDIAG" ;;
  *) echo "[FAIL] DIAGNOSTIC_MODE must be amp or ncc"; exit 2 ;;
esac
readonly REFERENCE_ROOT="${REFERENCE_ROOT:-results/stage5_diagnostics/S5_AMPDIAG_20260907T183818Z_2357762_41dc480ac467}"
readonly ATTEMPT="${PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)_$$_${EXPECTED_GIT_HEAD:0:12}"
readonly OUTPUT="results/stage5_diagnostics/$ATTEMPT"
readonly GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ "$EXPECTED_GIT_HEAD" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$EXPECTED_GIT_HEAD" ]] || { echo "[FAIL] Wrong HEAD"; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "[FAIL] Dirty tree"; exit 2; }
[[ ${#GPUS[@]} == 4 ]] || { echo "[FAIL] Exactly four distinct GPUs required"; exit 2; }
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${SEEN[$gpu]:-}" ]] || exit 2
  SEEN[$gpu]=1
done
# A shared read-only lock prevents the production runner taking its exclusive lock.
exec 9<"$SOURCE_ROOT/stage5.lock"
flock -sn 9 || { echo "[FAIL] Source run is still active"; exit 3; }
mkdir -p results/stage5_diagnostics results/exports
mkdir "$OUTPUT"
git rev-parse HEAD > "$OUTPUT/git_head.txt"
git status --porcelain=v1 --untracked-files=all > "$OUTPUT/git_status.txt"
printf '%s\n' "$SOURCE_RUN" > "$OUTPUT/source_run.txt"
printf 'PYBIN=%q GPU_LIST=%q EXPECTED_GIT_HEAD=%q DIAGNOSTIC_MODE=%q REFERENCE_ROOT=%q bash tools/runners/train/stage5_amp_diagnostic.sh\n' \
  "$PYBIN" "$GPU_LIST" "$EXPECTED_GIT_HEAD" "$DIAGNOSTIC_MODE" "$REFERENCE_ROOT" > "$OUTPUT/command.sh"
nvidia-smi > "$OUTPUT/nvidia-smi.txt"
pids=()
finish() {
  result=$?
  trap - EXIT INT TERM
  # Children are diagnostic processes started by this shell only.
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  printf '%s\n' "$result" > "$OUTPUT/exit_code.txt"
  (cd "$OUTPUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  archive="results/exports/${ATTEMPT}.tar.gz"
  if tar -czf "$archive" -C results/stage5_diagnostics "$ATTEMPT"; then
    sha256sum "$archive" > "${archive}.sha256"
    echo "[${DIAGNOSTIC_MODE^^} DIAG PACKAGE] $archive"
    cat "${archive}.sha256"
  else
    echo "[FAIL] Packaging failed; diagnostic files retained at $OUTPUT"
    result=1
  fi
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
variants=(F0 F2V F2S F2P)
for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  extra_args=()
  if [[ "$DIAGNOSTIC_MODE" == ncc ]]; then
    extra_args=(--ncc-audit --reference-report "$REFERENCE_ROOT/$variant.json")
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYBIN" -u -m tools.analysis.diagnose_stage5_amp \
    --repo-root "$REPO_ROOT" --expected-git-head "$EXPECTED_GIT_HEAD" \
    --source-root "$SOURCE_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
    --data-contract results/stage5_data/manifests/data_contract.json \
    --image-root results/stage5_data/image_only --variant "$variant" \
    --output "$OUTPUT/$variant.json" "${extra_args[@]}" > "$OUTPUT/$variant.log" 2>&1 &
  pids+=("$!")
  echo "[AMP DIAG START] $variant GPU=${GPUS[$i]} PID=$! $OUTPUT/$variant.log"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
pids=()
for variant in "${variants[@]}"; do
  tail -n 10 "$OUTPUT/$variant.log"
done
exit "$failed"
