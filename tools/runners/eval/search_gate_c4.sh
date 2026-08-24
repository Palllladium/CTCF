#!/usr/bin/env bash
# Gate C4: factorized descriptor-scale x search-reach study on IXI validation-58.
set -euo pipefail

# shellcheck source=tools/runners/eval/_search_gate_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_search_gate_common.sh"

GPU_LIST="${GPU_LIST:-2,3,4,5,6}"
PYBIN="${PYBIN:-python}"
PATHS_PROFILE="${PATHS_PROFILE:-3}"
# The authenticated frozen C3 source owns this value. A shell override would
# make the runner manifest disagree with the seed used by decision workers.
SEED=0
OUT_ROOT="${OUT_ROOT:-results/search_gate_c4}"
HEAVY_OUT_ROOT="${HEAVY_OUT_ROOT:-results/search_gate_c4_heavy}"
SOURCE_C3_DIR="${SOURCE_C3_DIR:-results/search_gate_c3/C3_DEVELOPMENT_20260822T212632Z_5d6c762a8a6f}"
SOURCE_C3_HEAVY_ROOT="${SOURCE_C3_HEAVY_ROOT:-results/search_gate_c3_heavy/C3_DEVELOPMENT_20260822T212632Z_5d6c762a8a6f}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
REMOTE_LOCATOR="${REMOTE_LOCATOR:-}"

sg_parse_gpu_list "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 || "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo "[FAIL] GPU_LIST must contain at least one unique GPU index" >&2
  exit 2
fi
GPU_CANONICAL="$(IFS=,; printf '%s' "${GPUS[*]}")"

sg_export_pythonpath
GIT_STATUS_AT_START="$(git status --porcelain=v1)"
if [[ -n "$GIT_STATUS_AT_START" ]]; then
  echo "[FAIL] Refusing to run from a dirty tree (tracked or untracked files):" >&2
  printf '%s\n' "$GIT_STATUS_AT_START" >&2
  echo "[HINT] Redirect nohup output outside the repository, for example to /tmp/search_gate_c4.log." >&2
  exit 1
fi
if [[ ! -d "$SOURCE_C3_DIR" || ! -f "$SOURCE_C3_DIR/c3_manifest.json" ]]; then
  echo "[FAIL] Frozen compact C3 source is missing: $SOURCE_C3_DIR" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_C3_HEAVY_ROOT" ]]; then
  echo "[FAIL] Frozen heavy C3 source is missing: $SOURCE_C3_HEAVY_ROOT" >&2
  exit 1
fi

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
STARTED_AT="$(sg_utc_started_at)"
RUN_ID="${RUN_ID:-C4_DEVELOPMENT_$(sg_utc_run_stamp)_$(sg_git_short_head)}"
ATTEMPT_ID="${ATTEMPT_ID:-A_$(sg_utc_run_stamp)_$$}"
if ! sg_is_safe_identifier "$RUN_ID" || ! sg_is_safe_identifier "$ATTEMPT_ID"; then
  echo "[FAIL] RUN_ID and ATTEMPT_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
RUN_ROOT="${OUT_ROOT}/${RUN_ID}"
HEAVY_RUN_ROOT="${HEAVY_OUT_ROOT}/${RUN_ID}"
RUN_ROOT_ABS="$(realpath -m "$RUN_ROOT")"
HEAVY_RUN_ROOT_ABS="$(realpath -m "$HEAVY_RUN_ROOT")"
if [[ "$HEAVY_RUN_ROOT_ABS" == "$RUN_ROOT_ABS" || "$HEAVY_RUN_ROOT_ABS" == "$RUN_ROOT_ABS/"* || \
      "$RUN_ROOT_ABS" == "$HEAVY_RUN_ROOT_ABS/"* ]]; then
  echo "[FAIL] Compact RUN_ROOT and HEAVY_RUN_ROOT must not overlap" >&2
  exit 2
fi
PACKAGE_ABS="$(pwd)/results/exports/${RUN_ID}.tar.gz"
if [[ -z "$REMOTE_LOCATOR" ]]; then
  REMOTE_LOCATOR="$(sg_default_remote_locator "$PACKAGE_ABS")"
fi

mkdir -p "$RUN_ROOT" "$RUN_ROOT/attempts" "$HEAVY_OUT_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
if ! flock -n 9; then
  echo "[FAIL] Another C4 process already holds RUN_ID=$RUN_ID" >&2
  exit 1
fi
ATTEMPT_ROOT="$RUN_ROOT/attempts/$ATTEMPT_ID"
if [[ -e "$ATTEMPT_ROOT" ]]; then
  echo "[FAIL] ATTEMPT_ID already exists: $ATTEMPT_ROOT" >&2
  exit 1
fi
mkdir -p "$ATTEMPT_ROOT"

{
  printf 'gpu_list=%s\n' "$GPU_CANONICAL"
  printf 'git_head=%s\n' "$HEAD"
  printf 'paths_profile=%s\n' "$PATHS_PROFILE"
  printf 'seed=%s\n' "$SEED"
  printf 'compact_run_root=%s\n' "$RUN_ROOT_ABS"
  printf 'heavy_run_root=%s\n' "$HEAVY_RUN_ROOT_ABS"
  printf 'source_c3_dir=%s\n' "$SOURCE_C3_DIR"
  printf 'source_c3_heavy_root=%s\n' "$SOURCE_C3_HEAVY_ROOT"
  printf 'min_free_gib=%s\n' "$MIN_FREE_GIB"
} > "$ATTEMPT_ROOT/runner_contract.txt"
if [[ ! -f "$RUN_ROOT/runner_contract.txt" ]]; then
  cp "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"
elif ! cmp -s "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"; then
  echo "[FAIL] Resume settings differ from the original C4 RUN_ID contract" >&2
  exit 1
fi

STARTED_FILE="$RUN_ROOT/started_at_utc.txt"
if [[ -f "$STARTED_FILE" ]]; then
  STARTED_AT="$(<"$STARTED_FILE")"
else
  printf '%s\n' "$STARTED_AT" > "$STARTED_FILE"
fi
{
  printf '#!/usr/bin/env bash\n'
  printf 'cd %q\n' "$(pwd)"
  printf 'GPU_LIST=%q PYBIN=%q PATHS_PROFILE=%q OUT_ROOT=%q HEAVY_OUT_ROOT=%q ' \
    "$GPU_LIST" "$PYBIN" "$PATHS_PROFILE" "$OUT_ROOT" "$HEAVY_OUT_ROOT"
  printf 'RUN_ID=%q ATTEMPT_ID=%q SOURCE_C3_DIR=%q SOURCE_C3_HEAVY_ROOT=%q ' \
    "$RUN_ID" "$ATTEMPT_ID" "$SOURCE_C3_DIR" "$SOURCE_C3_HEAVY_ROOT"
  printf 'MIN_FREE_GIB=%q REMOTE_LOCATOR=%q bash %q\n' "$MIN_FREE_GIB" "$REMOTE_LOCATOR" "$0"
} > "$ATTEMPT_ROOT/commands.sh"
git status --porcelain=v1 > "$ATTEMPT_ROOT/git_status.txt"
{
  "$PYBIN" -VV
  "$PYBIN" -m pip freeze || echo "[WARN] pip freeze unavailable"
  nvidia-smi || true
} > "$ATTEMPT_ROOT/environment.txt" 2>&1
for artifact in commands.sh git_status.txt environment.txt; do
  if [[ ! -f "$RUN_ROOT/$artifact" ]]; then
    cp "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"
  fi
done
{
  printf 'archive=%s\n' "$PACKAGE_ABS"
  printf 'sidecar=%s.sha256\n' "$PACKAGE_ABS"
} > "$RUN_ROOT/package_location.txt"
{
  printf 'heavy_run_root=%s\n' "$HEAVY_RUN_ROOT_ABS"
  printf 'source_c3_heavy_root=%s\n' "$(realpath -m "$SOURCE_C3_HEAVY_ROOT")"
  printf 'retention=RETAIN_BOTH_ON_H100_UNTIL_EXPLICIT_OPERATOR_DELETION\n'
  printf 'packaged=false\n'
} > "$RUN_ROOT/heavy_retention.txt"

STATUS="FAILED"
EXIT_CODE=1
PIDS=()

package_and_finalize() {
  local trap_code=$?
  local completed_at package
  trap - EXIT
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$trap_code" -eq 0 ]]; then
    STATUS="COMPLETE"
    EXIT_CODE=0
  else
    EXIT_CODE="$trap_code"
  fi
  git status --porcelain=v1 > "$ATTEMPT_ROOT/final_git_status.txt" 2>&1 || true
  if [[ ! -f "$RUN_ROOT/datasets.tsv" ]]; then
    printf 'dataset\tsplit\tcase_id\tpath\tbytes\tsha256\tmtime_utc\n' > "$RUN_ROOT/datasets.tsv"
  fi
  mkdir -p "$ATTEMPT_ROOT/prior_root_finalization"
  for artifact in run_manifest.json outputs.tsv; do
    if [[ -f "$RUN_ROOT/$artifact" ]]; then
      mv "$RUN_ROOT/$artifact" "$ATTEMPT_ROOT/prior_root_finalization/$artifact"
    fi
  done
  "$PYBIN" -m tools.analysis.run_artifacts finalize \
    --run-root "$RUN_ROOT" --run-id "$RUN_ID" --status "$STATUS" --exit-code "$EXIT_CODE" \
    --started-at "$STARTED_AT" --completed-at "$completed_at" --git-head "$HEAD" --branch "$BRANCH" \
    --gpu-index "${GPUS[0]}" --mode development --paths-profile "$PATHS_PROFILE" --seed "$SEED" \
    --time-steps 6 --expected-preflights 0 --no-strict-checkpoint-load --remote-locator "$REMOTE_LOCATOR" \
    || { STATUS="FAILED"; EXIT_CODE=1; }
  mkdir -p results/exports
  package="results/exports/${RUN_ID}.tar.gz"
  tar -czf "$package" -C "$OUT_ROOT" "$RUN_ID"
  (cd "$(dirname "$package")" && sha256sum "$(basename "$package")") > "${package}.sha256"
  echo "[PACKAGE] $(pwd)/$package"
  echo "[PACKAGE SIDECAR] $(pwd)/${package}.sha256"
  echo "[PACKAGE SHA-256] $(cat "${package}.sha256")"
  echo "[NEW C4 HEAVY RETAINED] $HEAVY_RUN_ROOT_ABS"
  echo "[SOURCE C3 HEAVY RETAINED] $(realpath -m "$SOURCE_C3_HEAVY_ROOT")"
  echo "[NOTICE] Neither heavy root was packaged or deleted. Retain both until an explicit operator decision."
  if [[ "$STATUS" != "COMPLETE" ]]; then
    echo "[FAILED] Send the compact package and its .sha256 sidecar for diagnosis." >&2
    exit "${EXIT_CODE:-1}"
  fi
  echo "[COMPLETE] Send the compact package and its .sha256 sidecar. Test-115 was not accessed."
}
trap package_and_finalize EXIT

echo "[RUN ID] $RUN_ID"
echo "[RUN ROOT] $RUN_ROOT_ABS"
echo "[HEAVY ROOT] $HEAVY_RUN_ROOT_ABS"
echo "[ATTEMPT ID] $ATTEMPT_ID"
echo "[HEAD] $HEAD"
echo "[GPU LIST] $GPU_CANONICAL"

echo "########## C4 fail-closed self-check ##########"
"$PYBIN" -m tools.dev.check_transactional_search --output "$ATTEMPT_ROOT/transactional_selfcheck.json"
"$PYBIN" -m tools.analysis.run_search_gate_c4 selfcheck --output "$ATTEMPT_ROOT/c4_selfcheck.json"
for artifact in transactional_selfcheck.json c4_selfcheck.json; do
  if [[ ! -f "$RUN_ROOT/$artifact" ]]; then
    cp "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"
  elif ! cmp -s "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"; then
    echo "[FAIL] Resume self-check differs from the original $artifact" >&2
    exit 1
  fi
done

echo "########## C4 source authentication and contracts ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c4 prepare \
  --run-root "$RUN_ROOT" \
  --heavy-root "$HEAVY_RUN_ROOT" \
  --source-c3-dir "$SOURCE_C3_DIR" \
  --source-c3-heavy-root "$SOURCE_C3_HEAVY_ROOT" \
  --num-shards "${#GPUS[@]}" \
  --physical-gpus "$GPU_CANONICAL" \
  --min-free-gib "$MIN_FREE_GIB"
SOURCE_CONTRACT_SHA256="$(sha256sum "$RUN_ROOT/source_contract.json" | awk '{print $1}')"
DECISION_CONTRACT_SHA256="$(sha256sum "$RUN_ROOT/decision_contract.json" | awk '{print $1}')"

echo "########## C4 label-free decision workers ##########"
mkdir -p "$RUN_ROOT/workers/decision/attempts/$ATTEMPT_ID"
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[$shard_index]}"
  log="$RUN_ROOT/workers/decision/attempts/$ATTEMPT_ID/worker_$(printf '%02d' "$shard_index").log"
  (
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYBIN" -m tools.analysis.run_search_gate_c4 decision-worker \
      --run-root "$RUN_ROOT" \
      --source-contract-sha256 "$SOURCE_CONTRACT_SHA256" \
      --decision-contract-sha256 "$DECISION_CONTRACT_SHA256" \
      --shard-index "$shard_index" --num-shards "${#GPUS[@]}" --gpu 0 \
      --physical-gpu "$physical_gpu" --attempt-id "$ATTEMPT_ID"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[DECISION WORKER] shard=$shard_index physical_gpu=$physical_gpu pid=$! log=$log"
done
worker_failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[FAIL] C4 decision shard $index failed; inspect its worker log." >&2
    worker_failed=1
  fi
done
PIDS=()
if [[ "$worker_failed" -ne 0 ]]; then
  exit 1
fi

echo "########## C4 immutable decision barrier ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c4 decision-barrier \
  --run-root "$RUN_ROOT" \
  --source-contract-sha256 "$SOURCE_CONTRACT_SHA256" \
  --decision-contract-sha256 "$DECISION_CONTRACT_SHA256" \
  --attempt-id "$ATTEMPT_ID"
BARRIER_SHA256="$(sha256sum "$RUN_ROOT/decision_barrier.json" | awk '{print $1}')"

echo "########## C4 post-freeze evaluation workers ##########"
mkdir -p "$RUN_ROOT/workers/evaluation/attempts/$ATTEMPT_ID"
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[$shard_index]}"
  log="$RUN_ROOT/workers/evaluation/attempts/$ATTEMPT_ID/worker_$(printf '%02d' "$shard_index").log"
  (
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYBIN" -m tools.analysis.run_search_gate_c4 evaluation-worker \
      --run-root "$RUN_ROOT" \
      --source-contract-sha256 "$SOURCE_CONTRACT_SHA256" \
      --decision-contract-sha256 "$DECISION_CONTRACT_SHA256" \
      --barrier-sha256 "$BARRIER_SHA256" \
      --shard-index "$shard_index" --num-shards "${#GPUS[@]}" --gpu 0 \
      --physical-gpu "$physical_gpu" --attempt-id "$ATTEMPT_ID"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[EVALUATION WORKER] shard=$shard_index physical_gpu=$physical_gpu pid=$! log=$log"
done
worker_failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[FAIL] C4 evaluation shard $index failed; inspect its worker log." >&2
    worker_failed=1
  fi
done
PIDS=()
if [[ "$worker_failed" -ne 0 ]]; then
  exit 1
fi

echo "########## C4 compact finalization ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c4 finalize \
  --run-root "$RUN_ROOT" \
  --source-contract-sha256 "$SOURCE_CONTRACT_SHA256" \
  --decision-contract-sha256 "$DECISION_CONTRACT_SHA256" \
  --barrier-sha256 "$BARRIER_SHA256" \
  --attempt-id "$ATTEMPT_ID"

exit 0
