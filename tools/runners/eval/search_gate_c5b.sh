#!/usr/bin/env bash
# Gate C5b V2: bounded pre-clip S4 amplitude bridge on IXI validation-58.
set -euo pipefail

# shellcheck source=tools/runners/eval/_search_gate_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_search_gate_common.sh"

GPU_LIST="${GPU_LIST:-2,3,4,5,6}"
PYBIN="${PYBIN:-python}"
PATHS_PROFILE="${PATHS_PROFILE:-3}"
OUT_ROOT="${OUT_ROOT:-results/search_gate_c5b}"
HEAVY_OUT_ROOT="${HEAVY_OUT_ROOT:-results/search_gate_c5b_heavy}"
SOURCE_C5_DIR="${SOURCE_C5_DIR:-results/search_gate_c5/C5_DEVELOPMENT_20260825T175112Z_242dde3281d2}"
SOURCE_C5_HEAVY_ROOT="${SOURCE_C5_HEAVY_ROOT:-results/search_gate_c5_heavy/C5_DEVELOPMENT_20260825T175112Z_242dde3281d2}"
MIN_FREE_GIB="${MIN_FREE_GIB:-50}"
REMOTE_LOCATOR="${REMOTE_LOCATOR:-}"
SEED=0

sg_parse_gpu_list "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 || "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo "[FAIL] GPU_LIST must contain at least one unique GPU index" >&2
  exit 2
fi
GPU_CANONICAL="$(IFS=,; printf '%s' "${GPUS[*]}")"
sg_export_pythonpath

GIT_STATUS_AT_START="$(git status --porcelain=v1)"
if [[ -n "$GIT_STATUS_AT_START" ]]; then
  echo "[FAIL] Refusing to run C5b from a dirty tree:" >&2
  printf '%s\n' "$GIT_STATUS_AT_START" >&2
  echo "[HINT] Redirect nohup output to /tmp/search_gate_c5b.log." >&2
  exit 1
fi
if [[ ! -f "$SOURCE_C5_DIR/c5_manifest.json" || ! -f "$SOURCE_C5_DIR/run_manifest.json" ]]; then
  echo "[FAIL] Frozen successful compact C5 source is absent: $SOURCE_C5_DIR" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_C5_HEAVY_ROOT" ]]; then
  echo "[FAIL] Frozen C5 heavy source is absent: $SOURCE_C5_HEAVY_ROOT" >&2
  exit 1
fi

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
STARTED_AT="$(sg_utc_started_at)"
RUN_ID="${RUN_ID:-C5B_DEVELOPMENT_$(sg_utc_run_stamp)_$(sg_git_short_head)}"
ATTEMPT_ID="${ATTEMPT_ID:-A_$(sg_utc_run_stamp)_$$}"
if ! sg_is_safe_identifier "$RUN_ID" || ! sg_is_safe_identifier "$ATTEMPT_ID"; then
  echo "[FAIL] RUN_ID and ATTEMPT_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

RUN_ROOT="$OUT_ROOT/$RUN_ID"
HEAVY_RUN_ROOT="$HEAVY_OUT_ROOT/$RUN_ID"
RUN_ROOT_ABS="$(realpath -m "$RUN_ROOT")"
HEAVY_RUN_ROOT_ABS="$(realpath -m "$HEAVY_RUN_ROOT")"
SOURCE_C5_DIR_ABS="$(realpath -m "$SOURCE_C5_DIR")"
SOURCE_C5_HEAVY_ROOT_ABS="$(realpath -m "$SOURCE_C5_HEAVY_ROOT")"
if [[ "$HEAVY_RUN_ROOT_ABS" == "$RUN_ROOT_ABS" || "$HEAVY_RUN_ROOT_ABS" == "$RUN_ROOT_ABS/"* || \
      "$RUN_ROOT_ABS" == "$HEAVY_RUN_ROOT_ABS/"* ]]; then
  echo "[FAIL] Compact and heavy C5b roots must not overlap" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$RUN_ROOT/attempts" "$HEAVY_OUT_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
if ! flock -n 9; then
  echo "[FAIL] Another C5b process holds RUN_ID=$RUN_ID" >&2
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
  printf 'protocol_id=%s\n' 'CTCF-SEARCH-GATE-C5B-V2'
  printf 'git_head=%s\n' "$HEAD"
  printf 'paths_profile=%s\n' "$PATHS_PROFILE"
  printf 'compact_run_root=%s\n' "$RUN_ROOT_ABS"
  printf 'heavy_run_root=%s\n' "$HEAVY_RUN_ROOT_ABS"
  printf 'source_c5_dir=%s\n' "$SOURCE_C5_DIR_ABS"
  printf 'source_c5_heavy_root=%s\n' "$SOURCE_C5_HEAVY_ROOT_ABS"
  printf 'min_free_gib=%s\n' "$MIN_FREE_GIB"
} > "$ATTEMPT_ROOT/runner_contract.txt"
if [[ ! -f "$RUN_ROOT/runner_contract.txt" ]]; then
  cp "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"
elif ! cmp -s "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"; then
  echo "[FAIL] Resume settings differ from the original C5b RUN_ID contract" >&2
  exit 1
fi

STARTED_FILE="$RUN_ROOT/started_at_utc.txt"
if [[ -f "$STARTED_FILE" ]]; then
  STARTED_AT="$(<"$STARTED_FILE")"
else
  printf '%s\n' "$STARTED_AT" > "$STARTED_FILE"
fi
{
  printf '#!/usr/bin/env bash\ncd %q\n' "$(pwd)"
  printf 'GPU_LIST=%q PYBIN=%q PATHS_PROFILE=%q OUT_ROOT=%q HEAVY_OUT_ROOT=%q ' \
    "$GPU_LIST" "$PYBIN" "$PATHS_PROFILE" "$OUT_ROOT" "$HEAVY_OUT_ROOT"
  printf 'SOURCE_C5_DIR=%q SOURCE_C5_HEAVY_ROOT=%q MIN_FREE_GIB=%q RUN_ID=%q ' \
    "$SOURCE_C5_DIR" "$SOURCE_C5_HEAVY_ROOT" "$MIN_FREE_GIB" "$RUN_ID"
  printf 'ATTEMPT_ID=%q REMOTE_LOCATOR=%q bash %q\n' "$ATTEMPT_ID" "$REMOTE_LOCATOR" "$0"
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

STATUS="FAILED"
EXIT_CODE=1
PIDS=()

package_and_finalize() {
  local trap_code=$?
  local completed_at package package_abs package_base
  trap - EXIT
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  PIDS=()
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$trap_code" -eq 0 ]]; then STATUS="COMPLETE"; EXIT_CODE=0; else EXIT_CODE="$trap_code"; fi
  git status --porcelain=v1 > "$ATTEMPT_ROOT/final_git_status.txt" 2>&1 || true
  if [[ ! -f "$RUN_ROOT/datasets.tsv" ]]; then
    printf 'dataset\tsplit\tcase_id\tpath\tbytes\tsha256\tmtime_utc\n' > "$RUN_ROOT/datasets.tsv"
  fi
  if [[ ! -f "$RUN_ROOT/heavy_retention.txt" ]]; then
    {
      printf 'source_c3_heavy=UNKNOWN_BEFORE_AUTHENTICATION\n'
      printf 'source_c4_heavy=UNKNOWN_BEFORE_AUTHENTICATION\n'
      printf 'source_c5_heavy=%s\n' "$SOURCE_C5_HEAVY_ROOT_ABS"
      printf 'target_c5b_heavy=%s\n' "$HEAVY_RUN_ROOT_ABS"
      printf 'retention=RETAIN_ALL_FOUR_ROOTS_UNTIL_EXPLICIT_OPERATOR_DECISION\npackaged=false\n'
    } > "$RUN_ROOT/heavy_retention.txt"
  fi
  mkdir -p "$ATTEMPT_ROOT/prior_finalization" results/exports
  for artifact in run_manifest.json outputs.tsv; do
    if [[ -f "$RUN_ROOT/$artifact" ]]; then mv "$RUN_ROOT/$artifact" "$ATTEMPT_ROOT/prior_finalization/$artifact"; fi
  done
  if [[ "$STATUS" == "COMPLETE" ]]; then
    package_base="$RUN_ID"
  else
    package_base="${RUN_ID}__${ATTEMPT_ID}__FAILED"
  fi
  package="results/exports/${package_base}.tar.gz"
  package_abs="$(pwd)/$package"
  if [[ -z "$REMOTE_LOCATOR" ]]; then
    REMOTE_LOCATOR="$(sg_default_remote_locator "$package_abs")"
  fi
  {
    printf 'archive=%s\n' "$package_abs"
    printf 'sidecar=%s.sha256\n' "$package_abs"
  } > "$RUN_ROOT/package_location.txt"
  "$PYBIN" -m tools.analysis.run_artifacts finalize \
    --run-root "$RUN_ROOT" --run-id "$RUN_ID" --status "$STATUS" --exit-code "$EXIT_CODE" \
    --started-at "$STARTED_AT" --completed-at "$completed_at" --git-head "$HEAD" --branch "$BRANCH" \
    --gpu-index "${GPUS[0]}" --mode development --paths-profile "$PATHS_PROFILE" --seed "$SEED" \
    --time-steps 6 --expected-preflights 0 --no-strict-checkpoint-load --remote-locator "$REMOTE_LOCATOR" \
    || { STATUS="FAILED"; EXIT_CODE=1; }
  if [[ -e "$package" || -e "${package}.sha256" ]]; then
    echo "[FAIL] Refusing to overwrite an existing C5b package: $package" >&2
    exit 1
  fi
  tar -czf "$package" -C "$OUT_ROOT" "$RUN_ID"
  (cd "$(dirname "$package")" && sha256sum "$(basename "$package")") > "${package}.sha256"
  echo "[PACKAGE] $(pwd)/$package"
  echo "[PACKAGE SIDECAR] $(pwd)/${package}.sha256"
  echo "[PACKAGE SHA-256] $(cat "${package}.sha256")"
  echo "[HEAVY ROOTS RETAINED] See $RUN_ROOT/heavy_retention.txt"
  if [[ "$STATUS" != "COMPLETE" ]]; then
    echo "[FAILED] Send the compact failed-attempt package for diagnosis." >&2
    exit "${EXIT_CODE:-1}"
  fi
  echo "[COMPLETE] Send the compact package and sidecar. Test-115 was not accessed."
}
trap package_and_finalize EXIT

echo "[RUN ID] $RUN_ID"
echo "[RUN ROOT] $RUN_ROOT_ABS"
echo "[HEAVY ROOT] $HEAVY_RUN_ROOT_ABS"
echo "[ATTEMPT ID] $ATTEMPT_ID"
echo "[HEAD] $HEAD"
echo "[GPU LIST] $GPU_CANONICAL"

echo "########## C5b fail-closed self-check ##########"
"$PYBIN" -m tools.dev.check_transactional_search --output "$ATTEMPT_ROOT/transactional_selfcheck.json"
"$PYBIN" -m tools.analysis.run_search_gate_c5b selfcheck --output "$ATTEMPT_ROOT/c5b_selfcheck.json"
for artifact in transactional_selfcheck.json c5b_selfcheck.json; do
  if [[ ! -f "$RUN_ROOT/$artifact" ]]; then
    cp "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"
  elif ! cmp -s "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"; then
    echo "[FAIL] Resume self-check differs: $artifact" >&2
    exit 1
  fi
done

echo "########## C5b successful-C5 authentication and contracts ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c5b prepare \
  --run-root "$RUN_ROOT" --heavy-root "$HEAVY_RUN_ROOT" \
  --source-c5-dir "$SOURCE_C5_DIR" --source-c5-heavy-root "$SOURCE_C5_HEAVY_ROOT" \
  --num-shards "${#GPUS[@]}" --physical-gpus "$GPU_CANONICAL" --min-free-gib "$MIN_FREE_GIB"
SOURCE_SHA="$(sha256sum "$RUN_ROOT/source_contract.json" | awk '{print $1}')"
DECISION_SHA="$(sha256sum "$RUN_ROOT/decision_contract.json" | awk '{print $1}')"

echo "########## C5b single-case real-H100 decision pilot ##########"
PILOT_LOG="$RUN_ROOT/workers/decision/attempts/$ATTEMPT_ID/pilot.log"
mkdir -p "$(dirname "$PILOT_LOG")"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYBIN" -m tools.analysis.run_search_gate_c5b decision-pilot \
  --run-root "$RUN_ROOT" --decision-contract-sha256 "$DECISION_SHA" \
  --num-shards "${#GPUS[@]}" --gpu 0 --physical-gpu "${GPUS[0]}" --attempt-id "$ATTEMPT_ID" \
  > "$PILOT_LOG" 2>&1
echo "[DECISION PILOT] physical_gpu=${GPUS[0]} log=$PILOT_LOG"

echo "########## C5b label-free decision workers ##########"
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[$shard_index]}"
  log="$RUN_ROOT/workers/decision/attempts/$ATTEMPT_ID/worker_$(printf '%02d' "$shard_index").log"
  (
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYBIN" -m tools.analysis.run_search_gate_c5b decision-worker \
      --run-root "$RUN_ROOT" --decision-contract-sha256 "$DECISION_SHA" \
      --shard-index "$shard_index" --num-shards "${#GPUS[@]}" --gpu 0 \
      --physical-gpu "$physical_gpu" --attempt-id "$ATTEMPT_ID"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[DECISION WORKER] shard=$shard_index physical_gpu=$physical_gpu pid=$! log=$log"
done
worker_failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[FAIL] C5b decision shard $index failed; inspect its worker log." >&2
    worker_failed=1
  fi
done
PIDS=()
if [[ "$worker_failed" -ne 0 ]]; then exit 1; fi

echo "########## C5b immutable label barrier ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c5b decision-barrier \
  --run-root "$RUN_ROOT" --decision-contract-sha256 "$DECISION_SHA" \
  --attempt-id "$ATTEMPT_ID"
BARRIER_SHA="$(sha256sum "$RUN_ROOT/decision_barrier.json" | awk '{print $1}')"

echo "########## C5b post-barrier evaluation contract ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c5b freeze-evaluation \
  --run-root "$RUN_ROOT" --source-contract-sha256 "$SOURCE_SHA" --decision-contract-sha256 "$DECISION_SHA" \
  --barrier-sha256 "$BARRIER_SHA"
EVALUATION_SHA="$(sha256sum "$RUN_ROOT/evaluation_contract.json" | awk '{print $1}')"

echo "########## C5b post-barrier evaluation workers ##########"
mkdir -p "$RUN_ROOT/workers/evaluation/attempts/$ATTEMPT_ID"
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[$shard_index]}"
  log="$RUN_ROOT/workers/evaluation/attempts/$ATTEMPT_ID/worker_$(printf '%02d' "$shard_index").log"
  (
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYBIN" -m tools.analysis.run_search_gate_c5b evaluation-worker \
      --run-root "$RUN_ROOT" --source-contract-sha256 "$SOURCE_SHA" --decision-contract-sha256 "$DECISION_SHA" \
      --barrier-sha256 "$BARRIER_SHA" --evaluation-contract-sha256 "$EVALUATION_SHA" \
      --shard-index "$shard_index" --num-shards "${#GPUS[@]}" --gpu 0 \
      --physical-gpu "$physical_gpu" --attempt-id "$ATTEMPT_ID"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[EVALUATION WORKER] shard=$shard_index physical_gpu=$physical_gpu pid=$! log=$log"
done
worker_failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[FAIL] C5b evaluation shard $index failed; inspect its worker log." >&2
    worker_failed=1
  fi
done
PIDS=()
if [[ "$worker_failed" -ne 0 ]]; then exit 1; fi

echo "########## C5b finalization ##########"
"$PYBIN" -m tools.analysis.run_search_gate_c5b finalize \
  --run-root "$RUN_ROOT" --source-contract-sha256 "$SOURCE_SHA" --decision-contract-sha256 "$DECISION_SHA" \
  --barrier-sha256 "$BARRIER_SHA" --evaluation-contract-sha256 "$EVALUATION_SHA" --attempt-id "$ATTEMPT_ID"
STATUS="COMPLETE"
EXIT_CODE=0
