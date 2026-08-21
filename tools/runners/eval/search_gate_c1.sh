#!/usr/bin/env bash
# Gate C1: multi-GPU exploration-19 now; confirmation-39 only after an explicit frozen-manifest handoff.
set -euo pipefail

MODE="${MODE:-exploration}"
GPU_LIST="${GPU_LIST:-2,3,4,5,6}"
PYBIN="${PYBIN:-python}"
PATHS_PROFILE="${PATHS_PROFILE:-3}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/search_gate_c1}"
IXI_CKPT="${IXI_CKPT:-results/P10_LONGRUN_VXM_UNIFIED_SVF_IXI/ckpt/best.pth}"
REMOTE_LOCATOR="${REMOTE_LOCATOR:-}"
KEEP_FIELDS="${KEEP_FIELDS:-0}"
EXPLORE_MANIFEST="${EXPLORE_MANIFEST:-}"
EXPLORE_MANIFEST_SHA256="${EXPLORE_MANIFEST_SHA256:-}"

case "$MODE" in
  selfcheck|exploration|confirmation|all) ;;
  *) echo "[FAIL] MODE must be selfcheck, exploration, confirmation, or all" >&2; exit 2 ;;
esac
if [[ "$KEEP_FIELDS" != "0" ]]; then
  echo "[FAIL] The standard C1 runner is compact-only; KEEP_FIELDS must remain 0" >&2
  exit 2
fi
if [[ "$MODE" == "confirmation" ]]; then
  if [[ -z "$EXPLORE_MANIFEST" || -z "$EXPLORE_MANIFEST_SHA256" ]]; then
    echo "[FAIL] confirmation requires EXPLORE_MANIFEST and EXPLORE_MANIFEST_SHA256" >&2
    exit 2
  fi
fi

IFS=',' read -r -a RAW_GPUS <<< "$GPU_LIST"
GPUS=()
for value in "${RAW_GPUS[@]}"; do
  gpu="${value//[[:space:]]/}"
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "[FAIL] GPU_LIST must be a comma-separated list of non-negative integers" >&2
    exit 2
  fi
  GPUS+=("$gpu")
done
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "[FAIL] GPU_LIST is empty" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo "[FAIL] GPU_LIST contains duplicate physical GPU indices" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
GIT_STATUS_AT_START="$(git status --porcelain=v1)"
if [[ -n "$GIT_STATUS_AT_START" ]]; then
  echo "[FAIL] Refusing to run from a dirty tree (tracked or untracked files):" >&2
  printf '%s\n' "$GIT_STATUS_AT_START" >&2
  echo "[HINT] Redirect nohup output outside the repository, for example to /tmp/search_gate_c1.log." >&2
  exit 1
fi

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODE_UPPER="$(printf '%s' "$MODE" | tr '[:lower:]' '[:upper:]')"
RUN_ID="${RUN_ID:-C1_${MODE_UPPER}_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)}"
RUN_ROOT="${OUT_ROOT}/${RUN_ID}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "[FAIL] RUN_ID contains unsupported characters" >&2
  exit 2
fi
ATTEMPT_ID="${ATTEMPT_ID:-A_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$ATTEMPT_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "[FAIL] ATTEMPT_ID contains unsupported characters" >&2
  exit 2
fi
PACKAGE_ABS="$(pwd)/results/exports/${RUN_ID}.tar.gz"
if [[ -z "$REMOTE_LOCATOR" ]]; then
  REMOTE_LOCATOR="H100_LOCAL_ARCHIVE=${PACKAGE_ABS};H100_LOCAL_SIDECAR=${PACKAGE_ABS}.sha256"
fi
mkdir -p "$RUN_ROOT/preflight" "$RUN_ROOT/attempts"
exec 9>"$RUN_ROOT/.run.lock"
if ! flock -n 9; then
  echo "[FAIL] Another C1 process already holds RUN_ID=$RUN_ID" >&2
  exit 1
fi
ATTEMPT_ROOT="$RUN_ROOT/attempts/$ATTEMPT_ID"
if [[ -e "$ATTEMPT_ROOT" ]]; then
  echo "[FAIL] ATTEMPT_ID already exists for this run: $ATTEMPT_ROOT" >&2
  exit 1
fi
mkdir -p "$ATTEMPT_ROOT/preflight"

GPU_CANONICAL="$(IFS=,; printf '%s' "${GPUS[*]}")"
{
  printf 'mode=%s\n' "$MODE"
  printf 'gpu_list=%s\n' "$GPU_CANONICAL"
  printf 'git_head=%s\n' "$HEAD"
  printf 'paths_profile=%s\n' "$PATHS_PROFILE"
  printf 'seed=%s\n' "$SEED"
  printf 'checkpoint=%s\n' "$IXI_CKPT"
  printf 'keep_fields=%s\n' "$KEEP_FIELDS"
  printf 'explore_manifest=%s\n' "$EXPLORE_MANIFEST"
  printf 'explore_manifest_sha256=%s\n' "$EXPLORE_MANIFEST_SHA256"
} > "$ATTEMPT_ROOT/runner_contract.txt"
if [[ ! -f "$RUN_ROOT/runner_contract.txt" ]]; then
  cp "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"
elif ! cmp -s "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"; then
  echo "[FAIL] Resume runner settings differ from the original RUN_ID contract" >&2
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
  printf 'MODE=%q GPU_LIST=%q PYBIN=%q PATHS_PROFILE=%q SEED=%q OUT_ROOT=%q RUN_ID=%q ATTEMPT_ID=%q ' \
    "$MODE" "$GPU_LIST" "$PYBIN" "$PATHS_PROFILE" "$SEED" "$OUT_ROOT" "$RUN_ID" "$ATTEMPT_ID"
  printf 'IXI_CKPT=%q REMOTE_LOCATOR=%q KEEP_FIELDS=%q EXPLORE_MANIFEST=%q ' \
    "$IXI_CKPT" "$REMOTE_LOCATOR" "$KEEP_FIELDS" "$EXPLORE_MANIFEST"
  printf 'EXPLORE_MANIFEST_SHA256=%q bash %q\n' "$EXPLORE_MANIFEST_SHA256" "$0"
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

STATUS="FAILED"
EXIT_CODE=1
EXPECTED_PREFLIGHTS=0
PIDS=()

package_and_finalize() {
  local trap_code=$?
  trap - EXIT
  local completed_at package
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
  for work_dir in "$RUN_ROOT"/exploration/cases/*/work "$RUN_ROOT"/confirmation/cases/*/work; do
    if [[ -d "$work_dir" ]]; then
      rm -rf -- "$work_dir"
    fi
  done
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$trap_code" -eq 0 ]]; then STATUS="COMPLETE"; EXIT_CODE=0; else EXIT_CODE="$trap_code"; fi
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
    --gpu-index "${GPUS[0]}" --mode "$MODE" --paths-profile "$PATHS_PROFILE" --seed "$SEED" \
    --time-steps 6 --expected-preflights "$EXPECTED_PREFLIGHTS" --remote-locator "$REMOTE_LOCATOR" \
    || { STATUS="FAILED"; EXIT_CODE=1; }

  mkdir -p results/exports
  package="results/exports/${RUN_ID}.tar.gz"
  tar -czf "$package" -C "$OUT_ROOT" "$RUN_ID"
  (cd "$(dirname "$package")" && sha256sum "$(basename "$package")") > "${package}.sha256"
  echo "[PACKAGE] $(pwd)/$package"
  echo "[PACKAGE SIDECAR] $(pwd)/${package}.sha256"
  echo "[PACKAGE SHA-256] $(cat "${package}.sha256")"
  if [[ "$STATUS" != "COMPLETE" ]]; then
    echo "[FAILED] Send the compact package and its .sha256 sidecar for diagnosis." >&2
    exit "${EXIT_CODE:-1}"
  fi
  echo "[COMPLETE] Send the compact package and its .sha256 sidecar. Heavy fields were not packaged by default."
}
trap package_and_finalize EXIT

echo "[RUN ID] $RUN_ID"
echo "[RUN ROOT] $RUN_ROOT"
echo "[ATTEMPT ID] $ATTEMPT_ID"
echo "[HEAD] $HEAD"
echo "[MODE] $MODE"
echo "[GPU LIST] $GPU_LIST"

echo "########## C1 fail-closed self-check ##########"
"$PYBIN" -m tools.dev.check_transactional_search --output "$ATTEMPT_ROOT/transactional_selfcheck.json"
"$PYBIN" -m tools.analysis.run_search_gate_c1 selfcheck --output "$ATTEMPT_ROOT/c1_selfcheck.json"
for artifact in transactional_selfcheck.json c1_selfcheck.json; do
  if [[ ! -f "$RUN_ROOT/$artifact" ]]; then
    cp "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"
  elif ! cmp -s "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"; then
    echo "[FAIL] Resume self-check differs from the original $artifact" >&2
    exit 1
  fi
done

if [[ "$MODE" == "selfcheck" ]]; then
  exit 0
fi

# `all` intentionally stops after exploration. Consuming validation-39 always requires an explicit
# confirmation mode plus a manually selected, hash-pinned exploration manifest.
STAGE="$MODE"
if [[ "$MODE" == "all" ]]; then
  STAGE="exploration"
  echo "[NOTICE] MODE=all means selfcheck + exploration; confirmation is never automatic."
fi

EXPECTED_PREFLIGHTS=1
"$PYBIN" -m tools.analysis.checkpoint_preflight \
  --checkpoint "$IXI_CKPT" \
  --ctcf-config CTCF-CascadeA-VM-Unified \
  --ctcf-l3-svf 1 \
  --time-steps 6 \
  --output "$ATTEMPT_ROOT/preflight/p10_ixi.json"
cp "$ATTEMPT_ROOT/preflight/p10_ixi.json" "$RUN_ROOT/preflight/p10_ixi.json"

PREPARE_ARGS=(
  prepare
  --stage "$STAGE"
  --run-root "$RUN_ROOT"
  --paths-profile "$PATHS_PROFILE"
  --checkpoint "$IXI_CKPT"
  --seed "$SEED"
  --num-shards "${#GPUS[@]}"
  --physical-gpus "$GPU_LIST"
)
if [[ "$STAGE" == "confirmation" ]]; then
  PREPARE_ARGS+=(
    --explore-manifest "$EXPLORE_MANIFEST"
    --explore-manifest-sha256 "$EXPLORE_MANIFEST_SHA256"
  )
fi
"$PYBIN" -m tools.analysis.run_search_gate_c1 "${PREPARE_ARGS[@]}"
cp "$RUN_ROOT/$STAGE/datasets.tsv" "$RUN_ROOT/datasets.tsv"
CONTRACT_SHA256="$(sha256sum "$RUN_ROOT/$STAGE/stage_contract.json" | awk '{print $1}')"

echo "########## C1 $STAGE: ${#GPUS[@]} deterministic case shards ##########"
mkdir -p "$RUN_ROOT/$STAGE/workers/attempts/$ATTEMPT_ID"
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[$shard_index]}"
  log="$RUN_ROOT/$STAGE/workers/attempts/$ATTEMPT_ID/worker_$(printf '%02d' "$shard_index").log"
  (
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYBIN" -m tools.analysis.run_search_gate_c1 worker \
      --stage "$STAGE" \
      --run-root "$RUN_ROOT" \
      --contract-sha256 "$CONTRACT_SHA256" \
      --shard-index "$shard_index" \
      --num-shards "${#GPUS[@]}" \
      --gpu 0 \
      --physical-gpu "$physical_gpu" \
      --attempt-id "$ATTEMPT_ID"
  ) > "$log" 2>&1 &
  worker_pid="$!"
  PIDS+=("$worker_pid")
  echo "[WORKER] shard=$shard_index physical_gpu=$physical_gpu pid=$worker_pid log=$log"
done

worker_failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[FAIL] C1 shard $index failed; inspect its worker log." >&2
    worker_failed=1
  fi
done
PIDS=()
if [[ "$worker_failed" -ne 0 ]]; then
  exit 1
fi

"$PYBIN" -m tools.analysis.run_search_gate_c1 finalize \
  --stage "$STAGE" \
  --run-root "$RUN_ROOT" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --attempt-id "$ATTEMPT_ID"

exit 0
