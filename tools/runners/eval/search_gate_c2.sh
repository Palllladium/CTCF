#!/usr/bin/env bash
# Gate C2: four frozen sequential search trajectories on the already-open IXI validation-58.
set -euo pipefail

# shellcheck source=tools/runners/eval/_search_gate_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_search_gate_common.sh"

GPU_LIST="${GPU_LIST:-2,3,4,5,6}"
PYBIN="${PYBIN:-python}"
PATHS_PROFILE="${PATHS_PROFILE:-3}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/search_gate_c2}"
IXI_CKPT="${IXI_CKPT:-results/P10_LONGRUN_VXM_UNIFIED_SVF_IXI/ckpt/best.pth}"
C1_EXPLORE_MANIFEST="${C1_EXPLORE_MANIFEST:-results/search_gate_c1/C1_EXPLORATION_20260821T121202Z_9275171a67a3/exploration/run_manifest.json}"
C1_EXPLORE_SHA256="${C1_EXPLORE_SHA256:-2e2084c66d33f165fb3a5995d05f535fb4b500e5d22072a4b811b1d388d94403}"
C1_CONFIRM_MANIFEST="${C1_CONFIRM_MANIFEST:-results/search_gate_c1/C1_CONFIRMATION_20260821T150103Z_9275171a67a3/confirmation/run_manifest.json}"
C1_CONFIRM_SHA256="${C1_CONFIRM_SHA256:-7ccf7b13f32c67b6821aeea31d742ade450d26d9f2dde56b286495479255defa}"
REMOTE_LOCATOR="${REMOTE_LOCATOR:-}"

sg_parse_gpu_list "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 || "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo "[FAIL] GPU_LIST must contain at least one unique GPU index" >&2
  exit 2
fi

sg_export_pythonpath
GIT_STATUS_AT_START="$(git status --porcelain=v1)"
if [[ -n "$GIT_STATUS_AT_START" ]]; then
  echo "[FAIL] Refusing to run from a dirty tree:" >&2
  printf '%s\n' "$GIT_STATUS_AT_START" >&2
  echo "[HINT] Redirect nohup output to /tmp/search_gate_c2.log, never into the repository root." >&2
  exit 1
fi
for path in "$C1_EXPLORE_MANIFEST" "$C1_CONFIRM_MANIFEST" "$IXI_CKPT"; do
  if [[ ! -f "$path" ]]; then
    echo "[FAIL] Required frozen input is missing: $path" >&2
    exit 1
  fi
done

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
STARTED_AT="$(sg_utc_started_at)"
RUN_ID="${RUN_ID:-C2_DEVELOPMENT_$(sg_utc_run_stamp)_$(sg_git_short_head)}"
RUN_ROOT="${OUT_ROOT}/${RUN_ID}"
ATTEMPT_ID="${ATTEMPT_ID:-A_$(sg_utc_run_stamp)_$$}"
if ! sg_is_safe_identifier "$RUN_ID" || ! sg_is_safe_identifier "$ATTEMPT_ID"; then
  echo "[FAIL] RUN_ID and ATTEMPT_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
PACKAGE_ABS="$(pwd)/results/exports/${RUN_ID}.tar.gz"
if [[ -z "$REMOTE_LOCATOR" ]]; then
  REMOTE_LOCATOR="$(sg_default_remote_locator "$PACKAGE_ABS")"
fi

mkdir -p "$RUN_ROOT/preflight" "$RUN_ROOT/attempts"
exec 9>"$RUN_ROOT/.run.lock"
if ! flock -n 9; then
  echo "[FAIL] Another C2 process already holds RUN_ID=$RUN_ID" >&2
  exit 1
fi
ATTEMPT_ROOT="$RUN_ROOT/attempts/$ATTEMPT_ID"
if [[ -e "$ATTEMPT_ROOT" ]]; then
  echo "[FAIL] ATTEMPT_ID already exists: $ATTEMPT_ROOT" >&2
  exit 1
fi
mkdir -p "$ATTEMPT_ROOT/preflight"

GPU_CANONICAL="$(IFS=,; printf '%s' "${GPUS[*]}")"
{
  printf 'gpu_list=%s\n' "$GPU_CANONICAL"
  printf 'git_head=%s\n' "$HEAD"
  printf 'paths_profile=%s\n' "$PATHS_PROFILE"
  printf 'seed=%s\n' "$SEED"
  printf 'checkpoint=%s\n' "$IXI_CKPT"
  printf 'c1_exploration_manifest=%s\n' "$C1_EXPLORE_MANIFEST"
  printf 'c1_exploration_sha256=%s\n' "$C1_EXPLORE_SHA256"
  printf 'c1_confirmation_manifest=%s\n' "$C1_CONFIRM_MANIFEST"
  printf 'c1_confirmation_sha256=%s\n' "$C1_CONFIRM_SHA256"
} > "$ATTEMPT_ROOT/runner_contract.txt"
if [[ ! -f "$RUN_ROOT/runner_contract.txt" ]]; then
  cp "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"
elif ! cmp -s "$ATTEMPT_ROOT/runner_contract.txt" "$RUN_ROOT/runner_contract.txt"; then
  echo "[FAIL] Resume settings differ from the original RUN_ID contract" >&2
  exit 1
fi

STARTED_FILE="$RUN_ROOT/started_at_utc.txt"
if [[ -f "$STARTED_FILE" ]]; then STARTED_AT="$(<"$STARTED_FILE")"; else printf '%s\n' "$STARTED_AT" > "$STARTED_FILE"; fi
{
  printf '#!/usr/bin/env bash\n'
  printf 'cd %q\n' "$(pwd)"
  printf 'GPU_LIST=%q PYBIN=%q PATHS_PROFILE=%q SEED=%q OUT_ROOT=%q RUN_ID=%q ATTEMPT_ID=%q ' \
    "$GPU_LIST" "$PYBIN" "$PATHS_PROFILE" "$SEED" "$OUT_ROOT" "$RUN_ID" "$ATTEMPT_ID"
  printf 'IXI_CKPT=%q C1_EXPLORE_MANIFEST=%q C1_EXPLORE_SHA256=%q ' \
    "$IXI_CKPT" "$C1_EXPLORE_MANIFEST" "$C1_EXPLORE_SHA256"
  printf 'C1_CONFIRM_MANIFEST=%q C1_CONFIRM_SHA256=%q REMOTE_LOCATOR=%q bash %q\n' \
    "$C1_CONFIRM_MANIFEST" "$C1_CONFIRM_SHA256" "$REMOTE_LOCATOR" "$0"
} > "$ATTEMPT_ROOT/commands.sh"
git status --porcelain=v1 > "$ATTEMPT_ROOT/git_status.txt"
{
  "$PYBIN" -VV
  "$PYBIN" -m pip freeze || echo "[WARN] pip freeze unavailable"
  nvidia-smi || true
} > "$ATTEMPT_ROOT/environment.txt" 2>&1
for artifact in commands.sh git_status.txt environment.txt; do
  if [[ ! -f "$RUN_ROOT/$artifact" ]]; then cp "$ATTEMPT_ROOT/$artifact" "$RUN_ROOT/$artifact"; fi
done
{
  printf 'archive=%s\n' "$PACKAGE_ABS"
  printf 'sidecar=%s.sha256\n' "$PACKAGE_ABS"
} > "$RUN_ROOT/package_location.txt"

STATUS="FAILED"
EXIT_CODE=1
EXPECTED_PREFLIGHTS=1
PIDS=()

package_and_finalize() {
  local trap_code=$?
  trap - EXIT
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  PIDS=()
  find "$RUN_ROOT/cases" -mindepth 2 -maxdepth 2 -type d -name work -exec rm -rf -- {} + 2>/dev/null || true
  local completed_at package
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$trap_code" -eq 0 ]]; then STATUS="COMPLETE"; EXIT_CODE=0; else EXIT_CODE="$trap_code"; fi
  git status --porcelain=v1 > "$ATTEMPT_ROOT/final_git_status.txt" 2>&1 || true
  if [[ ! -f "$RUN_ROOT/datasets.tsv" ]]; then
    printf 'dataset\tsplit\tcase_id\tpath\tbytes\tsha256\tmtime_utc\n' > "$RUN_ROOT/datasets.tsv"
  fi
  mkdir -p "$ATTEMPT_ROOT/prior_root_finalization"
  for artifact in run_manifest.json outputs.tsv; do
    if [[ -f "$RUN_ROOT/$artifact" ]]; then mv "$RUN_ROOT/$artifact" "$ATTEMPT_ROOT/prior_root_finalization/$artifact"; fi
  done
  "$PYBIN" -m tools.analysis.run_artifacts finalize \
    --run-root "$RUN_ROOT" --run-id "$RUN_ID" --status "$STATUS" --exit-code "$EXIT_CODE" \
    --started-at "$STARTED_AT" --completed-at "$completed_at" --git-head "$HEAD" --branch "$BRANCH" \
    --gpu-index "${GPUS[0]}" --mode development --paths-profile "$PATHS_PROFILE" --seed "$SEED" \
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
  echo "[COMPLETE] Send the compact package and its .sha256 sidecar. Test-115 was not accessed."
}
trap package_and_finalize EXIT

echo "[RUN ID] $RUN_ID"
echo "[RUN ROOT] $RUN_ROOT"
echo "[HEAD] $HEAD"
echo "[GPU LIST] $GPU_LIST"

"$PYBIN" -m tools.dev.check_transactional_search --output "$ATTEMPT_ROOT/transactional_selfcheck.json"
"$PYBIN" -m tools.analysis.run_search_gate_c2 selfcheck --output "$ATTEMPT_ROOT/c2_selfcheck.json"
cp "$ATTEMPT_ROOT/transactional_selfcheck.json" "$RUN_ROOT/transactional_selfcheck.json"
cp "$ATTEMPT_ROOT/c2_selfcheck.json" "$RUN_ROOT/c2_selfcheck.json"

"$PYBIN" -m tools.analysis.checkpoint_preflight \
  --checkpoint "$IXI_CKPT" \
  --ctcf-config CTCF-CascadeA-VM-Unified \
  --ctcf-l3-svf 1 \
  --time-steps 6 \
  --output "$ATTEMPT_ROOT/preflight/p10_ixi.json"
cp "$ATTEMPT_ROOT/preflight/p10_ixi.json" "$RUN_ROOT/preflight/p10_ixi.json"

"$PYBIN" -m tools.analysis.run_search_gate_c2 prepare \
  --run-root "$RUN_ROOT" \
  --paths-profile "$PATHS_PROFILE" \
  --checkpoint "$IXI_CKPT" \
  --seed "$SEED" \
  --num-shards "${#GPUS[@]}" \
  --physical-gpus "$GPU_LIST" \
  --c1-exploration-manifest "$C1_EXPLORE_MANIFEST" \
  --c1-exploration-sha256 "$C1_EXPLORE_SHA256" \
  --c1-confirmation-manifest "$C1_CONFIRM_MANIFEST" \
  --c1-confirmation-sha256 "$C1_CONFIRM_SHA256"
CONTRACT_SHA256="$(sha256sum "$RUN_ROOT/c2_contract.json" | awk '{print $1}')"

mkdir -p "$RUN_ROOT/workers/attempts/$ATTEMPT_ID"
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[$shard_index]}"
  log="$RUN_ROOT/workers/attempts/$ATTEMPT_ID/worker_$(printf '%02d' "$shard_index").log"
  (
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYBIN" -m tools.analysis.run_search_gate_c2 worker \
      --run-root "$RUN_ROOT" \
      --contract-sha256 "$CONTRACT_SHA256" \
      --shard-index "$shard_index" \
      --num-shards "${#GPUS[@]}" \
      --gpu 0 \
      --physical-gpu "$physical_gpu" \
      --attempt-id "$ATTEMPT_ID"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[WORKER] shard=$shard_index physical_gpu=$physical_gpu pid=$! log=$log"
done

worker_failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[FAIL] C2 shard $index failed; inspect its worker log." >&2
    worker_failed=1
  fi
done
PIDS=()
if [[ "$worker_failed" -ne 0 ]]; then exit 1; fi

"$PYBIN" -m tools.analysis.run_search_gate_c2 finalize \
  --run-root "$RUN_ROOT" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --attempt-id "$ATTEMPT_ID"

exit 0
