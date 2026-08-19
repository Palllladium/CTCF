#!/usr/bin/env bash
# Preregistered C0 search gate: self-check -> one OASIS smoke case -> fixed IXI validation-19.
set -euo pipefail

MODE="${MODE:-all}"
GPU="${GPU:-0}"
PYBIN="${PYBIN:-python}"
PATHS_PROFILE="${PATHS_PROFILE:-3}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/search_gate_c0}"
RUN_ID="${RUN_ID:-C0_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)}"
RUN_ROOT="${OUT_ROOT}/${RUN_ID}"
OAS_CKPT="${OAS_CKPT:-results/P16_W1_VXM_OASIS_LBL_DIG_J15/ckpt/best.pth}"
IXI_CKPT="${IXI_CKPT:-results/P10_LONGRUN_VXM_UNIFIED_SVF_IXI/ckpt/best.pth}"
REMOTE_LOCATOR="${REMOTE_LOCATOR:-PENDING_UPLOAD}"

case "$MODE" in
  selfcheck|smoke|development|all) ;;
  *) echo "[FAIL] MODE must be selfcheck, smoke, development, or all" >&2; exit 2 ;;
esac

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"

GIT_STATUS_AT_START="$(git status --porcelain=v1)"
if [[ -n "$GIT_STATUS_AT_START" ]]; then
  echo "[FAIL] Refusing to run from a dirty tree (tracked or untracked files):" >&2
  printf '%s\n' "$GIT_STATUS_AT_START" >&2
  exit 1
fi

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$RUN_ROOT/preflight"

{
  printf '#!/usr/bin/env bash\n'
  printf 'cd %q\n' "$(pwd)"
  printf 'MODE=%q GPU=%q PYBIN=%q PATHS_PROFILE=%q SEED=%q OUT_ROOT=%q RUN_ID=%q ' \
    "$MODE" "$GPU" "$PYBIN" "$PATHS_PROFILE" "$SEED" "$OUT_ROOT" "$RUN_ID"
  printf 'OAS_CKPT=%q IXI_CKPT=%q REMOTE_LOCATOR=%q bash %q\n' \
    "$OAS_CKPT" "$IXI_CKPT" "$REMOTE_LOCATOR" "$0"
} > "$RUN_ROOT/commands.sh"
git status --porcelain=v1 > "$RUN_ROOT/git_status.txt"
{
  "$PYBIN" -VV
  "$PYBIN" -m pip freeze || echo "[WARN] pip freeze unavailable"
  nvidia-smi || true
} > "$RUN_ROOT/environment.txt" 2>&1

STATUS="FAILED"
EXIT_CODE=1
EXPECTED_PREFLIGHTS=0

package_and_finalize() {
  local trap_code=$?
  trap - EXIT
  local completed_at package
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$trap_code" -eq 0 ]]; then STATUS="COMPLETE"; EXIT_CODE=0; else EXIT_CODE="$trap_code"; fi

  if [[ ! -f "$RUN_ROOT/datasets.tsv" ]]; then
    printf 'dataset\tsplit\tcase_id\tpath\tbytes\tsha256\tmtime_utc\n' > "$RUN_ROOT/datasets.tsv"
  fi
  "$PYBIN" -m tools.analysis.run_artifacts finalize \
    --run-root "$RUN_ROOT" --run-id "$RUN_ID" --status "$STATUS" --exit-code "$EXIT_CODE" \
    --started-at "$STARTED_AT" --completed-at "$completed_at" --git-head "$HEAD" --branch "$BRANCH" \
    --gpu-index "$GPU" --mode "$MODE" --paths-profile "$PATHS_PROFILE" --seed "$SEED" \
    --time-steps 6 --expected-preflights "$EXPECTED_PREFLIGHTS" --remote-locator "$REMOTE_LOCATOR" \
    || { STATUS="FAILED"; EXIT_CODE=1; }

  mkdir -p results/exports
  package="results/exports/${RUN_ID}.tar.gz"
  tar -czf "$package" -C "$OUT_ROOT" "$RUN_ID"
  sha256sum "$package" > "${package}.sha256"
  echo "[PACKAGE] $package"
  echo "[PACKAGE SHA-256] $(cat "${package}.sha256")"
  if [[ "$STATUS" != "COMPLETE" ]]; then
    echo "[FAILED] Send the compact package and its .sha256 sidecar for diagnosis." >&2
    exit "${EXIT_CODE:-1}"
  fi
  echo "[COMPLETE] Send the compact package and its .sha256 sidecar. Heavy fields were not packaged."
}
trap package_and_finalize EXIT

echo "[RUN ID] $RUN_ID"
echo "[RUN ROOT] $RUN_ROOT"
echo "[HEAD] $HEAD"
echo "[MODE] $MODE"
echo "########## C0.1 fail-closed self-check ##########"
"$PYBIN" -m tools.dev.check_transactional_search --output "$RUN_ROOT/selfcheck.json"

run_preflight() {
  local name="$1" checkpoint="$2"
  "$PYBIN" -m tools.analysis.checkpoint_preflight \
    --checkpoint "$checkpoint" \
    --ctcf-config CTCF-CascadeA-VM-Unified \
    --ctcf-l3-svf 1 \
    --time-steps 6 \
    --output "$RUN_ROOT/preflight/${name}.json"
}

run_stage() {
  local stage="$1" checkpoint="$2"
  "$PYBIN" -m tools.analysis.run_search_gate_c0 \
    --stage "$stage" \
    --run-root "$RUN_ROOT" \
    --gpu 0 \
    --paths-profile "$PATHS_PROFILE" \
    --checkpoint "$checkpoint" \
    --seed "$SEED"
}

if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  echo "########## C0.2 OASIS engineering smoke ##########"
  EXPECTED_PREFLIGHTS=$((EXPECTED_PREFLIGHTS + 1))
  run_preflight p16_j15 "$OAS_CKPT"
  run_stage smoke "$OAS_CKPT"
fi

if [[ "$MODE" == "development" || "$MODE" == "all" ]]; then
  echo "########## C0.3 fixed IXI-19 development gate ##########"
  EXPECTED_PREFLIGHTS=$((EXPECTED_PREFLIGHTS + 1))
  run_preflight p10_ixi "$IXI_CKPT"
  run_stage development "$IXI_CKPT"
fi

if [[ -f "$RUN_ROOT/smoke/datasets.tsv" && -f "$RUN_ROOT/development/datasets.tsv" ]]; then
  awk 'FNR == 1 && NR != 1 {next} {print}' \
    "$RUN_ROOT/smoke/datasets.tsv" "$RUN_ROOT/development/datasets.tsv" > "$RUN_ROOT/datasets.tsv"
elif [[ -f "$RUN_ROOT/smoke/datasets.tsv" ]]; then
  cp "$RUN_ROOT/smoke/datasets.tsv" "$RUN_ROOT/datasets.tsv"
elif [[ -f "$RUN_ROOT/development/datasets.tsv" ]]; then
  cp "$RUN_ROOT/development/datasets.tsv" "$RUN_ROOT/datasets.tsv"
fi

exit 0
