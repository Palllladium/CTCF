#!/usr/bin/env bash
# Reproducible ConvexAdam generation -> convention gate -> topology audit. One pair first, then all pairs.
set -euo pipefail

METHOD="${1:-convexadam}"
[[ "$METHOD" == "convexadam" ]] || { echo "supported method: convexadam" >&2; exit 2; }

PROFILE="${PROFILE:-2}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-1}"
FORCE="${FORCE:-0}"
PYBIN="${PYBIN:-python}"
if [[ -z "${REPAIR+x}" ]]; then
  [[ "$LIMIT" == "0" ]] && REPAIR=1 || REPAIR=0
fi
if [[ "$LIMIT" == "0" ]]; then
  TAG="$METHOD"
  EXPECTED=19
else
  TAG="${METHOD}_smoke${LIMIT}"
  EXPECTED="$LIMIT"
fi
OUT="results/external/${TAG}"
AUDIT="results/audit/${TAG}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
mkdir -p "$OUT" "$AUDIT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== external audit started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

force_flag=()
[[ "$FORCE" == "1" || "$LIMIT" != "0" ]] && force_flag=(--force)

echo "=== generate ${METHOD}: limit=${LIMIT} profile=${PROFILE} ==="
"$PYBIN" tools/analysis/generate_external_fields.py \
  --method "$METHOD" --paths "$PROFILE" --device "cuda:${GPU}" --limit "$LIMIT" --expected-count "$EXPECTED" \
  --out "$OUT" "${force_flag[@]}"

echo "=== audit canonical fields (float64 screen; exact report follows) ==="
"$PYBIN" tools/analysis/field_audit.py \
  --flows "$OUT/flows" --manifest "$OUT/manifest.json" --paths "$PROFILE" --source canonical --ds OASIS \
  --device "cuda:${GPU}" --repair "$REPAIR" --limit "$LIMIT" --eps 0.001 --work-eps 0.0011 --out "$AUDIT"

echo "=== exact predicate audit of original fields (predicate failure is a valid scientific result) ==="
set +e
"$PYBIN" -m utils.cert_exact --flow "$OUT/flows" --eps 0.001 --report "$AUDIT/exact_original.json"
original_rc=$?
set -e
if [[ "$original_rc" -ge 2 ]]; then
  echo "[FAIL] original exact audit was invalid or inconclusive" >&2
  exit 2
fi

if [[ "$REPAIR" == "1" ]]; then
  echo "=== exact predicate audit of repaired fields (all must pass) ==="
  "$PYBIN" -m utils.cert_exact --flow "$AUDIT/repaired_flows" --eps 0.001 \
    --report "$AUDIT/exact_repaired.json"
fi

if [[ "$LIMIT" != "0" ]]; then
  REPEAT_OUT="${OUT}_repeat"
  REPEAT_AUDIT="${AUDIT}_repeat"
  mkdir -p "$REPEAT_OUT" "$REPEAT_AUDIT"
  echo "=== controlled-CUDA repeat (bitwise identity is not claimed) ==="
  "$PYBIN" tools/analysis/generate_external_fields.py \
    --method "$METHOD" --paths "$PROFILE" --device "cuda:${GPU}" --limit "$LIMIT" --expected-count "$EXPECTED" \
    --out "$REPEAT_OUT" "${force_flag[@]}"
  "$PYBIN" tools/analysis/field_audit.py \
    --flows "$REPEAT_OUT/flows" --manifest "$REPEAT_OUT/manifest.json" --paths "$PROFILE" --source canonical \
    --ds OASIS --device "cuda:${GPU}" --repair 0 --limit "$LIMIT" --eps 0.001 --work-eps 0.0011 \
    --out "$REPEAT_AUDIT"
  set +e
  "$PYBIN" -m utils.cert_exact --flow "$REPEAT_OUT/flows" --eps 0.001 \
    --report "$REPEAT_AUDIT/exact_original.json"
  repeat_rc=$?
  set -e
  if [[ "$repeat_rc" -ge 2 ]]; then
    echo "[FAIL] repeated exact audit was invalid or inconclusive" >&2
    exit 2
  fi
  "$PYBIN" tools/analysis/compare_external_runs.py \
    --first-flows "$OUT/flows" --second-flows "$REPEAT_OUT/flows" \
    --first-manifest "$OUT/manifest.json" --second-manifest "$REPEAT_OUT/manifest.json" \
    --first-audit "$AUDIT/audit.csv" --second-audit "$REPEAT_AUDIT/audit.csv" \
    --first-exact "$AUDIT/exact_original.json" --second-exact "$REPEAT_AUDIT/exact_original.json" \
    --max-flow-max-delta 1.0 --max-flow-mean-delta 0.01 --max-dice-delta 0.001 \
    --max-fold-pct-delta 0.001 --max-bound-delta 0.01 --max-failure-relative-delta 0.05 \
    --report "$AUDIT/repeatability.json"
fi

if [[ "$REPAIR" == "1" ]]; then
  echo "PASS: return $OUT/{manifest.json,per_case.csv,run.log} and $AUDIT/{audit.csv,exact_original.json,exact_repaired.json}"
else
  echo "PASS: smoke complete; exact_repaired.json is created only by the full REPAIR=1 run"
fi
