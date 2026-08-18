#!/usr/bin/env bash
# Validate the recovered P18 and journal CTCF checkpoints under NATIVE_MANIFEST_V1.
# Heavy checkpoint bytes remain on the execution host; the exported package contains compact results only.
set -euo pipefail

GPU=0
MODE="all"
PATHS_PROFILE=3
PYBIN="${PYBIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash tools/runners/eval/validate_recovered_checkpoints.sh [options]

Options:
  --gpu N                 CUDA device index (default: 0)
  --mode all|p18|journal  Run both suites or one suite (default: all)
  --paths-profile N       Dataset path profile (default: 3)
  --python PATH           Python executable (default: $PYBIN or python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --paths-profile) PATHS_PROFILE="$2"; shift 2 ;;
    --python) PYBIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MODE" =~ ^(all|p18|journal)$ ]] || { echo "Invalid --mode: $MODE" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative integer" >&2; exit 2; }
[[ "$PATHS_PROFILE" =~ ^[0-9]+$ ]] || { echo "--paths-profile must be a positive integer" >&2; exit 2; }

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain=v1 --untracked-files=no)" ]]; then
  echo "Tracked files are dirty. Commit or revert them before a NATIVE_MANIFEST_V1 run." >&2
  git status --short --untracked-files=no >&2
  exit 3
fi

HEAD_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="RECOVERED_VALIDATION_${STAMP}_${HEAD_SHA:0:12}"
RUN_ROOT="results/validation/${RUN_ID}"
EXPORT_ROOT="results/exports"
PREFLIGHT_ROOT="$RUN_ROOT/preflight"
COMMANDS_FILE="$RUN_ROOT/commands.sh"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_COMPLETE=0

mkdir -p "$PREFLIGHT_ROOT" "$EXPORT_ROOT"
: > "$COMMANDS_FILE"

quote_command() {
  printf '%q ' "$@" >> "$COMMANDS_FILE"
  printf '\n' >> "$COMMANDS_FILE"
}

run_logged() {
  local label="$1"; shift
  quote_command "$@"
  echo "[RUN] $label"
  "$@" 2>&1 | tee "$RUN_ROOT/${label}.log"
}

write_dataset_manifest() {
  local -a cmd=(
    "$PYBIN" tools/analysis/run_artifacts.py dataset-manifest
    --paths-profile "$PATHS_PROFILE"
    --dataset-split OASIS:val
    --output "$RUN_ROOT/datasets.tsv"
  )
  [[ "$MODE" == "all" || "$MODE" == "journal" ]] && cmd+=(--dataset-split IXI:test)
  "${cmd[@]}"
}

capture_environment() {
  {
    printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'head=%s\n' "$HEAD_SHA"
    printf 'branch=%s\n' "$BRANCH"
    printf 'gpu=%s\n' "$GPU"
    printf 'mode=%s\n' "$MODE"
    printf 'paths_profile=%s\n' "$PATHS_PROFILE"
    printf 'python=%s\n' "$PYBIN"
    printf 'CTCF_DATA_DIR=%s\n' "${CTCF_DATA_DIR:-}"
    printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-}"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf '\n[python-version]\n'
    "$PYBIN" -V
    printf '\n[nvidia-smi-selected-gpu]\n'
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader -i "$GPU"
    printf '\n[python-packages]\n'
    "$PYBIN" -m pip freeze
  } > "$RUN_ROOT/environment.txt" 2>&1
  git status --porcelain=v1 --untracked-files=normal > "$RUN_ROOT/git_status.txt"
}

declare -A P18_SHA256=(
  [P18_ABL_VXM_OASIS_FULL]="11fe29ad9534dbe0a6f3c85b7a7838c0f38bd2b5b9c24218322c2c5fbd36920c"
  [P18_ABL_VXM_OASIS_NOICON]="ba435a83828f1d84a4ca0a11c3d910455957f5ed8b6db027f78cd9f9ad32acab"
  [P18_ABL_VXM_OASIS_NOJAC]="ed456f15fe1fa688333fcf1dd05289c6dc2276f14414595158afd0a075fa946a"
  [P18_ABL_VXM_OASIS_NOICON_NOJAC]="326a30d7dc632b2457326254ce988efa9ba2fe8877bf4423bba3193633cad8dd"
  [P18_ABL_VXM_OASIS_NOREG]="c4bb45c0d7eeaad0d84fb9e30ebf7c031f58211f9349b2180453c3427381c09d"
  [P18_ABL_VXM_OASIS_TRI_MEAN]="37864f2f3fb88756768e0c8d1108273b851660a6557bd1282e5b3797f153d42a"
  [P18_ABL_VXM_OASIS_TRI_ACTIVE]="e151c581a8345c58526cf09032b344b37d9844a71126f2c914fc15499b1f7311"
  [P18_ABL_VXM_OASIS_ICON_L2]="ce2240aa471c50a99aa963a4f301ac7799800a4706bc42f24d9cb4f534326e42"
  [P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.005]="9f7c3fc2d1826d2b9881213cc3cdd3a3d277860d6292b345a84d6dc718bce0e0"
  [P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.05]="bb7793105c8911de09e9ae2864baa9a4ba955a31ddbd058d2f3263eb53286946"
)

P18_EXPS=(
  P18_ABL_VXM_OASIS_FULL
  P18_ABL_VXM_OASIS_NOICON
  P18_ABL_VXM_OASIS_NOJAC
  P18_ABL_VXM_OASIS_NOICON_NOJAC
  P18_ABL_VXM_OASIS_NOREG
  P18_ABL_VXM_OASIS_TRI_MEAN
  P18_ABL_VXM_OASIS_TRI_ACTIVE
  P18_ABL_VXM_OASIS_ICON_L2
  P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.005
  P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.05
)

preflight() {
  local tag="$1" checkpoint="$2" config="$3" l3_svf="$4" expected_sha="$5"
  local -a cmd=(
    "$PYBIN" tools/analysis/checkpoint_preflight.py
    --checkpoint "$checkpoint"
    --ctcf-config "$config"
    --time-steps 6
    --output "$PREFLIGHT_ROOT/${tag}.json"
  )
  [[ -n "$l3_svf" ]] && cmd+=(--ctcf-l3-svf "$l3_svf")
  [[ -n "$expected_sha" ]] && cmd+=(--expected-sha256 "$expected_sha")
  run_logged "preflight_${tag}" "${cmd[@]}"
}

validate_result_dir() {
  local result_dir="$1" dataset="$2" split="$3"
  "$PYBIN" tools/analysis/run_artifacts.py validate-result \
    --datasets "$RUN_ROOT/datasets.tsv" \
    --result-dir "$result_dir" \
    --dataset "$dataset" \
    --split "$split"
}

aggregate_results() {
  local expected=24
  local -a patterns=(--summary-glob 'p18/*/summary.json' --summary-glob 'journal/*/summary.json')
  if [[ "$MODE" == "p18" ]]; then
    expected=20
    patterns=(--summary-glob 'p18/*/summary.json')
  elif [[ "$MODE" == "journal" ]]; then
    expected=4
    patterns=(--summary-glob 'journal/*/summary.json')
  fi
  "$PYBIN" tools/analysis/run_artifacts.py aggregate \
    --run-root "$RUN_ROOT" \
    "${patterns[@]}" \
    --expected-count "$expected" \
    --output "$RUN_ROOT/aggregate_metrics.csv"
}

finalize() {
  local exit_code=$?
  trap - EXIT
  set +e
  local status="FAILED"
  if [[ "$exit_code" -eq 0 && "$RUN_COMPLETE" -eq 1 ]]; then
    status="COMPLETE"
  fi
  local completed_at
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    printf 'status=%s\n' "$status"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'started_at_utc=%s\n' "$STARTED_AT"
    printf 'completed_at_utc=%s\n' "$completed_at"
  } > "$RUN_ROOT/completion.txt"

  local expected_preflights=12
  [[ "$MODE" == "p18" ]] && expected_preflights=10
  [[ "$MODE" == "journal" ]] && expected_preflights=2
  if ! "$PYBIN" tools/analysis/run_artifacts.py finalize \
    --run-root "$RUN_ROOT" \
    --run-id "$RUN_ID" \
    --status "$status" \
    --exit-code "$exit_code" \
    --started-at "$STARTED_AT" \
    --completed-at "$completed_at" \
    --git-head "$HEAD_SHA" \
    --branch "$BRANCH" \
    --gpu-index "$GPU" \
    --mode "$MODE" \
    --paths-profile "$PATHS_PROFILE" \
    --seed 0 \
    --time-steps 6 \
    --expected-preflights "$expected_preflights"
  then
    echo "[FINALIZATION FAILED] Could not build outputs.tsv/run_manifest.json" >&2
    status="FAILED"
    [[ "$exit_code" -ne 0 ]] || exit_code=90
    {
      printf 'status=%s\n' "$status"
      printf 'exit_code=%s\n' "$exit_code"
      printf 'started_at_utc=%s\n' "$STARTED_AT"
      printf 'completed_at_utc=%s\n' "$completed_at"
    } > "$RUN_ROOT/completion.txt"
  fi

  local package="$EXPORT_ROOT/${RUN_ID}.tar.gz"
  if ! tar -czf "$package.part" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")" \
    || ! mv "$package.part" "$package" \
    || ! sha256sum "$package" > "$package.sha256"; then
    echo "[FINALIZATION FAILED] Could not create or hash compact package" >&2
    [[ "$exit_code" -ne 0 ]] || exit_code=91
    status="FAILED"
  fi
  echo "[PACKAGE] $package"
  echo "[PACKAGE SHA-256] $(cat "$package.sha256")"
  if [[ "$status" == "COMPLETE" ]]; then
    echo "[COMPLETE] Upload the package and its .sha256 sidecar; do not upload checkpoints."
  else
    echo "[FAILED] Send the package and its .sha256 sidecar for diagnosis." >&2
  fi
  exit "$exit_code"
}
trap finalize EXIT

capture_environment
write_dataset_manifest

if [[ "$MODE" == "all" || "$MODE" == "p18" ]]; then
  for exp in "${P18_EXPS[@]}"; do
    preflight "$exp" "results/$exp/ckpt/best.pth" "CTCF-CascadeA-VM-Unified" 1 "${P18_SHA256[$exp]}"
  done
fi

if [[ "$MODE" == "all" || "$MODE" == "journal" ]]; then
  preflight "CTCF_UPD_OASIS_E500" "results/CTCF_UPD_OASIS_E500/ckpt/best.pth" "CTCF-CascadeA" "" ""
  preflight "CTCF_IXI_TUNED" "results/CTCF_IXI_TUNED/ckpt/best.pth" "CTCF-CascadeA" "" ""
fi

if [[ "$MODE" == "all" || "$MODE" == "p18" ]]; then
  run_logged "p18_eval" env \
    GPU="$GPU" PROFILE="--${PATHS_PROFILE}" PYBIN="$PYBIN" \
    OUT_ROOT="$RUN_ROOT/p18" FORCE=1 EPS=0.001 \
    bash tools/runners/train/loss_ablation_eval.sh
  for exp in "${P18_EXPS[@]}"; do
    validate_result_dir "$RUN_ROOT/p18/${exp}__FF" OASIS val
    validate_result_dir "$RUN_ROOT/p18/${exp}__REP" OASIS val
  done
fi

run_journal_eval() {
  local tag="$1" checkpoint="$2" dataset="$3" use_test="$4"
  local out="$RUN_ROOT/journal/$tag"
  local -a cmd=(
    "$PYBIN" -m experiments.inference
    --model ctcf
    --ckpt "$checkpoint"
    --ds "$dataset"
    "--${PATHS_PROFILE}"
    --gpu "$GPU"
    --strict_ckpt 1
    --deterministic
    --time_steps 6
    --ctcf_config CTCF-CascadeA
    --hd95
    --print_every 5
    --out_dir "$out"
  )
  [[ "$use_test" == "1" ]] && cmd+=(--use_test)
  run_logged "journal_${tag}" "${cmd[@]}"
  if [[ "$dataset" == "IXI" && "$use_test" == "1" ]]; then
    validate_result_dir "$out" IXI test
  else
    validate_result_dir "$out" "$dataset" val
  fi
}

if [[ "$MODE" == "all" || "$MODE" == "journal" ]]; then
  run_journal_eval "oas_ckpt_on_oasis" "results/CTCF_UPD_OASIS_E500/ckpt/best.pth" OASIS 0
  run_journal_eval "oas_ckpt_on_ixi" "results/CTCF_UPD_OASIS_E500/ckpt/best.pth" IXI 1
  run_journal_eval "ixi_ckpt_on_ixi" "results/CTCF_IXI_TUNED/ckpt/best.pth" IXI 1
  run_journal_eval "ixi_ckpt_on_oasis" "results/CTCF_IXI_TUNED/ckpt/best.pth" OASIS 0
fi

aggregate_results

if find "$RUN_ROOT" -type f \( -name '*.pth' -o -name '*.pth.tar' -o -name '*.npz' -o -name '*.png' \) \
  -print -quit | grep -q .; then
  echo "Heavy artifact unexpectedly appeared in compact run directory" >&2
  exit 4
fi

RUN_COMPLETE=1
