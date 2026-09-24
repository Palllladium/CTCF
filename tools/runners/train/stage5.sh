#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"

readonly PYBIN="${PYBIN:-python}"
readonly PHASE="${PHASE:-all}"
readonly GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
readonly OASIS_ALL_ROOT="${OASIS_ALL_ROOT:?Set OASIS_ALL_ROOT to the OASIS All394 directory. Test20 is forbidden.}"
readonly EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:?Set EXPECTED_GIT_HEAD to the exact committed Stage5 Git SHA.}"
readonly TRAINING_GIT_HEAD="${TRAINING_GIT_HEAD:-}"
readonly RUN_ID="${RUN_ID:?Set one stable RUN_ID and reuse it for every restart.}"
readonly REMOTE_HEAVY_LOCATOR="${REMOTE_HEAVY_LOCATOR:-PENDING_UPLOAD}"
readonly IMPORT_U0_RUN_ID="${IMPORT_U0_RUN_ID:-}"
readonly IMPORT_U0_COMPACT_ROOT="${IMPORT_U0_COMPACT_ROOT:-results/stage5/$IMPORT_U0_RUN_ID}"
readonly IMPORT_U0_HEAVY_ROOT="${IMPORT_U0_HEAVY_ROOT:-results/stage5_heavy/$IMPORT_U0_RUN_ID}"
readonly PRECISION_SOURCE_ROOT="${PRECISION_SOURCE_ROOT:-results/stage5_diagnostics/S5_PRECISIONDIAG_20260909T140904Z_3575584_cdede42ac60c}"
readonly PRECISION_CAPTURE_ROOT="${PRECISION_CAPTURE_ROOT:-results/stage5_heavy/S5_PRECISIONDIAG_20260909T140904Z_3575584_cdede42ac60c}"

readonly COMPACT_ROOT="${COMPACT_ROOT:-results/stage5/$RUN_ID}"
readonly HEAVY_ROOT="${HEAVY_ROOT:-results/stage5_heavy/$RUN_ID}"
readonly DATA_ROOT="${DATA_ROOT:-results/stage5_data}"
readonly MANIFEST_ROOT="$DATA_ROOT/manifests"
readonly IMAGE_ROOT="$DATA_ROOT/image_only"
readonly DATA_CONTRACT="$MANIFEST_ROOT/data_contract.json"
readonly PROTOCOL_ROOT="$COMPACT_ROOT/protocol"
readonly PROTOCOL="$PROTOCOL_ROOT/protocol.json"
readonly CHECKPOINT_ROOT="$HEAVY_ROOT/checkpoints"
readonly SOURCE_ROOT="$HEAVY_ROOT/source_fields"
readonly DECISION_ROOT="$HEAVY_ROOT/decisions"
readonly EVALUATION_ROOT="$COMPACT_ROOT/evaluation"
readonly BARRIER_ROOT="$COMPACT_ROOT/barriers"
readonly TRAINING_BARRIER="$BARRIER_ROOT/training_barrier.json"
readonly DECISION_BARRIER="$BARRIER_ROOT/decision_barrier.json"
readonly EVALUATION_BARRIER="$BARRIER_ROOT/evaluation_barrier.json"
readonly CONTINUATION="$COMPACT_ROOT/continuations/$EXPECTED_GIT_HEAD.json"
readonly SMOKE_REPORT="$COMPACT_ROOT/smoke/smoke_report.json"
readonly SMOKE_BARRIER="$BARRIER_ROOT/smoke_barrier.json"
readonly STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
readonly ATTEMPT_ID="A_$(date -u +%Y%m%dT%H%M%SZ)_$$"
readonly LOG_ROOT="$COMPACT_ROOT/logs/$ATTEMPT_ID"
readonly STATUS_ROOT="$COMPACT_ROOT/status/$ATTEMPT_ID"

readonly -a SEEDS=(0 1 2)
readonly -a VARIANTS=(F0 F2V F2S F2P F4P F24P A2P A24P)
readonly -a ALL_VARIANTS=(U0 F0 F2V F2S F2P F4P F24P A2P A24P)
ACTIVE_PIDS=()
if [[ ! "$GPU_LIST" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
  echo "[FAIL] GPU_LIST must contain one or more unique non-negative integer GPU indices." >&2
  exit 2
fi
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "[FAIL] GPU_LIST contains a duplicate GPU index: $gpu" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done
if [[ ! "$RUN_ID" =~ ^S5_[A-Z0-9]+_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}$ ]]; then
  echo "[FAIL] RUN_ID must be S5_<MODE>_<UTC>_<12-char-head>." >&2
  exit 2
fi
run_identity_head="$EXPECTED_GIT_HEAD"
if [[ "$PHASE" == "continue-evaluation" ]]; then
  if [[ ! "${TRAINING_GIT_HEAD:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[FAIL] continue-evaluation requires the exact TRAINING_GIT_HEAD." >&2
    exit 2
  fi
  run_identity_head="$TRAINING_GIT_HEAD"
elif [[ -n "${TRAINING_GIT_HEAD:-}" ]]; then
  echo "[FAIL] TRAINING_GIT_HEAD is only accepted for continue-evaluation." >&2
  exit 2
fi
if [[ "${RUN_ID##*_}" != "${run_identity_head:0:12}" ]]; then
  echo "[FAIL] RUN_ID suffix must match the execution HEAD, or TRAINING_GIT_HEAD for continue-evaluation." >&2
  exit 2
fi
case "$PHASE" in
  all|prepare|smoke|import-u0|train-u0|compare-precision|materialize-source|train-controller|decide|evaluate|package|continue-evaluation) ;;
  *) echo "[FAIL] Unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
if [[ -n "$IMPORT_U0_RUN_ID" ]]; then
  if [[ ! "$IMPORT_U0_RUN_ID" =~ ^S5_[A-Z0-9]+_[0-9]{8}T[0-9]{6}Z_(68df5b241042|ffd3090f6129)$ || "$IMPORT_U0_RUN_ID" == "$RUN_ID" ]]; then
    echo "[FAIL] U0 import requires a distinct source run from a supported U0 revision." >&2
    exit 2
  fi
elif [[ "$PHASE" == "import-u0" || "$PHASE" == "compare-precision" ]]; then
  echo "[FAIL] PHASE=$PHASE requires IMPORT_U0_RUN_ID." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$BARRIER_ROOT" "$HEAVY_ROOT"
exec 9>"$COMPACT_ROOT/stage5.lock"
if ! flock -n 9; then
  echo "[FAIL] Another process holds the Stage5 run lock: $COMPACT_ROOT/stage5.lock" >&2
  exit 3
fi

readonly -a GIT_ARGS=(--repo-root "$REPO_ROOT" --expected-git-head "$EXPECTED_GIT_HEAD")
PROTOCOL_ARGS=("${GIT_ARGS[@]}" --protocol "$PROTOCOL")
if [[ "$PHASE" == "continue-evaluation" ]]; then
  PROTOCOL_ARGS+=(--continuation "$CONTINUATION")
fi
readonly -a PROTOCOL_ARGS
readonly -a DATA_ARGS=(--data-contract "$DATA_CONTRACT" --image-root "$IMAGE_ROOT")

run_cli() {
  "$PYBIN" -m tools.analysis.run_stage5 "$@"
}

run_logged() {
  local log_file="$1"
  shift
  if [[ "$BASHPID" != "$$" ]]; then
    # A trapped TERM waits for the foreground worker to finish exiting before
    # this background shell is reaped and its logs are packaged.
    trap 'exit 143' TERM
  fi
  mkdir -p "$(dirname "$log_file")"
  echo "[START] $log_file"
  if "$@" >"$log_file" 2>&1; then
    echo "[PASS] $log_file"
  else
    local rc=$?
    echo "[FAIL] $log_file" >&2
    tail -n 80 "$log_file" >&2 || true
    return "$rc"
  fi
}

dependency_preflight() {
  "$PYBIN" -c 'import mamba_ssm, numpy, torch; print(f"[DEPENDENCY] numpy={numpy.__version__} torch={torch.__version__} mamba_ssm={mamba_ssm.__version__}")'
}

git_guard() {
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase data --target-root "$DATA_ROOT" >/dev/null
  local actual_head
  actual_head="$(git rev-parse HEAD)"
  if [[ "$actual_head" != "$EXPECTED_GIT_HEAD" ]]; then
    echo "[FAIL] Expected HEAD $EXPECTED_GIT_HEAD, found $actual_head" >&2
    return 1
  fi
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "[FAIL] Stage5 refuses a dirty Git tree." >&2
    git status --short >&2
    return 1
  fi
}

capture_provenance() {
  local attempt_root="$COMPACT_ROOT/attempts/$ATTEMPT_ID"
  mkdir -p "$attempt_root"
  {
    printf 'PHASE=%q ' "$PHASE"
    printf 'GPU_LIST=%q ' "$GPU_LIST"
    printf 'OASIS_ALL_ROOT=%q ' "$OASIS_ALL_ROOT"
    printf 'EXPECTED_GIT_HEAD=%q ' "$EXPECTED_GIT_HEAD"
    printf 'TRAINING_GIT_HEAD=%q ' "$TRAINING_GIT_HEAD"
    printf 'RUN_ID=%q ' "$RUN_ID"
    printf 'REMOTE_HEAVY_LOCATOR=%q ' "$REMOTE_HEAVY_LOCATOR"
    printf 'IMPORT_U0_RUN_ID=%q ' "$IMPORT_U0_RUN_ID"
    printf 'IMPORT_U0_COMPACT_ROOT=%q ' "$IMPORT_U0_COMPACT_ROOT"
    printf 'IMPORT_U0_HEAVY_ROOT=%q ' "$IMPORT_U0_HEAVY_ROOT"
    printf 'PRECISION_SOURCE_ROOT=%q ' "$PRECISION_SOURCE_ROOT"
    printf 'PRECISION_CAPTURE_ROOT=%q ' "$PRECISION_CAPTURE_ROOT"
    printf 'COMPACT_ROOT=%q ' "$COMPACT_ROOT"
    printf 'HEAVY_ROOT=%q ' "$HEAVY_ROOT"
    printf 'DATA_ROOT=%q ' "$DATA_ROOT"
    printf 'PYBIN=%q ' "$PYBIN"
    printf 'bash %q\n' "${BASH_SOURCE[0]}"
  } >"$attempt_root/commands.sh"
  {
    "$PYBIN" --version 2>&1
    "$PYBIN" -c 'import numpy; print(f"numpy={numpy.__version__}")'
    if "$PYBIN" -c 'import mamba_ssm, torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} mamba_ssm={mamba_ssm.__version__}")' 2>/dev/null; then
      :
    else
      echo "torch_or_mamba=NOT_REQUIRED_OR_UNAVAILABLE_FOR_THIS_PHASE"
    fi
    "$PYBIN" -m pip freeze --all
    uname -a
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  } >"$attempt_root/environment.txt"
  git rev-parse HEAD >"$attempt_root/git_head.txt"
  git branch --show-current >"$attempt_root/git_branch.txt"
  git status --porcelain=v1 --untracked-files=all >"$attempt_root/git_status.txt"
  {
    echo "checkpoint_root=$CHECKPOINT_ROOT"
    echo "source_field_root=$SOURCE_ROOT"
    echo "decision_output_root=$DECISION_ROOT"
    echo "image_root=$IMAGE_ROOT"
    echo "remote_locator=$REMOTE_HEAVY_LOCATOR"
    echo "retention_status=RETAIN_UNTIL_EXPLICIT_OPERATOR_DECISION"
  } >"$COMPACT_ROOT/heavy_retention.txt"
}

terminate_active_children() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}

# Poll only our worker PIDs, without requiring Bash 5.1 wait -n -p. A failed
# worker has finished its capture before it exits; stop its siblings immediately.
wait_for_batch() {
  local pid index
  local -a remaining=("$@")
  ACTIVE_PIDS=("$@")
  while [[ "${#remaining[@]}" -gt 0 ]]; do
    for index in "${!remaining[@]}"; do
      pid="${remaining[$index]}"
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      if wait "$pid"; then
        unset 'remaining[index]'
      else
        unset 'remaining[index]'
        ACTIVE_PIDS=("${remaining[@]}")
        terminate_active_children
        return 1
      fi
      ACTIVE_PIDS=("${remaining[@]}")
    done
    if [[ "${#remaining[@]}" -gt 0 ]]; then
      sleep 1
    fi
  done
  ACTIVE_PIDS=()
}

copy_compact_attestations() {
  mkdir -p \
    "$COMPACT_ROOT/data_attestations" \
    "$COMPACT_ROOT/training_attestations" \
    "$COMPACT_ROOT/source_attestations" \
    "$COMPACT_ROOT/decision"
  if [[ -d "$MANIFEST_ROOT" ]]; then
    local name
    for name in data_contract.json source_inventory.json split_manifest.json pair_manifest.json; do
      if [[ -f "$MANIFEST_ROOT/$name" ]]; then
        cp -f "$MANIFEST_ROOT/$name" "$COMPACT_ROOT/data_attestations/$name"
      fi
    done
  fi
  if [[ -d "$CHECKPOINT_ROOT" ]]; then
    while IFS= read -r -d '' path; do
      local relative="${path#"$CHECKPOINT_ROOT"/}"
      mkdir -p "$COMPACT_ROOT/training_attestations/$(dirname "$relative")"
      cp -f "$path" "$COMPACT_ROOT/training_attestations/$relative"
    done < <(find "$CHECKPOINT_ROOT" -type f \( \
      -name 'metrics.json' -o -name '*.sha256.json' -o -name 'failure.json' -o -name 'acknowledgement.json' \
      -o -path '*/telemetry/*/attempt.json' -o -path '*/telemetry/*/steps.jsonl' -o -path '*/telemetry/*/epochs.jsonl' \
    \) -print0)
  fi
  if [[ -d "$SOURCE_ROOT" ]]; then
    while IFS= read -r -d '' path; do
      local relative="${path#"$SOURCE_ROOT"/}"
      mkdir -p "$COMPACT_ROOT/source_attestations/$(dirname "$relative")"
      cp -f "$path" "$COMPACT_ROOT/source_attestations/$relative"
    done < <(find "$SOURCE_ROOT" -type f -name 'initial_report.json' -print0)
  fi
  if [[ -d "$DECISION_ROOT/records" ]]; then
    mkdir -p "$COMPACT_ROOT/decision/records"
    while IFS= read -r -d '' path; do
      cp -f "$path" "$COMPACT_ROOT/decision/records/$(basename "$path")"
    done < <(find "$DECISION_ROOT/records" -maxdepth 1 -type f -name '*.json' -print0)
  fi
  if [[ -d "$DECISION_ROOT/exact_reports" ]]; then
    mkdir -p "$COMPACT_ROOT/decision/exact_reports"
    while IFS= read -r -d '' path; do
      cp -f "$path" "$COMPACT_ROOT/decision/exact_reports/$(basename "$path")"
    done < <(find "$DECISION_ROOT/exact_reports" -maxdepth 1 -type f -name '*.json' -print0)
  fi
}

package_attempt() {
  local status="$1"
  local exit_code="$2"
  copy_compact_attestations
  local -a continuation_args=()
  if [[ "$PHASE" == "continue-evaluation" && -f "$CONTINUATION" ]]; then
    continuation_args=(--continuation "$CONTINUATION")
  fi
  run_cli finalize \
    "${GIT_ARGS[@]}" \
    "${continuation_args[@]}" \
    --run-root "$COMPACT_ROOT" \
    --run-id "$RUN_ID" \
    --attempt-id "$ATTEMPT_ID" \
    --status "$status" \
    --exit-code "$exit_code" \
    --started-at-utc "$STARTED_AT_UTC" \
    --remote-heavy-locator "$REMOTE_HEAVY_LOCATOR"
  local export_root="results/exports"
  run_cli package \
    "${GIT_ARGS[@]}" \
    --run-root "$COMPACT_ROOT" \
    --run-id "$RUN_ID" \
    --attempt-id "$ATTEMPT_ID" \
    --status "$status" \
    --export-root "$export_root"
  echo "[HEAVY RETAINED] $HEAVY_ROOT"
}

evaluation_products_complete() {
  local name
  for name in \
    evaluation_bundle.json \
    per_decision.csv \
    per_label.csv \
    geometry_metrics.csv \
    field_stage_diagnostics.csv \
    per_pair_metric.csv \
    planned_contrasts.csv \
    paired_effects_vs_u0.csv \
    decision_diagnostics.csv; do
    [[ -f "$EVALUATION_ROOT/products/$name" ]] || return 1
  done
}

FINALIZED=0
on_exit() {
  local rc=$?
  trap - EXIT
  terminate_active_children
  if [[ "$FINALIZED" -eq 0 ]]; then
    local status="PARTIAL"
    if [[ "$rc" -ne 0 ]]; then
      status="FAILED"
    elif [[ -f "$EVALUATION_BARRIER" ]] && evaluation_products_complete; then
      status="COMPLETE"
    fi
    package_attempt "$status" "$rc" || true
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT HUP
trap 'exit 143' TERM

prepare_phase() {
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase data --target-root "$DATA_ROOT"
  run_logged "$LOG_ROOT/prepare_data.log" run_cli prepare-data \
    "${GIT_ARGS[@]}" \
    --oasis-all-root "$OASIS_ALL_ROOT" \
    --manifest-root "$MANIFEST_ROOT" \
    --image-root "$IMAGE_ROOT"
  run_logged "$LOG_ROOT/prepare_protocol.log" run_cli prepare-protocol \
    "${GIT_ARGS[@]}" \
    --data-contract "$DATA_CONTRACT" \
    --output-root "$PROTOCOL_ROOT"
}

smoke_phase() {
  if [[ -f "$SMOKE_REPORT" && -f "$SMOKE_BARRIER" ]]; then
    run_cli freeze-smoke "${PROTOCOL_ARGS[@]}" --smoke-report "$SMOKE_REPORT" --output "$SMOKE_BARRIER"
    echo "[STAGE5 H100 SMOKE RESUME] $SMOKE_BARRIER"
    return 0
  fi
  if [[ -f "$SMOKE_REPORT" && ! -e "$SMOKE_BARRIER" ]]; then
    run_cli freeze-smoke "${PROTOCOL_ARGS[@]}" --smoke-report "$SMOKE_REPORT" --output "$SMOKE_BARRIER"
    echo "[STAGE5 H100 SMOKE RECOVERED] $SMOKE_BARRIER"
    return 0
  fi
  if [[ -e "$SMOKE_REPORT" || -e "$SMOKE_BARRIER" ]]; then
    echo "[FAIL] Partial Stage5 H100 smoke gate exists." >&2
    return 1
  fi
  run_logged "$LOG_ROOT/selfcheck.log" run_cli selfcheck
  local smoke_root="$HEAVY_ROOT/smoke/$ATTEMPT_ID"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" run_logged "$LOG_ROOT/h100_smoke.log" run_cli smoke \
    "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" \
    --output-root "$smoke_root" \
    --device cuda:0
  mkdir -p "$(dirname "$SMOKE_REPORT")"
  cp "$smoke_root/smoke_report.json" "$SMOKE_REPORT.part"
  mv "$SMOKE_REPORT.part" "$SMOKE_REPORT"
  run_cli freeze-smoke "${PROTOCOL_ARGS[@]}" --smoke-report "$SMOKE_REPORT" --output "$SMOKE_BARRIER"
}

import_u0_phase() {
  run_logged "$LOG_ROOT/import_u0.log" run_cli import-u0 \
    "${PROTOCOL_ARGS[@]}" \
    --source-protocol "$IMPORT_U0_COMPACT_ROOT/protocol/protocol.json" \
    --source-checkpoint-root "$IMPORT_U0_HEAVY_ROOT/checkpoints" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-manifest "$COMPACT_ROOT/imports/u0_import.json"
}

train_u0_phase() {
  local disk_phase="source"
  if [[ "$PHASE" == "compare-precision" ]]; then
    disk_phase="comparison"
  fi
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase "$disk_phase" --target-root "$HEAVY_ROOT"
  local -a pids=()
  local slot=0 seed
  # Seeds are logical experiment identities; GPU indices only choose worker devices.
  for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_logged "$LOG_ROOT/u0_seed_${seed}.log" run_cli train-u0 \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --seed "$seed" \
      --device cuda:0 &
    pids+=("$!")
    ACTIVE_PIDS=("${pids[@]}")
    slot=$((slot + 1))
    if [[ "$slot" -eq "${#GPUS[@]}" ]]; then
      wait_for_batch "${pids[@]}"
      pids=()
      slot=0
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_for_batch "${pids[@]}"
  fi
}

materialize_source_phase() {
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase source --target-root "$HEAVY_ROOT"
  local -a jobs=()
  local gpu_count="${#GPUS[@]}"
  local seed_count="${#SEEDS[@]}"
  local base=$((gpu_count / seed_count))
  local remainder=$((gpu_count % seed_count))
  local index seed shard count slot
  # Spread the GPUs over the declared seeds; the first `remainder` seeds get one extra
  # shard. Indexing by position, not by the seed value, keeps this correct if SEEDS changes.
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    count="$base"
    if [[ "$index" -lt "$remainder" ]]; then
      count=$((count + 1))
    fi
    if [[ "$count" -eq 0 ]]; then
      count=1
    fi
    for ((shard = 0; shard < count; shard++)); do
      jobs+=("$seed $shard $count")
    done
  done
  local -a pids=()
  local job
  slot=0
  for job in "${jobs[@]}"; do
    read -r seed shard count <<< "$job"
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_logged "$LOG_ROOT/source_s${seed}_${shard}of${count}.log" \
      run_cli materialize-source \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --source-root "$SOURCE_ROOT" \
      --seed "$seed" \
      --shard-index "$shard" \
      --num-shards "$count" \
      --device cuda:0 &
    pids+=("$!")
    ACTIVE_PIDS=("${pids[@]}")
    slot=$((slot + 1))
    if [[ "$slot" -eq "$gpu_count" ]]; then
      wait_for_batch "${pids[@]}"
      pids=()
      slot=0
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_for_batch "${pids[@]}"
  fi
}

# One wave: the variants at VARIANTS[start .. start+#GPUS-1], one per GPU, trained together.
# The GPU a slot uses is rotated by the seed so a slow device is not always paired with the
# same variant across the three seeds.
train_controller_wave() {
  local seed="$1" start="$2"
  local slot index variant physical_slot
  local -a pids=()
  for slot in "${!GPUS[@]}"; do
    index=$((start + slot))
    if [[ "$index" -ge "${#VARIANTS[@]}" ]]; then
      break
    fi
    variant="${VARIANTS[$index]}"
    physical_slot=$(((slot + seed) % ${#GPUS[@]}))
    CUDA_VISIBLE_DEVICES="${GPUS[$physical_slot]}" run_logged "$LOG_ROOT/controller_s${seed}_${variant}.log" \
      run_cli train-controller \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --seed "$seed" \
      --variant "$variant" \
      --device cuda:0 &
    pids+=("$!")
    ACTIVE_PIDS=("${pids[@]}")
  done
  wait_for_batch "${pids[@]}"
}

train_controller_phase() {
  local seed start
  for seed in "${SEEDS[@]}"; do
    run_cli init-controller "${PROTOCOL_ARGS[@]}" --checkpoint-root "$CHECKPOINT_ROOT" --seed "$seed"
  done
  for seed in "${SEEDS[@]}"; do
    echo "[STAGE5 CONTROLLER WAVE] seed=$seed"
    for ((start = 0; start < ${#VARIANTS[@]}; start += ${#GPUS[@]})); do
      train_controller_wave "$seed" "$start"
    done
  done
  run_cli freeze-training \
    "${PROTOCOL_ARGS[@]}" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output "$TRAINING_BARRIER"
}

compare_precision_phase() {
  prepare_phase
  # Import validates all completed U0 endpoints before the train-u0 command
  # adopts them. The comparison must never start a fresh 400-epoch U0 run.
  import_u0_phase
  train_u0_phase
  local seed
  for seed in "${SEEDS[@]}"; do
    run_cli init-controller "${PROTOCOL_ARGS[@]}" --checkpoint-root "$CHECKPOINT_ROOT" --seed "$seed"
  done
  run_logged "$LOG_ROOT/compare_precision.log" run_cli compare-precision \
    "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-root "$COMPACT_ROOT/comparison" \
    --heavy-root "$HEAVY_ROOT/comparison" \
    --run-id "$RUN_ID" \
    --gpu-list "$GPU_LIST" \
    --precision-source-root "$PRECISION_SOURCE_ROOT" \
    --capture-root "$PRECISION_CAPTURE_ROOT" &
  ACTIVE_PIDS=("$!")
  wait_for_batch "${ACTIVE_PIDS[@]}"
  echo "[PRECISION COMPARISON COMPLETE] $COMPACT_ROOT/comparison/summary.json"
  echo "[PRECISION COMPARISON STOP] No production training, decisions, or evaluation were started."
}

decision_worker() {
  trap 'exit 143' TERM
  local slot="$1"
  local queue_root="$2"
  local pending="$queue_root/pending"
  local claimed="$queue_root/claimed"
  local done_root="$queue_root/done"
  while true; do
    if compgen -G "$queue_root/FAILED.*" >/dev/null; then
      return 1
    fi
    local task="" candidate
    for candidate in "$pending"/*; do
      [[ -f "$candidate" ]] || continue
      task="${candidate##*/}"
      break
    done
    if [[ -z "$task" ]]; then
      return 0
    fi
    # Claim with atomic directory creation before moving the task. In particular,
    # concurrent mv on a vanished source is not a reliable lock on every platform.
    local claim_root="$claimed/$task"
    if ! mkdir "$claim_root" 2>/dev/null; then
      continue
    fi
    local claim="$claim_root/task.gpu${GPUS[$slot]}"
    if ! mv "$pending/$task" "$claim"; then
      echo "$task" >"$queue_root/FAILED.gpu${GPUS[$slot]}"
      return 1
    fi
    local seed variant
    read -r seed variant <"$claim"
    if ! CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_cli decide \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" \
      --training-barrier "$TRAINING_BARRIER" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --source-root "$SOURCE_ROOT" \
      --decision-root "$DECISION_ROOT" \
      --seed "$seed" \
      --variant "$variant" \
      --shard-index 0 \
      --num-shards 1 \
      --device cuda:0 >>"$LOG_ROOT/decision_gpu_${GPUS[$slot]}.log" 2>&1; then
      echo "$task" >"$queue_root/FAILED.gpu${GPUS[$slot]}"
      return 1
    fi
    mv "$claim" "$done_root/$task"
  done
}

decide_phase() {
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase full --target-root "$HEAVY_ROOT"
  local queue_root="$STATUS_ROOT/decision_queue"
  mkdir -p "$queue_root/pending" "$queue_root/claimed" "$queue_root/done"
  local index=0 seed variant
  for seed in "${SEEDS[@]}"; do
    for variant in "${ALL_VARIANTS[@]}"; do
      printf '%s %s\n' "$seed" "$variant" >"$queue_root/pending/$(printf '%03d' "$index")"
      index=$((index + 1))
    done
  done
  local -a pids=()
  local slot
  for slot in "${!GPUS[@]}"; do
    decision_worker "$slot" "$queue_root" &
    pids+=("$!")
    ACTIVE_PIDS=("${pids[@]}")
  done
  wait_for_batch "${pids[@]}"
  run_cli freeze-decision \
    "${PROTOCOL_ARGS[@]}" \
    --training-barrier "$TRAINING_BARRIER" \
    --source-root "$SOURCE_ROOT" \
    --decision-root "$DECISION_ROOT" \
    --output "$DECISION_BARRIER"
}

evaluate_phase() {
  local decision_sha
  decision_sha="$($PYBIN -c 'import json,sys; from tools.analysis.stage5.contracts import canonical_sha256; print(canonical_sha256(json.load(open(sys.argv[1], encoding="utf-8"))))' "$DECISION_BARRIER")"
  local -a pids=()
  local slot
  for slot in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_logged "$LOG_ROOT/evaluation_${slot}of${#GPUS[@]}.log" run_cli evaluate \
      "${PROTOCOL_ARGS[@]}" \
      --training-barrier "$TRAINING_BARRIER" \
      --decision-barrier "$DECISION_BARRIER" \
      --decision-barrier-sha256 "$decision_sha" \
      --data-contract "$DATA_CONTRACT" \
      --oasis-all-root "$OASIS_ALL_ROOT" \
      --source-root "$SOURCE_ROOT" \
      --decision-root "$DECISION_ROOT" \
      --evaluation-root "$EVALUATION_ROOT" \
      --shard-index "$slot" \
      --num-shards "${#GPUS[@]}" \
      --device cuda:0 &
    pids+=("$!")
    ACTIVE_PIDS=("${pids[@]}")
  done
  wait_for_batch "${pids[@]}"
  run_cli freeze-evaluation \
    "${PROTOCOL_ARGS[@]}" \
    --training-barrier "$TRAINING_BARRIER" \
    --decision-barrier "$DECISION_BARRIER" \
    --decision-barrier-sha256 "$decision_sha" \
    --data-contract "$DATA_CONTRACT" \
    --evaluation-root "$EVALUATION_ROOT" \
    --output "$EVALUATION_BARRIER"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" run_logged "$LOG_ROOT/aggregate.log" run_cli aggregate \
    "${PROTOCOL_ARGS[@]}" \
    --training-barrier "$TRAINING_BARRIER" \
    --decision-barrier "$DECISION_BARRIER" \
    --decision-barrier-sha256 "$decision_sha" \
    --data-contract "$DATA_CONTRACT" \
    --evaluation-barrier "$EVALUATION_BARRIER" \
    --source-root "$SOURCE_ROOT" \
    --decision-root "$DECISION_ROOT" \
    --evaluation-root "$EVALUATION_ROOT" \
    --output-root "$EVALUATION_ROOT/products" \
    --device cuda:0
}

continue_evaluation_phase() {
  run_logged "$LOG_ROOT/prepare_continuation.log" run_cli prepare-continuation \
    "${GIT_ARGS[@]}" --protocol "$PROTOCOL" "${DATA_ARGS[@]}" \
    --training-git-head "$TRAINING_GIT_HEAD" \
    --training-barrier "$TRAINING_BARRIER" \
    --checkpoint-root "$CHECKPOINT_ROOT" --source-root "$SOURCE_ROOT" \
    --decision-root "$DECISION_ROOT" --evaluation-root "$EVALUATION_ROOT" \
    --output "$CONTINUATION"
  decide_phase
  evaluate_phase
}

dependency_preflight
git_guard
capture_provenance
echo "[STAGE5] run_id=$RUN_ID phase=$PHASE head=$EXPECTED_GIT_HEAD"
if [[ "$PHASE" == "all" ]]; then
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase full --target-root "$HEAVY_ROOT"
fi

case "$PHASE" in
  prepare) prepare_phase ;;
  smoke) smoke_phase ;;
  import-u0) import_u0_phase ;;
  train-u0) train_u0_phase ;;
  compare-precision) compare_precision_phase ;;
  materialize-source) materialize_source_phase ;;
  train-controller) train_controller_phase ;;
  decide) decide_phase ;;
  evaluate) evaluate_phase ;;
  continue-evaluation) continue_evaluation_phase ;;
  package) ;;
  all)
    prepare_phase
    if [[ -n "$IMPORT_U0_RUN_ID" ]]; then
      import_u0_phase
    fi
    train_u0_phase
    train_controller_phase
    materialize_source_phase
    decide_phase
    evaluate_phase
    ;;
esac

FINALIZED=1
status="PARTIAL"
if [[ -f "$EVALUATION_BARRIER" ]] && evaluation_products_complete; then
  status="COMPLETE"
fi
package_attempt "$status" 0
echo "[$status] Stage5 phase $PHASE finished. Test20 members were not extracted, decoded, or evaluated."
