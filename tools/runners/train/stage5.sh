#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"

readonly PYBIN="${PYBIN:-python}"
readonly PHASE="${PHASE:-all}"
readonly GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
readonly OASIS_ALL_ROOT="${OASIS_ALL_ROOT:?Set OASIS_ALL_ROOT to the OASIS All394 directory. Test20 is forbidden.}"
readonly EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:?Set EXPECTED_GIT_HEAD to the exact committed Stage5 Git SHA.}"
readonly RUN_ID="${RUN_ID:?Set one stable RUN_ID and reuse it for every restart.}"
readonly REMOTE_HEAVY_LOCATOR="${REMOTE_HEAVY_LOCATOR:-PENDING_UPLOAD}"

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
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

if [[ ${#GPUS[@]} -lt ${#SEEDS[@]} || ${#GPUS[@]} -gt 8 ]]; then
  echo "[FAIL] Stage5 needs at least one GPU per seed (${#SEEDS[@]}) and at most eight." >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "[FAIL] GPU_LIST must contain three to eight unique non-negative integers." >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done
if [[ ! "$RUN_ID" =~ ^S5_[A-Z0-9]+_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}$ ]]; then
  echo "[FAIL] RUN_ID must be S5_<MODE>_<UTC>_<12-char-head>." >&2
  exit 2
fi
if [[ "${RUN_ID##*_}" != "${EXPECTED_GIT_HEAD:0:12}" ]]; then
  echo "[FAIL] RUN_ID suffix must equal the first 12 characters of EXPECTED_GIT_HEAD." >&2
  exit 2
fi
case "$PHASE" in
  all|prepare|smoke|train-u0|materialize-source|train-controller|decide|evaluate|package) ;;
  *) echo "[FAIL] Unknown PHASE=$PHASE" >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$BARRIER_ROOT" "$HEAVY_ROOT"
exec 9>"$COMPACT_ROOT/stage5.lock"
if ! flock -n 9; then
  echo "[FAIL] Another process holds the Stage5 run lock: $COMPACT_ROOT/stage5.lock" >&2
  exit 3
fi

readonly -a GIT_ARGS=(--repo-root "$REPO_ROOT" --expected-git-head "$EXPECTED_GIT_HEAD")
readonly -a PROTOCOL_ARGS=("${GIT_ARGS[@]}" --protocol "$PROTOCOL")
readonly -a DATA_ARGS=(--data-contract "$DATA_CONTRACT" --image-root "$IMAGE_ROOT")
readonly -a SMOKE_ARGS=(--smoke-barrier "$SMOKE_BARRIER" --smoke-report "$SMOKE_REPORT")

run_cli() {
  "$PYBIN" -m tools.analysis.run_stage5 "$@"
}

run_logged() {
  local log_file="$1"
  shift
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
    printf 'RUN_ID=%q ' "$RUN_ID"
    printf 'REMOTE_HEAVY_LOCATOR=%q ' "$REMOTE_HEAVY_LOCATOR"
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
    done < <(find "$CHECKPOINT_ROOT" -type f \( -name 'metrics.json' -o -name '*.sha256.json' \) -print0)
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
  run_cli finalize \
    "${GIT_ARGS[@]}" \
    --run-root "$COMPACT_ROOT" \
    --run-id "$RUN_ID" \
    --attempt-id "$ATTEMPT_ID" \
    --status "$status" \
    --exit-code "$exit_code" \
    --started-at-utc "$STARTED_AT_UTC" \
    --remote-heavy-locator "$REMOTE_HEAVY_LOCATOR"
  local export_root="results/exports"
  local archive_name="${RUN_ID}__${ATTEMPT_ID}__${status}.tar.gz"
  mkdir -p "$export_root"
  tar -czf "$export_root/.${archive_name}.part" -C "$(dirname "$COMPACT_ROOT")" "$(basename "$COMPACT_ROOT")"
  mv "$export_root/.${archive_name}.part" "$export_root/$archive_name"
  (cd "$export_root" && sha256sum "$archive_name" >"$archive_name.sha256")
  echo "[PACKAGE] $export_root/$archive_name"
  echo "[PACKAGE SIDECAR] $export_root/$archive_name.sha256"
  cat "$export_root/$archive_name.sha256"
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

train_u0_phase() {
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase source --target-root "$HEAVY_ROOT"
  local -a pids=()
  local slot seed
  # One GPU per declared seed. Do not reuse the slot index as the seed: the two only
  # coincide while SEEDS happens to be (0 1 2).
  for slot in "${!SEEDS[@]}"; do
    seed="${SEEDS[$slot]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_logged "$LOG_ROOT/u0_seed_${seed}.log" run_cli train-u0 \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" "${SMOKE_ARGS[@]}" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --seed "$seed" \
      --device cuda:0 &
    pids+=("$!")
  done
  ACTIVE_PIDS=("${pids[@]}")
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ACTIVE_PIDS=()
  [[ "$failed" -eq 0 ]]
}

materialize_source_phase() {
  run_cli disk-preflight "${GIT_ARGS[@]}" --phase source --target-root "$HEAVY_ROOT"
  local -a jobs=()
  local gpu_count="${#GPUS[@]}"
  local seed_count="${#SEEDS[@]}"
  local base=$((gpu_count / seed_count))
  local remainder=$((gpu_count % seed_count))
  local next_slot=0
  local index seed shard count slot
  # Spread the GPUs over the declared seeds; the first `remainder` seeds get one extra
  # shard. Indexing by position, not by the seed value, keeps this correct if SEEDS changes.
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    count="$base"
    if [[ "$index" -lt "$remainder" ]]; then
      count=$((count + 1))
    fi
    for ((shard = 0; shard < count; shard++)); do
      jobs+=("$seed $shard $count $next_slot")
      next_slot=$((next_slot + 1))
    done
  done
  local -a pids=()
  local job
  for job in "${jobs[@]}"; do
    read -r seed shard count slot <<< "$job"
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_logged "$LOG_ROOT/source_s${seed}_${shard}of${count}.log" \
      run_cli materialize-source \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" "${SMOKE_ARGS[@]}" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --source-root "$SOURCE_ROOT" \
      --seed "$seed" \
      --shard-index "$shard" \
      --num-shards "$count" \
      --device cuda:0 &
    pids+=("$!")
  done
  ACTIVE_PIDS=("${pids[@]}")
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ACTIVE_PIDS=()
  [[ "$failed" -eq 0 ]]
}

train_controller_phase() {
  local seed slot physical_slot variant start index
  for seed in "${SEEDS[@]}"; do
    run_cli init-controller "${PROTOCOL_ARGS[@]}" --checkpoint-root "$CHECKPOINT_ROOT" --seed "$seed"
  done
  for seed in "${SEEDS[@]}"; do
    echo "[STAGE5 CONTROLLER WAVE] seed=$seed"
    for ((start = 0; start < ${#VARIANTS[@]}; start += ${#GPUS[@]})); do
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
          "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" "${SMOKE_ARGS[@]}" \
          --checkpoint-root "$CHECKPOINT_ROOT" \
          --seed "$seed" \
          --variant "$variant" \
          --device cuda:0 &
        pids+=("$!")
      done
      ACTIVE_PIDS=("${pids[@]}")
      local failed=0
      for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
      done
      ACTIVE_PIDS=()
      [[ "$failed" -eq 0 ]]
    done
  done
  run_cli freeze-training \
    "${PROTOCOL_ARGS[@]}" "${SMOKE_ARGS[@]}" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output "$TRAINING_BARRIER"
}

decision_worker() {
  local slot="$1"
  local queue_root="$2"
  local pending="$queue_root/pending"
  local claimed="$queue_root/claimed"
  local done_root="$queue_root/done"
  while true; do
    local task
    task="$(find "$pending" -maxdepth 1 -type f -printf '%f\n' | sort | head -n 1)"
    if [[ -z "$task" ]]; then
      return 0
    fi
    local claim="$claimed/${task}.gpu${GPUS[$slot]}"
    if ! mv "$pending/$task" "$claim" 2>/dev/null; then
      continue
    fi
    local seed variant
    read -r seed variant <"$claim"
    if ! CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" run_cli decide \
      "${PROTOCOL_ARGS[@]}" "${DATA_ARGS[@]}" "${SMOKE_ARGS[@]}" \
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
  done
  ACTIVE_PIDS=("${pids[@]}")
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ACTIVE_PIDS=()
  [[ "$failed" -eq 0 ]]
  run_cli freeze-decision \
    "${PROTOCOL_ARGS[@]}" "${SMOKE_ARGS[@]}" \
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
      "${PROTOCOL_ARGS[@]}" "${SMOKE_ARGS[@]}" \
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
  done
  ACTIVE_PIDS=("${pids[@]}")
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ACTIVE_PIDS=()
  [[ "$failed" -eq 0 ]]
  run_cli freeze-evaluation \
    "${PROTOCOL_ARGS[@]}" "${SMOKE_ARGS[@]}" \
    --training-barrier "$TRAINING_BARRIER" \
    --decision-barrier "$DECISION_BARRIER" \
    --decision-barrier-sha256 "$decision_sha" \
    --data-contract "$DATA_CONTRACT" \
    --evaluation-root "$EVALUATION_ROOT" \
    --output "$EVALUATION_BARRIER"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" run_logged "$LOG_ROOT/aggregate.log" run_cli aggregate \
    "${PROTOCOL_ARGS[@]}" "${SMOKE_ARGS[@]}" \
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
  train-u0) train_u0_phase ;;
  materialize-source) materialize_source_phase ;;
  train-controller) train_controller_phase ;;
  decide) decide_phase ;;
  evaluate) evaluate_phase ;;
  package) ;;
  all)
    prepare_phase
    smoke_phase
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
