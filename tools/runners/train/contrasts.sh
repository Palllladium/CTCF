#!/usr/bin/env bash
# Paired IXI contrasts, frozen consensus and follow-up diagnostics.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-start}"
RUN_ID="${2:-B_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID"; exit 1; }
RUN="results/contrasts/$RUN_ID"
WORKFLOW="${WORKFLOW:-$(cat "$RUN/workflow.txt" 2>/dev/null || echo main)}"
[[ "$WORKFLOW" = main || "$WORKFLOW" = followup || "$WORKFLOW" = mechanism || "$WORKFLOW" = learning || "$WORKFLOW" = closure || "$WORKFLOW" = interventions ]] || { echo "Unknown WORKFLOW"; exit 1; }
if [ "$MODE" = stop ]; then
    [ -f "$RUN/runner.pid" ] || { echo "No matching live runner; no process was stopped."; exit 0; }
    read -r pid < "$RUN/runner.pid"
    if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "contrasts.sh _work $RUN_ID" &&
       [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ]; then
        kill -TERM -- "-$pid"
        echo "Stop requested; wait for the final package before resuming."
    else
        echo "No matching live runner; no process was stopped."
    fi
    exit 0
fi
: "${EXPECTED_GIT_HEAD:?Set the reviewed commit SHA}"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$EXPECTED_GIT_HEAD" ] && git diff --quiet HEAD -- || { echo "Unexpected/modified HEAD"; exit 1; }
if [ "$MODE" != _work ]; then
    case "$MODE" in
        start) [ ! -e "$RUN" ] || { echo "RUN_ID exists; use resume"; exit 1; } ;;
        resume) [ -f "$RUN/head.txt" ] || { echo "Unknown RUN_ID"; exit 1; } ;;
        *) echo "Usage: contrasts.sh [start|resume|stop] [RUN_ID]"; exit 1 ;;
    esac
    mkdir -p "$RUN"
    if [ -f "$RUN/workflow.txt" ]; then
        [ "$(cat "$RUN/workflow.txt")" = "$WORKFLOW" ] || { echo "Changed workflow"; exit 1; }
    else echo "$WORKFLOW" > "$RUN/workflow.txt"; fi
    LOG="$RUN/runner_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    nohup setsid bash tools/runners/train/contrasts.sh _work "$RUN_ID" > "$LOG" 2>&1 < /dev/null &
    echo "PID=$! RUN_ID=$RUN_ID LOG=$LOG"
    exit 0
fi
exec 9> "$RUN/.lock"
flock -n 9 || { echo "RUN_ID already active"; exit 1; }
echo "$$" > "$RUN/runner.pid"
if [ -f "$RUN/head.txt" ]; then
    [ "$(cat "$RUN/head.txt")" = "$HEAD" ] || { echo "RUN_ID belongs to another HEAD"; exit 1; }
else
    echo "$HEAD" > "$RUN/head.txt"
fi
PY="${PYBIN:-/data/mooncake/P/envs/ctcf/bin/python}"
PARENT="${PARENT_CHECKPOINT:-results/CALIBRATION_20261029T035516Z_78301dd6d80c_mamba294/ckpt/last.pth}"
DATA="${OASIS_ALL_ROOT:-/data/mooncake/P/OASIS_L2R_2021_task03/All}"
MANIFEST="${IXI_MANIFEST:-}"
read -r -a GPU <<< "${GPUS:-0 1 2 3}"
declare -A seen=()
for gpu in "${GPU[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && [ -z "${seen[$gpu]:-}" ] || { echo "Invalid/duplicate GPU: $gpu"; exit 1; }
    seen[$gpu]=1
done
[ "${#GPU[@]}" -gt 0 ] && [ -f "$PARENT" ] && [ -d "$DATA" ] || { echo "Missing GPU, parent or OASIS data"; exit 1; }
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOGS="$RUN/logs/$ATTEMPT"
mkdir -p "$LOGS" "$RUN/exit_codes"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/status.txt"; }
bind() {
    local digest
    digest="$(sha256sum "$2" | cut -d ' ' -f 1)" || return 1
    if [ -f "$RUN/$1.sha256" ]; then
        [ "$(cat "$RUN/$1.sha256")" = "$digest" ] || { echo "Changed $1"; return 1; }
    else echo "$digest" > "$RUN/$1.sha256"; fi
}
bind parent "$PARENT"
PIDS=()
REQUIRED=(pretests check parent b0 b1 b2 summary consensus)
if [ "$WORKFLOW" = followup ]; then REQUIRED=(pretests prepare profile controls anchor repair smooth summary repair_summary); fi
if [ "$WORKFLOW" = mechanism ]; then REQUIRED=(pretests prepare controls weights t2 smooth summary lumir_check lumir_profile lumir_train lumir_cert lumir_contrasts); fi
if [ "$WORKFLOW" = learning ]; then REQUIRED=(pretests preflight profile calibration smoke cost registrations teachers diagnostics students baselines summary); fi
if [ "$WORKFLOW" = closure ]; then REQUIRED=(pretests preflight targets recovery completion numerics stable_profile stable diagnostics certificates summary); fi
if [ "$WORKFLOW" = interventions ]; then REQUIRED=(pretests prepare recovery fields collect edits selection select supervision final hybrids summary); fi
finish() {
    local code=$? includes=() arm
    trap - EXIT INT
    trap '' TERM
    if [ "$(ps -o pgid= -p $$ | tr -d ' ')" = "$$" ]; then kill -TERM -- "-$$" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    for arm in "logs/${RUN_ID}"_* "results/${RUN_ID}"_* "results/lumir/${RUN_ID}"_*/attempts/* "results/lumir/${RUN_ID}"_*/certified_lumir25; do [ ! -d "$arm" ] || includes+=(--include-root "$arm"); done
    "$PY" -m tools.analysis.run_artifacts package --run-root "$RUN" --prefix "results/exports/${RUN_ID}__$ATTEMPT" \
        --git-head "$HEAD" --exit-code "$code" --label "CONTRASTS" \
        --required-steps "${REQUIRED[@]}" "${includes[@]}" || { [ "$code" != 0 ] || code=1; }
    rm -f "$RUN/runner.pid"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "RUN_ID=$RUN_ID HEAD=$HEAD GPUS=${GPU[*]} PID=$$ MANIFEST=$MANIFEST"
FREE_DEFAULT=200
if [ "$WORKFLOW" = followup ]; then FREE_DEFAULT=80; fi
if [ "$WORKFLOW" = mechanism ]; then FREE_DEFAULT=100; fi
if [ "$WORKFLOW" = learning ]; then FREE_DEFAULT=160; fi
if [ "$WORKFLOW" = interventions ]; then FREE_DEFAULT=800; fi
SPACE_CREDIT_ROOT=""
if [ "$WORKFLOW" = interventions ]; then SPACE_CREDIT_ROOT="$RUN"; fi
"$PY" -c 'import shutil,sys; from pathlib import Path; used=sum(p.stat().st_size for p in Path(sys.argv[2]).rglob("*") if p.is_file() and not p.is_symlink()) if sys.argv[2] else 0; sys.exit(shutil.disk_usage("results").free + used < int(sys.argv[1])*1024**3)' "${MIN_FREE_GIB:-$FREE_DEFAULT}" "$SPACE_CREDIT_ROOT"
job() {
    local task=$1 gpu=$2 code=0; shift 2
    if [[ "$task" != prepare && "$task" != preflight && "$task" != summary ]] && [ "$(cat "$RUN/exit_codes/$task.txt" 2>/dev/null || true)" = 0 ]; then log "SKIP $task (completed)"; return 0; fi
    log "START $task GPU=$gpu LOG=$LOGS/$task.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@" > "$LOGS/$task.log" 2>&1 || code=$?
    echo "$code" > "$RUN/exit_codes/$task.txt"
    if [ "$code" = 0 ]; then log "PASS $task"; else log "ERROR $task (exit $code)"; fi
    return "$code"
}
if [ "${PRETESTS:-1}" = 1 ]; then
    TESTS=(tools/analysis/tests/test_contrasts.py tools/analysis/tests/test_contrasts_runner.py tools/analysis/tests/test_precision.py)
    if [ "$WORKFLOW" = followup ]; then TESTS+=(tools/analysis/tests/test_contrast_checks.py tools/analysis/tests/test_repair_checks.py); fi
    if [ "$WORKFLOW" = mechanism ]; then TESTS+=(tools/analysis/tests/test_contrast_checks.py tools/analysis/tests/test_contrast_probes.py tools/analysis/tests/test_representation.py tools/analysis/tests/test_lumir_probes.py tools/analysis/tests/test_lumir.py tools/analysis/tests/test_distributed.py); fi
    if [ "$WORKFLOW" = learning ]; then TESTS=(tools/analysis/tests/test_matching.py tools/analysis/tests/test_matching_train.py tools/analysis/tests/test_matching_eval.py tools/analysis/tests/test_matching_diagnostics.py tools/analysis/tests/test_contrast_learning.py tools/analysis/tests/test_lpba.py tools/analysis/tests/test_learning_runner.py); fi
    if [ "$WORKFLOW" = closure ]; then TESTS=(tools/analysis/tests/test_ncc.py tools/analysis/tests/test_matching_eval.py tools/analysis/tests/test_contrast_audit.py tools/analysis/tests/test_closure_runner.py tools/analysis/tests/test_contrast_learning.py tools/analysis/tests/test_repair_checks.py tools/analysis/tests/test_precision.py::test_ncc_mode_cannot_silently_change_on_resume tools/analysis/tests/test_contrasts.py::test_native_trainer_executes_the_paired_step_with_a_fresh_optimizer); fi
    if [ "$WORKFLOW" = interventions ]; then TESTS=(tools/analysis/tests/test_weight_editing.py tools/analysis/tests/test_interventions_runner.py tools/analysis/tests/test_ncc.py tools/analysis/tests/test_contrasts.py::test_optional_representation_and_dose_preserve_legacy_resume_binding); fi
    job pretests "" "$PY" -m pytest -q "${TESTS[@]}" \
        tools/analysis/tests/test_run_artifacts.py::RunArtifactsTest::test_failed_run_can_be_packaged_without_site_packages
else echo 0 > "$RUN/exit_codes/pretests.txt"; log "SKIPPED pretests: PRETESTS=0"; fi
interventions() {
    local source="${SOURCE_RUN:-results/contrasts/BC_20261118T105250Z}"
    local external="${LPBA_MANIFEST:-/data/mooncake/P/LPBA40_prepared/manifest.json}"
    local status=0 code=0 seed pid fields_pid supervision_pid split lane slot final_suffix=""
    local -a common=(--out "$RUN" --manifest "$MANIFEST" --external_manifest "$external" --checkpoint "$PARENT" --source "$source" --history_run "${BM_SOURCE_RUN:-results/contrasts/BM_20261113T143420Z}" --oasis_root "$DATA") collect_pids=() selection_pids=() final_pids=() supervised=()
    [ "${#GPU[@]}" = 4 ] || { log "Interventions require four GPUs"; return 1; }
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    job prepare "" "$PY" -m experiments.weight_interventions prepare "${common[@]}" || return 1
    bind manifest "$MANIFEST" && bind lpba "$external" || return 1
    (
        code=0
        job recovery "${GPU[2]}" "$PY" -m experiments.weight_interventions recovery "${common[@]}" || code=1
        job fields "${GPU[2]}" "$PY" -m experiments.weight_interventions fields "${common[@]}" || code=1
        exit "$code"
    ) & fields_pid=$!
    (
        job dice_calibration "${GPU[3]}" "$PY" -m experiments.contrast_supervision calibrate --out "$RUN" --manifest "$MANIFEST" --checkpoint "$PARENT" --exp "${RUN_ID}_supervised" &&
        job dice_profile "${GPU[3]}" "$PY" -m experiments.contrast_supervision profile --out "$RUN" --manifest "$MANIFEST" --checkpoint "$PARENT" --exp "${RUN_ID}_supervised_profile" &&
        job supervision "${GPU[3]}" "$PY" -m experiments.contrast_supervision train --out "$RUN" --manifest "$MANIFEST" --checkpoint "$PARENT" --exp "${RUN_ID}_supervised"
    ) & supervision_pid=$!
    for seed in 0 1; do
        job "collect_s$seed" "${GPU[$seed]}" "$PY" -m experiments.weight_interventions collect "${common[@]}" --seed "$seed" & collect_pids+=($!)
    done
    code=0
    for pid in "${collect_pids[@]}"; do wait "$pid" || code=1; done
    echo "$code" > "$RUN/exit_codes/collect.txt"
    if [ "$code" = 0 ] && job edits "" "$PY" -m experiments.weight_interventions edits "${common[@]}"; then
        for seed in 0 1; do
            job "selection_s$seed" "${GPU[$seed]}" "$PY" -m experiments.weight_interventions selection "${common[@]}" --seed "$seed" & selection_pids+=($!)
        done
        code=0
        for pid in "${selection_pids[@]}"; do wait "$pid" || code=1; done
        echo "$code" > "$RUN/exit_codes/selection.txt"
        if [ "$code" != 0 ] || ! job select "" "$PY" -m experiments.weight_interventions select "${common[@]}"; then status=1; fi
    else status=1; fi
    if wait "$supervision_pid"; then
        supervised=(--supervised_checkpoint "results/${RUN_ID}_supervised/ckpt/last.pth")
        final_suffix="_with_s"
        job hybrids "${GPU[3]}" "$PY" -m experiments.weight_interventions hybrids "${common[@]}" || status=1
    else
        status=1
        log "HELD supervised readouts: calibration, memory profile or fitting failed; weight and field diagnostics continue"
        echo 125 > "$RUN/exit_codes/supervision.txt"
        echo 125 > "$RUN/exit_codes/hybrids.txt"
    fi
    if [ -f "$RUN/selection.json" ]; then
        for lane in 0 1 2; do
            slot=$lane; [ "$lane" != 2 ] || slot=3
            (
                code=0
                for split in sealed IXI LPBA OASIS; do
                    job "final_${split}_${lane}${final_suffix}" "${GPU[$slot]}" "$PY" -m experiments.weight_interventions final "${common[@]}" "${supervised[@]}" --split "$split" --lane "$lane" --lanes 3 || code=1
                done
                exit "$code"
            ) & final_pids+=($!)
        done
        code=0
        for pid in "${final_pids[@]}"; do wait "$pid" || code=1; done
        echo "$code" > "$RUN/exit_codes/final.txt"
        [ "$code" = 0 ] || status=1
    else
        echo 125 > "$RUN/exit_codes/final.txt"
        status=1
    fi
    wait "$fields_pid" || status=1
    job summary "" "$PY" -m experiments.weight_interventions summary "${common[@]}" || status=1
    return "$status"
}
if [ "$WORKFLOW" = interventions ]; then
    failed=0
    interventions || failed=1
    log "FINISHED exit=$failed"
    exit "$failed"
fi
closure() {
    local source="${SOURCE_RUN:-results/contrasts/B_LEARN_20261115T213527Z}"
    local base="${BASE_B1_CHECKPOINT:-results/BM_20261113T143420Z_b1_s1/ckpt/last.pth}"
    local b0="${BASE_B0_CHECKPOINT:-results/BM_20261113T143420Z_b0_s1/ckpt/last.pth}"
    local external="${LPBA_MANIFEST:-/data/mooncake/P/LPBA40_prepared/manifest.json}"
    local status=0 code=0 index lane count=2 train_lanes=1 train_slot=2 diagnostic_slot=3 pid
    local -a common=(--manifest "$MANIFEST" --source "$source" --checkpoint "$PARENT" --base_checkpoint "$base" --b0_checkpoint "$b0" --external_manifest "$external" --raw_root "${IXI_RAW_ROOT:-/data/mooncake/P/IXI_raw}" --threads "${PREP_THREADS:-32}") complete_pids=() other_pids=() train_pids=()
    [ "${#GPU[@]}" = 4 ] || [ "${#GPU[@]}" = 8 ] || { log "Closure requires 4 or 8 GPUs"; return 1; }
    if [ "${#GPU[@]}" = 8 ]; then count=4; train_lanes=3; train_slot=4; diagnostic_slot=7; fi
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    job preflight "" "$PY" -m experiments.contrast_audit preflight "${common[@]}" --out "$RUN" || return 1
    bind manifest "$MANIFEST" && bind lpba "$external" && bind base "$base" && bind b0 "$b0" || return 1
    for ((lane=0; lane<count; lane++)); do
        job "complete_${count}_$lane" "${GPU[$lane]}" "$PY" -m experiments.contrast_audit complete "${common[@]}" --out "$RUN" --lane "$lane" --lanes "$count" & complete_pids+=($!)
    done
    (
        code=0
        job teacher_audit "${GPU[$diagnostic_slot]}" "$PY" -m experiments.contrast_audit teachers "${common[@]}" --out "$RUN/teacher_audit" || code=1
        for ds in IXI LPBA OASIS; do
            case "$ds" in IXI) manifest="$MANIFEST" ;; LPBA) manifest="$external" ;; OASIS) manifest=datasets/splits/oasis_subject_disjoint_v2/pair_manifest.json ;; esac
            job "solver_$ds" "${GPU[$diagnostic_slot]}" "$PY" -m experiments.contrast_audit solvers "${common[@]}" --dataset "$ds" --manifest "$manifest" --oasis_root "$DATA" --out "$RUN/solver_curves/$ds" || code=1
        done
        for family in finetune scratch; do
            job "regional_$family" "${GPU[$diagnostic_slot]}" "$PY" -m experiments.contrast_audit regional "${common[@]}" --family "$family" --out "$RUN/regional/$family" || code=1
        done
        exit "$code"
    ) & other_pids+=($!)
    job recovery "${GPU[$train_slot]}" "$PY" -m experiments.contrast_audit recovery "${common[@]}" --out "$RUN/recovery" || status=1
    if job targets "" "$PY" -m experiments.contrast_audit targets "${common[@]}" --out "$RUN/targets"; then
        if job numerics "${GPU[$train_slot]}" "$PY" -m experiments.contrast_audit inputs "${common[@]}" --prepared "$RUN/targets" --out "$RUN/numerics" &&
           job stable_profile "${GPU[$train_slot]}" "$PY" -m experiments.contrast_audit profile "${common[@]}" --prepared "$RUN/targets" --out "$RUN/stable_profile"; then
            stable_arm() {
                local name=$1 gpu=$2 kind=${1%_s*} seed=${1##*_s} ds manifest certify code=0
                local weights="results/${RUN_ID}_$name/ckpt/last.pth" endpoint="$RUN/registration/$name/report.json"
                job "stable_$name" "$gpu" "$PY" -m experiments.contrast_learning train --manifest "$MANIFEST" --checkpoint "$PARENT" \
                    --prepared "$RUN/targets" --kind "$kind" --epochs 10 --seed "$seed" --ncc_mode stable \
                    --exp "${RUN_ID}_$name" --workers 8 --out "$RUN/registration/$name" || return 1
                job "raw_$name" "$gpu" "$PY" -m experiments.eval_matching evaluate --manifest "$MANIFEST" --base_checkpoint "$weights" \
                    --endpoint "$endpoint" --save_raw --out "$RUN/stable/raw/$name" || return 1
                job "cert_$name" "$gpu" "$PY" -m experiments.eval_matching evaluate --manifest "$MANIFEST" --base_checkpoint "$weights" \
                    --endpoint "$endpoint" --raw_from "$RUN/stable/raw/$name" --save_raw --certify 1 --out "$RUN/stable/IXI/$name" || code=1
                if [[ "$kind" = b0 || "$kind" = b1 ]]; then
                    for ds in LPBA OASIS; do
                        manifest="$external"; [ "$ds" != OASIS ] || manifest=datasets/splits/oasis_subject_disjoint_v2/pair_manifest.json
                        job "eval_${ds}_$name" "$gpu" "$PY" -m experiments.eval_matching evaluate --dataset "$ds" --manifest "$manifest" \
                            --oasis_root "$DATA" --base_checkpoint "$weights" --endpoint "$endpoint" --out "$RUN/stable/$ds/$name" || code=1
                    done
                fi
                return "$code"
            }
            for ((lane=0; lane<train_lanes; lane++)); do
                (
                    code=0; names=(b0_s0 b1_s0 normalised_t1_s0 resampled_t1_s0 shuffled_t2_s0 b0_s1 b1_s1)
                    for ((index=lane; index<${#names[@]}; index+=train_lanes)); do
                        stable_arm "${names[$index]}" "${GPU[$((train_slot+lane))]}" || code=1
                    done
                    exit "$code"
                ) & train_pids+=($!)
            done
        else
            log "HELD stable trainings: numerical audit or native training profile failed"
            echo 125 > "$RUN/exit_codes/stable.txt"; status=1
        fi
    else
        log "HELD stable trainings: targets unavailable"
        echo 125 > "$RUN/exit_codes/stable.txt"; status=1
    fi
    code=0; for pid in "${complete_pids[@]}"; do wait "$pid" || code=1; done
    echo "$code" > "$RUN/exit_codes/completion.txt"; [ "$code" = 0 ] || status=1
    code=0; for pid in "${other_pids[@]}"; do wait "$pid" || code=1; done
    echo "$code" > "$RUN/exit_codes/diagnostics.txt"; [ "$code" = 0 ] || status=1
    if [ "${#train_pids[@]}" -gt 0 ]; then
        code=0; for pid in "${train_pids[@]}"; do wait "$pid" || code=1; done
        echo "$code" > "$RUN/exit_codes/stable.txt"; [ "$code" = 0 ] || status=1
    fi
    job certificates "${GPU[0]}" "$PY" -m experiments.contrast_audit certificates "${common[@]}" \
        --completion_root "$RUN" --seqvp_root "${SEQVP_ROOT:-/data/mooncake/P/seq_vp_3d}" --out "$RUN/certificates" || status=1
    job summary "" "$PY" -m experiments.contrast_audit summary "${common[@]}" --out "$RUN" || status=1
    return "$status"
}
if [ "$WORKFLOW" = closure ]; then
    failed=0
    closure || failed=1
    log "FINISHED exit=$failed"
    exit "$failed"
fi
learning() {
    local source="${SOURCE_RUN:-results/contrasts/B_CHECK_20261110T211549Z}"
    local bm="${BM_SOURCE_RUN:-results/contrasts/BM_20261113T143420Z}"
    local base="${BASE_B1_CHECKPOINT:-results/BM_20261113T143420Z_b1_s1/ckpt/last.pth}"
    local b0="${BASE_B0_CHECKPOINT:-results/BM_20261113T143420Z_b0_s1/ckpt/last.pth}"
    local external="${LPBA_MANIFEST:-/data/mooncake/P/LPBA40_prepared/manifest.json}"
    local stride=1 code status=0 slot pid group name ds
    local -a common=(--manifest "$MANIFEST") arms=(base parent fine_b0 scratch_b0 scratch_b1 resampled_t1 shuffled_t2 t1_value t1_solver paired_value paired_solver paired_solver_small paired_solver_half) names weights
    [ "${#GPU[@]}" = 4 ] || [ "${#GPU[@]}" = 8 ] || { log "Learning dispatch requires 4 or 8 GPUs"; return 1; }
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    job preflight "" "$PY" -m experiments.contrast_learning preflight "${common[@]}" --checkpoint "$PARENT" \
        --source "$source" --bm_source "$bm" --base_checkpoint "$base" --b0_checkpoint "$b0" \
        --external_manifest "$external" --out "$RUN" || return 1
    bind manifest "$MANIFEST" && bind lpba "$external" && bind base "$base" && bind b0 "$b0" || return 1
    job profile_student_1 "${GPU[0]}" "$PY" -m experiments.train_matching profile "${common[@]}" \
        --kind student --mode solver --source paired --base_checkpoint "$base" --stride 1 \
        --out "$RUN/profile_student_1" --max_peak_gib 60 || return 1
    echo "$stride" > "$RUN/stride.txt"
    job profile_teacher "${GPU[0]}" "$PY" -m experiments.train_matching profile "${common[@]}" \
        --kind teacher --source paired --stride "$stride" --out "$RUN/profile_teacher" || return 1
    echo 0 > "$RUN/exit_codes/profile.txt"
    job calibration "${GPU[0]}" "$PY" -m experiments.matching_diagnostics calibrate "${common[@]}" \
        --stride 1 --out "$RUN/calibration" || return 1
    bind calibration "$RUN/calibration/calibration.json" || return 1
    job smoke_teacher "${GPU[0]}" "$PY" -m experiments.train_matching teacher "${common[@]}" \
        --source paired --stride "$stride" --updates 2 --save_every 2 --log_every 1 --out "$RUN/smoke/teacher" || return 1
    job smoke_student "${GPU[0]}" "$PY" -m experiments.train_matching student "${common[@]}" --source paired \
        --mode solver --stride "$stride" --updates 2 --save_every 2 --log_every 1 --teacher "$RUN/smoke/teacher/ckpt/last.pth" \
        --base_checkpoint "$base" --calibration "$RUN/calibration/calibration.json" --out "$RUN/smoke/student" || return 1
    job smoke_ixi "${GPU[0]}" "$PY" -m experiments.eval_matching evaluate --dataset IXI "${common[@]}" \
        --base_checkpoint "$base" --metric_checkpoint "$RUN/smoke/student/ckpt/last.pth" --certify 1 --limit 1 --out "$RUN/smoke/IXI" || return 1
    job smoke_lpba "${GPU[0]}" "$PY" -m experiments.eval_matching evaluate --dataset LPBA --manifest "$external" \
        --base_checkpoint "$base" --metric_checkpoint "$RUN/smoke/student/ckpt/last.pth" --limit 1 --out "$RUN/smoke/LPBA" || return 1
    job smoke_half "${GPU[0]}" "$PY" -m experiments.eval_matching evaluate --dataset IXI "${common[@]}" \
        --base_checkpoint "$base" --metric_checkpoint "$RUN/smoke/student/ckpt/last.pth" \
        --descriptor_stride 2 --limit 1 --out "$RUN/smoke/half" || return 1
    job smoke_oasis "${GPU[0]}" "$PY" -m experiments.eval_matching evaluate --dataset OASIS \
        --manifest datasets/splits/oasis_subject_disjoint_v2/pair_manifest.json --oasis_root "$DATA" \
        --base_checkpoint "$base" --metric_checkpoint "$RUN/smoke/student/ckpt/last.pth" --limit 1 --out "$RUN/smoke/OASIS" || return 1
    echo 0 > "$RUN/exit_codes/smoke.txt"
    job ctcf_profile "${GPU[0]}" "$PY" -m experiments.contrast_diagnostics profile "${common[@]}" --checkpoint "$PARENT" \
        --source "$source" --out "$RUN/ctcf_profile" --exp "${RUN_ID}_profile" --workers 8 --benchmark 1 || return 1
    job cost "" "$PY" -m experiments.contrast_learning cost --out "$RUN" || return 1
    job prepare "" "$PY" -m experiments.contrast_learning prepare "${common[@]}" --checkpoint "$PARENT" \
        --source "$source" --raw-root "${IXI_RAW_ROOT:-/data/mooncake/P/IXI_raw}" --threads "${PREP_THREADS:-32}" --out "$RUN/targets" || status=1
    evaluate_learning() {
        local name=$1 gpu=$2 weights=$3 metric=${4:-} endpoint=${5:-} ds manifest certify controls=0 failed=0
        local -a options=(--base_checkpoint "$weights")
        [ -z "$metric" ] || options+=(--metric_checkpoint "$metric")
        [ -z "$endpoint" ] || options+=(--endpoint "$endpoint")
        [ "$name" != paired_solver_half ] || options+=(--descriptor_stride 2)
        [ "$name" != paired_solver ] || controls=1
        for ds in IXI LPBA OASIS; do
            certify=0
            case "$ds" in
                IXI) manifest="$MANIFEST"; certify=1 ;;
                LPBA) manifest="$external" ;;
                OASIS) manifest=datasets/splits/oasis_subject_disjoint_v2/pair_manifest.json ;;
            esac
            job "eval_${ds}_$name" "$gpu" "$PY" -m experiments.eval_matching evaluate --dataset "$ds" --manifest "$manifest" \
                --oasis_root "$DATA" "${options[@]}" --out "$RUN/$ds/$name" --certify "$certify" \
                --controls "$([ "$ds" = IXI ] && echo "$controls" || echo 0)" || failed=1
        done
        return "$failed"
    }
    registration_learning() {
        local name=$1 gpu=$2
        if [[ "$name" != scratch_* ]] && [ "$(cat "$RUN/exit_codes/prepare.txt" 2>/dev/null)" != 0 ]; then
            log "HELD $name: auxiliary targets unavailable"; echo 125 > "$RUN/exit_codes/$name.txt"; return 1
        fi
        job "$name" "$gpu" "$PY" -m experiments.contrast_learning train "${common[@]}" --checkpoint "$PARENT" \
            --prepared "$RUN/targets" --kind "$name" --exp "${RUN_ID}_$name" --workers 8 --out "$RUN/registration/$name" || return 1
        evaluate_learning "$name" "$gpu" "results/${RUN_ID}_$name/ckpt/last.pth" "" "$RUN/registration/$name/report.json"
    }
    matching_learning() {
        local name=$1 gpu=$2 family=${1%_*} mode=${1##*_} width=16
        if [ "$name" = paired_solver_small ]; then family=paired; mode=solver; width=8; fi
        job "$name" "$gpu" "$PY" -m experiments.train_matching student "${common[@]}" --source "$family" --mode "$mode" \
            --teacher "$RUN/teachers/$family/ckpt/last.pth" --base_checkpoint "$base" --stride "$stride" --width "$width" \
            --calibration "$RUN/calibration/calibration.json" --out "$RUN/matching/$name" || return 1
        evaluate_learning "$name" "$gpu" "$base" "$RUN/matching/$name/ckpt/last.pth"
    }
    family_learning() {
        local family=$1 first=$2 second=$3 code=0 child
        job "teacher_$family" "$first" "$PY" -m experiments.train_matching teacher "${common[@]}" \
            --source "$family" --stride "$stride" --out "$RUN/teachers/$family" || return 1
        job "diagnostics_$family" "$first" "$PY" -m experiments.matching_diagnostics teacher "${common[@]}" \
            --teacher "$RUN/teachers/$family/ckpt/last.pth" --base_checkpoint "$base" \
            --out "$RUN/teacher_diagnostics/$family" || code=1
        if [ "$first" = "$second" ]; then
            matching_learning "${family}_value" "$first" || code=1
            matching_learning "${family}_solver" "$first" || code=1
        else
            matching_learning "${family}_value" "$first" & child=$!
            matching_learning "${family}_solver" "$second" || code=1
            wait "$child" || code=1
        fi
        if [ "$family" = paired ]; then
            matching_learning paired_solver_small "$first" || code=1
            if [ "$(cat "$RUN/exit_codes/paired_solver.txt" 2>/dev/null || true)" = 0 ]; then
                evaluate_learning paired_solver_half "$first" "$base" "$RUN/matching/paired_solver/ckpt/last.pth" || code=1
            else
                log "HELD paired_solver_half: trained main checkpoint unavailable"
            fi
        fi
        return "$code"
    }
    PIDS=()
    for slot in 0 1; do
        ( names=(scratch_b0 scratch_b1); registration_learning "${names[$slot]}" "${GPU[$slot]}" ) & PIDS+=($!)
    done
    if [ "${#GPU[@]}" = 8 ]; then
        registration_learning resampled_t1 "${GPU[2]}" & PIDS+=($!)
        registration_learning shuffled_t2 "${GPU[3]}" & PIDS+=($!)
        family_learning t1 "${GPU[4]}" "${GPU[5]}" & PIDS+=($!)
        family_learning paired "${GPU[6]}" "${GPU[7]}" & PIDS+=($!)
    else
        ( code=0; registration_learning resampled_t1 "${GPU[2]}" || code=1; family_learning t1 "${GPU[2]}" "${GPU[2]}" || code=1; exit "$code" ) & PIDS+=($!)
        ( code=0; registration_learning shuffled_t2 "${GPU[3]}" || code=1; family_learning paired "${GPU[3]}" "${GPU[3]}" || code=1; exit "$code" ) & PIDS+=($!)
    fi
    for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
    for group in registrations teachers diagnostics students; do
        code=0
        case "$group" in
            registrations) names=(scratch_b0 scratch_b1 resampled_t1 shuffled_t2) ;;
            teachers) names=(teacher_t1 teacher_paired) ;;
            diagnostics) names=(diagnostics_t1 diagnostics_paired) ;;
            students) names=(t1_value t1_solver paired_value paired_solver paired_solver_small) ;;
        esac
        for name in "${names[@]}"; do
            [ "$(cat "$RUN/exit_codes/$name.txt" 2>/dev/null || true)" = 0 ] || code=1
            if [ "$group" = students ] || [ "$group" = registrations ]; then
                for ds in IXI LPBA OASIS; do
                    [ "$(cat "$RUN/exit_codes/eval_${ds}_$name.txt" 2>/dev/null || true)" = 0 ] || code=1
                done
            fi
        done
        if [ "$group" = students ]; then
            for ds in IXI LPBA OASIS; do
                [ "$(cat "$RUN/exit_codes/eval_${ds}_paired_solver_half.txt" 2>/dev/null || true)" = 0 ] || code=1
            done
        fi
        echo "$code" > "$RUN/exit_codes/$group.txt"
    done
    PIDS=()
    for slot in 0 1 2; do
        ( names=(base parent fine_b0); weights=("$base" "$PARENT" "$b0"); evaluate_learning "${names[$slot]}" "${GPU[$slot]}" "${weights[$slot]}" ) & PIDS+=($!)
    done
    code=0
    for pid in "${PIDS[@]}"; do wait "$pid" || code=1; done
    echo "$code" > "$RUN/exit_codes/baselines.txt"
    [ "$code" = 0 ] || status=1
    job learning_summary "" "$PY" -m experiments.contrast_learning report --source "$source" --evaluations_root "$RUN/IXI" --out "$RUN" || status=1
    job summary "" "$PY" -m experiments.eval_matching report --out "$RUN" --required_arms "${arms[@]}" || status=1
    return "$status"
}
if [ "$WORKFLOW" = learning ]; then
    failed=0
    learning || failed=1
    log "FINISHED exit=$failed"
    exit "$failed"
fi
mechanism() {
    local source="${SOURCE_RUN:-results/contrasts/B_CHECK_20261110T211549Z}" original="${ORIGINAL_RUN:-results/contrasts/B_20261108T180120Z}"
    local -a common=(--source "$source" --original "$original" --manifest "$MANIFEST" --checkpoint "$PARENT" --out "$RUN")
    local status=0 pid lumir_pid="" slot
    [ "${#GPU[@]}" = 4 ] || [ "${#GPU[@]}" = 8 ] || { echo "Mechanism dispatch requires 4 or 8 GPUs"; return 1; }
    bind manifest "$MANIFEST" || return 1
    job prepare "" "$PY" -m experiments.contrast_probes prepare "${common[@]}" || return 1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
    lumir_lane() {
        local devices=$1 first=$2 code=0 resume=0 exp="${RUN_ID}_mind" raw
        raw="${LUMIR_RAW_CHECKPOINT:-results/CALIBRATION_20261029T035516Z_78301dd6d80c_lumir_control/ckpt/last.pth}"
        [ -f "$raw" ] || { log "HELD LUMIR: raw reference checkpoint missing"; return 1; }
        local -a options=(PYBIN="$PY" GPUS="$devices" GLOBAL_BATCH=4 BATCH_SIZE=1 PRECISION=bf16 CONFIG=CTCF-CascadeA-Mamba
            MODEL="--l3_svf 0 --input_representation mind6" TRAIN="--lr 0.0001 --seed 0" TIME_STEPS=6 PAIRS=lumir25 PRETESTS=0)
        job lumir_check "$first" "$PY" -m experiments.lumir_probes check --out "$RUN/lumir_check_$ATTEMPT" || return 1
        job lumir_profile "$first" env "${options[@]}" PROFILE=1 RESUME=0 PROFILE_UPDATES=20 PERF_WARMUP_UPDATES=5 \
            PROFILE_EXP="${RUN_ID}_mind_profile_$ATTEMPT" bash tools/runners/train/lumir.sh || return 1
        [ ! -f "$RUN/lumir_exp.txt" ] || exp="$(cat "$RUN/lumir_exp.txt")"
        if [ -f "results/$exp/ckpt/last.pth" ]; then resume=1
        elif [ -d "results/lumir/$exp" ]; then exp="${RUN_ID}_mind_$ATTEMPT"; fi
        echo "$exp" > "$RUN/lumir_exp.txt"
        job lumir_train "$first" env "${options[@]}" EXP="$exp" PROFILE=0 RESUME="$resume" STAGES=ramp EPOCHS=40 ITERS=1000 \
            bash tools/runners/train/lumir.sh || return 1
        job lumir_cert "$first" "$PY" -m experiments.export_lumir --ds LUMIR --paths 3 --config CTCF-CascadeA-Mamba \
            --l3_svf 0 --input_representation mind6 --time_steps 6 --precision bf16 --pairs lumir25 --certified 1 \
            --ckpt "results/$exp/ckpt/last.pth" --out "results/lumir/$exp/certified_lumir25" --name "${exp}_certified" || code=1
        job lumir_contrasts "$first" "$PY" -m experiments.lumir_probes contrasts --checkpoint "results/$exp/ckpt/last.pth" \
            --raw-checkpoint "$raw" --out "$RUN/lumir_contrasts" || code=1
        return "$code"
    }
    if [ "${#GPU[@]}" = 8 ]; then lumir_lane "${GPU[*]:4:4}" "${GPU[4]}" & lumir_pid=$!; fi
    PIDS=()
    for slot in 0 1 2 3; do
        (
            names=(half_t1 full_t1_t2 b0_s1 b1_s1); arms=(b1 b1 b0 b1); seeds=(0 0 1 1); epochs=(3 3 10 10)
            a=(0.5 1 1 0.5); b=(0 0.5 0 0.5); name=${names[$slot]}; code=0
            if job "$name" "${GPU[$slot]}" "$PY" -m experiments.contrast_diagnostics train --source "$source" --manifest "$MANIFEST" \
                --checkpoint "$PARENT" --out "$RUN/$name" --exp "${RUN_ID}_$name" --arm "${arms[$slot]}" --w_reg 1 \
                --seed "${seeds[$slot]}" --epochs "${epochs[$slot]}" --t1_weight "${a[$slot]}" --t2_weight "${b[$slot]}"; then
                readouts=(3); [ "${epochs[$slot]}" = 3 ] || readouts+=(10)
                for epoch in "${readouts[@]}"; do
                    job "cert_${name}_$epoch" "${GPU[$slot]}" "$PY" -m experiments.repair_diagnostics short \
                        --source "$original" --out "$RUN" --manifest "$MANIFEST" --arm-dir "$RUN/$name" --epoch "$epoch" || code=1
                done
            else code=1; fi
            exit "$code"
        ) & PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
    echo "$status" > "$RUN/exit_codes/controls.txt"
    PIDS=()
    for slot in 0 1 2; do
        (
            tasks=(weights t2 smooth)
            job "${tasks[$slot]}" "${GPU[$slot]}" "$PY" -m experiments.contrast_probes "${tasks[$slot]}" "${common[@]}"
        ) & PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
    job summary "" "$PY" -m experiments.contrast_probes summary "${common[@]}" || status=1
    if [ -n "$lumir_pid" ]; then wait "$lumir_pid" || status=1
    else lumir_lane "${GPU[*]}" "${GPU[0]}" || status=1; fi
    return "$status"
}
if [ "$WORKFLOW" = mechanism ]; then
    failed=0
    mechanism || failed=1
    log "FINISHED exit=$failed"
    exit "$failed"
fi
followup() {
    local source="${SOURCE_RUN:-results/contrasts/B_20261108T180120Z}" status=0 repair_pid slots slot name gpu repeat workers benchmark controls first pid
    local -a common=(--source "$source" --manifest "$MANIFEST" --checkpoint "$PARENT")
    local -a repair=(--source "$source" --manifest "$MANIFEST" --out "$RUN" --oasis-root "$DATA")
    [ -f "$MANIFEST" ] && [ -f "$source/head.txt" ] || { echo "Missing source run or manifest"; return 1; }
    bind manifest "$MANIFEST" || return 1
    bind source "$source/parent/recipe.json" || return 1
    job prepare "" "$PY" -m experiments.contrast_diagnostics prepare "${common[@]}" --out "$RUN" || return 1
    for workers in 0 8; do
        for benchmark in 0 1; do
            name="profile_${workers}_${benchmark}"
            job "$name" "${GPU[0]}" "$PY" -m experiments.contrast_diagnostics profile "${common[@]}" \
                --out "$RUN/$name" --exp "${RUN_ID}_${name}_$ATTEMPT" --workers "$workers" --benchmark "$benchmark" || status=1
        done
    done
    echo "$status" > "$RUN/exit_codes/profile.txt"
    short_control() {
        local name=$1 gpu=$2 arm=$3 reg=$4 seed=${5:-0}
        job "$name" "$gpu" "$PY" -m experiments.contrast_diagnostics train "${common[@]}" \
            --out "$RUN/$name" --exp "${RUN_ID}_$name" --arm "$arm" --w_reg "$reg" --seed "$seed" || return 1
        job "cert_$name" "$gpu" "$PY" -m experiments.repair_diagnostics short "${repair[@]}" --arm-dir "$RUN/$name"
    }
    if [ "${#GPU[@]}" -gt 1 ]; then
        job repair "${GPU[-1]}" "$PY" -m experiments.repair_diagnostics saved "${repair[@]}" & repair_pid=$!
        slots=$((${#GPU[@]}-1))
    else
        job repair "${GPU[0]}" "$PY" -m experiments.repair_diagnostics saved "${repair[@]}" || status=1
        slots=1; repair_pid=""
    fi
    PIDS=()
    for ((slot=0; slot<slots; slot++)); do
        (
            code=0; names=(b0_w4_s0 b0_w1_s0 b1_w1_s0 b2_w1_s0)
            for ((index=slot; index<4; index+=slots)); do
                name=${names[$index]}; arm=${name%%_*}; reg=${name:4:1}
                short_control "$name" "${GPU[$slot]}" "$arm" "$reg" || code=1
            done
            exit "$code"
        ) & PIDS+=($!)
    done
    controls=0
    for pid in "${PIDS[@]}"; do wait "$pid" || controls=1; done
    echo "$controls" > "$RUN/exit_codes/controls.txt"
    [ "$controls" = 0 ] || status=1
    job anchor "" "$PY" -m experiments.contrast_diagnostics anchor "${common[@]}" --out "$RUN" --exp "${RUN_ID}_b0_w4_s0" || status=1
    if [ -f "$RUN/anchor.json" ] && "$PY" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["add_b1_w4"])' "$RUN/anchor.json"; then
        short_control b1_w4_s0 "${GPU[0]}" b1 4 || status=1
    fi
    job decision "" "$PY" -m experiments.contrast_diagnostics decide "${common[@]}" --out "$RUN" || status=1
    if [ -f "$RUN/decision.json" ]; then
        repeat=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["repeat_arm"] or "")' "$RUN/decision.json")
        if [ -n "$repeat" ]; then
            short_control b0_w1_s1 "${GPU[0]}" b0 1 1 & first=$!
            if [ "$slots" -gt 1 ]; then gpu=${GPU[1]}; else wait "$first" || status=1; first=""; gpu=${GPU[0]}; fi
            short_control "${repeat}_w1_s1" "$gpu" "$repeat" 1 1 || status=1
            if [ -n "$first" ]; then wait "$first" || status=1; fi
        fi
    fi
    job smooth "${GPU[0]}" "$PY" -m experiments.contrast_diagnostics smooth "${common[@]}" --out "$RUN/smoothing" || status=1
    job summary "" "$PY" -m experiments.contrast_diagnostics summary "${common[@]}" --out "$RUN" || status=1
    if [ -n "$repair_pid" ]; then wait "$repair_pid" || status=1; fi
    job repair_summary "" "$PY" -m experiments.repair_diagnostics finish "${repair[@]}" || status=1
    return "$status"
}
if [ "$WORKFLOW" = followup ]; then
    failed=0
    followup || failed=1
    log "FINISHED exit=$failed"
    exit "$failed"
fi
consensus() {
    job consensus "$1" "$PY" -m experiments.dispatch1.eval_consensus --checkpoint "$PARENT" --data "$DATA" --out "$RUN/consensus"
}
failed=0
CONSENSUS_PID=""
if [ "${#GPU[@]}" -gt 1 ]; then
    consensus "${GPU[-1]}" & CONSENSUS_PID=$!
    slots=$((${#GPU[@]}-1))
else
    consensus "${GPU[0]}" || failed=1
    slots=1
fi
arm() {
    local name=$1 exp="${RUN_ID}_$1" weights="results/${RUN_ID}_$1/ckpt/last.pth" resume=()
    [ ! -f "$weights" ] || resume=(--resume "$weights")
    "$PY" -m experiments.contrasts train --exp "$exp" --manifest "$MANIFEST" --arm "$name" \
        --checkpoint "$PARENT" --out "$RUN/$name" "${resume[@]}" &&
    "$PY" -m experiments.contrasts evaluate --manifest "$MANIFEST" --checkpoint "$weights" --arm "$name" --out "$RUN/$name"
}
if [ "${RUN_B:-1}" = 1 ] && [ -f "$MANIFEST" ]; then
    bind manifest "$MANIFEST"
    if job check "${GPU[0]}" "$PY" -m experiments.contrasts check --manifest "$MANIFEST" --checkpoint "$PARENT" --out "$RUN/check" \
        --consensus_reserve_hours "${CONSENSUS_HOURS:-8}"; then
        for ((slot=0; slot<slots; slot++)); do
            (
                status=0
                names=(b0 b1 b2)
                for ((index=slot; index<3; index+=slots)); do job "${names[$index]}" "${GPU[$slot]}" arm "${names[$index]}" || status=1; done
                exit "$status"
            ) & PIDS+=($!)
        done
    else failed=1; fi
else
    log "HELD B: supply IXI_MANIFEST after image review; consensus is independent"
    failed=1
fi
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
if [ "$(cat "$RUN/exit_codes/check.txt" 2>/dev/null || true)" = 0 ]; then
    job parent "${GPU[0]}" "$PY" -m experiments.contrasts evaluate --manifest "$MANIFEST" --checkpoint "$PARENT" --out "$RUN/parent" || failed=1
fi
job summary "" "$PY" -m experiments.contrasts summary --out "$RUN" || failed=1
if [ -n "$CONSENSUS_PID" ]; then wait "$CONSENSUS_PID" || failed=1; fi
log "FINISHED exit=$failed"
exit "$failed"
