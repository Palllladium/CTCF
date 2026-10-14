#!/usr/bin/env bash
# CST/CoRe pilot (rules: PILOTS.md, "CST/CoRe pilot", in the memory registry): the corrected CST adaptation alone (cst)
# and with CoRe's affine-equivariance term (cst_eq), trained on OASIS without labels from one shared start.
#   1. The shared start on the first GPU (pair list, stage-0 pseudo labels, start check, measured cost of one arm), and
#      at the same time the P1 control's SDlogJ on 20 fixed training pairs on the second GPU.
#   2. Every (arm, seed) run on its own GPU, in waves when runs outnumber GPUs: train; calibrate the test-time refinement
#      on the same 20 training pairs; evaluate the 19 test pairs in both modes (solver, refined), FF and REP. Training
#      starts only when the start check passed and the measured cost per arm fits MAX_ARM_HOURS.
#   3. Dice groups, status.txt (one PASS/FAIL/SKIP line per step), one ZIP results/exports/<LABEL>_<stamp>.zip without
#      weights; the exit code is 1 when any step failed or training was skipped.
#   GPUS (2-4, default "0 1"), ARMS (default "cst cst_eq"), SEEDS (default "0"), PAIRS_PER_SUBJECT (default 2),
#   MAX_ARM_HOURS (default 10), LABEL (default CST)
#
#   GPUS="0 1" nohup bash tools/runners/train/cst_pilot.sh > cst_pilot.out 2>&1 &
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
read -r -a ARM_NAMES <<< "${ARMS:-cst cst_eq}"
read -r -a SEED_LIST <<< "${SEEDS:-0}"
PER_SUBJECT="${PAIRS_PER_SUBJECT:-2}"
MAX_ARM_HOURS="${MAX_ARM_HOURS:-10}"
LABEL="${LABEL:-CST}"
CONTROL=results/PILOT_P1_CONTROL/ckpt/last.pth
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=results/cst/$STAMP
STATUS=$RUN/status.txt
mkdir -p "$RUN" results/exports
incomplete=0  # training skipped: the run did not answer its question

if [[ ! -f "$CONTROL" ]]; then
  echo "[CST] $CONTROL missing: the P1 control sets the SDlogJ target of the calibration"
  exit 1
fi
for arm in "${ARM_NAMES[@]}"; do
  [[ "$arm" == cst || "$arm" == cst_eq ]] || { echo "[CST] unknown arm $arm (known: cst cst_eq)"; exit 1; }
done

log() { echo "[CST] $(date -u +%FT%TZ) $*"; }
mark() { echo "$1 $2" >> "$STATUS"; log "$1 $2"; }  # PASS|FAIL|SKIP step

step() {  # gpu name logfile python-args...
  local gpu="$1" name="$2" logfile="$3"
  shift 3
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$@" > "$logfile" 2>&1; then
    mark PASS "$name"
    return 0
  fi
  mark FAIL "$name (see $logfile)"
  return 1
}

run_arm() {  # gpu arm seed: train, calibrate, evaluate both modes
  local gpu="$1" arm="$2" seed="$3"
  local name="${arm}_s${seed}" dir="$RUN/${arm}_s${seed}"
  mkdir -p "$dir"
  step "$gpu" "$name training" "$dir/train.log" -m experiments.train_cst --arm "$arm" --seed "$seed" \
    --init "$RUN/init" --out "$dir" --paths 3 --gpu 0 --max_hours "$MAX_ARM_HOURS" || return
  if grep -qx "PASS control_sdlogj" "$STATUS"; then
    if step "$gpu" "$name calibration" "$dir/calibrate.log" -m experiments.eval_cst --ckpt "$dir/final.pth" \
      --calibrate "$dir/reg_fac.json" --control "$RUN/control_sdlogj.json" --paths 3 --gpu 0; then
      local calibration
      calibration=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['status'])" "$dir/reg_fac.json")
      if [[ "$calibration" != reached ]]; then
        mark SKIP "$name geometry match (calibration $calibration; refined results are diagnostic only)"
      fi
    fi
  else
    mark SKIP "$name calibration (no control SDlogJ)"
  fi
  step "$gpu" "$name solver eval" "$dir/eval_solver.log" -m experiments.eval_cst --ckpt "$dir/final.pth" \
    --mode solver --out "$dir/eval" --paths 3 --gpu 0
  if [[ -f "$dir/reg_fac.json" ]]; then
    step "$gpu" "$name refined eval" "$dir/eval_refined.log" -m experiments.eval_cst --ckpt "$dir/final.pth" \
      --mode refined --reg_fac_file "$dir/reg_fac.json" --out "$dir/eval" --paths 3 --gpu 0
  else
    mark SKIP "$name refined eval (no calibration)"
  fi
}

echo "HEAD $(git rev-parse HEAD) GPUS=${GPU_LIST[*]} ARMS=${ARM_NAMES[*]} SEEDS=${SEED_LIST[*]}" \
  "PAIRS_PER_SUBJECT=$PER_SUBJECT MAX_ARM_HOURS=$MAX_ARM_HOURS" > "$RUN/run_info.txt"
log "HEAD $(git rev-parse --short HEAD), run $RUN, GPUs ${GPU_LIST[*]}"
GPU_A=${GPU_LIST[0]}
GPU_B=${GPU_LIST[1]:-$GPU_A}
mkdir -p "$RUN/init"
step "$GPU_A" init "$RUN/init.log" -m experiments.train_cst --init_only --out "$RUN/init" --paths 3 --gpu 0 \
  --pairs_per_subject "$PER_SUBJECT" &
step "$GPU_B" control_sdlogj "$RUN/control_sdlogj.log" -m experiments.eval_cst \
  --control_out "$RUN/control_sdlogj.json" --control_ckpt "$CONTROL" --paths 3 --gpu 0 &
wait

if ! grep -qx "PASS init" "$STATUS"; then
  mark SKIP "training (shared start failed or its start check did not pass: see $RUN/init.log)"
  incomplete=1
else
  projected=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['projected_arm_hours'])" \
    "$RUN/init/init_report.json")
  if "$PY" -c "import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))" "$projected" "$MAX_ARM_HOURS"; then
    log "projected ${projected} h per arm"
    runs=()
    for seed in "${SEED_LIST[@]}"; do
      for arm in "${ARM_NAMES[@]}"; do
        runs+=("$arm $seed")
      done
    done
    i=0
    for run in "${runs[@]}"; do
      read -r arm seed <<< "$run"
      run_arm "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" "$arm" "$seed" &
      i=$((i + 1))
      if (( i % ${#GPU_LIST[@]} == 0 )); then
        wait
      fi
    done
    wait
  else
    mark SKIP "training (projected ${projected} h per arm exceeds MAX_ARM_HOURS=$MAX_ARM_HOURS)"
    incomplete=1
  fi
fi

shopt -s nullglob
scored=("$RUN"/*/eval/*/per_case.csv)
if (( ${#scored[@]} )); then
  "$PY" -m tools.analysis.dice_groups "${scored[@]}" | tee "$RUN/dice_groups.txt"
fi
archive="results/exports/${LABEL}_$STAMP.zip"
if ! "$PY" - "$archive" "$LABEL" "$RUN" <<'EOF'
import os, sys, zipfile
archive, label, root = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, _, files in os.walk(root):
        for name in files:
            if not name.endswith((".pth", ".npz")):
                path = os.path.join(folder, name)
                z.write(path, os.path.join(label, path))
EOF
then
  mark FAIL "archive $archive (results stay in $RUN)"
fi
failed=$(grep -c '^FAIL' "$STATUS")
log "status: $(grep -c '^PASS' "$STATUS") PASS, $failed FAIL, $(grep -c '^SKIP' "$STATUS") SKIP (see $STATUS)"
log "done: $archive"
(( failed == 0 && incomplete == 0 ))
