#!/usr/bin/env bash
# Round 2: a learned registration similarity next to NCC (rules: PILOTS.md, rounds 2 and 2b, in the memory registry).
# Stage 0 trains frozen feature nets on the unlabelled training images and checks them; stage 1 then trains
# VXM-100 from scratch with the PILOT_P1_CONTROL recipe (seed 0) plus lambda x the feature term, for every net
# that passed stage 0. Stage 1 starts only when stage 0 finished cleanly: every requested net trained and the
# checks ran. That is apart from the gate itself: a net that fails the gate drops only its own arm.
#   ARMS      feature nets, default "HARD RAND AE": HARD contrastive with NCC-hard negatives, RAND contrastive with
#             random negatives only, AE DeepSim-style autoencoder. Each arm trains its net, then its stage-1 run,
#             on its own GPU.
#   GPUS      one per arm plus a last one for diagnostics, default "0 1 2 3". The diagnostics GPU maps where P10
#             and J15 differ during the feature training (REGIME=1, default), then runs the stage-0 checks
#             (features, cost, calibrate, gate), then the diagnostic per-pair optimisation (tto) during stage 1;
#             ANATOMIX=1 adds the anatomix similarity alone to tto.
#   LABEL     names the export results/exports/<LABEL>_<stamp>.zip, default ROUND2.
# Each dispatch writes only under its own stamp (results/round2/<stamp>, experiments R2_<stamp>_*), and ends
# with status.txt: one PASS/FAIL/SKIP line per step; the exit code is 1 when any step failed.
#
#   GPUS="0 1 2 3" nohup bash tools/runners/train/round2.sh > round2.out 2>&1 &
#   ARMS=AE GPUS="0 1" REGIME=0 ANATOMIX=1 LABEL=ROUND2B nohup bash tools/runners/train/round2.sh > round2b.out 2>&1 &
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
read -r -a ARM_NAMES <<< "${ARMS:-HARD RAND AE}"
REGIME="${REGIME:-1}"
ANATOMIX="${ANATOMIX:-0}"
LABEL="${LABEL:-ROUND2}"
declare -A FEATURE_ARGS=(
  [HARD]="--kind contrastive --negatives ncc_hard"
  [RAND]="--kind contrastive --negatives random"
  [AE]="--kind autoencoder"
)
if (( ${#GPU_LIST[@]} < ${#ARM_NAMES[@]} + 1 )); then
  echo "[R2] need one GPU per arm (${ARM_NAMES[*]}) plus one for diagnostics, got GPUS=\"${GPU_LIST[*]}\""
  exit 1
fi
declare -A GPU_OF
for i in "${!ARM_NAMES[@]}"; do
  if [[ -z "${FEATURE_ARGS[${ARM_NAMES[$i]}]:-}" ]]; then
    echo "[R2] unknown arm ${ARM_NAMES[$i]} (known: ${!FEATURE_ARGS[*]})"
    exit 1
  fi
  GPU_OF[${ARM_NAMES[$i]}]=${GPU_LIST[$i]}
done
GPU_D=${GPU_LIST[${#GPU_LIST[@]}-1]}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=results/round2/$STAMP
DIAG=$RUN/diag
FEAT=$RUN/features
TAG=R2_$STAMP
STATUS=$RUN/status.txt
P10=results/P10_LONGRUN_VXM_UNIFIED_SVF_OASIS/ckpt/best.pth
J15=results/P16_W1_VXM_OASIS_LBL_DIG_J15/ckpt/best.pth
mkdir -p "$DIAG" "$FEAT" results/exports

if [[ ! -f "$P10" ]]; then
  echo "[R2] $P10 missing (restore it from MRI/experiment0206.tar.gz)"
  exit 1
fi
[[ -f "$J15" ]] || echo "[R2] WARNING $J15 missing: regime fails and tto runs from P10 only"

SCRATCH=(--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS --3 --gpu 0 --seed 0 --max_epoch 100
         --use_tb 0 --save_ckpt 1 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_jac 0.005)
EVAL=(--model ctcf --3 --strict_ckpt 1 --gpu 0 --ds OASIS --tto_mode none
      --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1)
REPAIR=(--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps 0.001)

log() { echo "[R2] $(date -u +%FT%TZ) $*"; }
mark() { echo "$1 $2" >> "$STATUS"; log "$1 $2"; }  # PASS|FAIL|SKIP step

features() {  # gpu name [train_features args]
  local gpu="$1" name="$2"
  shift 2
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.train_features --paths 3 --gpu 0 --out "$FEAT/$name" "$@" \
    > "$FEAT/$name.log" 2>&1; then
    mark PASS "features $name"
  else
    mark FAIL "features $name (see $FEAT/$name.log)"
  fi
}

diag() {  # gpu parts step-name [diag_round2 args]
  local gpu="$1" parts="$2" name="$3"
  shift 3
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m tools.analysis.diag_round2 --parts "$parts" --out "$DIAG" --paths 3 \
    --gpu 0 "$@" > "$DIAG/$name.log" 2>&1; then
    mark PASS "$name"
  else
    mark FAIL "$name (see $DIAG/$name.log, error_*.txt)"
  fi
}

arm() {  # gpu name: stage 1 for one feature net, with the mode and lambda fixed by the stage-0 gate
  local gpu="$1" name="$2" exp="${TAG}_$2" settings ckpt mode weight
  settings=$("$PY" - "$DIAG/stage0_gate.json" "$name" <<'EOF'
import json, sys
arm = json.load(open(sys.argv[1]))["arms"].get(sys.argv[2])
if arm and arm["pass"]:
    print(arm["checkpoint"], arm["mode"], repr(arm["lambda"]))
EOF
)
  if [[ -z "$settings" ]]; then
    mark SKIP "$exp (stage-0 gate failed or missing, see $DIAG/stage0_gate.json)"
    return
  fi
  read -r ckpt mode weight <<< "$settings"
  log "$exp: --feature_ckpt $ckpt --feature_mode $mode --w_anat $weight"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.train_CTCF "${SCRATCH[@]}" --anatomy_loss learned \
    --feature_ckpt "$ckpt" --feature_mode "$mode" --w_anat "$weight" --exp "$exp" > "$RUN/$exp.train.log" 2>&1; then
    mark FAIL "$exp training (see $RUN/$exp.train.log)"
    return
  fi
  mark PASS "$exp training"
  local last="results/$exp/ckpt/last.pth" scoring
  for scoring in FF REP; do
    local flags=()
    [[ "$scoring" == REP ]] && flags=("${REPAIR[@]}")
    if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" "${flags[@]}" --ckpt "$last" \
      --out_dir "$RUN/${exp}__$scoring" > "$RUN/${exp}__$scoring.log" 2>&1; then
      mark PASS "$exp $scoring"
    else
      mark FAIL "$exp $scoring (see $RUN/${exp}__$scoring.log)"
    fi
  done
}

placement=""
for name in "${ARM_NAMES[@]}"; do
  placement+="$name=${GPU_OF[$name]} "
done
echo "HEAD $(git rev-parse HEAD) ARMS=${ARM_NAMES[*]} GPUS=${GPU_LIST[*]} REGIME=$REGIME ANATOMIX=$ANATOMIX" \
  > "$RUN/run_info.txt"
log "HEAD $(git rev-parse --short HEAD), run $RUN, arms ${placement}diagnostics=$GPU_D"
if [[ "$REGIME" == 1 ]]; then
  diag "$GPU_D" regime diag_regime &
fi
for name in "${ARM_NAMES[@]}"; do
  read -r -a feature_args <<< "${FEATURE_ARGS[$name]}"
  features "${GPU_OF[$name]}" "$name" "${feature_args[@]}" &
done
wait

arms=()
untrained=()
for name in "${ARM_NAMES[@]}"; do
  if grep -qx "PASS features $name" "$STATUS" 2>/dev/null; then
    arms+=(--arm "$name=$FEAT/$name/feature_net.pth")
  else
    untrained+=("$name")
  fi
done
if (( ${#arms[@]} )); then
  diag "$GPU_D" features,cost,calibrate,gate diag_stage0 "${arms[@]}"
fi

if (( ${#untrained[@]} )); then
  mark SKIP "stage 1 (stage 0 incomplete: feature training failed for ${untrained[*]}; see $FEAT)"
elif ! grep -qx "PASS diag_stage0" "$STATUS" || [[ ! -f "$DIAG/stage0_gate.json" ]]; then
  mark SKIP "stage 1 (stage 0 incomplete: the stage-0 checks failed; see $DIAG)"
else
  for name in "${ARM_NAMES[@]}"; do
    arm "${GPU_OF[$name]}" "$name" &
  done
  tto_flags=()
  [[ "$ANATOMIX" == 1 ]] && tto_flags+=(--anatomix)
  diag "$GPU_D" tto diag_tto "${arms[@]}" "${tto_flags[@]}" &
  wait
fi

shopt -s nullglob
scored=("$RUN"/*/per_case.csv)
if (( ${#scored[@]} )); then
  "$PY" -m tools.analysis.dice_groups "${scored[@]}" | tee "$RUN/dice_groups.txt"
fi
archive="results/exports/${LABEL}_$STAMP.zip"
"$PY" - "$archive" "$LABEL" "$RUN" logs/"${TAG}"_* <<'EOF'
import os, sys, zipfile
archive, label, roots = sys.argv[1], sys.argv[2], sys.argv[3:]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                if not name.endswith((".pth", ".npz")):
                    path = os.path.join(folder, name)
                    z.write(path, os.path.join(label, path))
EOF
failed=$(grep -c '^FAIL' "$STATUS")
log "status: $(grep -c '^PASS' "$STATUS") PASS, $failed FAIL, $(grep -c '^SKIP' "$STATUS") SKIP (see $STATUS)"
log "done: $archive"
(( failed == 0 ))
