#!/usr/bin/env bash
# Round 2: a learned registration similarity next to NCC (rules: PILOTS.md, round 2, in the memory registry).
# Stage 0 trains three frozen feature nets on the unlabelled training images and checks them; stage 1 then trains
# VXM-100 from scratch with the PILOT_P1_CONTROL recipe (seed 0) plus lambda x the feature term, for every net
# that passed stage 0. Stage 1 starts only when stage 0 finished cleanly: all three nets trained and the checks
# ran. That is apart from the gate itself: a net that fails the gate drops only its own arm.
#   GPU A: contrastive net with NCC-hard negatives (HARD), then its stage-1 run
#   GPU B: contrastive net with random negatives only (RAND), then its stage-1 run
#   GPU C: DeepSim-style autoencoder (AE), then its stage-1 run
#   GPU D: where P10 and J15 differ (regime) during the feature training; then the stage-0 checks
#          (features, cost, calibrate, gate); then the diagnostic per-pair optimisation (tto) during stage 1
# Each dispatch writes only under its own stamp (results/round2/<stamp>, experiments R2_<stamp>_*), and ends
# with status.txt: one PASS/FAIL/SKIP line per step; the exit code is 1 when any step failed.
#
#   GPUS="0 1 2 3" nohup bash tools/runners/train/round2.sh > round2.out 2>&1 &
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r GPU_A GPU_B GPU_C GPU_D <<< "${GPUS:-0 1 2 3}"
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

log "HEAD $(git rev-parse --short HEAD), run $RUN, GPUs A=$GPU_A B=$GPU_B C=$GPU_C D=$GPU_D"
diag "$GPU_D" regime diag_regime &
features "$GPU_A" HARD --kind contrastive --negatives ncc_hard &
features "$GPU_B" RAND --kind contrastive --negatives random &
features "$GPU_C" AE --kind autoencoder &
wait

arms=()
untrained=()
for name in HARD RAND AE; do
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
  arm "$GPU_A" HARD &
  arm "$GPU_B" RAND &
  arm "$GPU_C" AE &
  diag "$GPU_D" tto diag_tto "${arms[@]}" &
  wait
fi

shopt -s nullglob
scored=("$RUN"/*/per_case.csv)
if (( ${#scored[@]} )); then
  "$PY" -m tools.analysis.dice_groups "${scored[@]}" | tee "$RUN/dice_groups.txt"
fi
archive="results/exports/ROUND2_$STAMP.zip"
"$PY" - "$archive" "$RUN" logs/"${TAG}"_* <<'EOF'
import os, sys, zipfile
archive, roots = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                if not name.endswith((".pth", ".npz")):
                    path = os.path.join(folder, name)
                    z.write(path, os.path.join("ROUND2", path))
EOF
failed=$(grep -c '^FAIL' "$STATUS")
log "status: $(grep -c '^PASS' "$STATUS") PASS, $failed FAIL, $(grep -c '^SKIP' "$STATUS") SKIP (see $STATUS)"
log "done: $archive"
(( failed == 0 ))
