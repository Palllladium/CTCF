#!/usr/bin/env bash
# Round 1 after pilots P1-P3: the diagnostics of tools/analysis/diag_round1.py and four short training runs in
# one dispatch on four GPUs. Each GPU works through its own queue; a failing step is recorded and the rest go on.
#   GPU A: diagnostics d0,d1,d2b,d4,m2, then d2 (per-pair optimisation), then the anatomix arm at share 2.
#   GPU B: d3 (gradients; writes the anatomix dose calibration), then the anatomix arm at share 0.5.
#   GPU C: P10 fine-tune with the NCC map averaged per labelled structure (oracle weighting, labels as weights).
#   GPU D: from-scratch VXM-100 control with seed 1, to see the run-to-run spread against PILOT_P1_CONTROL.
# Fine-tunes follow the P2 recipe (P10 weights, 50 epochs) and compare with PILOT_P2_CONTROL. "Share" is the
# anatomix gradient norm over the NCC gradient norm on the network at the P10 weights, measured by d3.
# Every run is scored FF and REP, d0 is repeated on the new checkpoints, and everything compact is zipped.
# Each dispatch writes only under its own stamp (results/round1/<stamp>, experiments R1_<stamp>_*), and ends
# with status.txt: one PASS/FAIL/SKIP line per step; the exit code is 1 when any step failed.
#
#   GPUS="0 1 2 3" nohup bash tools/runners/train/round1.sh > round1.out 2>&1 &
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r GPU_A GPU_B GPU_C GPU_D <<< "${GPUS:-0 1 2 3}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=results/round1/$STAMP
DIAG=$RUN/diag
TAG=R1_$STAMP
STATUS=$RUN/status.txt
P10=results/P10_LONGRUN_VXM_UNIFIED_SVF_OASIS/ckpt/best.pth
mkdir -p "$DIAG" results/exports

if [[ ! -f "$P10" ]]; then
  echo "[R1] $P10 missing (restore it from MRI/experiment0206.tar.gz)"
  exit 1
fi
# Missing packages cost only their parts: scipy the label-free components, anatomix d3's dose and both arms.
"$PY" -c "import scipy.ndimage" 2>/dev/null \
  || echo "[R1] WARNING scipy missing, component parts will be skipped: $PY -m pip install scipy"
"$PY" -c "import anatomix.model.network, huggingface_hub" 2>/dev/null \
  || echo "[R1] WARNING anatomix missing, the anatomix arms will be skipped (see tools/runners/train/pilot_p2.sh)"

FINETUNE=(--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS --3 --gpu 0 --seed 0 --max_epoch 50
          --use_tb 0 --save_ckpt 1 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_jac 0.005
          --init_ckpt "$P10" --finetune 1)
SCRATCH=(--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS --3 --gpu 0 --seed 1 --max_epoch 100
         --use_tb 0 --save_ckpt 1 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_jac 0.005)
EVAL=(--model ctcf --3 --strict_ckpt 1 --gpu 0 --ds OASIS --tto_mode none
      --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1)
REPAIR=(--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps 0.001)

log() { echo "[R1] $(date -u +%FT%TZ) $*"; }
mark() { echo "$1 $2" >> "$STATUS"; log "$1 $2"; }  # PASS|FAIL|SKIP step

diag() {  # gpu parts step-name [extra diag args]
  local gpu="$1" parts="$2" name="$3"
  shift 3
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m tools.analysis.diag_round1 --parts "$parts" --out "$DIAG" --paths 3 \
    --gpu 0 "$@" > "$DIAG/$name.log" 2>&1; then
    mark PASS "$name"
  else
    mark FAIL "$name (see $DIAG/$name.log, error_*.txt)"
  fi
}

arm() {  # gpu exp recipe-array-name [extra train args]
  local gpu="$1" exp="$2"
  local -n recipe="$3"
  shift 3
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.train_CTCF "${recipe[@]}" "$@" --exp "$exp" \
    > "$RUN/$exp.train.log" 2>&1; then
    mark FAIL "$exp training (see $RUN/$exp.train.log)"
    return
  fi
  mark PASS "$exp training"
  local ckpt="results/$exp/ckpt/last.pth" mode
  for mode in FF REP; do
    local flags=()
    [[ "$mode" == REP ]] && flags=("${REPAIR[@]}")
    if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" "${flags[@]}" --ckpt "$ckpt" \
      --out_dir "$RUN/${exp}__$mode" > "$RUN/${exp}__$mode.log" 2>&1; then
      mark PASS "$exp $mode"
    else
      mark FAIL "$exp $mode (see $RUN/${exp}__$mode.log)"
    fi
  done
}

anatomix_arm() {  # gpu share exp
  local w
  w=$("$PY" -c "import json; print(json.load(open('$DIAG/d3_calibration.json'))['w_anat']['$2'])" 2>/dev/null)
  if [[ -z "$w" ]]; then
    mark SKIP "$3 (no anatomix calibration from d3 in this run)"
    return
  fi
  log "$3 uses --w_anat $w (share $2)"
  arm "$1" "$3" FINETUNE --anatomy_loss foundation --w_anat "$w"
}

queue_a() {
  diag "$GPU_A" d0,d1,d2b,d4,m2 diag_a
  diag "$GPU_A" d2 diag_d2
  anatomix_arm "$GPU_A" 2.0 "${TAG}_ANATOMIX_SHARE2"
}

queue_b() {
  diag "$GPU_B" d3 diag_d3
  anatomix_arm "$GPU_B" 0.5 "${TAG}_ANATOMIX_SHARE05"
}

log "HEAD $(git rev-parse --short HEAD), run $RUN, GPUs A=$GPU_A B=$GPU_B C=$GPU_C D=$GPU_D"
queue_a &
queue_b &
arm "$GPU_C" "${TAG}_ORACLE_STRUCTURE_NCC" FINETUNE --ncc_regions labels &
arm "$GPU_D" "${TAG}_CONTROL_SEED1" SCRATCH --anatomy_loss none &
wait

names=()
extra=()
for exp in "${TAG}_ORACLE_STRUCTURE_NCC" "${TAG}_ANATOMIX_SHARE05" "${TAG}_ANATOMIX_SHARE2" "${TAG}_CONTROL_SEED1"; do
  if grep -qx "PASS $exp training" "$STATUS" 2>/dev/null; then
    names+=("$exp")
    extra+=(--extra "$exp=results/$exp/ckpt/last.pth")
  fi
done
if (( ${#names[@]} )); then
  diag "$GPU_A" d0 diag_arms --models "$(IFS=,; echo "${names[*]}")" "${extra[@]}" --out "$DIAG/arms"
fi

shopt -s nullglob
scored=("$RUN"/*/per_case.csv)
if (( ${#scored[@]} )); then
  "$PY" -m tools.analysis.dice_groups "${scored[@]}" | tee "$RUN/dice_groups.txt"
fi
archive="results/exports/ROUND1_$STAMP.zip"
"$PY" - "$archive" "$RUN" logs/"${TAG}"_* <<'EOF'
import os, sys, zipfile
archive, roots = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                if not name.endswith((".pth", ".npz")):
                    path = os.path.join(folder, name)
                    z.write(path, os.path.join("ROUND1", path))
EOF
failed=$(grep -c '^FAIL' "$STATUS")
log "status: $(grep -c '^PASS' "$STATUS") PASS, $failed FAIL, $(grep -c '^SKIP' "$STATUS") SKIP (see $STATUS)"
log "done: $archive"
(( failed == 0 ))
