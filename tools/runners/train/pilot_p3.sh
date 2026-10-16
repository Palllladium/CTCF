#!/usr/bin/env bash
# Pilot P3: train through the certified local clip. Both arms fine-tune the labelled P16 J15 cascade
# for 50 epochs with its own loss (soft-Dice labels, digital Jacobian penalty 15); the certified arm adds
# the soft-Dice of a certified-clipped random crop, with the clip's exact derivative. Afterwards each
# last.pth is scored three ways: feed-forward, the post-hoc repair chain, and the deployed operator
# (clip from identity, 2 sweeps) whose saved fields go through the exact verifier at eps 0.001.
# Success: Dice35 >= 0.903 on all 19 cases with every deployed field exactly certified.
#
#   GPUS="2 3" nohup bash tools/runners/train/pilot_p3.sh > pilot_p3.out 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r GPU_CONTROL GPU_CERT <<< "${GPUS:-2 3}"
INIT=results/P16_W1_VXM_OASIS_LBL_DIG_J15/ckpt/best.pth
INIT_SHA256=d21fcdc9e779b705dcda603ed88e58b1461835fa0cf75eddbc4bc70eff395940
SWEEPS=2
OUT=results/pilots/P3
mkdir -p "$OUT" results/exports

if [[ ! -f "$INIT" ]] || [[ "$(sha256sum "$INIT" | cut -d' ' -f1)" != "$INIT_SHA256" ]]; then
  echo "[P3] $INIT missing or not the P16 J15 checkpoint (restore it from MRI/large_experiment3107.tar.gz)"
  exit 1
fi

BASE=(--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS --3 --gpu 0 --seed 0 --max_epoch 50
      --use_tb 0 --save_ckpt 1 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_dice 1.0
      --jac_mode digital --w_jac 15 --init_ckpt "$INIT" --finetune 1)
EVAL=(--model ctcf --3 --strict_ckpt 1 --gpu 0 --ds OASIS --tto_mode none
      --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1)
REPAIR=(--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps 0.001)
CLIP=(--tto_clip_from_identity 1 --tto_clip_sweeps "$SWEEPS" --tto_tri_project_eps 0.0011 --save_flow 1)

train() {
  local gpu="$1" exp="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.train_CTCF "${BASE[@]}" "$@" --exp "$exp" \
    > "$OUT/$exp.train.log" 2>&1
}

score() {
  local gpu="$1" exp="$2" ckpt="results/$2/ckpt/last.pth"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" --ckpt "$ckpt" \
    --out_dir "$OUT/${exp}__FF" > "$OUT/${exp}__FF.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" "${REPAIR[@]}" --ckpt "$ckpt" \
    --out_dir "$OUT/${exp}__REP" > "$OUT/${exp}__REP.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" "${CLIP[@]}" --ckpt "$ckpt" \
    --out_dir "$OUT/${exp}__CLIP" > "$OUT/${exp}__CLIP.log" 2>&1
  # Exit 1 means some field failed the sufficient predicate: a result, not a crash.
  local code=0
  "$PY" -m utils.cert_exact --flow "$OUT/${exp}__CLIP/flows" --eps 0.001 \
    --report "$OUT/${exp}__CLIP/exact.json" > "$OUT/${exp}__CLIP.exact.log" 2>&1 || code=$?
  echo "[P3] $exp exact verifier exit code $code" | tee -a "$OUT/exact_codes.txt"
  if (( code >= 2 )); then return 1; fi
}

echo "[P3] $(date -u +%FT%TZ) HEAD $(git rev-parse --short HEAD), GPUs control=$GPU_CONTROL certified=$GPU_CERT"
train "$GPU_CONTROL" PILOT_P3_CONTROL & control=$!
train "$GPU_CERT" PILOT_P3_CERTIFIED --certified_steps "$SWEEPS" --w_cert 1.0 & cert=$!
failed=0
wait "$control" || failed=1
wait "$cert" || failed=1
if [[ "$failed" != 0 ]]; then
  echo "[P3] training failed, see $OUT/*.train.log"
  exit 1
fi

echo "[P3] $(date -u +%FT%TZ) training done, scoring last.pth"
score "$GPU_CONTROL" PILOT_P3_CONTROL & control=$!
score "$GPU_CERT" PILOT_P3_CERTIFIED & cert=$!
wait "$control"
wait "$cert"

"$PY" -m tools.analysis.dice_groups "$OUT"/*/per_case.csv | tee "$OUT/dice_groups.txt"
archive="results/exports/PILOT_P3_$(date -u +%Y%m%dT%H%M%SZ).zip"
"$PY" - "$archive" "$OUT" logs/PILOT_P3_CONTROL logs/PILOT_P3_CERTIFIED <<'EOF'
import os, sys, zipfile
archive, roots = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                if name.endswith(".npz"):
                    continue  # fields stay on the server; exact.json carries their verdicts and hashes
                path = os.path.join(folder, name)
                z.write(path, os.path.join("PILOT_P3", path))
EOF
echo "[P3] $(date -u +%FT%TZ) done: $archive"
