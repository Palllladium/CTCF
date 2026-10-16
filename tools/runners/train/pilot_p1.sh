#!/usr/bin/env bash
# Pilot P1: label-free GMM tissue target against an equal control (VXM Unified SVF, OASIS,
# 100 epochs from scratch, seed 0). Both arms train in parallel; afterwards each last.pth is
# scored feed-forward and after the certified repair chain, Dice is reported over 35/29/6 labels,
# and everything compact is zipped for transfer.
#
#   GPUS="0 1" nohup bash tools/runners/train/pilot_p1.sh > pilot_p1.out 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r GPU_CONTROL GPU_GMM <<< "${GPUS:-0 1}"
OUT=results/pilots/P1
mkdir -p "$OUT" results/exports

BASE=(--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS --3 --gpu 0 --seed 0 --max_epoch 100
      --use_tb 0 --save_ckpt 1 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_jac 0.005)
EVAL=(--model ctcf --3 --strict_ckpt 1 --gpu 0 --ds OASIS --tto_mode none
      --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1)
REPAIR=(--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps 0.001)

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
}

echo "[P1] $(date -u +%FT%TZ) HEAD $(git rev-parse --short HEAD), GPUs control=$GPU_CONTROL gmm=$GPU_GMM"
train "$GPU_CONTROL" PILOT_P1_CONTROL --anatomy_loss none & control=$!
train "$GPU_GMM" PILOT_P1_GMM --anatomy_loss gmm --w_anat 1.0 & gmm=$!
failed=0
wait "$control" || failed=1
wait "$gmm" || failed=1
if [[ "$failed" != 0 ]]; then
  echo "[P1] training failed, see $OUT/*.train.log"
  exit 1
fi

echo "[P1] $(date -u +%FT%TZ) training done, scoring last.pth"
score "$GPU_CONTROL" PILOT_P1_CONTROL & control=$!
score "$GPU_GMM" PILOT_P1_GMM & gmm=$!
wait "$control"
wait "$gmm"

"$PY" -m tools.analysis.dice_groups "$OUT"/*/per_case.csv | tee "$OUT/dice_groups.txt"
archive="results/exports/PILOT_P1_$(date -u +%Y%m%dT%H%M%SZ).zip"
"$PY" - "$archive" "$OUT" logs/PILOT_P1_CONTROL logs/PILOT_P1_GMM <<'EOF'
import os, sys, zipfile
archive, roots = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                z.write(path, os.path.join("PILOT_P1", path))
EOF
echo "[P1] $(date -u +%FT%TZ) done: $archive"
