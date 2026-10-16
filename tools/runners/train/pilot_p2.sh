#!/usr/bin/env bash
# Pilot P2: label-free anatomy on top of the trained unsupervised P10 VXM Unified SVF cascade.
# Every arm fine-tunes the same weights for 50 epochs with the P10 loss; the arms differ only in
# --anatomy_loss: control (none), latent (co-trained self-taught parcellation) and foundation (frozen
# anatomix features). Arms are spread over the given GPUs, run one after another on each GPU, and each is
# scored (last.pth, feed-forward and after the certified repair chain) as soon as it finishes.
# Decision rule: Dice35 gain over the control < 0.010 fails the arm, >= 0.015 is a signal.
#
#   GPUS="0 1" nohup bash tools/runners/train/pilot_p2.sh > pilot_p2.out 2>&1 &
#   ARMS="latent" GPUS="0" ...   # run a subset; the summary covers every P2 arm scored so far
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r -a ARM_LIST <<< "${ARMS:-control latent foundation}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
INIT=results/P10_LONGRUN_VXM_UNIFIED_SVF_OASIS/ckpt/best.pth
OUT=results/pilots/P2
mkdir -p "$OUT" results/exports

if [[ ! -f "$INIT" ]]; then
  echo "[P2] $INIT missing (restore it from MRI/experiment0206.tar.gz)"
  exit 1
fi
if [[ " ${ARM_LIST[*]} " == *" foundation "* ]] \
  && ! "$PY" -c "import anatomix.model.network, huggingface_hub" 2>/dev/null; then
  echo "[P2] foundation arm needs anatomix:" \
    "$PY -m pip install --no-deps 'git+https://github.com/neel-dey/anatomix.git@4c86aeeca5e4bf712d9e5775a01fadb37b30c13a'"
  exit 1
fi

BASE=(--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS --3 --gpu 0 --seed 0 --max_epoch 50
      --use_tb 0 --save_ckpt 1 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_jac 0.005
      --init_ckpt "$INIT" --finetune 1)
EVAL=(--model ctcf --3 --strict_ckpt 1 --gpu 0 --ds OASIS --tto_mode none
      --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1)
REPAIR=(--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps 0.001)

run_arm() {
  local gpu="$1" arm="$2" exp="PILOT_P2_${2^^}"
  local loss="$arm"
  [[ "$arm" == control ]] && loss=none
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.train_CTCF "${BASE[@]}" --anatomy_loss "$loss" --w_anat 1.0 \
    --exp "$exp" > "$OUT/$exp.train.log" 2>&1
  local ckpt="results/$exp/ckpt/last.pth"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" --ckpt "$ckpt" \
    --out_dir "$OUT/${exp}__FF" > "$OUT/${exp}__FF.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.inference "${EVAL[@]}" "${REPAIR[@]}" --ckpt "$ckpt" \
    --out_dir "$OUT/${exp}__REP" > "$OUT/${exp}__REP.log" 2>&1
  echo "[P2] $(date -u +%FT%TZ) $exp done"
}

worker() {
  local gpu="$1"; shift
  local arm
  for arm in "$@"; do
    run_arm "$gpu" "$arm" || { echo "[P2] $arm failed on GPU $gpu, see $OUT/"; return 1; }
  done
}

echo "[P2] $(date -u +%FT%TZ) HEAD $(git rev-parse --short HEAD), init sha256 $(sha256sum "$INIT" | cut -d' ' -f1)"
pids=()
for i in "${!GPU_LIST[@]}"; do
  assigned=()
  for j in "${!ARM_LIST[@]}"; do
    (( j % ${#GPU_LIST[@]} == i )) && assigned+=("${ARM_LIST[$j]}")
  done
  (( ${#assigned[@]} )) || continue
  echo "[P2] GPU ${GPU_LIST[$i]}: ${assigned[*]}"
  worker "${GPU_LIST[$i]}" "${assigned[@]}" & pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done

"$PY" -m tools.analysis.dice_groups "$OUT"/*/per_case.csv | tee "$OUT/dice_groups.txt"
archive="results/exports/PILOT_P2_$(date -u +%Y%m%dT%H%M%SZ).zip"
"$PY" - "$archive" "$OUT" logs/PILOT_P2_* <<'EOF'
import os, sys, zipfile
archive, roots = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                z.write(path, os.path.join("PILOT_P2", path))
EOF
echo "[P2] $(date -u +%FT%TZ) done: $archive (failed=$failed)"
exit "$failed"
