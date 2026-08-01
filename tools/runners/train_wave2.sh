#!/usr/bin/env bash
# Phase 17, Wave 2 — TRILINEAR fold penalty (Stage B): train the DEPLOYED warp fold-free (OASIS, 500ep).
#
# Wave 1 penalised the DIGITAL criterion (the wrong one); this penalises the trilinear grid_sample warp we
# actually certify, via a differentiable hinge on its Bernstein coefficients. Two variants x three weights,
# one run per card (6 cards). Winner picked later by POST-repair Dice + repair cost vs the Wave-1 digital
# models (does targeting the RIGHT criterion beat the digital penalty, and does it shrink/kill the repair?).
#
#   bernstein  sound 27-coefficient hinge (aligned with the certificate, heavier: ~27 det evals + backward)
#   sampled    cheaper 5^3 interior-lattice proxy
#   RUN=1..3  bernstein  w_jac 1 / 5 / 15
#   RUN=4..6  sampled    w_jac 1 / 5 / 15
#
# Runs ONE experiment in the foreground; wrap per card. Resumable (re-run the same command). The bernstein
# penalty is memory/compute-heavier than digital — SMOKE-test one card first to rule out OOM/NaN:
#   CUDA_VISIBLE_DEVICES=2 RUN=1 SMOKE=1 MAX_EPOCH=3 bash tools/runners/train_wave2.sh
#
#   # the wave, one run per card GPU2-7:
#   cards="2 3 4 5 6 7"; r=1
#   for dev in $cards; do
#     CUDA_VISIBLE_DEVICES=$dev RUN=$r nohup bash tools/runners/train_wave2.sh >/dev/null 2>>w2_errors.log &
#     r=$((r+1))
#   done
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
MAX_EPOCH="${MAX_EPOCH:-500}"
SMOKE="${SMOKE:-0}"
RUN="${RUN:?set RUN=1..6}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"

COMMON="--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS ${PROFILE} --gpu ${GPU} \
        --max_epoch ${MAX_EPOCH} --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_dice 1.0 \
        --seed 0 --use_tb 1 --save_ckpt 1"

case "$RUN" in
  1) EXP=P17_W2_VXM_OASIS_TRI_BERN_J1;  JAC="--jac_mode trilinear --tri_pen_mode bernstein --w_jac 1" ;;
  2) EXP=P17_W2_VXM_OASIS_TRI_BERN_J5;  JAC="--jac_mode trilinear --tri_pen_mode bernstein --w_jac 5" ;;
  3) EXP=P17_W2_VXM_OASIS_TRI_BERN_J15; JAC="--jac_mode trilinear --tri_pen_mode bernstein --w_jac 15" ;;
  4) EXP=P17_W2_VXM_OASIS_TRI_SAMP_J1;  JAC="--jac_mode trilinear --tri_pen_mode sampled --w_jac 1" ;;
  5) EXP=P17_W2_VXM_OASIS_TRI_SAMP_J5;  JAC="--jac_mode trilinear --tri_pen_mode sampled --w_jac 5" ;;
  6) EXP=P17_W2_VXM_OASIS_TRI_SAMP_J15; JAC="--jac_mode trilinear --tri_pen_mode sampled --w_jac 15" ;;
  *) echo "RUN must be 1..6"; exit 1 ;;
esac

[[ "$SMOKE" == "1" ]] && EXP="${EXP}_SMOKE"

RESUME=""
last="results/${EXP}/ckpt/last.pth"
if [[ -f "$last" ]]; then RESUME="--resume $last"; echo ">>> resuming ${EXP} from $last"; fi

echo ">>> ${EXP} | visible GPU=${GPU} | ${MAX_EPOCH}ep | ${JAC}"
# shellcheck disable=SC2086
exec "${PYBIN}" -m experiments.train_CTCF $COMMON $JAC --exp "$EXP" $RESUME
