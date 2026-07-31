#!/usr/bin/env bash
# Phase 16, Wave 1 — labelled-regime operating point on OASIS (VxM Unified SVF, 500ep).
#
# With a hard test-time projection now guaranteeing zero digital folds, the train-time penalty's
# job is no longer "minimise folds" but "keep the field cheaply projectable without capping Dice".
# So this is a w_jac sweep under weak supervision; the winner is picked later by POST-PROJECTION
# Dice (feed-forward + tools/runners/tto_project.sh), not by feed-forward folds.
#
#   RUN=1  labels, NO digital penalty   (central, w_jac 0.005) — max Dice / max folds baseline
#   RUN=2  labels + digital  w_jac=1     — light
#   RUN=3  labels + digital  w_jac=5     — current candidate
#   RUN=4  labels + digital  w_jac=15    — heavy
#
# Runs ONE experiment in the foreground; wrap it per card. Resumable: re-run the SAME command and
# it continues from results/<EXP>/ckpt/last.pth (written every epoch). Interrupt with Ctrl-C / kill.
#
#   # smoke today on a free card (3 epochs, separate _SMOKE dir), then re-run to test resume:
#   CUDA_VISIBLE_DEVICES=2 RUN=1 SMOKE=1 MAX_EPOCH=3 bash tools/runners/train_wave1.sh
#
#   # the wave tomorrow, one run per card GPU4-7:
#   CUDA_VISIBLE_DEVICES=4 RUN=1 nohup bash tools/runners/train_wave1.sh > w1r1.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=5 RUN=2 nohup bash tools/runners/train_wave1.sh > w1r2.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=6 RUN=3 nohup bash tools/runners/train_wave1.sh > w1r3.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=7 RUN=4 nohup bash tools/runners/train_wave1.sh > w1r4.log 2>&1 &
set -e

GPU="${GPU:-0}"                 # index WITHIN CUDA_VISIBLE_DEVICES (keep 0; the mask picks the card)
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
MAX_EPOCH="${MAX_EPOCH:-500}"
SMOKE="${SMOKE:-0}"
RUN="${RUN:?set RUN=1|2|3|4}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"   # H-box default for profile --3; override to change

COMMON="--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS ${PROFILE} --gpu ${GPU} \
        --max_epoch ${MAX_EPOCH} --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_dice 1.0 \
        --seed 0 --use_tb 1 --save_ckpt 1"

case "$RUN" in
  1) EXP=P16_W1_VXM_OASIS_LBL_NODIG;   JAC="--jac_mode central --w_jac 0.005" ;;
  2) EXP=P16_W1_VXM_OASIS_LBL_DIG_J1;  JAC="--jac_mode digital --w_jac 1" ;;
  3) EXP=P16_W1_VXM_OASIS_LBL_DIG_J5;  JAC="--jac_mode digital --w_jac 5" ;;
  4) EXP=P16_W1_VXM_OASIS_LBL_DIG_J15; JAC="--jac_mode digital --w_jac 15" ;;
  *) echo "RUN must be 1..4"; exit 1 ;;
esac

[[ "$SMOKE" == "1" ]] && EXP="${EXP}_SMOKE"   # separate dir so a smoke never pollutes a real run

RESUME=""
last="results/${EXP}/ckpt/last.pth"
if [[ -f "$last" ]]; then RESUME="--resume $last"; echo ">>> resuming ${EXP} from $last"; fi

echo ">>> ${EXP} | visible GPU=${GPU} | ${MAX_EPOCH}ep | ${JAC}"
# exec so Ctrl-C / kill reach python directly and stop cleanly.
# shellcheck disable=SC2086
exec "${PYBIN}" -m experiments.train_CTCF $COMMON $JAC --exp "$EXP" $RESUME
