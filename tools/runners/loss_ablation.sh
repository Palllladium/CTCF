#!/usr/bin/env bash
# Loss-component ablation (TRAIN, 100ep). Closes the "is each loss term necessary / did we optimise the wrong
# criterion" question. 100ep (NOT 500) is reliable for RANKING (project rule: longrun only confirms the winner);
# confirm the top config(s) at 500ep afterwards. Anchor = weak-sup VxM-Unified-SVF OASIS, identical to the
# Wave-1/2 setup so numbers compare. Metric of record is measured LATER by loss_ablation_eval.sh = POST-repair
# certified Dice (the guarantee lives at inference), NOT feed-forward Dice.
#
# Two live hypotheses this matrix decides:
#   (a) the inference certificate/repair may make ICON and/or the digital Jacobian penalty REDUNDANT -> dropping
#       them frees capacity and may RECOVER Dice (an improvement, not just an ablation). Runs 2/3/4.
#   (b) we penalised the DIGITAL Jacobian for years, but digital != the deployed trilinear fold. Runs 6/7 swap to
#       the corrected trilinear target; run 7 uses reduce=active (P17 used mean, which smears sparse folds into a
#       near-zero signal) -- the sharp test of whether targeting the RIGHT criterion finally beats digital.
#
# Runs ONE experiment in the foreground; wrap per card (RUN=1..8). Resumable: re-run the SAME command, continues
# from results/<EXP>/ckpt/last.pth. SMOKE first on one card:
#   CUDA_VISIBLE_DEVICES=2 RUN=1 SMOKE=1 MAX_EPOCH=3 bash tools/runners/loss_ablation.sh
#   # the ablation, one run per card GPU2-...:
#   r=1; for dev in 2 3 4 5 6 7; do
#     CUDA_VISIBLE_DEVICES=$dev RUN=$r nohup bash tools/runners/loss_ablation.sh > abl_r${r}.log 2>&1 &
#     r=$((r+1)); done   # (8 runs, 6 cards: launch 1-6, then 7-8 as cards free)
set -e

GPU="${GPU:-0}"                 # index WITHIN CUDA_VISIBLE_DEVICES (keep 0; the mask picks the card)
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
MAX_EPOCH="${MAX_EPOCH:-100}"
SMOKE="${SMOKE:-0}"
RUN="${RUN:?set RUN=1..8}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"

# Anchor: weak-sup (w_dice 1.0) VxM Unified SVF OASIS. Per-run flags override w_reg/w_icon/w_jac/jac_mode/
# tri_pen_*/icon_mode only. FULL = the current operating point (digital penalty at w_jac 5 = Stage-B best).
COMMON="--config CTCF-CascadeA-VM-Unified --l3_svf 1 --ds OASIS ${PROFILE} --gpu ${GPU} \
        --max_epoch ${MAX_EPOCH} --w_ncc 1.0 --w_dice 1.0 --seed 0 --use_tb 1 --save_ckpt 1"

case "$RUN" in
  1) EXP=P18_ABL_VXM_OASIS_FULL;        LOSS="--w_reg 1.0 --w_icon 0.05 --w_jac 5 --jac_mode digital" ;;
  2) EXP=P18_ABL_VXM_OASIS_NOICON;      LOSS="--w_reg 1.0 --w_icon 0.0  --w_jac 5 --jac_mode digital" ;;
  3) EXP=P18_ABL_VXM_OASIS_NOJAC;       LOSS="--w_reg 1.0 --w_icon 0.05 --w_jac 0 --jac_mode central" ;;
  4) EXP=P18_ABL_VXM_OASIS_NOICON_NOJAC;LOSS="--w_reg 1.0 --w_icon 0.0  --w_jac 0 --jac_mode central" ;;
  5) EXP=P18_ABL_VXM_OASIS_NOREG;       LOSS="--w_reg 0.0 --w_icon 0.05 --w_jac 5 --jac_mode digital" ;;
  6) EXP=P18_ABL_VXM_OASIS_TRI_MEAN;    LOSS="--w_reg 1.0 --w_icon 0.05 --w_jac 5 --jac_mode trilinear --tri_pen_mode bernstein --tri_pen_reduce mean" ;;
  7) EXP=P18_ABL_VXM_OASIS_TRI_ACTIVE;  LOSS="--w_reg 1.0 --w_icon 0.05 --w_jac 5 --jac_mode trilinear --tri_pen_mode bernstein --tri_pen_reduce active" ;;
  8) EXP=P18_ABL_VXM_OASIS_ICON_L2;     LOSS="--w_reg 1.0 --w_icon 0.05 --w_jac 5 --jac_mode digital --icon_mode l2" ;;
  *) echo "RUN must be 1..8"; exit 1 ;;
esac

[[ "$SMOKE" == "1" ]] && EXP="${EXP}_SMOKE"   # separate dir so a smoke never pollutes a real run

RESUME=""
last="results/${EXP}/ckpt/last.pth"
if [[ -f "$last" ]]; then RESUME="--resume $last"; echo ">>> resuming ${EXP} from $last"; fi

echo ">>> ${EXP} | visible GPU=${GPU} | ${MAX_EPOCH}ep | ${LOSS}"
# exec so Ctrl-C / kill reach python directly.
# shellcheck disable=SC2086
exec "${PYBIN}" -m experiments.train_CTCF $COMMON $LOSS --exp "$EXP" $RESUME
