#!/usr/bin/env bash
# Deployed-warp Jacobian figure. Saves the CURRENT model's flow twice -- feed-forward and certified-repair --
# then renders signed trilinear det J + witnessed-fold mask + Bernstein margin (before/after). The local
# flow_0440_0441.npz is the OLD 0.8208 model (Paper 1); this regenerates on a current checkpoint. Pair 0440_0441
# matches Paper 1's figure. Inference saves ALL cases' flows; we plot the one pair.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
EXP="${EXP:-P16_W1_VXM_OASIS_LBL_DIG_J15}"     # a current checkpoint (0.903 regime); override for another
PAIR="${PAIR:-0440_0441}"
EPS="${EPS:-0.001}"
FF_DIR="${FF_DIR:-results/jacfig_ff}"
REP_DIR="${REP_DIR:-results/jacfig_rep}"
OUT="${OUT:-results/figs/deployed_jacobian_${PAIR}.png}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5 --ds OASIS --tto_mode none"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
CKPT="$(ck "$EXP")"
[[ -f "$CKPT" ]] || { echo "[MISS] no ckpt for $EXP at $CKPT" >&2; exit 1; }

echo "########## Jacobian figure | $EXP | pair $PAIR ##########"
# feed-forward: save flow, NO repair (the folds the central log|det| map hides are here)
# shellcheck disable=SC2086
[[ -f "$FF_DIR/flows/flow_${PAIR}.npz" ]] || \
  "${PYBIN}" -m experiments.inference $COMMON $VM --ckpt "$CKPT" --out_dir "$FF_DIR" --save_flow --tto_project 0 --tto_tri_project 0
# certified repair: save the repaired flow (folds gone, tri_cert_bound >= eps)
# shellcheck disable=SC2086
[[ -f "$REP_DIR/flows/flow_${PAIR}.npz" ]] || \
  "${PYBIN}" -m experiments.inference $COMMON $VM --ckpt "$CKPT" --out_dir "$REP_DIR" --save_flow $CHAIN

echo ">>> rendering $OUT"
"${PYBIN}" tools/paper/plot_deployed_jacobian.py \
  --ff  "$FF_DIR/flows/flow_${PAIR}.npz" \
  --rep "$REP_DIR/flows/flow_${PAIR}.npz" \
  --eps "$EPS" --out "$OUT"
echo ">>> done: $OUT"
