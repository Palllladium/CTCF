#!/usr/bin/env bash
# ONE command for the whole loss ablation: TRAIN all 8 configs (bounded-parallel across cards) -> EVAL each
# (feed-forward + certified repair) -> print the post-repair TABLE. Kill it (Ctrl-C / kill) and re-run the SAME
# command to continue: finished trainings are no-ops (the trainer resumes and exits immediately once epoch_start
# >= max_epoch), finished evals skip (summary.csv), incomplete work resumes from ckpt/last.pth. It just reuses
# loss_ablation.sh (train, per RUN) and loss_ablation_eval.sh (eval), so the per-config logic is already tested.
#
#   PROFILE=--3 bash tools/runners/train/loss_ablation_all.sh                 # 100ep, cards 2-7, eval on card 2
#   CARDS="0 1 2 3" MAX_EPOCH=100 PROFILE=--3 bash tools/runners/train/loss_ablation_all.sh
#   SMOKE=1 MAX_EPOCH=3 CARDS="2" bash tools/runners/train/loss_ablation_all.sh  # smoke: 8 tiny runs on one card
set -euo pipefail

PROFILE="${PROFILE:---3}"
MAX_EPOCH="${MAX_EPOCH:-100}"
CARDS="${CARDS:-2 3 4 5 6 7}"     # physical GPUs; 8 configs run in bounded-parallel batches of |CARDS|
EVAL_GPU="${EVAL_GPU:-2}"         # single card for the (fast, sequential) inference eval
NRUNS="${NRUNS:-8}"
SMOKE="${SMOKE:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a CARD_ARR <<< "$CARDS"
ncards=${#CARD_ARR[@]}
echo "########## loss-ablation ALL | ${NRUNS} configs x ${MAX_EPOCH}ep | cards [${CARDS}] | eval GPU ${EVAL_GPU} ##########"

# ---- Phase 1: train, at most |CARDS| in parallel; wait for each batch before the next ----
r=1
while [[ $r -le $NRUNS ]]; do
  launched=()
  pids=()
  for ((c=0; c<ncards && r<=NRUNS; c++)); do
    card=${CARD_ARR[$c]}
    echo ">>> [train] RUN=$r on physical GPU ${card}  (errors -> abl_err_r${r}.log; full per-epoch log in logs/<EXP>/)"
    # stdout -> /dev/null (the trainer already writes the full per-epoch history to logs/<EXP>/logfile.log);
    # only stderr (NaN warnings / tracebacks) is captured, so the returned logs are errors-only, not noise.
    CUDA_VISIBLE_DEVICES="$card" RUN="$r" PROFILE="$PROFILE" MAX_EPOCH="$MAX_EPOCH" SMOKE="$SMOKE" \
      nohup bash "$HERE/loss_ablation.sh" > /dev/null 2> "abl_err_r${r}.log" &
    launched+=("$r")
    pids+=("$!")
    r=$((r+1))
  done
  echo ">>> [train] waiting for batch: RUN(s) ${launched[*]} ..."
  failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[FAIL] RUN=${launched[$i]} (see abl_err_r${launched[$i]}.log)" >&2
      failed=1
    fi
  done
  [[ "$failed" == "0" ]] || exit 1
  echo ">>> [train] batch done"
done
echo ">>> [train] ALL trainings finished (or resumed-complete)"

# ---- Phase 2: eval (feed-forward + certified repair) + the post-repair table ----
echo ">>> [eval] scoring the 8 checkpoints (feed-forward + certified repair) on GPU ${EVAL_GPU}"
GPU="$EVAL_GPU" PROFILE="$PROFILE" SMOKE="$SMOKE" bash "$HERE/loss_ablation_eval.sh"
echo ">>> loss-ablation ALL complete. Table above; re-run this command any time to resume/refresh."
