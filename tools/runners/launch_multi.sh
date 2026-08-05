#!/usr/bin/env bash
# Launch a sharded eval runner across several GPUs, RESUMABLE against cards being reclaimed mid-run.
# Each GPU runs one shard (NSHARD=<#gpus>, SHARD=i); CUDA_VISIBLE_DEVICES pins the card so the runner's
# GPU=0 maps to it. Every inference writes <tag>/summary.csv only when that whole tag finishes, and the
# runners SKIP a tag whose summary.csv already exists -- so completion is tracked per tag, independent of
# which shard produced it.
#
# Usage:
#   GPUS="2 3 4 5 6 7" PROFILE=--3 bash tools/runners/launch_multi.sh gateb_clip.sh
#
# If a card is taken away, its shard dies and its in-flight tag is left unfinished (no summary.csv). To
# recover: STOP any still-running shards for this runner (pkill -f <runner>), then re-run this SAME command
# with whatever GPUs are free now (GPUS="..."). Finished tags SKIP; each remaining tag maps to exactly one
# shard in the new run, so there is no double-work and no collision. Repeat until the final table is full.
# After all shards finish, an NSHARD=1 aggregation pass runs (everything SKIPs) and prints the runner's table.
RUNNER="$1"
GPUS="${GPUS:-2 3 4 5 6 7}"
PROFILE="${PROFILE:---3}"
LOGDIR="${LOGDIR:-logs/multi}"

if [[ -z "$RUNNER" || ! -f "tools/runners/$RUNNER" ]]; then
  echo "usage: GPUS=\"2 3 4\" PROFILE=--3 bash tools/runners/launch_multi.sh <runner.sh>" >&2
  echo "  runner must exist under tools/runners/ (e.g. gateb_clip.sh, injectivity.sh, stageb_repair_eval.sh)" >&2
  exit 2
fi
mkdir -p "$LOGDIR"
read -r -a arr <<< "$GPUS"
n=${#arr[@]}
echo "runner=$RUNNER profile=$PROFILE shards=$n gpus=(${arr[*]}) logs=$LOGDIR"

pids=()
for i in "${!arr[@]}"; do
  g="${arr[$i]}"
  log="$LOGDIR/${RUNNER%.sh}_shard${i}_gpu${g}.log"
  echo "  shard $i on GPU $g -> $log"
  CUDA_VISIBLE_DEVICES="$g" NSHARD="$n" SHARD="$i" GPU=0 PROFILE="$PROFILE" \
    nohup bash "tools/runners/$RUNNER" > "$log" 2>&1 &
  pids+=("$!")
done

echo "launched ${#pids[@]} shards (pids: ${pids[*]}). waiting..."
for p in "${pids[@]}"; do wait "$p" || echo "[warn] shard pid $p exited non-zero (card reclaimed? re-run to resume)"; done

echo; echo "==== aggregation pass on GPU ${arr[0]} (SKIPs done tags, mops up any leftovers, prints the table) ===="
CUDA_VISIBLE_DEVICES="${arr[0]}" PROFILE="$PROFILE" GPU=0 bash "tools/runners/$RUNNER"
