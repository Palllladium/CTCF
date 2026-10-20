#!/usr/bin/env bash
# Dispatch 2: im2grid-Lite over a lambda grid, the P1 control fine-tuned at lower w_reg, direct and second pass of
# Mamba NoSVF, the known-warp EPE of the dispatch-1 arm h2_B and the first updates of the H1 matchers.
#   GPUS (default "0 1 2 3 4 5 6 7"; with fewer GPUs the eight job slots share them in turn), D1 (the dispatch-1 run
#   folder with the lambda files).
#
#   nohup bash tools/runners/train/dispatch2.sh > dispatch2.out 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git diff --quiet HEAD -- || { echo "[dispatch2] tracked files differ from HEAD: commit or restore them"; exit 1; }

PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
read -r -a GPU <<< "${GPUS:-0 1 2 3 4 5 6 7}"
D1="${D1:-results/dispatch1/20261017T094254Z}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="results/dispatch2/$STAMP"
NODE="--paths 3 --gpu 0"
P1_CKPT=results/PILOT_P1_CONTROL/ckpt/last.pth
P1_MODEL="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
H2_B_CKPT=results/D1_20261017T094254Z_h2_B/ckpt/last.pth
MAMBA_CKPT=results/P10_LONGRUN_MAMBA_NOSVF_OASIS/ckpt/best.pth
MAMBA_FF=results/SEDM/inference/P10_LONGRUN_MAMBA_NOSVF_OASIS/per_case.csv

[ "${#GPU[@]}" -ge 1 ] || { echo "[dispatch2] GPUS is empty"; exit 1; }
for f in "$P1_CKPT" "$H2_B_CKPT" "$D1/im2grid_prep/lambda.json" "$D1/h1_prep_D3/lambda.json" "$D1/h1_prep_E3/lambda.json"; do
    [ -f "$f" ] || { echo "[dispatch2] missing $f"; exit 1; }
done
mkdir -p "$RUN"
echo "$(date -u +%FT%TZ) HEAD $(git rev-parse HEAD) GPUS=${GPU[*]} D1=$D1 PY=$PY" > "$RUN/run_info.txt"
"$PY" -m pytest -q -p no:cacheprovider tools/analysis/tests/dispatch1 > "$RUN/pre_tests.txt" 2>&1 \
    || { echo "[dispatch2] unit tests failed: $RUN/pre_tests.txt"; exit 1; }
REPAIR="$("$PY" -c 'from experiments.core.field_scoring import repair_flags; print(repair_flags())')"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/status.txt"; }

# job NAME GPU EXP COMMAND...: the commands run in order on one GPU; the training logs of EXP are kept either way.
job() {
    local name=$1 gpu=$2 exp=$3 status=PASS
    shift 3
    mkdir -p "$RUN/$name"
    log "START $name on GPU $gpu"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        for command in "$@"; do
            echo "+ $command"
            eval "$command" || exit 1
        done
    ) > "$RUN/$name/log.txt" 2>&1 || status=ERROR
    if [ "$exp" != - ]; then
        [ -d "logs/$exp" ] && cp -r "logs/$exp" "$RUN/$name/training_logs"
        [ -f "results/$exp/metrics.csv" ] && cp "results/$exp/metrics.csv" "$RUN/$name/"
    fi
    log "$status $name"
}

i=0
for scale in 0.0625 0.015625 0.00390625 0.0009765625 0; do
    name="im2grid_s$scale" exp="D2_${STAMP}_im2grid_s$scale"
    job "$name" "${GPU[$((i % ${#GPU[@]}))]}" "$exp" \
        "$PY -m experiments.dispatch1.train_im2grid --lambda_file $D1/im2grid_prep/lambda.json --lambda_scale $scale --exp $exp --iters 39400 --seed 0 $NODE" \
        "$PY -m experiments.dispatch1.train_im2grid --eval_ckpt results/$exp/ckpt/last.pth --out $RUN/$name $NODE" &
    i=$((i + 1))
done

for w_reg in 0.3 0.1; do
    name="ctcf_wreg$w_reg" exp="D2_${STAMP}_ctcf_wreg$w_reg"
    inference="$PY -m experiments.inference --model ctcf --strict_ckpt 1 --ds OASIS --tto_mode none $NODE --ckpt results/$exp/ckpt/last.pth $P1_MODEL"
    job "$name" "${GPU[$((i % ${#GPU[@]}))]}" "$exp" \
        "$PY -m experiments.train_CTCF --config CTCF-CascadeA-VM-Unified --l3_svf 1 --seed 0 --ds OASIS --use_tb 0 --w_ncc 1.0 --w_icon 0.05 --w_reg $w_reg --w_jac 0.005 --finetune 1 --init_ckpt $P1_CKPT --max_epoch 100 --save_ckpt 1 --exp $exp $NODE" \
        "$inference --out_dir $RUN/$name/FF" \
        "$inference $REPAIR --out_dir $RUN/$name/REP" &
    i=$((i + 1))
done

(
    gpu="${GPU[$((7 % ${#GPU[@]}))]}"
    if [ -f "$MAMBA_CKPT" ] && [ -f "$MAMBA_FF" ] && "$PY" -c "import mamba_ssm" 2> /dev/null; then
        job mamba_recursion "$gpu" - \
            "$PY -m experiments.dispatch1.eval_bridge --ckpt $MAMBA_CKPT --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0 --check_ff $MAMBA_FF --limit_bank 3 --out $RUN/mamba_recursion $NODE"
    else
        log "SKIP mamba_recursion (checkpoint, historical FF table or mamba_ssm missing)"
    fi
    job epe_h2_B "$gpu" - \
        "$PY -m experiments.dispatch1.eval_known_warp --ckpt $H2_B_CKPT $P1_MODEL --out $RUN/epe_h2_B/epe.json $NODE"
    for mode in D3 E3; do
        for lr in 3e-4 3e-5; do
            job "h1_steps_${mode}_lr$lr" "$gpu" - \
                "$PY -m experiments.dispatch1.diag_h1_steps --mode $mode --lambda_file $D1/h1_prep_$mode/lambda.json --lr $lr --out $RUN/h1_steps_${mode}_lr$lr/steps.json $NODE"
        done
    done
) &

wait
"$PY" -m tools.analysis.dice_groups $(find "$RUN" -path '*/REP/per_case.csv' | sort) > "$RUN/dice_groups.txt" 2>&1 || true
mkdir -p results/exports
"$PY" - "$RUN" "results/exports/DISPATCH2_$STAMP.zip" << 'EOF'
import sys
import zipfile
from pathlib import Path

run, archive = Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.suffix not in (".pth", ".pt"):
            z.write(path, Path(Path(archive).stem) / path.relative_to(run))
EOF
log "done: results/exports/DISPATCH2_$STAMP.zip; failed: $(grep -c ' ERROR ' "$RUN/status.txt" || true)"
