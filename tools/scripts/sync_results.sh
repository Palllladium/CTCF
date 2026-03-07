#!/usr/bin/env bash
# Download experiment artifacts from a remote GPU machine.
# Usage (run on local machine):
#   bash tools/scripts/sync_results.sh user@remote-ip [--key ~/.ssh/id_rsa] [--with-ckpt]
set -euo pipefail

LOCAL_DIR="./remote_results/$(date +%Y%m%d_%H%M%S)"
REMOTE_DIR="~/CTCF"
WITH_CKPT=0

if [ $# -lt 1 ]; then
  echo "Usage: bash tools/scripts/sync_results.sh user@remote-ip [--key path/to/key] [--with-ckpt]"
  exit 1
fi

REMOTE_HOST="$1"
shift

SSH_KEY_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --key) SSH_KEY_ARG="-i $2"; shift 2 ;;
    --with-ckpt) WITH_CKPT=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCP_OPTS="-r -C ${SSH_KEY_ARG}"
SSH_CMD="ssh ${SSH_KEY_ARG} ${REMOTE_HOST}"

echo "=== Syncing results from $REMOTE_HOST ==="
echo "Local destination: $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

echo "[1/4] Downloading ablation_results.txt..."
scp $SCP_OPTS "${REMOTE_HOST}:${REMOTE_DIR}/ablation_results.txt" "$LOCAL_DIR/" 2>/dev/null || echo "  (not found)"

echo "[2/4] Downloading log files..."
$SSH_CMD "find ${REMOTE_DIR}/logs -name 'logfile.log'" 2>/dev/null | while read -r f; do
  REL=$(echo "$f" | sed "s|${REMOTE_DIR}/||")
  mkdir -p "$LOCAL_DIR/$(dirname "$REL")"
  scp $SCP_OPTS "${REMOTE_HOST}:${f}" "$LOCAL_DIR/${REL}" 2>/dev/null
done

echo "[3/4] Downloading TensorBoard events..."
$SSH_CMD "find ${REMOTE_DIR}/logs -name 'events.out.tfevents.*'" 2>/dev/null | while read -r f; do
  REL=$(echo "$f" | sed "s|${REMOTE_DIR}/||")
  mkdir -p "$LOCAL_DIR/$(dirname "$REL")"
  scp $SCP_OPTS "${REMOTE_HOST}:${f}" "$LOCAL_DIR/${REL}" 2>/dev/null
done

if [ "$WITH_CKPT" = "1" ]; then
  echo "[4/4] Downloading best checkpoints..."
  $SSH_CMD "find ${REMOTE_DIR}/results -path '*/ckpt/best.pth'" 2>/dev/null | while read -r f; do
    REL=$(echo "$f" | sed "s|${REMOTE_DIR}/||")
    mkdir -p "$LOCAL_DIR/$(dirname "$REL")"
    scp $SCP_OPTS "${REMOTE_HOST}:${f}" "$LOCAL_DIR/${REL}" 2>/dev/null
  done
else
  echo "[4/4] Skipping checkpoints (use --with-ckpt to enable)."
fi

echo ""
echo "=== Sync complete ==="
echo "Results saved to: $LOCAL_DIR"
echo ""
echo "View TensorBoard locally:"
echo "  tensorboard --logdir $LOCAL_DIR/logs"
echo ""
if [ -f "$LOCAL_DIR/ablation_results.txt" ]; then
  echo "Quick summary:"
  cat "$LOCAL_DIR/ablation_results.txt"
fi
