#!/usr/bin/env bash
# Upload the current CTCF working tree snapshot to a remote machine.
# Run this on your LOCAL PC (Git Bash / WSL).
#
# Usage:
#   bash tools/scripts/upload_project.sh user@remote-ip [--key ~/.ssh/id_ed25519] [--remote-dir ~/CTCF]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash tools/scripts/upload_project.sh user@remote-ip [--key path] [--remote-dir ~/CTCF]"
  exit 1
fi

REMOTE_HOST="$1"
shift

SSH_KEY_ARG=""
REMOTE_DIR="~/CTCF"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key) SSH_KEY_ARG="-i $2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCHIVE_NAME="ctcf_project_$(date +%Y%m%d_%H%M%S).tar.gz"
ARCHIVE_PATH="${TMPDIR:-/tmp}/${ARCHIVE_NAME}"
SSH_CMD="ssh ${SSH_KEY_ARG} ${REMOTE_HOST}"
SCP_CMD="scp ${SSH_KEY_ARG} -C"

cleanup() {
  rm -f "$ARCHIVE_PATH"
}
trap cleanup EXIT

echo "=== Packing current project tree ==="
(cd "$WORK_DIR" && tar -czf "$ARCHIVE_PATH" \
  --exclude=".git" \
  --exclude=".idea" \
  --exclude=".venv" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude="__pycache__" \
  --exclude="datasets" \
  --exclude="logs" \
  --exclude="results" \
  --exclude="remote_results" \
  --exclude="figures" \
  --exclude="paper/out" \
  .)

SIZE="$(du -h "$ARCHIVE_PATH" | cut -f1)"
echo "Archive ready: $ARCHIVE_PATH ($SIZE)"

echo "=== Uploading project snapshot to $REMOTE_HOST:$REMOTE_DIR ==="
$SSH_CMD "mkdir -p ${REMOTE_DIR}"
$SCP_CMD "$ARCHIVE_PATH" "${REMOTE_HOST}:${REMOTE_DIR}/"
$SSH_CMD "cd ${REMOTE_DIR} && tar -xzf ${ARCHIVE_NAME} && rm ${ARCHIVE_NAME}"

echo ""
echo "============================================"
echo "  Project upload complete!"
echo "  Remote dir: ${REMOTE_DIR}"
echo "  Next: bash tools/scripts/remote_setup.sh --work-dir ${REMOTE_DIR} --skip-repo-sync"
echo "============================================"
