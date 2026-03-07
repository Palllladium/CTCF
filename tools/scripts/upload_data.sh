#!/usr/bin/env bash
# Upload OASIS/IXI datasets to remote GPU machine.
# Run this on your LOCAL PC (Git Bash / WSL).
#
# Usage:
#   bash tools/scripts/upload_data.sh user@remote-ip [--key ~/.ssh/id_rsa] [--datasets oasis,ixi]
#
# Prerequisites:
#   Create archives first (in the same folder as the dataset dirs):
#     cd C:/Users/user/Documents/Education/MasterWork/datasets
#     tar -czf OASIS_L2R_2021_task03.tar.gz OASIS_L2R_2021_task03/
#     tar -czf IXI_data.tar.gz IXI_data/
set -euo pipefail

LOCAL_DATA_DIR="C:/Users/user/Documents/Education/MasterWork/datasets"
REMOTE_DATA_DIR="/data"

# --- Parse args ---
if [ $# -lt 1 ]; then
  echo "Usage: bash tools/scripts/upload_data.sh user@remote-ip [--key path] [--datasets oasis,ixi]"
  exit 1
fi

REMOTE_HOST="$1"; shift
SSH_KEY_ARG=""
DATASETS="oasis"  # default: only OASIS

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key) SSH_KEY_ARG="-i $2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SSH_CMD="ssh ${SSH_KEY_ARG} ${REMOTE_HOST}"
SCP_CMD="scp ${SSH_KEY_ARG} -C"

# Create remote data directory
$SSH_CMD "sudo mkdir -p ${REMOTE_DATA_DIR} && sudo chmod 777 ${REMOTE_DATA_DIR}"

if [[ "$DATASETS" == *"oasis"* ]]; then
  ARCHIVE="${LOCAL_DATA_DIR}/OASIS_L2R_2021_task03.tar.gz"
  if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: Archive not found: $ARCHIVE"
    echo "Create it first:"
    echo "  cd '${LOCAL_DATA_DIR}'"
    echo "  tar -czf OASIS_L2R_2021_task03.tar.gz OASIS_L2R_2021_task03/"
    exit 1
  fi
  SIZE=$(du -h "$ARCHIVE" | cut -f1)
  echo "=== Uploading OASIS ($SIZE) ==="
  $SCP_CMD "$ARCHIVE" "${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
  echo "=== Unpacking OASIS on remote ==="
  $SSH_CMD "cd ${REMOTE_DATA_DIR} && tar -xzf OASIS_L2R_2021_task03.tar.gz && rm OASIS_L2R_2021_task03.tar.gz"
  echo "OASIS ready at ${REMOTE_DATA_DIR}/OASIS_L2R_2021_task03/"
fi

if [[ "$DATASETS" == *"ixi"* ]]; then
  ARCHIVE="${LOCAL_DATA_DIR}/IXI_data.tar.gz"
  if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: Archive not found: $ARCHIVE"
    echo "Create it first:"
    echo "  cd '${LOCAL_DATA_DIR}'"
    echo "  tar -czf IXI_data.tar.gz IXI_data/"
    exit 1
  fi
  SIZE=$(du -h "$ARCHIVE" | cut -f1)
  echo "=== Uploading IXI ($SIZE) ==="
  $SCP_CMD "$ARCHIVE" "${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
  echo "=== Unpacking IXI on remote ==="
  $SSH_CMD "cd ${REMOTE_DATA_DIR} && tar -xzf IXI_data.tar.gz && rm IXI_data.tar.gz"
  echo "IXI ready at ${REMOTE_DATA_DIR}/IXI_data/"
fi

echo ""
echo "============================================"
echo "  Upload complete!"
echo "  Remote data dir: ${REMOTE_DATA_DIR}"
echo "  Next: bash tools/scripts/remote_setup.sh --data-dir ${REMOTE_DATA_DIR}"
echo "============================================"
