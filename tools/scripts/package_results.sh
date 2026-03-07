#!/usr/bin/env bash
# Pack local experiment outputs into a single archive for sharing.
# Default mode: only ablation_results.txt + logs/*/logfile.log.
# By default archive is written to repo root.
set -euo pipefail

WORK_DIR="${CTCF_WORK_DIR:-}"
OUT_DIR=""
ARCHIVE_NAME=""
EXP_GLOB="*"
WITH_CKPT=0
WITH_TB=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --archive-name) ARCHIVE_NAME="$2"; shift 2 ;;
    --exp-glob) EXP_GLOB="$2"; shift 2 ;;
    --with-ckpt) WITH_CKPT="$2"; shift 2 ;;
    --with-tb) WITH_TB="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$WITH_CKPT" != "0" && "$WITH_CKPT" != "1" ]]; then
  echo "--with-ckpt must be 0 or 1"
  exit 1
fi
if [[ "$WITH_TB" != "0" && "$WITH_TB" != "1" ]]; then
  echo "--with-tb must be 0 or 1"
  exit 1
fi

if [ -z "$WORK_DIR" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
if [ -z "$OUT_DIR" ]; then
  OUT_DIR="$WORK_DIR"
fi
if [ -z "$ARCHIVE_NAME" ]; then
  ARCHIVE_NAME="ctcf_results_$(date +%Y%m%d_%H%M%S).tar.gz"
fi

if [ ! -d "$WORK_DIR" ]; then
  echo "Work dir not found: $WORK_DIR"
  exit 1
fi
if [ ! -f "$WORK_DIR/experiments/train_CTCF.py" ]; then
  echo "Not a CTCF repo root (missing experiments/train_CTCF.py): $WORK_DIR"
  exit 1
fi

if [[ "$ARCHIVE_NAME" = /* ]]; then
  ARCHIVE_PATH="$ARCHIVE_NAME"
else
  ARCHIVE_PATH="$OUT_DIR/$ARCHIVE_NAME"
fi

TMP_LIST="$(mktemp)"
cleanup() { rm -f "$TMP_LIST"; }
trap cleanup EXIT

if [ -f "$WORK_DIR/ablation_results.txt" ]; then
  echo "ablation_results.txt" >> "$TMP_LIST"
fi

if [ -d "$WORK_DIR/logs" ]; then
  while IFS= read -r f; do
    exp_name="$(basename "$(dirname "$f")")"
    if [[ "$exp_name" == $EXP_GLOB ]]; then
      echo "$f" >> "$TMP_LIST"
    fi
  done < <(cd "$WORK_DIR" && find logs -mindepth 2 -maxdepth 2 -type f -name "logfile.log" | sort)
fi

if [ "$WITH_CKPT" = "1" ] && [ -d "$WORK_DIR/results" ]; then
  while IFS= read -r f; do
    echo "$f" >> "$TMP_LIST"
  done < <(cd "$WORK_DIR" && find results -type f \( -name "best.pth" -o -name "epoch_*.pth" \) | sort)
fi

if [ "$WITH_TB" = "1" ] && [ -d "$WORK_DIR/logs" ]; then
  while IFS= read -r f; do
    echo "$f" >> "$TMP_LIST"
  done < <(cd "$WORK_DIR" && find logs -type f -name "events.out.tfevents.*" | sort)
fi

count="$(wc -l < "$TMP_LIST" | tr -d ' ')"
if [ "$count" = "0" ]; then
  echo "Nothing to pack (no matching files found)."
  exit 1
fi

mkdir -p "$(dirname "$ARCHIVE_PATH")"
(cd "$WORK_DIR" && tar -czf "$ARCHIVE_PATH" -T "$TMP_LIST")

echo "Archive created: $ARCHIVE_PATH"
echo "Files packed: $count"
