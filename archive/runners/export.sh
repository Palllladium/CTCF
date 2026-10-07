#!/usr/bin/env bash
# Export the P18 loss-ablation best checkpoints.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""

cd "$(git rev-parse --show-toplevel)"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="results/exports/P18_CHECKPOINTS_${stamp}"
mkdir -p "$out"

exps=(
  P18_ABL_VXM_OASIS_FULL
  P18_ABL_VXM_OASIS_NOICON
  P18_ABL_VXM_OASIS_NOJAC
  P18_ABL_VXM_OASIS_NOICON_NOJAC
  P18_ABL_VXM_OASIS_NOREG
  P18_ABL_VXM_OASIS_TRI_MEAN
  P18_ABL_VXM_OASIS_TRI_ACTIVE
  P18_ABL_VXM_OASIS_ICON_L2
  P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.005
  P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.05
)

low_priority() {
  if command -v ionice >/dev/null 2>&1; then
    ionice -c3 nice -n 19 "$@"
  else
    nice -n 19 "$@"
  fi
}

manifest="$out/manifest.tsv"
missing_log="$out/missing.txt"
printf 'experiment\tstatus\trelative_path\tbytes\tmtime_utc\tsha256\n' > "$manifest"
: > "$missing_log"

present_exps=()
paths=()
total=0
for exp in "${exps[@]}"; do
  path="results/${exp}/ckpt/best.pth"
  if [[ -s "$path" ]]; then
    present_exps+=("$exp")
    paths+=("$path")
    total=$((total + $(stat -c '%s' -- "$path")))
  else
    if [[ -e "$path" ]]; then
      reason="exists but is empty"
    else
      reason="not found on disk"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$exp" "MISSING" "$path" "" "" "" >> "$manifest"
    printf '%s\t%s\t%s\n' "$exp" "$path" "$reason" >> "$missing_log"
    echo "[MISSING] $path ($reason) — recorded, continuing" >&2
  fi
done

(( ${#paths[@]} > 0 )) || {
  echo "[NOTHING TO EXPORT] none of the ${#exps[@]} checkpoints is present" >&2
  cat "$missing_log" >&2
  exit 2
}

available="$(df -PB1 "$out" | awk 'NR==2 {print $4}')"
(( available > total + 1073741824 )) || {
  echo "[NO SPACE] required=$((total + 1073741824)) available=$available" >&2
  exit 3
}

declare -A signatures
for i in "${!paths[@]}"; do
  exp="${present_exps[$i]}"
  path="${paths[$i]}"
  size="$(stat -c '%s' -- "$path")"
  mtime_epoch="$(stat -c '%Y' -- "$path")"
  signatures["$path"]="${size}:${mtime_epoch}"

  line="$(low_priority sha256sum -- "$path")"
  sha="${line%% *}"

  [[ "$(stat -c '%s:%Y' -- "$path")" == "${signatures[$path]}" ]] || {
    echo "[CHANGED DURING HASH] $path" >&2
    exit 4
  }

  mtime_utc="$(date -u -d "@${mtime_epoch}" +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$exp" "PRESENT" "$path" "$size" "$mtime_utc" "$sha" >> "$manifest"
done

{
  printf 'exported_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'current_head_at_export=%s\n' "$(git rev-parse HEAD)"
  printf 'note=current HEAD is export context, not historical training HEAD\n'
  printf 'checkpoints_requested=%s\n' "${#exps[@]}"
  printf 'checkpoints_present=%s\n' "${#paths[@]}"
} > "$out/export_context.txt"

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'uname=%s\n' "$(uname -a)"
  printf '\n[nvidia-smi -L]\n'
  nvidia-smi -L 2>&1 || printf 'nvidia-smi unavailable\n'
  printf '\n[python]\n'
  command -v python >/dev/null 2>&1 && python -V 2>&1 || printf 'python not on PATH\n'
  printf '\n[conda list]\n'
  if conda list -p /data/mooncake/P/envs/ctcf 2>/dev/null; then
    :
  elif conda list -n ctcf 2>/dev/null; then
    :
  else
    printf 'conda list failed for both -p /data/mooncake/P/envs/ctcf and -n ctcf\n'
  fi
} > "$out/env_h100.txt" 2>&1

git status --porcelain=v1 > "$out/git_status_at_export.txt"

tmp="$out/P18_best_checkpoints.tar.part"
final="$out/P18_best_checkpoints.tar"

low_priority tar --format=posix -cf "$tmp" \
  "${paths[@]}" \
  "$manifest" \
  "$missing_log" \
  "$out/export_context.txt" \
  "$out/env_h100.txt" \
  "$out/git_status_at_export.txt"

for path in "${paths[@]}"; do
  [[ "$(stat -c '%s:%Y' -- "$path")" == "${signatures[$path]}" ]] || {
    echo "[CHANGED DURING ARCHIVE] $path" >&2
    exit 5
  }
done

mv -- "$tmp" "$final"
low_priority sha256sum -- "$final" > "$final.sha256"

echo "[READY] $final"
echo "[SUMMARY] present=${#paths[@]} missing=$(( ${#exps[@]} - ${#paths[@]} )) of ${#exps[@]}"
cat "$manifest"
cat "$final.sha256"
if [[ -s "$missing_log" ]]; then
  echo "[MISSING LIST]"
  cat "$missing_log"
fi