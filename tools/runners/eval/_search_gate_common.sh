#!/usr/bin/env bash
# Shared, side-effect-free shell mechanics for the C0/C1/C2/C3 search-gate runners.
#
# Only helpers whose behaviour is byte-identical across the runners live here.  Anything
# where the gates genuinely differ - the dirty-tree hint text, the GPU-list aggregate
# checks, MODE handling, preflight expectations, the staging plan - stays in the runner
# that owns it, so no message or exit code changes when this file is sourced.
#
# This file is sourced, never executed:  source "$(dirname "${BASH_SOURCE[0]}")/_search_gate_common.sh"

# Put the repository root on PYTHONPATH without discarding an inherited value.
sg_export_pythonpath() {
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
}

# Compact UTC stamp used inside RUN_ID and ATTEMPT_ID.
sg_utc_run_stamp() {
  date -u +%Y%m%dT%H%M%SZ
}

# ISO-8601 UTC instant recorded as the start of a run.
sg_utc_started_at() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

# Twelve-character HEAD used to make a RUN_ID traceable to the code that produced it.
sg_git_short_head() {
  git rev-parse --short=12 HEAD
}

# Parse a comma-separated GPU list into the caller's GPUS array.
#
# Only the per-item integer check lives here because it is identical in C1 and C2.  The
# aggregate checks (empty list, duplicate indices) stay with the caller: C1 reports them
# as two distinct failures and C2 as one, and those messages are part of each contract.
sg_parse_gpu_list() {
  local raw_list="$1"
  local value gpu
  local -a raw_items=()
  IFS=',' read -r -a raw_items <<< "$raw_list"
  GPUS=()
  for value in "${raw_items[@]}"; do
    gpu="${value//[[:space:]]/}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "[FAIL] GPU_LIST must be a comma-separated list of non-negative integers" >&2
      exit 2
    fi
    GPUS+=("$gpu")
  done
}

# True when a RUN_ID or ATTEMPT_ID is safe to use as a path component.
sg_is_safe_identifier() {
  [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]]
}

# Default locator recorded when the operator did not supply REMOTE_LOCATOR.
sg_default_remote_locator() {
  local package_abs="$1"
  printf 'H100_LOCAL_ARCHIVE=%s;H100_LOCAL_SIDECAR=%s.sha256' "$package_abs" "$package_abs"
}
