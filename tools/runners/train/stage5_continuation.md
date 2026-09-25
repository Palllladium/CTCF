# Continue a completed Stage 5 training run with corrected evaluation code

Use `PHASE=continue-evaluation` when the frozen training revision differs from
the checked-out evaluation revision. Keep the original `RUN_ID`, compact root,
heavy root and data cache. Set `EXPECTED_GIT_HEAD` to the exact new commit and
`TRAINING_GIT_HEAD` to the original protocol's commit.

The runner authenticates the complete checkpoint matrix and source field files,
writes an immutable continuation binding, then runs decisions, the decision
barrier, evaluation, aggregation and ZIP packaging. It does not run training,
source inference or an extra smoke phase. Labels remain inaccessible until the
complete decision barrier has been validated.

The continuation rejects changes to model, training, data and search operator
modules. Only the reviewed post-training modules are permitted to differ. Each decision
records both code revisions and the continuation digest; the final manifest
also identifies the original training revision. Existing training metadata and
source certificates are not rewritten.

Example environment (provide the two exact commits and original run ID):

```bash
nohup env \
  PHASE=continue-evaluation \
  GPU_LIST=0,1,2,3 \
  PYBIN=/path/to/environment/bin/python \
  OASIS_ALL_ROOT=/path/to/OASIS/All \
  EXPECTED_GIT_HEAD="$EVALUATION_HEAD" \
  TRAINING_GIT_HEAD="$TRAINING_HEAD" \
  RUN_ID="$RUN_ID" \
  bash tools/runners/train/stage5.sh >"$LOG" 2>&1 &
```

One worker runs per selected GPU. The number and physical indices may change on
restart. Repeat the same command with the same code and `RUN_ID` and a fresh log
filename. Verified completed decisions/evaluations are reused. An interrupted
decision publication is recovered from its immutable journal; unfinished field
computation is repeated. The run lock prevents simultaneous runners for that run.

A new evaluation commit may reuse decisions written under an earlier continuation
of the same run. `prepare-continuation` authenticates each one (its fields, its
parent continuation, both code revisions and, for pre-margin decisions, that the
clip ran on the unchanged nominal 0.0011 path) and lists the approved digests in
the new binding. Decisions that were never journaled are recomputed.

The safety transaction uses the nominal working margin 0.0011 whenever the stored
source affords it. A certified source whose verified bound lies below 0.0011 uses
that bound instead; the claim 0.001 and the exact certification of the saved float32
result (or byte-identical rollback to the source) are unchanged. The preflight
reports how many sources need the reduced margin before any GPU work starts.

The main log reports phase and worker completion. Detailed progress is in
`logs/<attempt>/prepare_continuation.log`, `decision_gpu_<index>.log`, and
`evaluation_<shard>of<count>.log`. Completion requires a `COMPLETE` manifest and
the evaluation products; a process exit or a single worker's `PASS` is insufficient.

A2P/A24P participate in the full common matrix. Their exact reports additionally
contain descriptive raw-head and attenuation statistics from the same forward
pass. No extra training/backward pass is performed. Observation time and staging
allocations are excluded from the decision performance measurements. These
statistics describe the endpoint policy; they do not establish why training
gradients grew.
