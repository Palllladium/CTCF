# Paired IXI and frozen consensus

The runner uses the existing CTCF trainer. `experiments.contrasts.RECIPE` is the single source for training,
readiness and checkpoint validation: B0=T1 input/loss, B1=T1 input and T1+T2 loss, B2=T1+T2 input/loss;
matched two-channel models, Mamba294 initial weights, seed0, BF16, LR2e-5, w_reg4, 100x399 updates, fixed last.
Primary evaluation is IXI Val58: 29 disjoint pairs, both directions, Dice30 with SegNorm. Directions are averaged
within pairs. IXI-pretrained weights must not enter challenge tracks that prohibit external training data.

## Prepare data on the node

Look for original preprocessing transforms/aligned NIfTI files first. Otherwise the tool estimates affine
processed-T1 -> raw-T1 and rigid raw-T1 -> raw-T2 pull transforms in physical LPS. T2 starts from header alignment
and is resampled once after composition. T1 is unchanged; T2 uses a T1-derived mask, P1/P99 scaling and no N4.
The 397 included Train subjects use FreeSurfer labels only to check laterality after the fitted map, never in registration losses or
normalisation. Val/Test labels do not select orientation. This use of Train labels for integrity QC must be disclosed.

Two Train subjects are explicitly excluded from all B arms because their original T2 scans truncate brain tissue:
IXI539 (`subject_481`, posterior brain) and IXI556 (`subject_494`, cerebellum/brainstem). The source crosswalk and
their QC reports remain intact; `manifest.json` records both reasons in `training_exclusions`. The matched cohort
is Train397/Val58; the training budget remains 100x399 updates. Each epoch covers all 397 moving subjects once,
then draws two more from a new permutation. The pair stream remains identical across B0/B1/B2.

Copy the processed Train/Val directories, crosswalk CSV and extracted official raw T1/T2 files to the node.
`RAW` must contain the filenames from the CSV directly. No downloads run automatically.

```bash
PY=/data/mooncake/P/envs/ctcf/bin/python
"$PY" -m pip install SimpleITK matplotlib
OUT=/data/mooncake/P/IXI_paired_v2
COMMON=(--mapping /data/mooncake/P/ixi_processed_to_original.csv --processed /data/mooncake/P/IXI_data --raw /data/mooncake/P/IXI_raw)
# Revalidate the orientation already found on Train with the new fitting recipe:
"$PY" -m tools.analysis.prepare_ixi calibrate "${COMMON[@]}" --out "$OUT" --geometry /data/mooncake/P/IXI_paired_qc_fixed/geometry_0.json
GEOMETRY="$OUT/geometry_0.json"
"$PY" -m tools.analysis.prepare_ixi prepare "${COMMON[@]}" --out "$OUT" --geometry "$GEOMETRY" --limit 8
```

Calibration with `--geometry` revalidates one supplied orientation. Without it, a cheap coarse search proposes
48 signed axis permutations before checking finalists. Every final Train-panel subject must pass full-resolution
correlation and laterality. Calibration, preparation and wrong-ID controls share one affine recipe: 24 coarse MI
starts (scale 0.70..1.05 in steps of 0.05; LPS superior shifts -10/0/+10 mm), followed by fine fitting in physical
coordinates. Affine start/refinement selection uses freshly evaluated MI on a common grid. Correlation is used
for orientation approval and data QC. Nested composites are flattened before refinement/export.
Correlation remains an intensity QC measure, not independent proof of identity.
Inspect `orientation_scores.json`, Train `alignment.png` and `qc.json` before Val preparation.
Geometry files start with provisional QC limits of 0.99 brain coverage, 0.8 T1 correlation, 20 mm rigid centre
motion and 20 degrees rotation. Review/freeze limits on these Train cases; these are QC defaults, not validated
population limits. Changing geometry/limits requires a new output directory. Left/right QC uses relative
centroids after the full transform, not the sign of absolute coordinates. Image identity still needs review.
Old recipe caches/geometries are rejected. QC flags make preparation exit nonzero after saving reports.
`preparation_profile_0.json` extrapolates measured CPU preparation time to 457 subjects; it is only an estimate.

After the fixed eight pass, `prepare --limit 30` adds Train cases until the panel has 10 per site, preserving
the initial eight. Inspect that panel before full preparation; it does not prove every subject will pass.

```bash
# After Train QC/limits are frozen; keep GEOMETRY pointing to the approved file:
FINAL="$OUT"
"$PY" -m tools.analysis.prepare_ixi prepare "${COMMON[@]}" --out "$FINAL" --geometry "$GEOMETRY" --limit 397
# Continue to Val only after all Train cases pass, without changing geometry/limits:
"$PY" -m tools.analysis.prepare_ixi prepare "${COMMON[@]}" --out "$FINAL" --geometry "$GEOMETRY"
"$PY" -m tools.analysis.prepare_ixi identity "${COMMON[@]}" --out "$FINAL" --geometry "$GEOMETRY"
"$PY" -m tools.analysis.prepare_ixi finalise "${COMMON[@]}" --out "$FINAL"
```

The last command intentionally writes `review_required.json` and stops. Return it with the requested alignment
and wrong-ID previews for review. After review, the project team supplies `$FINAL/reviewed_ids.txt` and the final command;
the node operator does not manually assign IDs or approve unchecked cases.
The 11 manually resolved crosswalk cases and order-only cases are automatically required. Correct failures;
additional subjects cannot be silently dropped. The final manifest freezes Train397/Val58, exclusions, pairings and source hashes.

```bash
"$PY" -m tools.analysis.prepare_ixi finalise "${COMMON[@]}" --out "$FINAL" --reviewed "$FINAL/reviewed_ids.txt"
```

If sequential preparation is too slow, split only the full `prepare` stage using `--shards N --shard i`
(i=0..N-1), with separate logs and sufficient CPU cores. Finalise after every shard finishes. Optional
`--transforms DIR` reads `IXIxxx_processed_to_t1.tfm` and `IXIxxx_t1_to_t2.tfm`: SimpleITK pull transforms in LPS.
Raw FreeSurfer `.xfm` files require explicit coordinate conversion and cannot be passed directly.

## Dispatch after code review and commit

```bash
cd /data/mooncake/P/CTCF
git pull --ff-only
export EXPECTED_GIT_HEAD=<reviewed-code-SHA>
export PYBIN=/data/mooncake/P/envs/ctcf/bin/python
export IXI_MANIFEST=/data/mooncake/P/IXI_paired_v2/manifest.json
export GPUS="0 1 2 3"
bash tools/runners/train/contrasts.sh start
```

The launcher prints RUN_ID/PID/log and backgrounds itself. Four GPUs run three B arms and consensus independently;
fewer cards queue arms without sharing a card. `RUN_B=0` permits consensus while IXI preparation is unfinished;
the package remains incomplete for B. Resume later with `RUN_B=1` and the reviewed manifest at the same code SHA.
Readiness checks FP32 function preservation (BF16 differences are diagnostic), Train gradient dose, full-size
steps, validation, raw/repair/certificate chain and the 72 GPU-hour budget including an 8-hour consensus reserve.
A negative certificate verdict is recorded; exceptions in the chain hold B. Parent evaluation follows B training/evaluation
without waiting for consensus. The final package waits for both queues.

Consensus uses two Train-only banks K1/4/16, direct, Train-calibrated smoothing and a separate pair-excluded Dev
bank. Every variant retains FF/REP Dice under the same repair. Exact certification covers REP of direct/smooth/
both Train K16 means and FF of direct: 500 certificates for 100 directions, not 1800. K1/K4/Dev results are diagnostic,
with no certified claim. Cost is still profiled after two directions; fewer certificates do not remove repair cost.

```bash
RUN_ID=B_<UTC>
tail -n 30 -f "results/contrasts/$RUN_ID/status.txt"
tail -n 25 results/contrasts/"$RUN_ID"/logs/*/*.log
# Stop only the owned process group; wait for packaging before resuming:
bash tools/runners/train/contrasts.sh stop "$RUN_ID"
GPUS="4 5 6 7" bash tools/runners/train/contrasts.sh resume "$RUN_ID"
```

Do not pull/modify the running checkout. Resume restores checkpoints and reuses completed jobs; parent, manifest
and SHA bindings cannot change. Raw scores survive repair/certificate failure. Return `results/exports/B_*__COMPLETE.zip`,
its `.sha256` and outer `runner_*.log` (FAILED/PARTIAL variants on failure/interruption). Weights/fields remain in
`results/<RUN_ID>_{b0,b1,b2}/ckpt/` and `results/contrasts/<RUN_ID>/`; nothing is deleted automatically.
