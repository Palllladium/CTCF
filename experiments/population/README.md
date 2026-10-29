# Population atlas reference

This is a research implementation of a canonical population atlas, not a validated MRI registration recipe.
The current observation model is the normalized Gaussian reference; its suitability remains to be measured.

`geometry.py` retains every coefficient of an orthonormal 3D Haar velocity transform, including odd-size carries.
Zero padding supplies the velocity collar. The subject pull map is `alpha_i = Exp(v_i)` and the fixed-to-moving
pair map is `Exp(-v_m) o Exp(v_f)`. Numerical integration does not itself certify that map.

`model.py` implements three nested prior-mean families, selected by `train --arm`:

- `A0`: learned unconditional spatial mean `mu0(T)`.
- `A_lin`: `mu0(T) + L(context)`.
- `A1`: `mu0(T) + L(context) + g(T, context) - g(T, 0)`.

Here `context` is the reconstructed coarser velocity in voxel units. `L` is a bias-free 5x5x5 convolution,
with the same receptive field as `g`. `L` and the final layer of `g` start at zero, so seeded branches share
their initial field distributions. Inactive heads are frozen and excluded from the weight penalty. Nesting
does not guarantee an ordering of heldout Dice: optimization, regularization and generalization can differ.
An improvement of `A_lin` over `A0` can reflect ordinary smoothness; it does not isolate anatomy.

All use the same image encoder and autoregressive Gaussian posterior. Every posterior head receives the image
features and sampled coarser state. Deterministic encoding follows conditional means; it is neither the exact
joint posterior mean nor the joint mode. Coarse-to-fine initial scales are explicit arguments. A common scale
for every Haar band gives a white velocity prior, not a smooth one.

`appearance.py` integrates positive gain, offset and 26 zero-DC DCT nuisance coefficients exactly. The marginal
retains the Gaussian normalizer, determinant and truncation probabilities. Small matrices and reductions use
FP64. Parameters and geometry use FP32, with optional TF32 for convolutions; training uses no FP16 autocast or scaler.
Forward chunking does not establish a small backward memory footprint; full-volume cost must be profiled.

## Entry points

Run each command with `--help` for its required numerical parameters. No MRI hyperparameter recipe is implicit.

| Module (`python -m experiments.population.<module>`) | Inputs and outputs |
|---|---|
| `probe` | Single template/subject label pickles; label-driven SVF, raw quality diagnostics and Haar energies. `--steps` applies at each explicitly specified smoothing scale. |
| `predictability` | Train/heldout `probe` directories; a coarsest-conditioned ridge readout with raw coefficient errors and label Dice. |
| `observation` | A fixed template, subject and optional canonical flow; appearance marginal and descriptive residual statistics. |
| `train` | Single-subject Train pickles; joint atlas/encoder/prior/noise fit, `last.pth`, per-epoch `metrics.csv`. |
| `evaluate` | Checkpoint plus paired OASIS pickles, or heldout single-subject images for importance diagnostics. |
| `profile` | Worker for the preliminary full-volume dispatch described below. |
| `calibrate` | Expanded image-only calibration, cached subject MAP, and separate label-driven geometry diagnostics. |

Single-subject pickles contain `(image, segmentation)`; main model training uses only the image. The caller must
establish subject-disjoint splits and a common affine-aligned voxel grid. Pickles contain no raw-world affine.
Paired OASIS evaluation expects `(moving, fixed, moving_seg, fixed_seg)` with dense labels 0..35 and scores Dice35.
Path checks do not detect renamed copies or repeated biological subjects.

For `train`, one epoch visits each supplied subject once. A train-mean image initializes the atlas. During warm-up
the generative prior is frozen; all branches receive the same warm-up. The per-subject objective is

```
(NLL_i + KL_i + global_penalty / number_of_train_subjects) / number_of_voxels
```

The global penalty contains atlas roughness, Gaussian priors on prior-network weights/scale logits, and an
Inverse-Gamma prior on the noise variance. Scaling applies to the whole objective, without a separate beta on KL.
Resume repeats the same command with `--resume` and the new total `--epochs`; recipe and CPU/CUDA backend must match.
The checkpoint supplies model, optimizer, completed history and random states. CPU resume is tested exactly;
this does not promise bitwise reproducibility of nondeterministic CUDA operations.

`evaluate --pairs ... --map-steps ... --map-lr ... --repair 1 --certify 1` freezes globals and starts MAP from
the encoder. Adam optimizes the full interior velocity in native voxel units; the prior is evaluated after
the orthonormal Haar transform. This changes optimization coordinates, not the objective or degrees of freedom.
Both MAP flags are required; `--map-steps 0` explicitly selects encoder-only inference. The report retains the
best visited objective, its data/prior terms and per-scale velocity changes. It writes FF/REP fields and geometric/Dice
tables. Repair is terminal and preserves the zero boundary; exact verification reloads the stored FP32 fields.
Predicate failure and resource exhaustion remain separate outcomes. A positive cell predicate alone is not a
global homeomorphism claim; the report also checks the zero boundary.

`evaluate --subjects ... --importance-samples ... --map-steps ... --map-lr ... --repair 0 --certify 0` performs no label evaluation or model
selection. It reports a finite-sample importance estimate and effective sample size. Proposal failure and bias
of the log estimate remain possible, especially in a high-dimensional field.

The `predictability` diagnostic is supervised through its label-fitted input fields. The reference self-fit is
reported for quality control and excluded from regression. Supply either a ridge fixed before inspecting heldout
results, or a grid selected by inner-Train leave-one-subject-out fine-coefficient MSE. Heldout Dice does not select
the ridge. Global linear prediction from the coarsest band and the local nonlinear A1 prior are different,
non-nested functional classes; neither result predicts the other. Point-substitution Dice is not an upper bound
on a soft prior's utility. The ratio `f` is omitted unless a positive denominator floor is supplied explicitly;
it is not a rejection gate. Raw label-fit maps are not treated as ground-truth deformations.

## Preliminary server dispatch

`tools/runners/train/population.sh` runs ten GPU tasks, one process per selected GPU:

- three short fits (`A0`, `A_lin`, `A1`), each two epochs over four subjects, including one warm-up epoch:
  eight updates, followed by 40-step MAP fits on two inner-heldout subjects;
- six multiscale label fits, including the reference self-fit, followed by the K2' ridge readout;
- K3: appearance/residual diagnostics with the frozen Mamba parent, in FP32/TF32.

In parallel, one CPU task checks the new global certificate on all 19 OASIS and 115 IXI materialized fields.
It uses `results/psi_cert/{oas_w4,ixi_w4}/exact_psi.json` and matching `flows/`. Missing Psi payload can be
reconstructed from `results/collar/{OAS_COLLAR_w4,IXI_COLLAR_w4}/flows/` only if its SHA matches the old report.
The cell certificate is reused only for the same bytes; the worst lower-bound case of each set is also certified
from scratch. Missing fields, changed bytes or failed checks make this task fail while GPU tasks continue.
`CERTIFICATE_ROOT` overrides the root `results` directory. Boundary/operator tests run in the pretests as well.

All six subjects come deterministically from the existing frozen OASIS Train split: four inner-Train and two
inner-heldout. The latter are held out only from the small A fit; the historical parent may have seen them.
No Dev/Test subject is used. The Train mean, split, selected input bytes, parent bytes and diagnostic settings
are recorded in `config.json` (or `parent_binding.json` after a delayed parent upload) and checked on continuation.
Residual spectra are descriptive: residuals mix
anatomy, geometry, interpolation and appearance, rather than measuring noise alone.

This package measures full-volume cost and catches fit/IO failures. Eight updates include warm-up and cold
kernels; they do not establish steady-state throughput or a scientific comparison of branches. The Gaussian
reference and its diagnostic hyperparameters are not selected as the final observation model. Main trainings
do not start automatically.

The runner requires `EXPECTED_GIT_HEAD`; `PYBIN`, `OASIS_ALL_ROOT`, `PARENT_CHECKPOINT`, `RUN_ID` and `GPUS`
configure its environment. `GPUS` accepts free numeric device IDs separated by spaces or commas. With the same
`RUN_ID`, completed tasks are skipped only when their completion marker and report both exist. Failed tasks
restart in a new attempt directory, preserving earlier outputs; GPU count and IDs may change. Do not run the
same `RUN_ID` concurrently. The parent checkpoint must exist before dispatch starts.

Per-task logs and compact reports are under `results/population/<RUN_ID>/`. The final single-root ZIP and SHA-256
sidecar are written to `results/exports/`; arrays and checkpoints stay on the server. `COMPLETE` requires all ten
GPU tasks, the CPU certificate checks, pretests and the summary to succeed. GPU topology is included when
`nvidia-smi` is available. Interrupted or failed attempts still produce a compact package.

New branch heads change the layout of earlier debug A checkpoints, which must be regenerated. Existing CTCF
and Mamba checkpoints are unaffected.

Training and evaluation outputs remain exploratory until an observation model, budgets, thresholds and an
independent evaluation protocol are fixed. Synthetic tests and crop-level IO checks do not establish MRI quality.

## Expanded calibration dispatch

`tools/runners/train/calibration.sh` runs one fixed package on disjoint GPU pools. Its defaults allocate seven
GPUs: `A_GPUS="1 2"`, `MAMBA_GPU=3`, `LUMIR_GPUS="4 5 6 7"`. IDs may change; LUMIR needs four devices and
the other two pools need at least one each. Overlapping pools are rejected. Do not run two instances with the
same `RUN_ID`, or update the checkout while it is running.
The worker holds a nonblocking `flock` on `<run>/.lock` before touching logs/status or starting jobs. A second
worker for that RUN_ID is rejected; `resume` also checks the lock before backgrounding. The lock file stays
after exit but its existence does not mean it is held. Do not delete it to bypass an active run.

The A cohort is nested within the frozen 294-subject Train split: 64 and 278 training subjects, with the same
16 inner-heldout subjects. Seven image-only fits use a shared 64-subject initial atlas, appearance settings,
and 320-update prior warm-up:

| Prior | Subjects | Width | Seeds | Epochs |
|---|---:|---:|---|---:|
| A_lin, A1 | 64 | 8 | 0, 1 | 100 |
| A_lin, A1 | 278 | 8 | 0 | 50 |
| A1 | 64 | 16 | 0 | 100 |

Training retains epoch-50/100 and update-3200/6400 milestones when reached. Intra-epoch update snapshots
are evaluation-only: they cannot resume an exact subject-shuffle trajectory. Last checkpoints resume from
the last complete epoch. Full inner-heldout MAP evaluation uses each final checkpoint, plus update 6400 for
the 278-subject fits, matching the update budget of 64x100. Epoch-50 and update-3200 weights are retained
for further readouts; no additional 500-step MAP evaluation is scheduled for them.

For each assessed checkpoint, each inner-heldout subject is fitted once for 500 native-velocity MAP updates.
Steps 0/40/100/200/500 are saved and scored as raw fixed-step diagnostics; final FF/REP evaluation uses the
best data-plus-prior objective visited, never Dice-based state selection. Pair composition and exact cell and
boundary checks reuse the common operators. A calculation completing does not imply its certificate passed:
inspect each field's global verdict and repair cost. Cached final velocities and pair fields are hash-checked.

All 294 Train subjects also receive independent label-driven fits. The reference self-fit is excluded from
regression. A global ridge curve uses 16/32/63/127/277 subjects and selects its penalty by inner-Train LOO;
three local priors use the same existing A0/A_lin/A1 heads, conditioned on true previous scales. These are
supervised diagnostics, not image-only training or a rejection gate. One final A_lin fit also receives a spatial
predictive diagnostic: 32-cube targets, four-voxel excluded halos, encoder/MAP fitted only to visible voxels,
and normalized conditional appearance likelihood versus an identity predictor. Geometry remains plug-in;
this is not exact model evidence, a comparison of observation families, or an automatic model choice.
Failed label fits skip only ridge/local-prior diagnostics: all seven image-only fits still run. Their failure
remains visible in the package status. A historical Mamba reference is scored on exactly the same 16 cyclic
pairs, in TF32, before waiting for Mamba294; `PARENT_CHECKPOINT` selects its weights (default P10 `best.pth`).
Its legacy flow is materialized into sampler-map coordinates before raw/repair scoring. No identity-boundary
assumption or global certificate is claimed for that parent. It saw these images during training, and the
reference is not an equal-inference-time optimization comparison.

A fresh Mamba NoSVF trains from scratch on the 294 Train subjects, with both pair partners restricted to
Train. Its fixed recipe is BF16, batch 1, LR 1e-4, 500 epochs, last checkpoint, and validation on the frozen
`pair_manifest.json`: 50 Dev pairs in both directions. The split and pair manifests are copied and hashed.
The trainer also writes a Dev-selected `best.pth`; only `last.pth` belongs to this comparison.
It sees the inner-heldout 16, so comparisons with it there are descriptive; the clean comparison is on Dev100.
The historically inspected OASIS-19 is not a new independent final test.
Mamba294 versus historical P10 is not an isolated effect of the data split: update counts, training schedule,
resume history and potentially precision differ. The historical log does not establish its precision.

Before heavy concurrent workloads, five sequential LUMIR profiles measure full-stage BF16 on one GPU/global8,
four GPUs/global8, four GPUs/global4, and four GPUs/micro2/global8, plus a four-GPU TF32/global8 reference.
Each uses 5 warm-up and 20 measured updates. Micro2 may OOM; optional profile failures do not choose a recipe
or block the independently checked production configuration. Eight-GPU scaling is not measured in this allocation.

The node readiness check then compares two continuous runs with actual TERM/resume (four epochs, ten updates
per epoch, global4, BF16). It exercises off/partial/full cascade graphs, optimizer and per-rank RNG restoration,
CSV continuity, all official export members and one independently reproduced displacement. Engineering limits
for the first continuous/resumed exported field are RMS <=0.05 and maximum <=0.5 voxels; same-checkpoint export
reproduction requires maximum <=1e-4 voxels. The continuous repeat is tested against the same fixed bounds;
excess noise is reported separately from a resume mismatch. Two repeats do not estimate a reliable noise
distribution, and cannot silently enlarge the bounds. These are operational tolerances, not scientific equivalence claims.
A supplemental global8 ramp run uses five updates per epoch: both batch sizes see 160 pairs with matching
per-epoch stage progress and LR. This short quality contrast is descriptive and does not choose the long recipe.

After the mandatory global4 readiness PASS, the authorized LUMIR control starts automatically: Mamba NoSVF,
four GPUs, BF16, microbatch 1/global4, LR 1e-4, 40 epochs x1000 updates =160,000 pair presentations, with the
normal cascade ramp. `NCCL_NVLS_ENABLE=0` is fixed throughout; DDP and other NCCL communication remain enabled.
A readiness failure blocks this control; a failed optional profile or an independent A/Mamba task does not.

After code review and commit/push, replace `<reviewed-commit-sha>` below with the actual reviewed commit.
The launcher checks imports of nibabel/scipy/pytest in the existing environment; it does not install or upgrade
packages. Linux `flock` must be available. P10's checkpoint path is checked immediately, before backgrounding
or starting any training; supply `PARENT_CHECKPOINT` for a different location. The worker prints `df` and requires 200 GiB free at startup (`MIN_FREE_GIB` explicitly overrides this
rough initial allowance, including on resume). This is not a guarantee against other jobs filling the disk.
Existing processed OASIS
All and the LUMIR 2024/2025 layout documented in `lumir.sh` must already be present.

```bash
cd /data/mooncake/P/CTCF &&
git pull --ff-only &&
EXPECTED_GIT_HEAD="<reviewed-commit-sha>" \
A_GPUS="1 2" MAMBA_GPU=3 LUMIR_GPUS="4 5 6 7" \
PYBIN=/data/mooncake/P/envs/ctcf/bin/python PRETESTS=1 \
bash tools/runners/train/calibration.sh start
```

The script backgrounds the worker and prints PID/RUN_ID/LOG. Use `tail -f <printed-LOG>` to follow it.
That assignment-only PID/RUN_ID/LOG line can be pasted back into the shell to set the variables for monitoring.
The SHA above is a placeholder, not a launchable value. Follow task logs under
`results/calibration/$RUN_ID/logs/`, the common `status.txt`, and long-training logs
`logs/${RUN_ID}_mamba294/logfile.log` and `logs/$(cat results/calibration/$RUN_ID/lumir_exp.txt)/logfile.log`.
`Ctrl+C` exits `tail` only. `kill -TERM <printed-PID>` stops
the wrapper and its owned workers and creates a PARTIAL package. Resume with the same RUN_ID and HEAD:

```bash
cd /data/mooncake/P/CTCF &&
EXPECTED_GIT_HEAD="<same-reviewed-commit-sha>" \
A_GPUS="1 2" MAMBA_GPU=3 LUMIR_GPUS="4 5 6 7" \
PYBIN=/data/mooncake/P/envs/ctcf/bin/python PRETESTS=1 \
bash tools/runners/train/calibration.sh resume "<original-RUN_ID>"
```

Completed tasks are skipped, interrupted training restarts from its last completed epoch, and incomplete
subject fits are recomputed. If LUMIR was interrupted before its first checkpoint, a fresh EXP is bound in
`lumir_exp.txt`; the old directory remains intact. Prior attempts are retained. A new RUN_ID starts a new experiment.
To change GPU assignments, first stop the wrapper and wait for its PARTIAL package and workers to exit, then
resume the same RUN_ID/HEAD with new disjoint physical IDs. A_GPUS may contain one or more space-separated IDs;
MAMBA_GPU is one ID and LUMIR_GPUS must contain exactly four. Thus the full package needs at least six cards.
For six, use A_GPUS="2", MAMBA_GPU=3, LUMIR_GPUS="4 5 6 7"; for eight fully free cards use A_GPUS="0 1 2"
with the other two pools unchanged. LUMIR's GPU count/global batch are part of the fixed recipe, not a live toggle.

As soon as A and LUMIR finish, return `results/exports/${RUN_ID}__<attempt>__A_LUMIR__COMPLETE.zip`, its
`.sha256` and the outer log. This archive's required-step list explicitly excludes Mamba294; it may still run.
After Mamba finishes, return the final `${RUN_ID}__<attempt>__COMPLETE.zip` and `.sha256`. Failure/interruption
packages use FAILED/PARTIAL. The compact archive includes A diagnostics and external LUMIR/Mamba logs and
manifests. Checkpoints, subject velocities, pair fields and official submission ZIPs remain on the node.
Official long-control displacements are under
`results/lumir/<EXP from lumir_exp.txt>/attempts/<attempt>/export_lumir25/`; they are not automatically submitted.
A technically COMPLETE archive is not a verdict
that A improves Dice or that its fields all passed certification.
