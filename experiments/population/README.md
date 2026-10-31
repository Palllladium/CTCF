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

`tools/runners/train/calibration.sh` runs one package on disjoint GPU pools. Its defaults allocate six GPUs:
`A_GPUS="2"`, `MAMBA_GPU=3`, `LUMIR_GPUS="4 5 6 7"`. IDs follow `nvidia-smi` order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`).
LUMIR needs exactly four devices and the other two pools at least one each; overlapping pools are rejected.
Do not run two instances with the same `RUN_ID`, or update the checkout while it is running.
The worker holds a nonblocking `flock` on `<run>/.lock` before touching logs/status or starting jobs. A second
worker for that RUN_ID is rejected; `resume` also checks the lock before backgrounding. The lock file stays
after exit but its existence does not mean it is held. Do not delete it to bypass an active run.

A jobs share one queue of `A_SLOTS` (default 3) slots per GPU: tasks peak at 7-14 GiB of 80, and certificates run on
the CPU. Each stage waits for the previous one; a failed job stays visible and cancels nothing else. When LUMIR or
Mamba finish or fail, their GPUs join the queue. A job killed by SIGKILL restarts from its own checkpoints up to
`KILL_RETRIES` times (default 2), after `KILL_RETRY_SECONDS` (default 120). Stopping the wrapper stops the lanes first,
so an intentional stop is never undone. The run has no time limit. Times of single jobs measured earlier do not
predict this queue, because three jobs share a GPU.
Temporary files go to `results/tmp`, because the system disk of the node may be nearly full.

### A

The A cohort is nested within the frozen 294-subject Train split: 64 and 278 training subjects, with the same
16 inner-heldout subjects. Image-only fits use a shared 64-subject initial atlas, appearance settings,
and 320-update prior warm-up:

| Prior | Subjects | Width | Seeds | Epochs | Updates |
|---|---:|---:|---|---:|---:|
| A_lin, A1 | 64 | 8 | 0, 1 | 100, then 300 | 6400, then 19200 |
| A0 | 64 | 8 | 0 | 100, then 300 | 6400, then 19200 |
| A_lin, A1 | 278 | 8 | 0 | 100 | 27800 |
| A1 | 64 | 16 | 0 | 100 | 6400 |

Training NLL at 6400 updates is still falling, so a comparison there alone is a transient reading. The extension
continues each 64-subject arm in `arms/<arm>/ext/` from a copy of `last.pth`, leaving the evaluated checkpoint and
its cached MAP byte-identical. Intra-epoch update snapshots are evaluation-only: they cannot resume an exact
subject-shuffle trajectory.

Evaluation points, each with the same 500-step MAP on the 16 inner-heldout subjects and 16 cyclic pairs:

| Point | Arms | Certificate |
|---|---|---|
| `last` | every 64-subject arm (6400 updates); 278-subject arms (27800) | yes |
| `update_6400` | 278-subject arms, matching 64x100 | yes |
| `update_19200` | 278-subject arms, matching 64x300 | no |
| `ext/last` | extended arms (19200) | yes |
| `ext/epoch_200`, `update_3200` | seed-0 A_lin and A1: a Dice-against-updates curve | no |

Each subject is fitted once for each checkpoint. Steps 0/40/100/200/500 are scored as raw fixed-step diagnostics; final
FF/REP evaluation uses the best data-plus-prior objective visited, never Dice-based state selection. The step arrays are
removed once their readouts exist; final velocities and pair fields stay and are hash-checked on reuse.
A collar-zero pair map fixes the box boundary, but CUDA division can round (N-1)/(N-1) below one and leak nanovoxels
onto a face. That rounding is removed before any bytes are written, and its size is reported; a larger boundary
value refuses the zero-boundary repair. The raw field and its Dice are written first, so a repair or certificate
failure cannot lose them. Such a failure stays explicit: the pair records it, the task exits non-zero after writing
every completed score, and the next attempt recomputes only that pair. A negative certificate verdict is an outcome
of a completed step.

Before any other evaluation one real full-size pair from `lin64_s0` passes the whole chain. `pair_check.json` is
written only if the raw chain held, with status `PASS` or, when repair or certification raised, `RAW_ONLY`. Without
that file every evaluation job is held with exit code 125, because it would fail the same way, while training goes on.
With `RAW_ONLY` evaluation runs, since its raw scores are valid, and the task still ends non-zero.

Further readouts on the same evaluation machinery:

- `diagnostics.json`: per-stage detail RMS, the RMS of each prior-mean term, the detail energy each cumulative sum
  explains at the encoder and MAP states, learned band scales, noise, across-subject spread by scale, and the distance
  of each pair field from the label-driven pair map, whole-brain and by scale. These describe what an arm uses;
  explained energy at an inferred state is not predictive evidence, and the label-driven map is not a true deformation.
- `atlas_preview.png`: atlas, one subject and the warped atlas, for visual inspection.
- `<point>_train`: 8 training subjects of the arm, raw Dice and MAP terms, the training side of a generalization gap.
- `predictive.json` on seed-0 A_lin and A1 at `last` and `ext/last`: 32-cube targets, four-voxel excluded halos,
  encoder/MAP fitted only to visible voxels, normalized conditional appearance likelihood versus an identity predictor.
  Geometry remains plug-in; this is not exact model evidence or an automatic model choice.
- `last_dev` on 278-subject arms: 100 subject fits and the frozen 50 Dev pairs in both directions, the clean comparison
  with Mamba294. `last_oasis19`: descriptive placement on the long-inspected OASIS-19; no setting is chosen from it.

All 294 Train subjects also receive independent label-driven fits. The reference self-fit is excluded from
regression. A global ridge curve uses 16/32/63/127/277 subjects and selects its penalty by inner-Train LOO;
three local priors use the same existing A0/A_lin/A1 heads, conditioned on true previous scales. These are
supervised diagnostics, not image-only training or a rejection gate. Failed label fits skip only ridge/local-prior
diagnostics; their failure remains visible in the package status.

### Mamba references

A fresh Mamba NoSVF trains from scratch on the 294 Train subjects, with both pair partners restricted to Train: BF16,
batch 1, LR 1e-4, validation on the frozen `pair_manifest.json` (50 Dev pairs in both directions), last checkpoint.
The trainer also writes a Dev-selected `best.pth`; only `last.pth` belongs to this comparison.
Its first 52 epochs ran with a 500-epoch LR horizon. Once the full cascade is on, an epoch costs about 320 s, so the
run continues with `--max_epoch 250 --schedule_max_epoch 500`. The cascade ramp is unchanged, the LR curve is not: LR
drops from 1e-4 to about 9.4e-5 at epoch 52 and reaches its 2e-5 floor at about epoch 215, where the 500-epoch recipe
would hold 1e-4 until epoch 75. This is an approved change of recipe, not a plain continuation. The historical P10
curve (Dice 0.8304 at epoch 200, 0.8314 at 300, 0.8315 at 499 on OASIS-19) is the reason for accepting this horizon,
not evidence for this model.

`--task reference` scores a checkpoint with the same FF/REP columns, in TF32, on `heldout`, `dev` or `oasis19`:
Mamba294 after its training, the historical P10 (`PARENT_CHECKPOINT`, default `best.pth`) on `dev` and `oasis19`.
Mamba294 and P10 saw the 16 inner-heldout subjects, and P10 saw Dev. Those readings are descriptive; the clean
comparison is Mamba294 against A on Dev. Mamba294 versus P10 is not an isolated effect of the data split: update
counts, schedule, resume history and potentially precision differ. No global certificate is claimed for references.

### LUMIR

Five sequential profiles measure full-stage BF16 on one GPU/global8, four GPUs/global8, four GPUs/global4 and four
GPUs/micro2/global8, plus a four-GPU TF32/global8 reference: 5 warm-up and 20 measured updates. Optional profile
failures do not choose a recipe or block the production configuration.

The node readiness check then runs two continuous trainings and one actual TERM/resume (four epochs, ten updates per
epoch, global4, BF16). Mandatory: real interruption and continuation, CSV continuity, off/partial/full cascade graphs,
finite state and equal optimizer counters, every official export member, the restored model, optimizer and per-rank RNG
digests equal to the interrupted checkpoint, and one exported displacement reproduced from the same checkpoint within
1e-4 voxels. The spread between independent trainings is written to `agreement.json` only: it measures nondeterminism,
and one repeat cannot turn it into a resume criterion. A supplemental global8 ramp is a descriptive short contrast.

After the mandatory readiness PASS the authorized control starts automatically: Mamba NoSVF, four GPUs, BF16,
microbatch 1/global4, LR 1e-4, 40 epochs x1000 updates = 160,000 pair presentations and 40,000 updates, with the normal
cascade ramp. `NCCL_NVLS_ENABLE=0` is fixed throughout. A readiness failure blocks this control and hands its GPUs to A.
The training monitor is Dice over 9 coarse classes on 18 T1 pairs, where zero displacement already scores 0.727:
a trend, not the official fine-label Dice.

From the same last checkpoint two official 36-pair submissions are written, neither is submitted automatically:
`attempts/<attempt>/export_lumir25/` (raw) and `certified_lumir25/`. The latter tapers each field to an identity boundary
over 4 voxels, repairs it with that boundary fixed, rereads the written NIfTI bytes and records the cell and boundary
verdicts, the change to the field and, on labelled pairs, coarse Dice and fold counts before and after. `recipe.json`
binds the folder to the checkpoint hash and the settings, so a resumed export cannot mix fields of different weights.

Right after readiness, one labelled pair of the readiness checkpoint passes this export (`readiness/certified_check/`).
That checks the code path on real data, on a barely trained field, not how the repair behaves on a trained one. Its
failure holds only `certified_lumir25/`: the control and its raw ZIP do not depend on it, and the checkpoint stays.

### Running

After code review and commit/push, replace `<reviewed-commit-sha>` below with the actual reviewed commit.
The launcher checks imports of nibabel/scipy/pytest in the existing environment; it does not install or upgrade
packages. Linux `flock` must be available. P10's checkpoint path is checked immediately; supply `PARENT_CHECKPOINT` for a
different location. The worker prints `df` and requires 200 GiB free at startup (`MIN_FREE_GIB` overrides this rough
allowance, including on resume). This is not a guarantee against other jobs filling the disk. Existing processed OASIS
All and Test and the LUMIR 2024/2025 layout documented in `lumir.sh` must already be present.

```bash
cd /data/mooncake/P/CTCF &&
git pull --ff-only &&
EXPECTED_GIT_HEAD="<reviewed-commit-sha>" \
A_GPUS="2" MAMBA_GPU=3 LUMIR_GPUS="4 5 6 7" \
PYBIN=/data/mooncake/P/envs/ctcf/bin/python PRETESTS=1 \
bash tools/runners/train/calibration.sh start
```

The script backgrounds the worker and prints PID/RUN_ID/LOG. Use `tail -f <printed-LOG>` to follow it.
That assignment-only PID/RUN_ID/LOG line can be pasted back into the shell to set the variables for monitoring.
Follow task logs under `results/calibration/$RUN_ID/logs/`, the common `status.txt`, and long-training logs
`logs/${RUN_ID}_mamba294/logfile.log` and `logs/$(cat results/calibration/$RUN_ID/lumir_exp.txt)/logfile.log`.
`Ctrl+C` exits `tail` only. `kill -TERM <printed-PID>` stops the wrapper and its owned workers and creates a PARTIAL
package. Resume with the same RUN_ID; a later commit continues only with `ALLOW_NEW_HEAD=1`, and `heads.txt` records
which HEAD each attempt ran:

```bash
cd /data/mooncake/P/CTCF &&
EXPECTED_GIT_HEAD="<reviewed-commit-sha>" ALLOW_NEW_HEAD=1 \
A_GPUS="2" MAMBA_GPU=3 LUMIR_GPUS="4 5 6 7" \
PYBIN=/data/mooncake/P/envs/ctcf/bin/python PRETESTS=1 \
bash tools/runners/train/calibration.sh resume "<original-RUN_ID>"
```

Completed tasks are skipped, interrupted training restarts from its last completed epoch, and incomplete subject fits
are recomputed. If LUMIR was interrupted before its first checkpoint, a fresh EXP is bound in `lumir_exp.txt`; the old
directory remains intact. Prior attempts are retained. A new RUN_ID starts a new experiment. To change GPU assignments,
first stop the wrapper and wait for its PARTIAL package and workers to exit. LUMIR's GPU count/global batch are part of
the fixed recipe, not a live toggle.

Return each archive with its `.sha256` as it appears in `results/exports/`, and the outer log:
`${RUN_ID}__<attempt>__A64__*.zip` (the 64-subject Dice, with certificates), `...__LUMIR__*.zip` (after the control),
`...__A_LUMIR__*.zip` (after A and LUMIR), and the final `${RUN_ID}__<attempt>__*.zip`. Required-step lists differ, so an
early archive may be COMPLETE while other lanes run. The compact archives include A diagnostics and external LUMIR/Mamba
logs and manifests; checkpoints, subject velocities, pair fields and official submission ZIPs remain on the node.
A technically COMPLETE archive is not a verdict that A improves Dice or that its fields all passed certification.
