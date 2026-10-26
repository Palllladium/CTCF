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
