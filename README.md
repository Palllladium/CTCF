# CTCF: Cascade Transformer for Coarse-to-Fine Unsupervised Medical Image Registration

> **Status (June 2026):** This is the published **v1.0.0** code for the CTCF paper; the
> results below are stable. Active development continues on a feature branch — the cascade
> is being generalized into a backbone-pluggable framework (VoxelMorph, LKU-Net, MambaMorph,
> VMambaMorph, EfficientMorph) and the codebase is undergoing a refactor. Some file paths in
> this README may differ from the current tree until the next tagged release.

A three-level coarse-to-fine cascade framework for unsupervised deformable 3D medical image registration.

**Paper (preprint):** [CTCF: A Three-Level Coarse-to-Fine Cascade for Unsupervised Deformable Medical Image Registration](https://doi.org/10.20944/preprints202604.0580.v1) — Preprints.org, 2026.

**Pretrained checkpoints:** [10.5281/zenodo.19665292](https://doi.org/10.5281/zenodo.19665292) — CTCF and both baselines on OASIS + IXI.

## Architecture

CTCF wraps a lightweight coarse-and-refine envelope around an existing single-pass registration backbone (TransMorph-DCA):

- **Level 1** — CoarseFlowNet (3.19M params): convolutional U-Net at 1/4 resolution for global alignment.
- **Level 2** — Swin-DCA + SR decoder (287.11M params): dual-stream Swin Transformer with DCA at 1/2 resolution.
- **Level 3** — FlowRefiner (5.66M params): error-driven convolutional U-Net at 1/2 resolution using NCC error maps.

Total: **295.96M** parameters. Levels 1 and 3 add only **3.0%** overhead over the TransMorph-DCA backbone.

A smoothstep warmup schedule gradually activates the outer cascade levels during training.

![CTCF Architecture](figures/architecture_ctcf_pipeline.png)

<details>
<summary>Building blocks</summary>

![Building blocks](figures/architecture_ctcf_blocks.png)
</details>

## Results

All models trained **unsupervised** (NCC + regularization, no segmentation labels during training).

### OASIS (19 test pairs)

| Method | Dice | HD95 | SDlogJ | Fold% | Params |
|--------|------|------|--------|-------|--------|
| TransMorph-DCA | 0.8145 | 1.848 | 0.0805 | 0.264 | 283.93M |
| UTSRMorph (Large) | 0.8172 | 1.890 | 0.1015 | 0.890 | 421.50M |
| **CTCF (ours)** | **0.8208** | **1.790** | **0.0797** | 0.523 | 295.96M |

### IXI (115 test subjects)

| Method | Dice | HD95 | SDlogJ | Fold% | Params |
|--------|------|------|--------|-------|--------|
| TransMorph-DCA | 0.7456 | 3.504 | 0.0874 | 1.153 | 283.93M |
| UTSRMorph (IXI-Large) | 0.7602 | 3.012 | 0.0627 | 0.677 | 152.23M |
| **CTCF (ours)** | **0.7624** | **2.843** | **0.0594** | **0.561** | 295.96M |

All Dice improvements are statistically significant (p < 0.001, Wilcoxon signed-rank test).

### Visual Comparison

| OASIS | IXI |
|:-----:|:---:|
| ![OASIS boxplot](figures/boxplot_oasis.png) | ![IXI boxplot](figures/boxplot_ixi.png) |

### Qualitative Examples

| OASIS | IXI |
|:-----:|:---:|
| ![Qualitative OASIS](figures/qualitative_oasis_v2.png) | ![Qualitative IXI](figures/qualitative_ixi_v2.png) |

## Installation

```bash
conda env create -f environment.yml
conda activate ctcf
```

## Datasets

Both datasets are used in their `.pkl`-format preprocessed versions
redistributed by the [TransMorph project][transmorph-repo]:

- **OASIS** — Learn2Reg 2021 Task 3 preprocessing (skull stripping, bias-field
  correction, affine alignment to MNI 152, FreeSurfer segmentation of 35
  labels). Download via TransMorph's [OASIS page][transmorph-oasis] (~1.3 GB).
- **IXI** — FreeSurfer-segmented T1 volumes (30 anatomical labels) with a
  template atlas from CycleMorph. Redistributed under CC BY-SA 3.0 Unported.
  Download via TransMorph's [IXI page][transmorph-ixi].

Both datasets come preprocessed to 160×192×224.

If you use these data, please cite:

- **OASIS:** Marcus et al., *J. Cogn. Neurosci.* 19:1498–1507 (2007);
  Hoopes et al., *IPMI 2021* (preprocessing / HyperMorph release).
- **IXI:** the IXI consortium at <https://brain-development.org/ixi-dataset/>.
- **TransMorph** (for both `.pkl` distributions): Chen et al., *Med. Image
  Anal.* 82:102615 (2022).
- **CycleMorph** (IXI atlas): Kim et al., *Med. Image Anal.* 71:102036 (2021).

[transmorph-repo]: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration
[transmorph-oasis]: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md
[transmorph-ixi]: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md

## Path Configuration

All training and inference scripts read dataset locations from the `PATHS` dict
in [experiments/core/train_runtime.py](experiments/core/train_runtime.py).
Before running anything, open that file and edit profile `1` so that the
`train_dir`, `val_dir`, `test_dir` (IXI only), and `atlas_path` (IXI only) keys
point to your local OASIS / IXI data roots. Every command below uses the `--1`
flag to select this profile.

## Pretrained Checkpoints

Pretrained weights for CTCF and both baselines on both datasets are hosted on
Zenodo: **[10.5281/zenodo.19665292](https://doi.org/10.5281/zenodo.19665292)**
(record page: <https://zenodo.org/records/19665292>).

The bundle is split into three archives — one per model family (`CTCF`,
`TM-DCA`, `UTSRMorph`). Download all three and extract them into a common
parent directory; each archive contributes one model subtree to a shared
`Checkpoints/` root, yielding:

```
Checkpoints/
├── CTCF/
│   ├── OASIS/best.pth
│   └── IXI/best.pth
├── TM-DCA/
│   ├── OASIS/best.pth
│   └── IXI/best.pth
└── UTSRMorph/
    ├── OASIS/best.pth
    └── IXI/best.pth
```

All inference commands below assume this layout; if you place `Checkpoints/`
elsewhere, just pass the appropriate path via `--ckpt`.

## Quick Start

### Training

```bash
# CTCF
python -m experiments.train_CTCF --ds OASIS --1
python -m experiments.train_CTCF --ds IXI --1

# Baselines
python -m experiments.train_TransMorphDCA --ds OASIS --1
python -m experiments.train_UTSRMorph --ds OASIS --1
```

### Inference — Reproducing the Paper Metrics

**CTCF (ours):**

```bash
# OASIS — 19 test pairs
python -m experiments.inference --model ctcf \
  --ckpt Checkpoints/CTCF/OASIS/best.pth \
  --ds OASIS --1 --hd95

# IXI — 115 test subjects
python -m experiments.inference --model ctcf \
  --ckpt Checkpoints/CTCF/IXI/best.pth \
  --ds IXI --1 --use_test --hd95
```

**TransMorph-DCA:**

```bash
python -m experiments.inference --model tm-dca \
  --ckpt Checkpoints/TM-DCA/OASIS/best.pth \
  --ds OASIS --1 --hd95

python -m experiments.inference --model tm-dca \
  --ckpt Checkpoints/TM-DCA/IXI/best.pth \
  --ds IXI --1 --use_test --hd95
```

**UTSRMorph** (the config key differs between OASIS and IXI):

```bash
python -m experiments.inference --model utsrmorph \
  --ckpt Checkpoints/UTSRMorph/OASIS/best.pth \
  --ds OASIS --1 --hd95 --utsr_config UTSRMorph-Large

python -m experiments.inference --model utsrmorph \
  --ckpt Checkpoints/UTSRMorph/IXI/best.pth \
  --ds IXI --1 --use_test --hd95 --utsr_config UTSRMorph-IXI-Large
```

Per-case metrics are written to
`results/infer/<DS>/<model>/best/per_case.csv`, and aggregate mean±std to
`summary.json` alongside it.

### Cross-Dataset Zero-Shot (Table 6 in paper)

```bash
bash tools/run_cross_inference.sh --paths-profile 1 --gpu 0
```

### Common Inference Flags

| Flag | Purpose |
|------|---------|
| `--ds OASIS` / `--ds IXI` | Dataset selector |
| `--use_test` | IXI only — evaluate on the 115-subject test split instead of the 58-subject val split |
| `--hd95` | Add HD95 to the reported metrics (Dice and SDlogJ / Fold% are always computed) |
| `--utsr_config` | `UTSRMorph-Large` for OASIS, `UTSRMorph-IXI-Large` for IXI |
| `--save_pngs` | Save qualitative preview PNGs |
| `--save_flow` | Save the predicted flow fields as compressed `.npz` |

### Ablation Experiments

All ablation rounds from the paper can be reproduced with a single script:

```bash
# Run a specific round
bash tools/ablation.sh R1 --gpu 0

# Run all rounds sequentially
bash tools/ablation.sh all
```

Rounds:
- R1 (loss/strategy),
- R2 (L3 tuning),
- R3 (L1 capacity),
- R4 (cascade decomposition),
- R5 (resolution scaling),
- R6 (capacity ablation).

### Key Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max_epoch` | 500 | Training epochs |
| `--w_reg` | auto | Diffusion regularization weight (IXI=4.0, others=1.0) |
| `--w_icon` | 0.05 | ICON loss weight |
| `--w_jac` | 0.005 | Jacobian penalty weight |
| `--l1_base_ch` | 32 | Level 1 base channels |
| `--l3_base_ch` | 64 | Level 3 base channels |
| `--l3_error_mode` | ncc | Error map: `absdiff`, `gradmag`, or `ncc` |
| `--time_steps` | 6 | L2 integration steps |

## Project Structure

```
models/CTCF/
  model.py          # CTCF_CascadeA: main forward pass, composes L1+L2+L3 flows
  stages.py         # L1 (CoarseFlowNetQuarter), L2 (CTCF_DCA_CoreHalf), L3 (FlowRefiner3D)
  configs.py        # CtcfConfig dataclass
  blocks.py         # Swin Transformer blocks, DCA attention

models/TransMorph_DCA/  # Baseline: TransMorph-DCA
models/UTSRMorph/       # Baseline: UTSRMorph

experiments/
  train_CTCF.py         # CTCF training (Runner class, CLI args)
  train_TransMorphDCA.py
  train_UTSRMorph.py
  inference.py          # Unified inference and evaluation
  core/
    train_runtime.py    # Path profiles, run_train() entry point
    train_rules.py      # Dataset defaults (cascade schedule, hyperparams)
    model_adapters.py   # CLI args -> CtcfConfig bridge

utils/
  losses.py         # NCC, ICON, Jacobian, diffusion regularization
  field.py          # Flow composition, warping, identity grid
  validation.py     # Dice, SDlogJ, fold% evaluation
  spatial.py        # SpatialTransformer

datasets/
  OASIS.py          # OASIS dataloader (414 volumes, 35 regions)
  IXI.py            # IXI dataloader (576 volumes, 30 regions)

tools/
  ablation.sh   # Unified ablation runner (R1-R6)
  count_params.py   # Parameter counting utility
  compute_stats.py          # Statistical tests (Wilcoxon, Hodges-Lehmann)
  paper/            # Figure generation scripts
```

## Notes

- `logs/` and `results/` are not version-controlled.
- Baselines use original authors' codebases with minimal modifications (data loaders and logging only).
- CTCF uses bidirectional training (forward + backward per iteration).

## Citation

Main paper (preprint):

```bibtex
@article{pasenko2026ctcf,
  author  = {Pasenko, Daniil V. and Davydov, Roman},
  title   = {{CTCF}: A Three-Level Coarse-to-Fine Cascade for Unsupervised Deformable Medical Image Registration},
  journal = {Preprints},
  year    = {2026},
  doi     = {10.20944/preprints202604.0580.v1}
}
```

Earlier conference version (ElCon-CN 2026):

```bibtex
@inproceedings{pasenko2026ctcf_elcon,
  author    = {Pasenko, Daniil V.},
  title     = {{CTCF}: Cascaded Transformer with Cross-Attention and Super-Resolution for Unsupervised Medical Image Registration},
  booktitle = {2026 ElCon Conference of Young Researchers (ElCon-CN)},
  pages     = {120--127},
  year      = {2026},
  doi       = {10.1109/ElCon-CN69892.2026.11414003}
}
```
