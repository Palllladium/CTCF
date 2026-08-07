# Produce external registration fields for the deployed-warp topology audit (`--2` box)

Goal: show the digital≠trilinear fold gap is **universal** — other methods' deployed fields also fold
trilinearly where their reported metric reads 0. We need each method's **displacement field** on our 19 OASIS
test pairs, plus its **own warped moving image** (so our adapter can validate its convention, not guess it).

Start with **one method, one pair** to de-risk the convention, then scale to all 19 × 2 methods.

Pairs (moving→fixed), same as our eval: `0438_0439, 0439_0440, …, 0456_0457` (consecutive OASIS test subjects).
OASIS test volumes on `--2` are the same ones our runners use (see `experiments/core/path_profiles.py:PATHS`).

---

## Method 1 — FireANTs (easiest: training-free package)

```bash
conda activate ctcf                 # or a fresh env
pip install fireants                # per the FireANTs README (Nature Commun. 2026 release)
```

For each pair `MOV_FIX` (e.g. `0440_0441`): register moving→fixed with a diffeomorphic (SyN-style) preset per
their README, then obtain three things from the result object:
- `disp`   — the displacement field (note whether it is in **voxels or mm**, and the **axis order**),
- `warped` — moving resampled by the transform (their own warp),
- (moving, fixed are the inputs).

Then save in our canonical format (this helper does NOT convert — it records raw + metadata):

```bash
python tools/foreign/save_external_field.py \
    --method fireants --pair 0440_0441 \
    --disp   <their_disp.nii.gz> \
    --moving <moving_0440.nii.gz> --fixed <fixed_0441.nii.gz> \
    --warped <their_warped_moving.nii.gz> \
    --units voxel --axis_order DHW --sign fwd --align_corners unknown \
    --note "FireANTs <preset>, disp=<moving->fixed>, produced by <function>"
```

Fill `--units/--axis_order/--sign` from the FireANTs docs (best guess is fine — the `warped` image lets us
verify it). If it is easier to stay in Python, `import tools.foreign.save_external_field as S; S.save_field(...)`
with numpy arrays straight from the result object.

---

## Method 2 — ConvexAdam-MINDSSC (a search engine)

```bash
git clone https://github.com/multimodallearning/convexAdam
cd convexAdam && pip install -r requirements.txt      # (Apache-2.0)
```

Their `l2r_2021_convexAdam_task3_*` script **is the OASIS pipeline** (L2R Task 3 = OASIS). Run the
**MIND-SSC** variant (not the nnUNet-feature one) on each pair, save the displacement field + warped moving,
then the same `save_external_field.py` call with `--method convexadam`.

---

## What to send back

The whole `results/foreign/` directory:
```
results/foreign/fireants/0440_0441.npz          # {disp, moving, fixed, warped_moving, meta}
results/foreign/convexadam/0440_0441.npz
...
```
Each `.npz` already carries the images + metadata, so nothing else is needed. **Even a single
`results/foreign/fireants/0440_0441.npz` is enough for me to write and validate the adapter** before you
batch the rest — send that one first.

## What happens next (our side)
1. Write a `to_contract()` adapter per method in `tools/analysis/field_audit.py` (sign/axes/units/align_corners).
2. **Validate it** by checking `warp(moving, adapted_disp) ≈ warped_moving` (bit-close) — the real convention
   test, stronger than a synthetic ±1-voxel shift.
3. Run the audit → the G2 table: "method reports 0 folds (central/digital) but folds trilinearly in Y% of
   cells, repaired at Z Dice cost", across methods = the discovery is universal.
