from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import IXI, OASIS
from experiments.core.cli_ctcf import ctcf_overrides_from_args
from experiments.core.inference_metrics import metric_profile_for, write_results, write_trace
from experiments.core.model_adapters import get_model_adapter
from experiments.core.path_profiles import get_dataset_paths
from utils import (
    NumpyType,
    RegisterModel,
    SegNorm,
    dice_per_label,
    hd95_mean_labels,
    mk_grid_img,
    setup_device,
)
from utils.tto import TTOConfig, refine_flow


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str, strict: bool) -> None:
    """Load checkpoint weights; on strict mismatch, warn and fall back to a tolerant load."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt

    if strict:
        try:
            model.load_state_dict(sd, strict=True)
            return
        except RuntimeError as e:
            print(f"[WARN] Strict checkpoint load failed: {e}")
            print("[WARN] Falling back to tolerant load. Pass --strict_ckpt 0 to silence this warning.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (first 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (first 10): {unexpected[:10]}")


def build_infer_dataset(ds_key: str, files: list[str], atlas_path: str | None):
    """Build the validation/test dataset for inference (intensity + segmentation)."""
    if ds_key == "IXI":
        tfm = transforms.Compose([SegNorm(), NumpyType((np.float32, np.int16))])
        return IXI.IXIBrainInferDataset(files, atlas_path, transforms=tfm)
    if ds_key == "OASIS":
        tfm = transforms.Compose([NumpyType((np.float32, np.int16))])
        return OASIS.OASISBrainInferDataset(files, transforms=tfm)
    raise ValueError(f"Unsupported dataset '{ds_key}' for inference.")


def build_infer_model(args, device):
    """Resolve the adapter and build its model with per-model CLI config keys."""
    adapter = get_model_adapter(args.model)
    # fmt: off
    match adapter.key:
        case "tm-dca": model = adapter.build(time_steps=args.time_steps, config_key=args.tm_config)
        case "utsrmorph": model = adapter.build(config_key=args.utsr_config)
        case "ctcf":
            model = adapter.build(
                time_steps=args.time_steps,
                config_key=args.ctcf_config,
                **ctcf_overrides_from_args(args, prefix="ctcf_"),
            )
        case "voxelmorph": model = adapter.build(config_key=args.vxm_config)
        case "lkunet": model = adapter.build(config_key=args.lku_config)
        case "efficientmorph": model = adapter.build(config_key=args.em_config)
        case "mambamorph": model = adapter.build(config_key=args.mamba_config, diffeomorphic=bool(args.mamba_diffeo))
        case "vmambamorph": model = adapter.build(config_key=args.vmamba_config)
        case "corrmlp": model = adapter.build(img_size=tuple(args.img_size))
        case "sacb": model = adapter.build(img_size=tuple(args.img_size))
        case _: raise ValueError(f"Unknown model: {args.model}")
    # fmt: on
    return adapter, model.to(device)


def save_preview(
    out_png: str,
    x: torch.Tensor,
    y: torch.Tensor,
    x_seg: torch.Tensor,
    y_seg: torch.Tensor,
    def_seg: torch.Tensor,
    def_grid: torch.Tensor | None,
) -> None:
    """Save orthogonal preview panels for fixed/moving/segmentation/grid."""
    import matplotlib.pyplot as plt

    fixed = y.detach().cpu().numpy()[0, 0]
    moving = x.detach().cpu().numpy()[0, 0]
    fixed_seg = y_seg.detach().cpu().numpy()[0, 0]
    warped_seg = def_seg.detach().cpu().numpy()[0, 0]
    grid_vol = None if def_grid is None else def_grid.detach().cpu().numpy()[0, 0]
    d, h, w = fixed.shape
    cz, cy, cx = d // 2, h // 2, w // 2

    def slices(vol):
        return (vol[cz], vol[:, cy, :], vol[:, :, cx])

    rows = [slices(fixed), slices(moving), slices(fixed_seg), slices(warped_seg)]
    titles = ["Fixed", "Moving", "Fixed seg", "Warped seg"]
    if grid_vol is not None:
        rows.append(slices(grid_vol))
        titles.append("Deformed grid")

    fig = plt.figure(figsize=(12, 2.4 * len(rows)))
    n_rows = len(rows)
    for r, (axial, coronal, sagittal) in enumerate(rows):
        ax1 = fig.add_subplot(n_rows, 3, r * 3 + 1)
        ax2 = fig.add_subplot(n_rows, 3, r * 3 + 2)
        ax3 = fig.add_subplot(n_rows, 3, r * 3 + 3)
        gray = "gray" if r in (0, 1, 4) else None
        ax1.imshow(axial, cmap=gray)
        ax2.imshow(coronal, cmap=gray)
        ax3.imshow(sagittal, cmap=gray)
        ax1.set_title(f"{titles[r]} (ax)")
        ax2.set_title(f"{titles[r]} (cor)")
        ax3.set_title(f"{titles[r]} (sag)")
        for ax in (ax1, ax2, ax3):
            ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


class InferRunner:
    """Standalone inference runner for case-level metrics, CSV summary and optional previews."""

    def __init__(self, args):
        self.args = args
        self.device = setup_device(args.gpu, seed=args.seed, deterministic=args.deterministic)

        ds_key = args.ds.upper()
        self.ds_key = ds_key
        ds_paths = get_dataset_paths(args.paths, ds_key)
        if args.use_test and "test_dir" in ds_paths:
            self.test_dir = ds_paths["test_dir"]
        else:
            self.test_dir = ds_paths["val_dir"]
        self.test_files = sorted(glob.glob(os.path.join(self.test_dir, "*.pkl")))
        if not self.test_files:
            raise RuntimeError(f"No .pkl files found in test_dir: {self.test_dir}")

        profile = metric_profile_for(ds_key)
        self.labels = profile.labels
        self.hd95_labels = profile.labels
        self.jac_metrics = profile.jac_metrics
        self.jac_log = profile.jac_log

        atlas_path = str(ds_paths["atlas_path"]).rstrip("/\\") if ds_key == "IXI" else None
        ds = build_infer_dataset(ds_key, self.test_files, atlas_path)
        self.loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.adapter, self.model = build_infer_model(args, self.device)
        self.name = self.adapter.key

        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")
        self.ckpt_path = args.ckpt
        load_checkpoint_state(self.model, self.ckpt_path, strict=bool(args.strict_ckpt))
        self.model.eval()

        self.tto = TTOConfig(
            mode=args.tto_mode,
            steps=args.tto_steps,
            lr=args.tto_lr,
            w_reg=args.tto_w_reg,
            w_jac=args.tto_w_jac,
            jac_mode=args.tto_jac_mode,
            jac_eps=args.tto_jac_eps,
            topo_mask=bool(args.tto_topo_mask),
            svf_int_steps=args.tto_svf_int_steps,
            lr_schedule=args.tto_lr_schedule,
            use_mask=bool(args.tto_mask),
            kan_degree=args.tto_kan_degree,
            kan_k=args.tto_kan_k,
            snapshot_at=tuple(args.tto_trace or ()),
            stop_mode=args.tto_stop,
            fold_k=args.tto_fold_k,
            fold_delta=args.tto_fold_delta,
            fold_check_every=args.tto_fold_check_every,
            plateau_window=args.tto_plateau_window,
            plateau_rel=args.tto_plateau_rel,
        )
        if self.tto.enabled:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _score(self, flow, x_seg, y_seg, reg_nearest) -> dict:
        """Dice + Jacobian metrics for one flow field."""
        with torch.no_grad():
            def_seg = reg_nearest((x_seg.float(), flow.float()))
            dice_lbl = dice_per_label(def_seg.long(), y_seg.long(), labels=self.labels)
            row = {"dice_mean": float(np.mean(dice_lbl))}
            row.update(self.jac_metrics(flow, x_seg))
        return row, def_seg, dice_lbl

    def run(self):
        """Execute inference over the dataset and save per-case metrics."""
        args = self.args
        if args.out_dir:
            out_dir = args.out_dir
        else:
            ckpt_name = Path(self.ckpt_path).stem
            if self.tto.enabled:
                ckpt_name = f"{ckpt_name}__{self.tto.slug()}"
            out_dir = os.path.join("results", "infer", self.ds_key, self.name, ckpt_name)
        os.makedirs(out_dir, exist_ok=True)
        png_dir = os.path.join(out_dir, "png") if args.save_pngs else None
        flow_dir = os.path.join(out_dir, "flows") if args.save_flow else None
        if png_dir is not None:
            os.makedirs(png_dir, exist_ok=True)
        if flow_dir is not None:
            os.makedirs(flow_dir, exist_ok=True)

        print(f"[INFO] Model: {self.name}")
        print(f"[INFO] Checkpoint: {self.ckpt_path}")
        print(f"[INFO] Test dir: {self.test_dir}")
        print(f"[INFO] Out dir: {out_dir}")

        rows = []
        trace_rows = []
        reg_nearest = None
        reg_bilin = None

        for idx, batch in enumerate(self.loader):
            x, y, x_seg, y_seg = [t.to(self.device, non_blocking=True) for t in batch]
            vol_shape = tuple(x.shape[2:])
            if reg_nearest is None:
                reg_nearest = RegisterModel(vol_shape, mode="nearest").to(self.device)
                reg_bilin = RegisterModel(vol_shape, mode="bilinear").to(self.device)
            stem = Path(self.test_files[idx]).stem
            cid = stem[2:] if stem.startswith("p_") else stem

            t0 = time.perf_counter()
            with torch.no_grad():
                flow = self.adapter.forward(self.model, x, y)
            t_fwd = time.perf_counter() - t0

            snapshots = {}
            if self.tto.enabled:
                result = refine_flow(
                    flow.float(),
                    x.float(),
                    y.float(),
                    self.tto,
                    reg_bilin.spatial_trans,
                    mask=x_seg,
                )
                flow, snapshots = result.flow, result.snapshots
            dt = time.perf_counter() - t0

            row, def_seg, dice_lbl = self._score(flow, x_seg, y_seg, reg_nearest)
            row = {"case_id": cid, "time_sec": dt, **row}
            if self.tto.enabled:
                # Numeric only: write_results averages every column it is given.
                row["tto_steps"] = result.steps_run
                row["tto_stopped_early"] = float(result.stop_reason != "fixed")
                row["tto_folds_start"] = result.folds_start
                row["tto_folds_end"] = result.folds_end
                row["tto_fold_budget"] = result.fold_budget
                row["fwd_sec"] = t_fwd

            if args.hd95:
                with torch.no_grad():
                    row["hd95_mean"] = hd95_mean_labels(
                        def_seg.long(),
                        y_seg.long(),
                        labels=self.hd95_labels,
                        spacing=(1.0, 1.0, 1.0),
                    )
            for lbl, v in zip(self.labels, dice_lbl, strict=True):
                row[f"dice_lbl_{lbl}"] = float(v)
            rows.append(row)

            for step, snap in snapshots.items():
                srow, _, _ = self._score(snap, x_seg, y_seg, reg_nearest)
                trace_rows.append({"case_id": cid, "tto_step": step, **srow})

            if args.save_flow:
                np.savez_compressed(os.path.join(flow_dir, f"flow_{cid}.npz"), flow=flow.detach().cpu().numpy())
            if args.save_pngs and (args.png_limit < 0 or idx < args.png_limit):
                with torch.no_grad():
                    grid = mk_grid_img(flow.float(), grid_step=8, line_thickness=1)
                    def_grid = reg_bilin((grid.float(), flow.float()))
                save_preview(os.path.join(png_dir, f"{cid}.png"), x, y, x_seg, y_seg, def_seg, def_grid)

            if (idx + 1) % max(1, args.print_every) == 0:
                msg = (
                    f"[{idx + 1:03d}/{len(self.loader):03d}] {cid} "
                    f"dice={row['dice_mean']:.4f} time={dt:.3f}s{self.jac_log(row)}"
                )
                if args.hd95:
                    msg += f" hd95={row['hd95_mean']:.4f}"
                print(msg)

        write_results(rows, out_dir, model_name=self.name, ckpt_path=self.ckpt_path, test_dir=self.test_dir)
        if trace_rows:
            write_trace(trace_rows, out_dir)
        if args.save_pngs:
            print(f"[SAVED] png dir: {png_dir}")
        if args.save_flow:
            print(f"[SAVED] flow dir: {flow_dir}")
