import argparse
import csv
import glob
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import OASIS, IXI
from experiments.core.train_runtime import PATHS, add_common_args
from experiments.core.model_adapters import get_model_adapter
from utils import (
    dice_per_label, fold_percent_from_flow, hd95_mean_labels,
    logdet_std_from_flow, mk_grid_img, setup_device, NumpyType, RegisterModel, SegNorm, dice_val
)


LABELS_BY_DS = {
    "OASIS": tuple(range(1, 36)),
    "IXI": tuple(range(0, 46)),
}

HD95_LABELS_BY_DS = {
    "OASIS": tuple(range(1, 36)),
    "IXI": tuple(range(1, 46)),
}


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str, *, strict: bool) -> None:
    """Load checkpoint weights into model with strict/non-strict key matching."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt

    if strict:
        model.load_state_dict(sd, strict=True)
        return

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing: print(f"[WARN] Missing keys: {len(missing)} (first 10): {missing[:10]}")
    if unexpected: print(f"[WARN] Unexpected keys: {len(unexpected)} (first 10): {unexpected[:10]}")


def save_preview(out_png: str, x: torch.Tensor, y: torch.Tensor, x_seg: torch.Tensor, y_seg: torch.Tensor, def_seg: torch.Tensor, def_grid: Optional[torch.Tensor]):
    """Save orthogonal preview panels for fixed/moving/segmentation/grid."""
    import matplotlib.pyplot as plt

    x = x.detach().cpu().numpy()[0, 0]
    y = y.detach().cpu().numpy()[0, 0]
    ys = y_seg.detach().cpu().numpy()[0, 0]
    ds = def_seg.detach().cpu().numpy()[0, 0]
    dg = None if def_grid is None else def_grid.detach().cpu().numpy()[0, 0]
    d, h, w = y.shape
    cz, cy, cx = d // 2, h // 2, w // 2

    def slices(vol):
        return (vol[cz], vol[:, cy, :], vol[:, :, cx])

    rows = [slices(y), slices(x), slices(ys), slices(ds)]
    titles = ["Fixed", "Moving", "Fixed seg", "Warped seg"]
    if dg is not None:
        rows.append(slices(dg))
        titles.append("Deformed grid")

    fig = plt.figure(figsize=(12, 2.4 * len(rows)))
    n_rows = len(rows)
    for r, (a, c, s) in enumerate(rows):
        ax1 = fig.add_subplot(n_rows, 3, r * 3 + 1)
        ax2 = fig.add_subplot(n_rows, 3, r * 3 + 2)
        ax3 = fig.add_subplot(n_rows, 3, r * 3 + 3)
        ax1.imshow(a, cmap="gray" if r in (0, 1, 4) else None)
        ax2.imshow(c, cmap="gray" if r in (0, 1, 4) else None)
        ax3.imshow(s, cmap="gray" if r in (0, 1, 4) else None)
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
        self.test_dir = PATHS[int(args.paths)][args.ds.upper()]["val_dir"]
        self.test_files = sorted(glob.glob(os.path.join(self.test_dir, "*.pkl")))
        if not self.test_files:
            raise RuntimeError(f"No .pkl files found in test_dir: {self.test_dir}")

        ds_key = args.ds.upper()
        self.labels = LABELS_BY_DS[ds_key]
        self.hd95_labels = HD95_LABELS_BY_DS[ds_key]
        self.ixi_multiclass_dice = ds_key == "IXI"
        if ds_key == "OASIS":
            tfm = transforms.Compose([NumpyType((np.float32, np.int16))])
            ds = OASIS.OASISBrainInferDataset(self.test_files, transforms=tfm)
        else:
            atlas_path = str(PATHS[int(args.paths)][ds_key]["atlas_path"]).rstrip("/\\")
            tfm = transforms.Compose([SegNorm(), NumpyType((np.float32, np.int16))])
            ds = IXI.IXIBrainInferDataset(self.test_files, atlas_path, transforms=tfm)
        self.loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        self.adapter = get_model_adapter(args.model)
        self.name = self.adapter.key
        match self.name:
            case "tm-dca": self.model = self.adapter.build(time_steps=int(args.time_steps), config_key=args.tm_config).to(self.device)
            case "utsrmorph": self.model = self.adapter.build(config_key=args.utsr_config).to(self.device)
            case "ctcf": self.model = self.adapter.build(time_steps=int(args.time_steps), config_key=args.ctcf_config).to(self.device)
            case _: raise ValueError(f"Unknown model: {args.model}")

        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")
        self.ckpt_path = args.ckpt
        load_checkpoint_state(self.model, self.ckpt_path, strict=bool(int(args.strict_ckpt)))
        self.model.eval()


    def run(self):
        """Execute inference over dataset and save per-case metrics."""
        args = self.args
        ckpt_name = Path(self.ckpt_path).stem
        out_dir = os.path.join("results", "infer", args.ds.upper(), self.name, ckpt_name)
        os.makedirs(out_dir, exist_ok=True)
        png_dir = os.path.join(out_dir, "png") if args.save_pngs else None
        flow_dir = os.path.join(out_dir, "flows") if args.save_flow else None
        if png_dir is not None: os.makedirs(png_dir, exist_ok=True)
        if flow_dir is not None: os.makedirs(flow_dir, exist_ok=True)

        print(f"[INFO] Model: {self.name}")
        print(f"[INFO] Checkpoint: {self.ckpt_path}")
        print(f"[INFO] Test dir: {self.test_dir}")
        print(f"[INFO] Out dir: {out_dir}")

        rows = []
        reg_nearest = None
        reg_bilin = None

        with torch.no_grad():
            for idx, batch in enumerate(self.loader):
                x, y, x_seg, y_seg = [t.to(self.device, non_blocking=True) for t in batch]
                vol_shape = tuple(x.shape[2:])
                if reg_nearest is None:
                    reg_nearest = RegisterModel(vol_shape, mode="nearest").to(self.device)
                    reg_bilin = RegisterModel(vol_shape, mode="bilinear").to(self.device)
                stem = Path(self.test_files[idx]).stem
                cid = stem[2:] if stem.startswith("p_") else stem
                
                t0 = time.perf_counter()
                flow = self.adapter.forward(self.model, x, y)
                def_seg = reg_nearest((x_seg.float(), flow.float()))
                dt = time.perf_counter() - t0
                dice_lbl = dice_per_label(def_seg.long(), y_seg.long(), labels=self.labels)
                dice_mean = float(dice_val(def_seg.long(), y_seg.long(), 46).item()) if self.ixi_multiclass_dice else float(np.mean(dice_lbl))

                row = {
                    "case_id": cid,
                    "dice_mean": dice_mean,
                    "fold_percent": fold_percent_from_flow(flow),
                    "logdet_std": logdet_std_from_flow(flow),
                    "time_sec": dt,
                }

                if args.hd95:
                    row["hd95_mean"] = hd95_mean_labels(def_seg.long(), y_seg.long(), labels=self.hd95_labels, spacing=(1.0, 1.0, 1.0))
                for lbl, v in zip(self.labels, dice_lbl):
                    row[f"dice_lbl_{lbl}"] = float(v)
                rows.append(row)

                if args.save_flow: np.savez_compressed(os.path.join(flow_dir, f"flow_{cid}.npz"), flow=flow.detach().cpu().numpy())
                if args.save_pngs and (args.png_limit < 0 or idx < args.png_limit):
                    grid = mk_grid_img(flow.float(), grid_step=8, line_thickness=1)
                    def_grid = reg_bilin((grid.float(), flow.float()))
                    save_preview(os.path.join(png_dir, f"{cid}.png"), x, y, x_seg, y_seg, def_seg, def_grid)

                if (idx + 1) % max(1, args.print_every) == 0:
                    msg = f"[{idx+1:03d}/{len(self.loader):03d}] {cid} dice={row['dice_mean']:.4f} fold%={row['fold_percent']:.4f} time={dt:.3f}s"
                    if args.hd95:
                        msg += f" hd95={row['hd95_mean']:.4f}"
                    print(msg)

        self._save_results(rows, out_dir)


    def _save_results(self, rows, out_dir):
        """Persist per-case and aggregated summaries to CSV/JSON."""
        args = self.args
        header = ["case_id", "dice_mean", "fold_percent", "logdet_std", "time_sec"]
        
        if args.hd95: header.append("hd95_mean")
        header += [f"dice_lbl_{lbl}" for lbl in self.labels]
        per_case_path = os.path.join(out_dir, "per_case.csv")
        
        with open(per_case_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

        n = len(rows)
        keys = ["dice_mean", "fold_percent", "logdet_std", "time_sec"] + (["hd95_mean"] if args.hd95 else [])
        metrics = {}
        for k in keys:
            arr = np.array([r[k] for r in rows], dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            sem = std / math.sqrt(n) if n > 1 else 0.0
            metrics[k] = {"mean": mean, "std": std, "sem": sem, "ci95": 1.96 * sem}

        summary = {
            "model": self.name,
            "ckpt_path": self.ckpt_path,
            "test_dir": self.test_dir,
            "n_cases": n,
            "metrics": metrics,
        }
        summary_json = os.path.join(out_dir, "summary.json")
        summary_csv = os.path.join(out_dir, "summary.csv")
        
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std", "sem", "ci95"])
            for k, v in metrics.items():
                w.writerow([k, f"{v['mean']:.6f}", f"{v['std']:.6f}", f"{v['sem']:.6f}", f"{v['ci95']:.6f}"])

        print(f"\n[SAVED] per-case: {per_case_path}")
        print(f"[SAVED] summary: {summary_json}")
        print(f"[SAVED] summary: {summary_csv}")
        if self.args.save_pngs: print(f"[SAVED] png dir: {os.path.join(out_dir, 'png')}")
        if self.args.save_flow: print(f"[SAVED] flow dir: {os.path.join(out_dir, 'flows')}")


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, mode="infer")
    p.add_argument("--model", required=True, choices=["tm-dca", "utsrmorph", "ctcf"], help="Model family to run.")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth).")
    p.add_argument("--strict_ckpt", type=int, choices=[0, 1], default=1, help="Strict checkpoint key matching (1) or tolerant load (0).")
    
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic backend settings.")
    p.add_argument("--print_every", type=int, default=1, help="Console print period (cases).")
    p.add_argument("--save_flow", action="store_true", help="Save flow fields to compressed .npz files.")
    p.add_argument("--save_pngs", action="store_true", help="Save per-case preview PNGs.")
    p.add_argument("--png_limit", type=int, default=5, help="Max PNG count (-1 for all cases).")
    
    p.add_argument("--hd95", action="store_true", help="Compute HD95 in addition to Dice/fold metrics.")
    p.add_argument("--time_steps", type=int, default=12, help="Integration steps for velocity-based models.")
    p.add_argument("--tm_config", type=str, default="TransMorph-3-LVL", help="TransMorph-DCA config key.")
    p.add_argument("--utsr_config", type=str, default="UTSRMorph-Large", help="UTSRMorph config key.")
    p.add_argument("--ctcf_config", type=str, default="CTCF-CascadeA", help="CTCF config key.")
    return p.parse_args()


def main():
    args = parse_args()
    InferRunner(args).run()


if __name__ == "__main__":
    main()
