from __future__ import annotations

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, compute_fig

ITER_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("ncc", "ncc"),
    ("reg", "reg"),
    ("icon", "icon"),
    ("jac", "jac"),
    ("dice_tr", "dice_tr"),
)


def write_tb_images(writer: SummaryWriter, last_vis: dict, epoch: int) -> None:
    """Write segmentation and grid previews to TensorBoard."""
    if not last_vis: return

    def_out = last_vis.get("def_seg")
    def_grid = last_vis.get("def_grid")
    x_vis = last_vis.get("x_seg")
    y_vis = last_vis.get("y_seg")
    if def_out is None or def_grid is None or x_vis is None or y_vis is None: return

    plt.switch_backend("agg")
    figures = {
        "Grid": compute_fig(def_grid),
        "Input": compute_fig(x_vis),
        "Ground truth": compute_fig(y_vis),
        "Prediction": compute_fig(def_out),
    }
    for tag, fig in figures.items():
        writer.add_figure(tag, fig, epoch)
        plt.close(fig)


def format_iter_log(
    meters: dict[str, AverageMeter],
    it: int,
    train_total: int,
    lr_now: float,
) -> str:
    main = "all" if "all" in meters else next(iter(meters.keys()))
    parts = [f"Iter {it:4d} / {train_total:4d} | {main}(avg)={meters[main].avg:.4f}"]

    extras = []
    for key, label in ITER_LOG_KEYS:
        if key in meters: extras.append(f"{label}={meters[key].val:.4f}")
    if extras: parts.append("| " + " ".join(extras))

    parts.append(f"| lr={lr_now:.3e}")
    return " ".join(parts)


def format_metric_suffix(ndvp: float | None, sdlogj: float | None) -> str:
    parts = []
    if ndvp is not None: parts.append(f" | ndv%={ndvp:.2f}")
    if sdlogj is not None: parts.append(f" | sdlogj={sdlogj:.4f}")
    return "".join(parts)


def format_train_suffix(meters: dict[str, AverageMeter]) -> str:
    if "alpha_l1" not in meters: return ""
    return f" | a1={meters['alpha_l1'].val:.3f} a3={meters['alpha_l3'].val:.3f} w={meters['warm'].val:.3f}"
