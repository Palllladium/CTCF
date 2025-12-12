from __future__ import annotations

import os
import sys
import glob
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted


class Logger:
    """
    Mirrors stdout to both console and logs/<exp>/logfile.log
    """
    def __init__(self, log_dir: str, filename: str = "logfile.log"):
        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.log = open(self.log_path, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # required for python logging compatibility
        self.log.flush()


def attach_stdout_logger(log_dir: str) -> Logger:
    """
    Redirect sys.stdout to Logger(log_dir). Returns created logger (so you can keep ref if needed).
    """
    logger = Logger(log_dir)
    sys.stdout = logger
    return logger


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    gpu_id: int
    gpu_name: Optional[str]


def setup_device(gpu_id: int = 0, seed: int = 0, deterministic: bool = False) -> DeviceInfo:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        name = torch.cuda.get_device_name(gpu_id)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        print(f"Number of GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"     GPU #{i}: {torch.cuda.get_device_name(i)}")
        print(f"Currently using: {name}")
        print(f"If the GPU is available? True")
        return DeviceInfo(device=torch.device("cuda", gpu_id), gpu_id=gpu_id, gpu_name=name)
    else:
        print("CUDA not available, using CPU.")
        return DeviceInfo(device=torch.device("cpu"), gpu_id=-1, gpu_name=None)


@dataclass(frozen=True)
class ExperimentPaths:
    exp_dir: str
    log_dir: str


def make_exp_dirs(exp_name: str) -> ExperimentPaths:
    exp_root = exp_name.rstrip("/\\")
    exp_dir = os.path.join("results", exp_root)
    log_dir = os.path.join("logs", exp_root)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return ExperimentPaths(exp_dir=exp_dir, log_dir=log_dir)


def adjust_learning_rate_poly(
    optimizer: optim.Optimizer,
    epoch: int,
    max_epochs: int,
    init_lr: float,
    power: float = 0.9
) -> float:
    """
    Polynomial LR schedule: lr = init_lr * (1 - epoch/max_epochs)^power
    """
    new_lr = float(np.round(init_lr * np.power(1 - (epoch / max_epochs), power), 8))
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    return new_lr


def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str, max_model_num: int = 8) -> None:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)

    model_lists = natsorted(glob.glob(os.path.join(save_dir, "*")))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, "*")))


def load_checkpoint_if_exists(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


@dataclass(frozen=True)
class PerfInfo:
    epoch_time_sec: float
    mean_iter_time_ms: float
    peak_gpu_mem_gib: Optional[float]


def perf_epoch_start() -> float:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def perf_epoch_end(t0: float, iters: int, iter_time_sum: float) -> PerfInfo:
    epoch_time = time.perf_counter() - t0
    mean_iter_ms = (iter_time_sum / max(1, iters)) * 1000.0
    peak = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return PerfInfo(epoch_time_sec=epoch_time, mean_iter_time_ms=mean_iter_ms, peak_gpu_mem_gib=peak)


def comput_fig(img: torch.Tensor) -> plt.Figure:
    """
    16 axial slices from [B, C, D, H, W], assumes B=1, C=1.
    """
    arr = img.detach().float().cpu().numpy()[0, 0]
    z0 = min(48, max(0, arr.shape[0] - 16))
    arr = arr[z0:z0 + 16]

    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(arr.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis("off")
        plt.imshow(arr[i, :, :], cmap="gray")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step: int, line_thickness: int = 1, grid_sz=(160, 192, 224), device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Binary 3D grid [1, 1, D, H, W] for deformation visualization.
    grid_sz must match the image spatial shape.
    """
    d, h, w = grid_sz
    grid_img = np.zeros((d, h, w), dtype=np.float32)

    for j in range(0, h, grid_step):
        jj = min(h - 1, j + line_thickness - 1)
        grid_img[:, jj, :] = 1.0
    for i in range(0, w, grid_step):
        ii = min(w - 1, i + line_thickness - 1)
        grid_img[:, :, ii] = 1.0

    out = torch.from_numpy(grid_img[None, None, ...])
    if device is not None:
        out = out.to(device, non_blocking=True)
    return out