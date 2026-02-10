import os
import sys
import glob
import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted


# ----------------------------- Logging ----------------------------- #
class Logger:
    """
    Mirrors stdout (and optionally stderr) to both console and a log file in log_dir.
    """
    def __init__(self, log_dir: str, filename: str = "logfile.log"):
        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.log = open(self.log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass


def attach_stdout_logger(log_dir: str, filename: str = "logfile.log", mirror_stderr: bool = True) -> Logger:
    """
    Redirect sys.stdout to Logger(log_dir). Optionally mirrors sys.stderr as well.
    Returns created logger (keep ref if you want to close it manually).
    """
    logger = Logger(log_dir, filename=filename)
    sys.stdout = logger
    if mirror_stderr:
        sys.stderr = logger
    return logger


# ----------------------------- Device ----------------------------- #
@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    gpu_id: int
    gpu_name: Optional[str]


def setup_device(gpu_id: int = 0, seed: int = 0, deterministic: bool = False) -> torch.device:
    """
    Sets seeds + CUDA flags and returns torch.device.
    IMPORTANT: Must return torch.device (NOT DeviceInfo), so callers can do model.to(device).
    """
    # seeds (full set for reproducibility)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(f"gpu_id={gpu_id} is out of range (available: 0..{gpu_count - 1})")

        torch.cuda.set_device(gpu_id)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # not all versions support this, keep silent if not
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.benchmark = True

        print(f"Number of GPU: {gpu_count}")
        for i in range(gpu_count):
            print(f"     GPU #{i}: {torch.cuda.get_device_name(i)}")
        name = torch.cuda.get_device_name(gpu_id)
        print(f"Currently using: {name}")
        print("If the GPU is available? True")

        return torch.device(f"cuda:{gpu_id}")

    print("CUDA not available, using CPU.")
    return torch.device("cpu")


def get_device_info(gpu_id: int = 0) -> DeviceInfo:
    """
    Optional helper if you still want metadata in some scripts.
    Not used by training loops that call model.to(device).
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(f"gpu_id={gpu_id} is out of range (available: 0..{gpu_count - 1})")
        name = torch.cuda.get_device_name(gpu_id)
        return DeviceInfo(device=torch.device(f"cuda:{gpu_id}"), gpu_id=gpu_id, gpu_name=name)
    return DeviceInfo(device=torch.device("cpu"), gpu_id=-1, gpu_name=None)


# ----------------------------- Experiment paths ----------------------------- #
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


# ----------------------------- LR schedule ----------------------------- #
def adjust_learning_rate_poly(
    optimizer: optim.Optimizer,
    epoch: int,
    max_epochs: int,
    init_lr: float,
    power: float = 0.9,
) -> float:
    """
    Polynomial LR schedule: lr = init_lr * (1 - epoch/max_epochs)^power
    """
    max_epochs = max(1, int(max_epochs))
    epoch = int(epoch)
    new_lr = float(np.round(init_lr * np.power(1 - (epoch / max_epochs), power), 8))
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    return new_lr


# ----------------------------- Checkpoints ----------------------------- #
def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str, max_model_num: int = 8) -> None:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)

    # keep at most N newest (by natural sort)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, "*")))
    while len(model_lists) > int(max_model_num):
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, "*")))


def load_checkpoint_if_exists(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Optional[Dict[str, Any]]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=map_location)

    # accept either full dict or raw state_dict
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt if isinstance(ckpt, dict) else {"state_dict": state_dict}


# ----------------------------- Perf helpers ----------------------------- #
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
    mean_iter_ms = (iter_time_sum / max(1, int(iters))) * 1000.0
    peak = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return PerfInfo(epoch_time_sec=epoch_time, mean_iter_time_ms=mean_iter_ms, peak_gpu_mem_gib=peak)


# ----------------------------- Visualization helpers ----------------------------- #
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


def mk_grid_img(flow: torch.Tensor, grid_step: int = 8, line_thickness: int = 1) -> torch.Tensor:
    """
    Create binary 3D grid [1,1,D,H,W] that matches flow spatial shape.
    flow: [B,3,D,H,W] or [B,D,H,W,3] (we only need D,H,W)
    """
    if flow.dim() == 5 and flow.shape[1] in (2, 3):
        d, h, w = map(int, flow.shape[-3:])
        device = flow.device
    elif flow.dim() == 5 and flow.shape[-1] in (2, 3):
        d, h, w = map(int, flow.shape[1:4])
        device = flow.device
    else:
        raise ValueError(f"Unsupported flow shape: {tuple(flow.shape)}")

    grid_step = max(1, int(grid_step))
    line_thickness = max(1, int(line_thickness))

    grid_img = torch.zeros((1, 1, d, h, w), dtype=torch.float32, device=device)

    # lines along H (y)
    for j in range(0, h, grid_step):
        j0 = j
        j1 = min(h, j + line_thickness)
        grid_img[:, :, :, j0:j1, :] = 1.0

    # lines along W (x)
    for i in range(0, w, grid_step):
        i0 = i
        i1 = min(w, i + line_thickness)
        grid_img[:, :, :, :, i0:i1] = 1.0

    return grid_img