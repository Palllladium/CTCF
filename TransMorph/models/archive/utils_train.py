import os, sys, json, datetime, platform, logging
import torch

def dice_bin(a, b, eps=1e-6):
    inter = (a * b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)

@torch.no_grad()
def validate(model, val_loader, warp_fn, sdlogj_fn, target_shape=None, device='cuda'):
    import numpy as np
    model.eval()
    dices_epoch = []
    for data in val_loader:
        F_img = data[0].to(device).float()
        M_img = data[1].to(device).float()
        if len(data) < 4:
            continue
        F_seg = data[2].to(device).long()
        M_seg = data[3].to(device).long()

        if (target_shape is not None) and (F_img.shape[-3:] != target_shape):
            F_img = torch.nn.functional.interpolate(F_img, size=target_shape, mode='trilinear', align_corners=True)
            M_img = torch.nn.functional.interpolate(M_img, size=target_shape, mode='trilinear', align_corners=True)
            F_seg = torch.nn.functional.interpolate(F_seg.float(), size=target_shape, mode='nearest').long()
            M_seg = torch.nn.functional.interpolate(M_seg.float(), size=target_shape, mode='nearest').long()

        _, phi = model(F_img, M_img)
        M_seg_warp = warp_fn(M_seg.float(), phi, mode='nearest').long()

        labels = torch.unique(F_seg)
        labels = labels[(labels > 0) & (labels <= 255)]
        if labels.numel() == 0:
            continue
        per_label = []
        for lbl in labels:
            f = (F_seg == lbl).float()
            m = (M_seg_warp == lbl).float()
            per_label.append(dice_bin(m, f).item())
        dices_epoch.append(float(np.mean(per_label)))

    return float(np.mean(dices_epoch)) if len(dices_epoch) else 0.0

def get_logger(log_file: str, name: str = "train"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def snapshot_env(to_path: str):
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "sm": (torch.cuda.get_device_capability() if torch.cuda.is_available() else None),
    }
    with open(to_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info