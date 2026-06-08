from __future__ import annotations

import os
from typing import Any

import torch

from utils import (
    IXI_VOI_LABELS,
    OASIS_VOI_LABELS,
    adjust_learning_rate_poly,
    adjust_lr_ctcf_schedule,
    dice_val,
    dice_val_subset,
    load_checkpoint_if_exists,
)


def build_val_dice_fn(args, ds_key: str):
    """Pick the validation Dice function matching the dataset key."""
    if ds_key == "SYNTH":
        num_labels = getattr(args, "synth_num_labels", 36)
        return lambda p, t: dice_val(p, t, num_labels)
    if ds_key == "IXI":
        return lambda p, t: dice_val_subset(p, t, labels=IXI_VOI_LABELS)
    return lambda p, t: dice_val_subset(p, t, labels=OASIS_VOI_LABELS)


def resume_from_ckpt(
    args,
    runner,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    best_dsc_init: float,
) -> tuple[int, float]:
    """Load resume checkpoint into runner + scaler; return (epoch_start, best_dsc)."""
    if not args.resume:
        return 0, best_dsc_init

    ckpt = load_checkpoint_if_exists(
        args.resume,
        model=runner.model,
        optimizer=runner.optimizer,
        map_location=device,
    )
    if ckpt is None:
        return 0, best_dsc_init

    epoch_start = int(ckpt.get("epoch", -1)) + 1
    best_dsc = float(ckpt.get("best_dsc", best_dsc_init))
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    print(f">>> Resumed from {args.resume} @ epoch {epoch_start}, best={best_dsc:.4f}")
    return epoch_start, best_dsc


def select_lr_policy(runner, optimizer, epoch: int, max_epoch: int, init_lr: float) -> float:
    """Drive the optimizer LR through the policy attached to the runner."""
    if getattr(runner, "lr_policy", "") == "ctcf":
        return adjust_lr_ctcf_schedule(optimizer, epoch, max_epoch, init_lr)
    return adjust_learning_rate_poly(optimizer, epoch, max_epoch, init_lr)


def save_ckpt(state: dict[str, Any], ckpt_dir: str, is_best: bool) -> None:
    """Persist `state` as last.pth and (if `is_best`) best.pth."""
    torch.save(state, os.path.join(ckpt_dir, "last.pth"))
    if is_best: torch.save(state, os.path.join(ckpt_dir, "best.pth"))
