"""
Unified log parser for CTCF, TM-DCA, and UTSRMorph training logs.

Handles three log formats:
  - CTCF:      [epoch NNN] val_dice=X best=X | j<=0%=X | sdlogj=X | a1=X a3=X w=X
  - CTCF (IXI): [epoch NNN] val_dice=X best=X | j<=0%=X | ndv%=X | a1=X a3=X w=X
  - TM-DCA:    val DSC: X | fold%: X          (preceded by "Epoch N loss X")
  - UTSRMorph: val DSC: X | fold%: X          (preceded by "Epoch N loss X")

Iteration-level logs:
  - CTCF:      Iter N / M | all(avg)=X | ncc=X reg=X icon=X jac=X | lr=X
  - TM-DCA:    Iter N / M | loss(avg)=X | last NCC=X DSC=X REG=X | lr=X
  - UTSRMorph: Iter N / M | loss(avg)=X | last NCC=X REG=X | lr=X
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EpochRecord:
    epoch: int
    val_dice: float
    best_dice: float
    fold_pct: Optional[float] = None  # j<=0% or fold%
    sdlogj: Optional[float] = None
    ndv_pct: Optional[float] = None  # IXI only
    alpha_l1: Optional[float] = None  # CTCF only
    alpha_l3: Optional[float] = None  # CTCF only
    warm: Optional[float] = None  # CTCF only
    train_loss: Optional[float] = None  # from "Epoch N loss X" line


@dataclass
class IterRecord:
    epoch: int
    iteration: int
    total_iters: int
    loss_avg: float
    ncc: Optional[float] = None
    reg: Optional[float] = None
    icon: Optional[float] = None
    jac: Optional[float] = None
    cyc: Optional[float] = None
    dsc: Optional[float] = None  # TM-DCA DSC loss component
    lr: Optional[float] = None


@dataclass
class TrainingLog:
    experiment: str = ""
    dataset: str = ""
    epochs: list[EpochRecord] = field(default_factory=list)
    iters: list[IterRecord] = field(default_factory=list)
    model_type: str = ""  # "ctcf", "tm-dca", "utsrmorph"


# regex patterns
# CTCF epoch: [epoch NNN] val_dice=X best=X | j<=0%=X | sdlogj=X | a1=X a3=X w=X
# TM-DCA IXI: [epoch NNN] train_loss=X val_dsc=X fold%=X best=X
_RE_CTCF_EPOCH = re.compile(
    r"\[epoch\s+(\d+)\]\s+"
    r"(?:train_loss=[-\d.eE+]+\s+)?"
    r"val_(?:dice|dsc)=([\d.]+)\s+"
    r"(?:fold%=([\d.]+)\s+)?"
    r"best=([\d.]+)"
    r"(?:\s*\|\s*(?:j<=0%|fold%)=([\d.]+))?"
    r"(?:\s*\|\s*ndv%=([\d.]+))?"
    r"(?:\s*\|\s*sdlogj=([\d.]+))?"
    r"(?:\s*\|\s*a1=([\d.]+)\s+a3=([\d.]+)\s+w=([\d.]+))?"
)

# TM-DCA / UTSRMorph epoch: val DSC: X | fold%: X
_RE_BASELINE_VAL = re.compile(r"val DSC:\s+([\d.]+)\s*\|\s*fold%:\s+([\d.]+)")

# "Epoch N loss X" line (for train loss and epoch number in baseline logs)
_RE_EPOCH_LOSS = re.compile(r"Epoch\s+(\d+)\s+loss\s+([-\d.eE+]+)")

# CTCF iter: Iter N / M | all(avg)=X | ncc=X reg=X icon=X [cyc=X] jac=X | lr=X
_RE_CTCF_ITER = re.compile(
    r"Iter\s+(\d+)\s*/\s*(\d+)\s*\|\s*all\(avg\)=([-\d.eE+]+)\s*\|"
    r"\s*ncc=([-\d.eE+]+)\s+reg=([-\d.eE+]+)\s+icon=([-\d.eE+]+)"
    r"(?:\s+cyc=([-\d.eE+]+))?"
    r"\s+jac=([-\d.eE+]+)\s*\|\s*lr=([-\d.eE+]+)"
)

# TM-DCA iter: Iter N / M | loss(avg)=X | last NCC=X DSC=X REG=X | lr=X
_RE_TMDCA_ITER = re.compile(
    r"Iter\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss\(avg\)=([-\d.eE+]+)\s*\|"
    r"\s*last NCC=([-\d.eE+]+)\s+DSC=([-\d.eE+]+)\s+REG=([-\d.eE+]+)\s*\|\s*lr=([-\d.eE+]+)"
)

# UTSRMorph iter: Iter N / M | loss(avg)=X | last NCC=X REG=X | lr=X
_RE_UTSR_ITER = re.compile(
    r"Iter\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss\(avg\)=([-\d.eE+]+)\s*\|"
    r"\s*last NCC=([-\d.eE+]+)\s+REG=([-\d.eE+]+)\s*\|\s*lr=([-\d.eE+]+)"
)

# Header: >>> Experiment: NAME | ds=DATASET
_RE_HEADER = re.compile(r">>>\s*Experiment:\s*(\S+)(?:\s*\|\s*ds=(\S+))?")

# "Training Starts (epoch NNN)" — used for epoch counting
_RE_TRAIN_START = re.compile(r"Training Starts \(epoch\s+(\d+)\)")


def _detect_model_type(path: Path, lines: list[str]) -> str:
    """Heuristic model type detection from log content and path."""
    name = path.parent.name.upper() if path.name == "logfile.log" else path.stem.upper()
    if "TMDCA" in name or "TM-DCA" in name or "TM_DCA" in name:
        return "tm-dca"
    if "UTSR" in name:
        return "utsrmorph"
    # check content
    for line in lines[:20]:
        if "all(avg)=" in line:
            return "ctcf"
        if "DSC=" in line and "NCC=" in line:
            return "tm-dca"
        if "NCC=" in line and "REG=" in line and "DSC=" not in line:
            return "utsrmorph"
    return "ctcf"


def parse_log(path: str | Path, *, parse_iters: bool = False) -> TrainingLog:
    """Parse a training logfile and return structured data.

    Args:
        path: Path to logfile.log.
        parse_iters: If True, also parse per-iteration loss lines (slower).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    log = TrainingLog()
    log.model_type = _detect_model_type(path, lines)

    # parse header
    for line in lines[:5]:
        m = _RE_HEADER.match(line)
        if m:
            log.experiment = m.group(1)
            log.dataset = m.group(2) or ""
            break

    current_epoch = 0
    current_train_loss = None
    best_dice_so_far = 0.0

    for line in lines:
        # ── track epoch starts ──
        m_ts = _RE_TRAIN_START.search(line)
        if m_ts:
            current_epoch = int(m_ts.group(1))
            continue

        # ── CTCF / structured epoch line ──
        m = _RE_CTCF_EPOCH.search(line)
        if m:
            # Groups: (1)epoch, (2)val_dice, (3)inline_fold%, (4)best,
            #         (5)pipe_fold%, (6)ndv%, (7)sdlogj, (8)a1, (9)a3, (10)w
            fold = m.group(3) or m.group(5)  # inline fold% or pipe-separated
            rec = EpochRecord(
                epoch=int(m.group(1)),
                val_dice=float(m.group(2)),
                best_dice=float(m.group(4)),
                fold_pct=float(fold) if fold else None,
                ndv_pct=float(m.group(6)) if m.group(6) else None,
                sdlogj=float(m.group(7)) if m.group(7) else None,
                alpha_l1=float(m.group(8)) if m.group(8) else None,
                alpha_l3=float(m.group(9)) if m.group(9) else None,
                warm=float(m.group(10)) if m.group(10) else None,
            )
            log.epochs.append(rec)
            continue

        # ── "Epoch N loss X" ──
        m = _RE_EPOCH_LOSS.search(line)
        if m:
            current_epoch = int(m.group(1))
            current_train_loss = float(m.group(2))
            continue

        # ── TM-DCA / UTSRMorph val line ──
        m = _RE_BASELINE_VAL.search(line)
        if m:
            dice = float(m.group(1))
            fold = float(m.group(2))
            best_dice_so_far = max(best_dice_so_far, dice)
            rec = EpochRecord(
                epoch=current_epoch,
                val_dice=dice,
                best_dice=best_dice_so_far,
                fold_pct=fold,
                train_loss=current_train_loss,
            )
            log.epochs.append(rec)
            current_train_loss = None
            continue

        # ── iteration lines (optional) ──
        if not parse_iters:
            continue

        m = _RE_CTCF_ITER.search(line)
        if m:
            log.iters.append(
                IterRecord(
                    epoch=current_epoch,
                    iteration=int(m.group(1)),
                    total_iters=int(m.group(2)),
                    loss_avg=float(m.group(3)),
                    ncc=float(m.group(4)),
                    reg=float(m.group(5)),
                    icon=float(m.group(6)),
                    cyc=float(m.group(7)) if m.group(7) else None,
                    jac=float(m.group(8)),
                    lr=float(m.group(9)),
                )
            )
            continue

        m = _RE_TMDCA_ITER.search(line)
        if m:
            log.iters.append(
                IterRecord(
                    epoch=current_epoch,
                    iteration=int(m.group(1)),
                    total_iters=int(m.group(2)),
                    loss_avg=float(m.group(3)),
                    ncc=float(m.group(4)),
                    dsc=float(m.group(5)),
                    reg=float(m.group(6)),
                    lr=float(m.group(7)),
                )
            )
            continue

        m = _RE_UTSR_ITER.search(line)
        if m:
            log.iters.append(
                IterRecord(
                    epoch=current_epoch,
                    iteration=int(m.group(1)),
                    total_iters=int(m.group(2)),
                    loss_avg=float(m.group(3)),
                    ncc=float(m.group(4)),
                    reg=float(m.group(5)),
                    lr=float(m.group(6)),
                )
            )

    return log


def parse_ablation_summary(path: str | Path) -> list[dict]:
    """Parse ablation_N_results.txt into a list of dicts.

    Returns list of: {"name": str, "time_sec": int, "last_dice": float, "best_dice": float}
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    results = []
    # Pattern: [N] NAME: completed in Xs
    re_run = re.compile(r"\[(\d+)\]\s+(\S+):\s+completed in (\d+)s")
    re_dice = re.compile(r"Last Dice:\s+([\d.]+),\s*Best Dice:\s+([\d.]+)")
    pending_name = None
    pending_time = None
    for line in lines:
        m = re_run.search(line)
        if m:
            pending_name = m.group(2)
            pending_time = int(m.group(3))
            continue
        m = re_dice.search(line)
        if m and pending_name:
            results.append(
                {
                    "name": pending_name,
                    "time_sec": pending_time,
                    "last_dice": float(m.group(1)),
                    "best_dice": float(m.group(2)),
                }
            )
            pending_name = None
    return results
