from dataclasses import dataclass
from typing import Dict, List


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else float(x)


def move_towards(prev: float, target: float, max_delta: float) -> float:
    d = float(target) - float(prev)
    if d > max_delta:
        return float(prev) + float(max_delta)
    if d < -max_delta:
        return float(prev) - float(max_delta)
    return float(target)


@dataclass
class CTCFControllerCfg:
    """
    CTCF adaptive epoch-level schedule configuration.

    fold_soft / fold_hard:
        Validation fold% thresholds (percent, 0..100). These thresholds gate refinement
        and auxiliary objectives and drive reactive strengthening of JAC.

    Dice stability detector:
        Uses a rolling window of validation Dice values. Dice is considered stable when:
          - std(window) <= stab_std_tol
          - recent_gain(window) <= stab_gain_tol

    Rate limits:
        Absolute per-epoch deltas. This guarantees smooth curves (no jumps) on TB.
    """
    fold_soft: float = 5.0
    fold_hard: float = 10.0

    alpha_l3_start: float = 0.10
    alpha_l3_rate: float = 0.02

    icon_rate: float = 0.01
    cyc_rate: float = 0.005

    jac_rate_base: float = 0.005
    jac_rate_boost: float = 0.03
    jac_mul_min: float = 0.20
    jac_mul_max: float = 2.00

    cyc_mul_max: float = 0.20

    hist_len: int = 12
    stab_window: int = 7
    stab_std_tol: float = 0.005
    stab_gain_tol: float = 0.002


@dataclass
class CTCFKnobs:
    """
    Per-epoch knobs produced by the controller.

    alpha_* are architectural gates passed into the model.
    w_*_mul are multipliers applied to user-provided base weights (args.w_*):
        W_icon = args.w_icon * w_icon_mul
        W_cyc  = args.w_cyc  * w_cyc_mul
        W_jac  = args.w_jac  * w_jac_mul
    """
    alpha_l1: float = 1.0
    alpha_l3: float = 0.10
    w_icon_mul: float = 0.0
    w_cyc_mul: float = 0.0
    w_jac_mul: float = 0.20


class CTCFController:
    """
    Adaptive epoch-level controller for CTCF (no rollback, no reset).

    Contract:
      - on_val_end(epoch, val_dice, val_fold_percent) is called once per epoch after validate()
      - get() returns knobs for the next epoch (1-epoch delay feedback loop)
      - all knob trajectories are rate-limited => smooth TB curves, no phase shocks
    """

    def __init__(self, cfg: CTCFControllerCfg):
        self.cfg = cfg
        self.knobs = CTCFKnobs(
            alpha_l1=1.0,
            alpha_l3=float(cfg.alpha_l3_start),
            w_icon_mul=0.0,
            w_cyc_mul=0.0,
            w_jac_mul=float(cfg.jac_mul_min),
        )
        self.phase: str = "S0"
        self._dice_hist: List[float] = []
        self._fold_hist: List[float] = []


    def get(self) -> CTCFKnobs:
        return self.knobs


    def tb_scalars(self) -> Dict[str, float]:
        phase_map = {"S0": 0.0, "S1": 1.0, "S2": 2.0, "S3": 3.0}
        return {
            "sched/alpha_l1": float(self.knobs.alpha_l1),
            "sched/alpha_l3": float(self.knobs.alpha_l3),
            "sched/w_icon_mul": float(self.knobs.w_icon_mul),
            "sched/w_cyc_mul": float(self.knobs.w_cyc_mul),
            "sched/w_jac_mul": float(self.knobs.w_jac_mul),
            "sched/phase": float(phase_map.get(self.phase, -1.0)),
        }


    def on_val_end(self, *, epoch: int, val_dice: float, val_fold_percent: float) -> None:
        self.push_hist(val_dice, val_fold_percent)

        dice_stable = self.dice_is_stable()
        foldp = float(val_fold_percent)

        fold_bad_soft = foldp > float(self.cfg.fold_soft)
        fold_bad_hard = foldp > float(self.cfg.fold_hard)

        self.update_jac(foldp)

        if self.phase == "S0":
            if dice_stable and (not fold_bad_soft):
                self.phase = "S1"

        elif self.phase == "S1":
            if (self.knobs.alpha_l3 >= 0.85) and dice_stable and (not fold_bad_soft):
                self.phase = "S2"

        elif self.phase == "S2":
            if (self.knobs.w_icon_mul >= 0.85) and dice_stable and (not fold_bad_hard):
                self.phase = "S3"

        self.knobs.alpha_l1 = 1.0

        if (self.phase in ("S1", "S2", "S3")) and (not fold_bad_soft):
            self.knobs.alpha_l3 = move_towards(self.knobs.alpha_l3, 1.0, float(self.cfg.alpha_l3_rate))

        if (self.phase in ("S2", "S3")) and (not fold_bad_soft):
            self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, 1.0, float(self.cfg.icon_rate))

        if (self.phase == "S3") and (not fold_bad_soft):
            self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, float(self.cfg.cyc_mul_max), float(self.cfg.cyc_rate))

        if fold_bad_hard:
            # freeze non-topology knobs; JAC keeps reacting via update_jac()
            self.knobs.alpha_l3 = float(self.knobs.alpha_l3)
            self.knobs.w_icon_mul = float(self.knobs.w_icon_mul)
            self.knobs.w_cyc_mul = float(self.knobs.w_cyc_mul)

        self.clamp_all()


    def push_hist(self, dice: float, foldp: float) -> None:
        self._dice_hist.append(float(dice))
        self._fold_hist.append(float(foldp))
        if len(self._dice_hist) > int(self.cfg.hist_len):
            self._dice_hist.pop(0)
        if len(self._fold_hist) > int(self.cfg.hist_len):
            self._fold_hist.pop(0)


    def dice_is_stable(self) -> bool:
        W = int(self.cfg.stab_window)
        if len(self._dice_hist) < W:
            return False

        w = self._dice_hist[-W:]
        mean = sum(w) / float(W)
        var = sum((x - mean) ** 2 for x in w) / float(W)
        std = var ** 0.5

        half = max(2, W // 2)
        prev = w[:half]
        curr = w[-half:]
        prev_m = sum(prev) / float(len(prev))
        curr_m = sum(curr) / float(len(curr))
        gain = curr_m - prev_m

        return (std <= float(self.cfg.stab_std_tol)) and (gain <= float(self.cfg.stab_gain_tol))


    def update_jac(self, fold_percent: float) -> None:
        foldp = float(fold_percent)
        if foldp <= float(self.cfg.fold_soft):
            rate = float(self.cfg.jac_rate_base)
        elif foldp <= float(self.cfg.fold_hard):
            rate = float(self.cfg.jac_rate_boost) * 0.5
        else:
            rate = float(self.cfg.jac_rate_boost)

        self.knobs.w_jac_mul = move_towards(self.knobs.w_jac_mul, float(self.cfg.jac_mul_max), rate)


    def clamp_all(self) -> None:
        self.knobs.alpha_l1 = 1.0
        self.knobs.alpha_l3 = clamp(self.knobs.alpha_l3, float(self.cfg.alpha_l3_start), 1.0)
        self.knobs.w_icon_mul = clamp(self.knobs.w_icon_mul, 0.0, 1.0)
        self.knobs.w_cyc_mul = clamp(self.knobs.w_cyc_mul, 0.0, float(self.cfg.cyc_mul_max))
        self.knobs.w_jac_mul = clamp(self.knobs.w_jac_mul, float(self.cfg.jac_mul_min), float(self.cfg.jac_mul_max))