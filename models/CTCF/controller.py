from dataclasses import dataclass


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else float(x)


def move_towards(prev, target, delta):
    d = float(target) - float(prev)
    if d > delta:
        return float(prev) + float(delta)
    if d < -delta:
        return float(prev) - float(delta)
    return float(target)


@dataclass
class CTCFControllerCfg:
    jac_soft: float = 0.35
    jac_hard: float = 0.80
    jac_recover: float = 0.20
    icon_rate: float = 0.01
    cyc_rate: float = 0.005
    icon_decay_bad: float = 0.03
    cyc_decay_bad: float = 0.01
    jac_rate_base: float = 0.005
    jac_rate_boost: float = 0.03
    jac_rate_relax: float = 0.01
    jac_mul_min: float = 0.20
    jac_mul_max: float = 2.00
    cyc_mul_max: float = 0.20
    icon_mul_max: float = 1.00
    icon_start_jac_max: float = 0.50
    cyc_start_jac_max: float = 0.40
    hist_len: int = 12
    stab_window: int = 7
    stab_std_tol: float = 0.005
    stab_gain_tol: float = 0.002


@dataclass
class CTCFKnobs:
    alpha_l3: float = 1.0
    w_icon_mul: float = 0.0
    w_cyc_mul: float = 0.0
    w_jac_mul: float = 0.20


class CTCFController:
    def __init__(self, cfg: CTCFControllerCfg):
        self.cfg = cfg
        self.knobs = CTCFKnobs(w_jac_mul=float(cfg.jac_mul_min))
        self.phase = "S0"
        self._dice = []


    def get(self):
        return self.knobs


    def tb_scalars(self):
        phase = {"S0": 0.0, "S1": 1.0, "S2": 2.0, "S3": 3.0}
        return {
            "sched/alpha_l3": float(self.knobs.alpha_l3),
            "sched/w_icon_mul": float(self.knobs.w_icon_mul),
            "sched/w_cyc_mul": float(self.knobs.w_cyc_mul),
            "sched/w_jac_mul": float(self.knobs.w_jac_mul),
            "sched/phase": float(phase.get(self.phase, -1.0)),
        }


    def on_val_end(self, *, epoch: int, val_dice: float, val_jac_percent: float):
        del epoch
        self._push_dice(val_dice)
        stable = self._stable()
        jac = float(val_jac_percent)
        bad_soft = jac > float(self.cfg.jac_soft)
        bad_hard = jac > float(self.cfg.jac_hard)

        self._update_jac(jac)

        if self.phase == "S0" and stable and not bad_soft:
            self.phase = "S1"
        elif self.phase == "S1" and self.knobs.w_icon_mul >= 0.8 and stable and jac <= float(self.cfg.icon_start_jac_max):
            self.phase = "S2"
        elif self.phase == "S2" and self.knobs.w_cyc_mul >= 0.12 and stable and jac <= float(self.cfg.cyc_start_jac_max):
            self.phase = "S3"

        if bad_hard:
            self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, 0.0, float(self.cfg.cyc_decay_bad))
            self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, 0.0, float(self.cfg.icon_decay_bad))
        else:
            if self.phase in ("S1", "S2", "S3"):
                if jac <= float(self.cfg.icon_start_jac_max):
                    self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, float(self.cfg.icon_mul_max), float(self.cfg.icon_rate))
                elif bad_soft:
                    self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, 0.0, float(self.cfg.icon_decay_bad))
            if self.phase in ("S2", "S3"):
                if jac <= float(self.cfg.cyc_start_jac_max):
                    self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, float(self.cfg.cyc_mul_max), float(self.cfg.cyc_rate))
                elif bad_soft:
                    self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, 0.0, float(self.cfg.cyc_decay_bad))

        self.knobs.w_icon_mul = clamp(self.knobs.w_icon_mul, 0.0, float(self.cfg.icon_mul_max))
        self.knobs.w_cyc_mul = clamp(self.knobs.w_cyc_mul, 0.0, float(self.cfg.cyc_mul_max))
        self.knobs.w_jac_mul = clamp(self.knobs.w_jac_mul, float(self.cfg.jac_mul_min), float(self.cfg.jac_mul_max))


    def _push_dice(self, v):
        self._dice.append(float(v))
        if len(self._dice) > int(self.cfg.hist_len):
            self._dice.pop(0)


    def _stable(self):
        w = int(self.cfg.stab_window)
        if len(self._dice) < w:
            return False
        x = self._dice[-w:]
        m = sum(x) / float(w)
        var = sum((v - m) ** 2 for v in x) / float(w)
        std = var ** 0.5
        h = max(2, w // 2)
        g = (sum(x[-h:]) / float(h)) - (sum(x[:h]) / float(h))
        return (std <= float(self.cfg.stab_std_tol)) and (g <= float(self.cfg.stab_gain_tol))


    def _update_jac(self, jac):
        if jac > float(self.cfg.jac_hard):
            tgt, rate = float(self.cfg.jac_mul_max), float(self.cfg.jac_rate_boost)
        elif jac > float(self.cfg.jac_soft):
            tgt, rate = float(self.cfg.jac_mul_max), float(self.cfg.jac_rate_boost) * 0.5
        elif jac < float(self.cfg.jac_recover):
            tgt, rate = float(self.cfg.jac_mul_min), float(self.cfg.jac_rate_relax)
        else:
            tgt = max(float(self.cfg.jac_mul_min), self.knobs.w_jac_mul - float(self.cfg.jac_rate_base))
            rate = float(self.cfg.jac_rate_base)
        self.knobs.w_jac_mul = move_towards(self.knobs.w_jac_mul, tgt, rate)
