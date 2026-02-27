from dataclasses import dataclass


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def move_towards(prev, target, delta):
    d = target - prev
    if d > delta:
        return prev + delta
    if d < -delta:
        return prev - delta
    return target


@dataclass
class CTCFControllerCfg:
    max_epoch: int = 60
    jac_soft: float = 0.35
    jac_hard: float = 0.80
    jac_relax: float = 0.20
    jac_rate_up: float = 0.02
    jac_rate_down: float = 0.01
    jac_mul_min: float = 0.20
    jac_mul_max: float = 0.50
    icon_rate: float = 0.01
    icon_decay_bad: float = 0.03
    icon_mul_max: float = 1.00
    icon_start_jac_max: float = 0.50
    cyc_rate: float = 0.005
    cyc_decay_bad: float = 0.01
    cyc_mul_max: float = 0.20
    cyc_start_jac_max: float = 0.35
    s1_at: float = 0.25
    s2_at: float = 0.45
    s3_at: float = 0.75


    @classmethod
    def for_ds(cls, ds: str, max_epoch: int):
        ds_key = ds.upper()
        me = int(max_epoch)
        
        if ds_key == "IXI":
            return cls(
                max_epoch=me,
                jac_soft=0.12,
                jac_hard=0.35,
                jac_relax=0.03,
                jac_rate_up=0.015,
                jac_rate_down=0.008,
                jac_mul_min=0.20,
                jac_mul_max=0.40,
                icon_rate=0.015,
                icon_decay_bad=0.025,
                icon_mul_max=0.80,
                icon_start_jac_max=0.18,
                cyc_rate=0.003,
                cyc_decay_bad=0.008,
                cyc_mul_max=0.10,
                cyc_start_jac_max=0.12,
                s1_at=0.08,
                s2_at=0.22,
                s3_at=0.65,
            )
        
        if ds_key == "OASIS":
            return cls(
                max_epoch=me,
                jac_soft=0.35,
                jac_hard=0.80,
                jac_relax=0.20,
                jac_rate_up=0.02,
                jac_rate_down=0.01,
                jac_mul_min=0.20,
                jac_mul_max=0.50,
                icon_rate=0.01,
                icon_decay_bad=0.03,
                icon_mul_max=0.90,
                icon_start_jac_max=0.50,
                cyc_rate=0.004,
                cyc_decay_bad=0.01,
                cyc_mul_max=0.16,
                cyc_start_jac_max=0.35,
                s1_at=0.20,
                s2_at=0.40,
                s3_at=0.70,
            )
        return cls(max_epoch=me)


@dataclass
class CTCFKnobs:
    alpha_l3: float = 1.0
    w_icon_mul: float = 0.0
    w_cyc_mul: float = 0.0
    w_jac_mul: float = 0.20


class CTCFController:
    def __init__(self, cfg: CTCFControllerCfg):
        self.cfg = cfg
        self.knobs = CTCFKnobs(w_jac_mul=cfg.jac_mul_min)
        self.phase = "S0"


    def get(self):
        return self.knobs


    def tb_scalars(self):
        phase = {"S0": 0.0, "S1": 1.0, "S2": 2.0, "S3": 3.0}
        return {
            "sched/alpha_l3": self.knobs.alpha_l3,
            "sched/w_icon_mul": self.knobs.w_icon_mul,
            "sched/w_cyc_mul": self.knobs.w_cyc_mul,
            "sched/w_jac_mul": self.knobs.w_jac_mul,
            "sched/phase": phase.get(self.phase, -1.0),
        }


    def on_val_end(self, *, epoch: int, val_dice: float, val_jac_percent: float):
        del val_dice
        jac = val_jac_percent
        bad_soft = jac > self.cfg.jac_soft
        bad_hard = jac > self.cfg.jac_hard

        self._update_jac(jac)
        self._update_phase(epoch=epoch, jac=jac)

        if bad_hard:
            self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, 0.0, self.cfg.icon_decay_bad)
            self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, 0.0, self.cfg.cyc_decay_bad)
        else:
            if self.phase in ("S1", "S2", "S3"):
                if jac <= self.cfg.icon_start_jac_max: self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, self.cfg.icon_mul_max, self.cfg.icon_rate)
                elif bad_soft: self.knobs.w_icon_mul = move_towards(self.knobs.w_icon_mul, 0.0, self.cfg.icon_decay_bad)
            if self.phase == "S3":
                if jac <= self.cfg.cyc_start_jac_max: self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, self.cfg.cyc_mul_max, self.cfg.cyc_rate)
                elif bad_soft: self.knobs.w_cyc_mul = move_towards(self.knobs.w_cyc_mul, 0.0, self.cfg.cyc_decay_bad)

        self.knobs.w_icon_mul = clamp(self.knobs.w_icon_mul, 0.0, self.cfg.icon_mul_max)
        self.knobs.w_cyc_mul = clamp(self.knobs.w_cyc_mul, 0.0, self.cfg.cyc_mul_max)
        self.knobs.w_jac_mul = clamp(self.knobs.w_jac_mul, self.cfg.jac_mul_min, self.cfg.jac_mul_max)


    def _epoch_gate(self, epoch: int, at: float):
        if at >= 1.0: return False
        return epoch >= int(round(self.cfg.max_epoch * at))


    def _update_phase(self, *, epoch: int, jac: float):
        if self.phase == "S0" and self._epoch_gate(epoch, self.cfg.s1_at) and jac <= self.cfg.icon_start_jac_max: self.phase = "S1"
        if self.phase == "S1" and self._epoch_gate(epoch, self.cfg.s2_at) and jac <= self.cfg.cyc_start_jac_max: self.phase = "S2"
        if self.phase == "S2" and self._epoch_gate(epoch, self.cfg.s3_at) and jac <= self.cfg.cyc_start_jac_max: self.phase = "S3"
        if jac > self.cfg.jac_hard and self.phase in ("S2", "S3"): self.phase = "S1"


    def _update_jac(self, jac: float):
        if jac > self.cfg.jac_hard: tgt, rate = self.cfg.jac_mul_max, self.cfg.jac_rate_up
        elif jac > self.cfg.jac_soft:
            mid = self.cfg.jac_mul_min + 0.5 * (self.cfg.jac_mul_max - self.cfg.jac_mul_min)
            tgt, rate = mid, self.cfg.jac_rate_up * 0.5
        elif jac < self.cfg.jac_relax: tgt, rate = self.cfg.jac_mul_min, self.cfg.jac_rate_down
        else: tgt, rate = self.cfg.jac_mul_min, self.cfg.jac_rate_down
        self.knobs.w_jac_mul = move_towards(self.knobs.w_jac_mul, tgt, rate)
