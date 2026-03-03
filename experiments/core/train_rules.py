"""Train-time schedules and governors used by CTCF runner."""


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _move_towards(prev: float, target: float, delta: float) -> float:
    d = target - prev
    if d > delta: return prev + delta
    if d < -delta: return prev - delta
    return target


def _ramp(epoch: int, start: int, end: int) -> float:
    if epoch <= start: return 0.0
    if epoch >= end: return 1.0
    return float(epoch - start) / float(max(1, end - start))


class CtcfTrainRules:
    """Deterministic ICON/CYC ramps + validation-driven jacobian governor."""

    PROFILES = {
        "OASIS": {
            "icon_start_frac": 0.00,
            "icon_warm_frac": 0.075,
            "cyc_start_frac": 0.15,
            "cyc_warm_frac": 0.10,
            "aux_until_frac": 0.25,
            "aux_l1_start": 0.12,
            "aux_l2_start": 0.08,
            "jac_soft": 0.35,
            "jac_hard": 0.80,
            "jac_relax": 0.20,
            "jac_mul_min": 0.20,
            "jac_mul_max": 0.50,
            "jac_rate_up": 0.02,
            "jac_rate_down": 0.01,
        },
        "IXI": {
            "icon_start_frac": 0.00,
            "icon_warm_frac": 0.075,
            "cyc_start_frac": 2.00,
            "cyc_warm_frac": 0.10,
            "aux_until_frac": 0.25,
            "aux_l1_start": 0.12,
            "aux_l2_start": 0.08,
            "jac_soft": 0.12,
            "jac_hard": 0.35,
            "jac_relax": 0.03,
            "jac_mul_min": 0.20,
            "jac_mul_max": 0.40,
            "jac_rate_up": 0.015,
            "jac_rate_down": 0.008,
        },
        "SYNTH": {
            "icon_start_frac": 0.00,
            "icon_warm_frac": 0.075,
            "cyc_start_frac": 0.15,
            "cyc_warm_frac": 0.10,
            "aux_until_frac": 0.25,
            "aux_l1_start": 0.12,
            "aux_l2_start": 0.08,
            "jac_soft": 0.35,
            "jac_hard": 0.80,
            "jac_relax": 0.20,
            "jac_mul_min": 0.20,
            "jac_mul_max": 0.50,
            "jac_rate_up": 0.02,
            "jac_rate_down": 0.01,
        },
    }


    def __init__(self, *, ds: str, max_epoch: int):
        max_epoch = max(1, int(max_epoch))
        profile = self.PROFILES[str(ds).upper()]

        self.icon_start_epoch = int(round(profile["icon_start_frac"] * max_epoch))
        self.icon_warm_epochs = max(1, int(round(profile["icon_warm_frac"] * max_epoch)))
        self.cyc_start_epoch = int(round(profile["cyc_start_frac"] * max_epoch))
        self.cyc_warm_epochs = max(1, int(round(profile["cyc_warm_frac"] * max_epoch)))
        self.aux_until = max(0, int(round(profile["aux_until_frac"] * max_epoch)))

        self.aux_l1_start = profile["aux_l1_start"]
        self.aux_l2_start = profile["aux_l2_start"]
        self.jac_soft = profile["jac_soft"]
        self.jac_hard = profile["jac_hard"]
        self.jac_relax = profile["jac_relax"]
        self.jac_mul_min = profile["jac_mul_min"]
        self.jac_mul_max = profile["jac_mul_max"]
        self.jac_rate_up = profile["jac_rate_up"]
        self.jac_rate_down = profile["jac_rate_down"]

        self.w_icon_mul = 0.0
        self.w_cyc_mul = 0.0
        self.w_jac_mul = self.jac_mul_min


    def on_epoch_start(self, epoch: int):
        self.w_icon_mul = _ramp(epoch, self.icon_start_epoch, self.icon_start_epoch + self.icon_warm_epochs)
        self.w_cyc_mul = _ramp(epoch, self.cyc_start_epoch, self.cyc_start_epoch + self.cyc_warm_epochs)


    def update_from_val(self, val_jac_percent: float):
        jac = float(val_jac_percent)
        if jac > self.jac_hard: target, rate = self.jac_mul_max, self.jac_rate_up
        elif jac > self.jac_soft:
            target = self.jac_mul_min + 0.5 * (self.jac_mul_max - self.jac_mul_min)
            rate = self.jac_rate_up * 0.5
        elif jac < self.jac_relax: target, rate = self.jac_mul_min, self.jac_rate_down
        else: target, rate = self.jac_mul_min, self.jac_rate_down

        self.w_jac_mul = _move_towards(self.w_jac_mul, target, rate)
        self.w_jac_mul = _clamp(self.w_jac_mul, self.jac_mul_min, self.jac_mul_max)


    def aux_lambdas(self, epoch: int):
        if self.aux_until <= 0 or int(epoch) >= self.aux_until:
            return 0.0, 0.0
        t = float(epoch) / float(max(1, self.aux_until - 1))
        return self.aux_l1_start * (1.0 - t), self.aux_l2_start * (1.0 - t)


def apply_ctcf_dataset_defaults(args):
    ds_key = str(args.ds).upper()

    if args.w_reg is None:
        args.w_reg = 4.0 if ds_key == "IXI" else 1.0
