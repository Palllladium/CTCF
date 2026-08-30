"""The frozen Stage-5 training configurations and the one controller construction site.

Six call sites used to spell out ``width`` and ``free_residual_limit_voxels`` by hand and
rely on the model's own default collar. Building every Stage-5 controller here means the
protocol geometry reaches the module that applies it, instead of being restated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from experiments.stage5.losses import ControllerLossConfig
from experiments.stage5.safety import COLLAR_WIDTH
from models.CTCF.controller import Stage5SpatialController
from tools.analysis.stage5.primitives import canonical_sha256, require_finite, require_int

STAGE5_U0_FIXED_EPOCH = 400
STAGE5_CONTROLLER_FIXED_EPOCH = 100
STAGE5_SEEDS = (0, 1, 2)
STAGE5_AMP_INITIAL_SCALE = 65536.0
STAGE5_AMP_GROWTH_INTERVAL = 1_000_000


@dataclass(frozen=True, slots=True)
class U0TrainingConfig:
    fixed_epoch: int = STAGE5_U0_FIXED_EPOCH
    learning_rate: float = 1e-4
    time_steps: int = 6
    config_key: str = "CTCF-CascadeA-Mamba"
    w_ncc: float = 1.0
    w_reg: float = 1.0
    w_icon: float = 0.05
    w_jac: float = 0.005
    amp_initial_scale: float = STAGE5_AMP_INITIAL_SCALE
    amp_growth_interval: int = STAGE5_AMP_GROWTH_INTERVAL

    def __post_init__(self) -> None:
        endpoint = f"the frozen {STAGE5_U0_FIXED_EPOCH}-epoch U0 endpoint"
        if require_int(self.fixed_epoch, endpoint, error=ValueError) != STAGE5_U0_FIXED_EPOCH:
            raise ValueError(f"Stage5 U0 must use the frozen {STAGE5_U0_FIXED_EPOCH}-epoch endpoint")
        architecture = "the frozen CTCF-CascadeA-Mamba time_steps"
        if (
            self.config_key != "CTCF-CascadeA-Mamba"
            or require_int(self.time_steps, architecture, error=ValueError) != 6
        ):
            raise ValueError("Stage5 U0 must use the frozen CTCF-CascadeA-Mamba/time_steps=6 architecture")
        for name in ("learning_rate", "w_ncc", "w_reg", "w_icon", "w_jac", "amp_initial_scale"):
            require_finite(getattr(self, name), f"Stage5 U0 {name}", minimum=0.0, error=ValueError)
        if self.learning_rate <= 0.0 or self.w_ncc <= 0.0:
            raise ValueError("Stage5 U0 learning rate and NCC weight must be positive")
        if self.amp_initial_scale != STAGE5_AMP_INITIAL_SCALE:
            raise ValueError("Stage5 U0 must use the frozen AMP initial scale")
        growth = require_int(self.amp_growth_interval, "Stage5 U0 AMP growth interval", minimum=1, error=ValueError)
        if growth != STAGE5_AMP_GROWTH_INTERVAL:
            raise ValueError("Stage5 U0 must use the frozen AMP growth interval")


@dataclass(frozen=True, slots=True)
class ControllerTrainingConfig:
    fixed_epoch: int = STAGE5_CONTROLLER_FIXED_EPOCH
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    free_residual_limit_voxels: float = 2.0
    width: int = 16
    loss: ControllerLossConfig = field(default_factory=ControllerLossConfig)
    amp_initial_scale: float = STAGE5_AMP_INITIAL_SCALE
    amp_growth_interval: int = STAGE5_AMP_GROWTH_INTERVAL

    def __post_init__(self) -> None:
        # ControllerLossConfig validates its own weights; re-checking them here would put
        # one object's contract in two places.
        endpoint = f"the frozen {STAGE5_CONTROLLER_FIXED_EPOCH}-epoch controller endpoint"
        if require_int(self.fixed_epoch, endpoint, error=ValueError) != STAGE5_CONTROLLER_FIXED_EPOCH:
            raise ValueError(f"Stage5 controllers must use the frozen {STAGE5_CONTROLLER_FIXED_EPOCH}-epoch endpoint")
        for name in ("learning_rate", "weight_decay", "free_residual_limit_voxels", "amp_initial_scale"):
            require_finite(getattr(self, name), f"Stage5 controller {name}", minimum=0.0, error=ValueError)
        if self.learning_rate <= 0.0 or self.free_residual_limit_voxels <= 0.0:
            raise ValueError("invalid Stage5 controller optimizer contract")
        require_int(self.width, "Stage5 controller width", minimum=4, error=ValueError)
        if self.amp_initial_scale != STAGE5_AMP_INITIAL_SCALE:
            raise ValueError("Stage5 controllers must use the frozen AMP initial scale")
        growth = require_int(
            self.amp_growth_interval,
            "Stage5 controller AMP growth interval",
            minimum=1,
            error=ValueError,
        )
        if growth != STAGE5_AMP_GROWTH_INTERVAL:
            raise ValueError("Stage5 controllers must use the frozen AMP growth interval")


def build_stage5_controller(config: ControllerTrainingConfig) -> Stage5SpatialController:
    """Construct the controller every Stage-5 phase uses, carrying the frozen collar."""
    return Stage5SpatialController(
        width=config.width,
        free_residual_limit_voxels=config.free_residual_limit_voxels,
        collar_width=COLLAR_WIDTH,
    )


def config_sha256(config: U0TrainingConfig | ControllerTrainingConfig) -> str:
    return canonical_sha256(asdict(config))


def require_seed(seed: int) -> int:
    if require_int(seed, "Stage5 seed", error=ValueError) not in STAGE5_SEEDS:
        raise ValueError(f"Stage5 seed must be one of {STAGE5_SEEDS}")
    return seed


__all__ = [
    "STAGE5_AMP_GROWTH_INTERVAL",
    "STAGE5_AMP_INITIAL_SCALE",
    "STAGE5_CONTROLLER_FIXED_EPOCH",
    "STAGE5_SEEDS",
    "STAGE5_U0_FIXED_EPOCH",
    "ControllerTrainingConfig",
    "U0TrainingConfig",
    "build_stage5_controller",
    "config_sha256",
    "require_seed",
]
