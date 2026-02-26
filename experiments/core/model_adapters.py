from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class ModelAdapter:
    """Unified adapter interface: model build + forward flow."""
    key = ""

    def build(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class TmDcaAdapter(ModelAdapter):
    key = "tm-dca"

    def build(self, *, time_steps: int = 12, config_key: str = "TransMorph-3-LVL") -> torch.nn.Module:
        from models.TransMorph_DCA.model import CONFIGS, TransMorphCascadeAd

        cfg = deepcopy(CONFIGS[config_key])
        model = TransMorphCascadeAd(cfg, int(time_steps))
        model.cfg = cfg
        return model

    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True, pool: int = 2) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            flow_half = model((F.avg_pool3d(x, pool), F.avg_pool3d(y, pool)))
        return F.interpolate(flow_half.float(), scale_factor=pool, mode="trilinear", align_corners=False) * float(pool)


class UtsrMorphAdapter(ModelAdapter):
    key = "utsrmorph"

    def build(self, *, config_key: str = "UTSRMorph-Large") -> torch.nn.Module:
        from models.UTSRMorph.model import CONFIGS, UTSRMorph

        cfg = deepcopy(CONFIGS[config_key])
        model = UTSRMorph(cfg)
        model.cfg = cfg
        return model

    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(torch.cat((x, y), dim=1))
        return flow


class CtcfAdapter(ModelAdapter):
    key = "ctcf"

    def build(
        self,
        *,
        time_steps: int = 12,
        config_key: str = "CTCF-CascadeA",
        use_level1: Optional[bool] = None,
        use_level3: Optional[bool] = None,
        use_checkpoint: Optional[bool] = None,
        synth_img_size: Optional[Tuple[int, int, int]] = None,
        synth_dwin: Optional[Tuple[int, int, int]] = None,
    ) -> torch.nn.Module:
        from models.CTCF.model import CONFIGS, CTCF_CascadeA

        cfg = deepcopy(CONFIGS[config_key])
        cfg.time_steps = int(time_steps)

        if synth_img_size is not None:
            d, h, w = (int(v) for v in synth_img_size)
            cfg.img_size = (d, h, w)
            cfg.window_size = (d // 32, h // 32, w // 32)

        if synth_dwin is not None:
            cfg.dwin_size = tuple(int(v) for v in synth_dwin)

        for k, v in {"use_level1": use_level1, "use_level3": use_level3, "use_checkpoint": use_checkpoint}.items():
            if v is not None:
                setattr(cfg, k, bool(v))

        model = CTCF_CascadeA(cfg)
        model.cfg = cfg
        return model

    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y, return_all=False, alpha_l1=1.0, alpha_l3=1.0)
        return flow.float()


def get_model_adapter(name: str) -> ModelAdapter:
    """Resolve adapter by model name."""
    key = name.strip().lower()
    match key:
        case "tm-dca": return TmDcaAdapter()
        case "utsrmorph": return UtsrMorphAdapter()
        case "ctcf": return CtcfAdapter()
        case _: raise ValueError(f"Unknown model: {name}")
