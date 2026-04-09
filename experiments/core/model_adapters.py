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
        use_checkpoint: Optional[bool] = None,
        synth_img_size: Optional[Tuple[int, int, int]] = None,
        synth_dwin: Optional[Tuple[int, int, int]] = None,
        l1_base_ch: Optional[int] = None,
        l3_base_ch: Optional[int] = None,
        l3_error_mode: Optional[str] = None,
        prealign_encoder: Optional[bool] = None,
        # GEN2 enhancements (architectural)
        l3_iters: Optional[int] = None,
        l3_full_res: Optional[bool] = None,
        learned_upsample: Optional[bool] = None,
        l2_l3_skip: Optional[bool] = None,
        l1_half_res: Optional[bool] = None,
        l2_full_res: Optional[bool] = None,
        l1_l2_skip: Optional[bool] = None,
        l3_compose: Optional[bool] = None,
        l3_svf: Optional[bool] = None,
        # GEN2.5 enhancements (capacity)
        l3_cab: Optional[bool] = None,
        l3_context_blocks: Optional[int] = None,
        l3_gate: Optional[bool] = None,
        l3_unshared: Optional[bool] = None,
        l1_cab: Optional[bool] = None,
    ) -> torch.nn.Module:
        from models.CTCF.model import CONFIGS, CTCF_CascadeA

        cfg = deepcopy(CONFIGS[config_key])
        cfg.time_steps = int(time_steps)

        if synth_img_size is not None:
            d, h, w = (int(v) for v in synth_img_size)
            cfg.img_size = (d, h, w)
            cfg.window_size = (d // 32, h // 32, w // 32)

        if synth_dwin is not None: cfg.dwin_size = tuple(int(v) for v in synth_dwin)
        if use_checkpoint is not None: cfg.use_checkpoint = bool(use_checkpoint)
        if l1_base_ch is not None: cfg.level1_base_ch = int(l1_base_ch)
        if l3_base_ch is not None: cfg.level3_base_ch = int(l3_base_ch)
        if l3_error_mode is not None: cfg.level3_error_mode = str(l3_error_mode)
        if prealign_encoder is not None: cfg.prealign_encoder = bool(prealign_encoder)

        # GEN2 (architectural)
        if l3_iters is not None: cfg.l3_iters = int(l3_iters)
        if l3_full_res is not None: cfg.l3_full_res = bool(l3_full_res)
        if learned_upsample is not None: cfg.learned_upsample = bool(learned_upsample)
        if l2_l3_skip is not None: cfg.l2_l3_skip = bool(l2_l3_skip)
        if l1_half_res is not None: cfg.l1_half_res = bool(l1_half_res)
        if l2_full_res is not None: cfg.l2_full_res = bool(l2_full_res)
        if l1_l2_skip is not None: cfg.l1_l2_skip = bool(l1_l2_skip)
        if l3_compose is not None: cfg.l3_compose = bool(l3_compose)
        if l3_svf is not None: cfg.l3_svf = bool(l3_svf)

        # GEN2.5 (capacity)
        if l3_cab is not None: cfg.l3_cab = bool(l3_cab)
        if l3_context_blocks is not None: cfg.l3_context_blocks = int(l3_context_blocks)
        if l3_gate is not None: cfg.l3_gate = bool(l3_gate)
        if l3_unshared is not None: cfg.l3_unshared = bool(l3_unshared)
        if l1_cab is not None: cfg.l1_cab = bool(l1_cab)

        model = CTCF_CascadeA(cfg)
        model.cfg = cfg
        return model

    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y, return_all=False, alpha_l1=1.0)
        return flow.float()


def get_model_adapter(name: str) -> ModelAdapter:
    """Resolve adapter by model name."""
    key = name.strip().lower()
    match key:
        case "tm-dca": return TmDcaAdapter()
        case "utsrmorph": return UtsrMorphAdapter()
        case "ctcf": return CtcfAdapter()
        case _: raise ValueError(f"Unknown model: {name}")
