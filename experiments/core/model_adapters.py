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
        from models.TransMorph_DCA.configs import CONFIGS
        from models.TransMorph_DCA.model import TransMorphCascadeAd

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
        from models.UTSRMorph.configs import CONFIGS
        from models.UTSRMorph.model import UTSRMorph

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
        l3_iters: Optional[int] = None,
        l3_unshared: Optional[bool] = None,
        l1_half_res: Optional[bool] = None,
        l2_full_res: Optional[bool] = None,
        l3_full_res: Optional[bool] = None,
        l3_svf: Optional[bool] = None,
        l3_num_heads: Optional[int] = None,
    ) -> torch.nn.Module:
        from models.CTCF.configs import CONFIGS
        from models.CTCF.model import CTCFCascadeA

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
        if l3_num_heads is not None: cfg.level3_num_heads = int(l3_num_heads)

        if l3_iters is not None: cfg.l3_iters = int(l3_iters)
        if l3_unshared is not None: cfg.l3_unshared = bool(l3_unshared)
        if l1_half_res is not None: cfg.l1_half_res = bool(l1_half_res)
        if l2_full_res is not None: cfg.l2_full_res = bool(l2_full_res)
        if l3_full_res is not None: cfg.l3_full_res = bool(l3_full_res)
        if l3_svf is not None: cfg.l3_svf = bool(l3_svf)

        model = CTCFCascadeA(cfg)
        model.cfg = cfg
        return model


    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y, alpha_l1=1.0)
        return flow.float()


class VoxelMorphAdapter(ModelAdapter):
    key = "voxelmorph"

    def build(self, *, config_key: str = "VxmDense", img_size: Optional[Tuple[int, int, int]] = None) -> torch.nn.Module:
        from models.VoxelMorph.configs import CONFIGS
        from models.VoxelMorph.wrapper import VxmDense

        cfg = deepcopy(CONFIGS[config_key])
        img_size = tuple(cfg.img_size) if img_size is None else tuple(int(v) for v in img_size)
        model = VxmDense(
            vol_size=img_size,
            enc_nf=list(cfg.enc_nf),
            dec_nf=list(cfg.dec_nf),
            int_steps=int(cfg.int_steps),
        )
        model.cfg = cfg
        return model


    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y)
        return flow.float()


class LkuNetAdapter(ModelAdapter):
    key = "lkunet"

    def build(self, *, config_key: str = "LKU-8", img_size: Optional[Tuple[int, int, int]] = None) -> torch.nn.Module:
        from models.LKUNet.configs import CONFIGS
        from models.LKUNet.wrapper import LkuNetSolo

        cfg = deepcopy(CONFIGS[config_key])
        img_size = tuple(cfg.img_size) if img_size is None else tuple(int(v) for v in img_size)
        model = LkuNetSolo(
            vol_size=img_size,
            in_channel=int(cfg.in_channel),
            n_classes=int(cfg.n_classes),
            start_channel=int(cfg.start_channel),
        )
        model.cfg = cfg
        return model


    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        """Return voxel-unit flow; LKU-Net predicts normalized grid flow."""
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            flow_normalized = model.unet(x, y)
        flow_normalized = flow_normalized.float()
        d, h, w = flow_normalized.shape[-3:]
        scale = torch.tensor(
            [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0],
            device=flow_normalized.device,
            dtype=flow_normalized.dtype,
        ).view(1, 3, 1, 1, 1)
        return flow_normalized * scale


class EfficientMorphAdapter(ModelAdapter):
    key = "efficientmorph"

    def build(self, *, config_key: str = "EfficientMorph_2x3_2", img_size: Optional[Tuple[int, int, int]] = None) -> torch.nn.Module:
        from models.EfficientMorph.wrapper import EfficientMorphSolo

        model = EfficientMorphSolo(config_key=config_key, img_size=img_size)
        return model


    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        use_amp = bool(amp and torch.cuda.is_available())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y)
        return flow.float()


class MambaMorphAdapter(ModelAdapter):
    key = "mambamorph"

    def build(self, *, config_key: str = "MambaMorph", diffeomorphic: bool = True, img_size: Optional[Tuple[int, int, int]] = None) -> torch.nn.Module:
        from models.MambaMorph.wrapper import MambaMorphSolo

        model = MambaMorphSolo(config_key=config_key, diffeomorphic=diffeomorphic, img_size=img_size)
        return model


    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            _, flow = model(x, y)
        return flow.float()


class VMambaMorphAdapter(ModelAdapter):
    key = "vmambamorph"

    def build(self, *, config_key: str = "VMambaMorph", img_size: Optional[Tuple[int, int, int]] = None) -> torch.nn.Module:
        from models.VMambaMorph.wrapper import VMambaMorphSolo

        model = VMambaMorphSolo(config_key=config_key, img_size=img_size)
        return model


    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            _, flow = model(x, y)
        return flow.float()


def get_model_adapter(name: str) -> ModelAdapter:
    """Resolve adapter by model name."""
    key = name.strip().lower()
    match key:
        case "tm-dca": return TmDcaAdapter()
        case "utsrmorph": return UtsrMorphAdapter()
        case "ctcf": return CtcfAdapter()
        case "voxelmorph" | "vxm" | "vxmdense": return VoxelMorphAdapter()
        case "lkunet": return LkuNetAdapter()
        case "efficientmorph": return EfficientMorphAdapter()
        case "mambamorph": return MambaMorphAdapter()
        case "vmambamorph": return VMambaMorphAdapter()
        case _: raise ValueError(f"Unknown model: {name}")
