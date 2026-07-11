from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F


def _apply_optional_overrides(cfg: Any, **overrides: Any) -> None:
    """Assign each override to cfg as an attribute, skipping entries whose value is None."""
    for name, value in overrides.items():
        if value is None:
            continue
        setattr(cfg, name, value)


class ModelAdapter:
    """Unified adapter interface: model build + forward flow."""

    key = ""

    def build(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class TmDcaAdapter(ModelAdapter):
    key = "tm-dca"

    def build(
        self,
        time_steps: int = 12,
        config_key: str = "TransMorph-3-LVL",
    ) -> torch.nn.Module:
        from models.TransMorph_DCA.configs import CONFIGS
        from models.TransMorph_DCA.model import TransMorphCascadeAd

        cfg = deepcopy(CONFIGS[config_key])
        model = TransMorphCascadeAd(cfg, time_steps)
        model.cfg = cfg
        return model

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
        pool: int = 2,
    ) -> torch.Tensor:
        use_amp = amp and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            flow_half = model((F.avg_pool3d(x, pool), F.avg_pool3d(y, pool)))
        return F.interpolate(
            flow_half.float(),
            scale_factor=pool,
            mode="trilinear",
            align_corners=False,
        ) * float(pool)


class UtsrMorphAdapter(ModelAdapter):
    key = "utsrmorph"

    def build(self, config_key: str = "UTSRMorph-Large") -> torch.nn.Module:
        from models.UTSRMorph.configs import CONFIGS
        from models.UTSRMorph.model import UTSRMorph

        cfg = deepcopy(CONFIGS[config_key])
        model = UTSRMorph(cfg)
        model.cfg = cfg
        return model

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        use_amp = amp and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(torch.cat((x, y), dim=1))
        return flow


class CtcfAdapter(ModelAdapter):
    key = "ctcf"

    def build(
        self,
        time_steps: int = 12,
        config_key: str = "CTCF-CascadeA",
        use_checkpoint: bool | None = None,
        synth_img_size: tuple[int, int, int] | None = None,
        synth_dwin: tuple[int, int, int] | None = None,
        l1_base_ch: int | None = None,
        l3_base_ch: int | None = None,
        l3_error_mode: str | None = None,
        l3_corr_mode: str | None = None,
        l3_iters: int | None = None,
        l3_unshared: bool | None = None,
        l1_half_res: bool | None = None,
        l2_full_res: bool | None = None,
        l3_full_res: bool | None = None,
        l3_svf: bool | None = None,
        l3_num_heads: int | None = None,
    ) -> torch.nn.Module:
        from models.CTCF.configs import CONFIGS
        from models.CTCF.model import CTCFCascadeA

        cfg = deepcopy(CONFIGS[config_key])
        cfg.time_steps = time_steps

        if synth_img_size is not None:
            d, h, w = synth_img_size
            cfg.img_size = (d, h, w)
            cfg.window_size = (d // 32, h // 32, w // 32)

        dwin_size = None if synth_dwin is None else tuple(synth_dwin)
        _apply_optional_overrides(
            cfg=cfg,
            dwin_size=dwin_size,
            use_checkpoint=use_checkpoint,
            level1_base_ch=l1_base_ch,
            level3_base_ch=l3_base_ch,
            level3_error_mode=l3_error_mode,
            level3_corr_mode=l3_corr_mode,
            level3_num_heads=l3_num_heads,
            l3_iters=l3_iters,
            l3_unshared=l3_unshared,
            l1_half_res=l1_half_res,
            l2_full_res=l2_full_res,
            l3_full_res=l3_full_res,
            l3_svf=l3_svf,
        )

        model = CTCFCascadeA(cfg)
        model.cfg = cfg
        return model

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        use_amp = amp and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y, alpha_l1=1.0)
        return flow.float()


class VoxelMorphAdapter(ModelAdapter):
    key = "voxelmorph"

    def build(
        self,
        config_key: str = "VxmDense",
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.VoxelMorph.configs import CONFIGS
        from models.VoxelMorph.wrapper import VxmDense

        cfg = deepcopy(CONFIGS[config_key])
        img_size = tuple(cfg.img_size) if img_size is None else tuple(img_size)
        model = VxmDense(
            vol_size=img_size,
            enc_nf=list(cfg.enc_nf),
            dec_nf=list(cfg.dec_nf),
            int_steps=cfg.int_steps,
        )
        model.cfg = cfg
        return model

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        use_amp = amp and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y)
        return flow.float()


class LkuNetAdapter(ModelAdapter):
    key = "lkunet"

    def build(
        self,
        config_key: str = "LKU-8",
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.LKUNet.configs import CONFIGS
        from models.LKUNet.wrapper import LkuNetSolo

        cfg = deepcopy(CONFIGS[config_key])
        img_size = tuple(cfg.img_size) if img_size is None else tuple(img_size)
        model = LkuNetSolo(
            vol_size=img_size,
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
        )
        model.cfg = cfg
        return model

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        """Return voxel-unit flow; LKU-Net's UNet predicts normalised [-1,1] flow."""
        use_amp = amp and torch.cuda.is_available()
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

    def build(
        self,
        config_key: str = "EfficientMorph_2x3_2",
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.EfficientMorph.wrapper import EfficientMorphSolo

        return EfficientMorphSolo(config_key=config_key, img_size=img_size)

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        use_amp = amp and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y)
        return flow.float()


class MambaMorphAdapter(ModelAdapter):
    key = "mambamorph"

    def build(
        self,
        config_key: str = "MambaMorph",
        diffeomorphic: bool = True,
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.MambaMorph.wrapper import MambaMorphSolo

        return MambaMorphSolo(
            config_key=config_key,
            diffeomorphic=diffeomorphic,
            img_size=img_size,
        )

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        del amp
        with torch.autocast(device_type="cuda", enabled=False):
            _, flow = model(x, y)
        return flow.float()


class VMambaMorphAdapter(ModelAdapter):
    key = "vmambamorph"

    def build(
        self,
        config_key: str = "VMambaMorph",
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.VMambaMorph.wrapper import VMambaMorphSolo

        return VMambaMorphSolo(config_key=config_key, img_size=img_size)

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        del amp
        with torch.autocast(device_type="cuda", enabled=False):
            _, flow = model(x, y)
        return flow.float()


class CorrMLPAdapter(ModelAdapter):
    key = "corrmlp"

    def build(
        self,
        enc_channels: int = 8,
        dec_channels: int = 16,
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.CorrMLP.wrapper import CorrMLPSolo

        del img_size  # CorrMLP is fully convolutional; input size is not needed at build time
        return CorrMLPSolo(enc_channels=enc_channels, dec_channels=dec_channels, use_checkpoint=False)

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        use_amp = amp and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = model(x, y)
        return flow.float()


class SACBAdapter(ModelAdapter):
    key = "sacb"

    def build(
        self,
        num_k: int = 7,
        ch_scale: int = 4,
        img_size: tuple[int, int, int] | None = None,
    ) -> torch.nn.Module:
        from models.SACB.wrapper import SACBSolo

        img_size = (160, 192, 224) if img_size is None else tuple(img_size)
        return SACBSolo(img_size=img_size, num_k=num_k, ch_scale=ch_scale)

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        del amp  # SACB's kmeans_gpu breaks under fp16 autocast; force fp32 (see train_SACB)
        with torch.autocast(device_type="cuda", enabled=False):
            _, flow = model(x, y)
        return flow.float()


ADAPTERS: dict[str, type[ModelAdapter]] = {
    "tm-dca": TmDcaAdapter,
    "utsrmorph": UtsrMorphAdapter,
    "ctcf": CtcfAdapter,
    "voxelmorph": VoxelMorphAdapter,
    "lkunet": LkuNetAdapter,
    "efficientmorph": EfficientMorphAdapter,
    "mambamorph": MambaMorphAdapter,
    "vmambamorph": VMambaMorphAdapter,
    "corrmlp": CorrMLPAdapter,
    "sacb": SACBAdapter,
}


def get_model_adapter(name: str) -> ModelAdapter:
    """Resolve adapter by model name."""
    key = name.strip().lower()
    if key not in ADAPTERS:
        raise ValueError(f"Unknown model: {name}")
    return ADAPTERS[key]()
