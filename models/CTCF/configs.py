from __future__ import annotations

import ml_collections


def get_ctcf_config() -> ml_collections.ConfigDict:
    """Paper 1 CTCF Swin-DCA cascade preset (half-res L2/L3, L3 base_ch=64)."""
    c = ml_collections.ConfigDict()

    c.if_transskip = True
    c.if_convskip = True
    c.in_chans = 1
    c.img_size = (160, 192, 224)
    c.patch_size = 4

    c.embed_dim = 96
    c.depths = (4, 4, 5)
    c.num_heads = (8, 8, 8)
    c.window_size = (5, 6, 7)
    c.dwin_size = (7, 5, 3)

    c.mlp_ratio = 4
    c.pat_merg_rf = 4

    c.qkv_bias = False
    c.drop_rate = 0.0
    c.drop_path_rate = 0.3
    c.ape = False
    c.spe = False
    c.rpe = True
    c.patch_norm = True
    c.use_checkpoint = False
    c.out_indices = (0, 1, 2)

    c.reg_head_chan = 16
    c.time_steps = 6

    c.backbone = "swin-dca"

    c.use_level1 = True
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = True
    c.level3_base_ch = 64
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1
    c.level3_corr_mode = "none"

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = False
    c.l3_full_res = False
    c.l3_svf = False

    return c


def _unified_cascade_base(
    backbone: str,
    use_cascade: bool = True,
    l3_svf: bool = False,
) -> ml_collections.ConfigDict:
    """Phase 7+ unified cascade scaffold: full-res L2/L3, L3 base_ch=32, NCC error mode."""
    c = ml_collections.ConfigDict()

    c.backbone = backbone
    c.img_size = (160, 192, 224)
    c.time_steps = 0

    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 32
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1
    c.level3_corr_mode = "none"

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = l3_svf

    return c


def get_ctcf_vm_legacy_config(use_cascade: bool = True) -> ml_collections.ConfigDict:
    """Legacy VxM cascade preset (Paper 1 protocol). Kept for P9/historical checkpoints."""
    c = ml_collections.ConfigDict()

    c.backbone = "vxm"
    c.img_size = (160, 192, 224)
    c.time_steps = 0

    c.vxm = ml_collections.ConfigDict()
    c.vxm.enc_nf = [16, 32, 32, 32]
    c.vxm.dec_nf = [32, 32, 32, 32, 32, 16, 16]
    c.vxm.int_steps = 7

    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 64
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1
    c.level3_corr_mode = "none"

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = False
    c.l3_full_res = False
    c.l3_svf = False

    return c


def get_ctcf_vm_unified_config(use_cascade: bool = True) -> ml_collections.ConfigDict:
    """VxM cascade preset on the unified protocol."""
    c = _unified_cascade_base(
        backbone="vxm",
        use_cascade=use_cascade,
        l3_svf=False,
    )

    c.vxm = ml_collections.ConfigDict()
    c.vxm.enc_nf = [16, 32, 32, 32]
    c.vxm.dec_nf = [32, 32, 32, 32, 32, 16, 16]
    c.vxm.int_steps = 7

    return c


def get_ctcf_lku_config(
    config_key: str = "LKU-8",
    use_cascade: bool = True,
) -> ml_collections.ConfigDict:
    """LKU-Net cascade preset. `config_key` selects the LKU width variant."""
    c = _unified_cascade_base(
        backbone="lku",
        use_cascade=use_cascade,
        l3_svf=False,
    )
    c.lku_config = config_key
    return c


def get_ctcf_mamba_config(use_cascade: bool = True) -> ml_collections.ConfigDict:
    """MambaMorph cascade preset. L3-SVF on by default."""
    c = _unified_cascade_base(
        backbone="mamba",
        use_cascade=use_cascade,
        l3_svf=True,
    )
    c.mamba_config = "MambaMorph"
    return c


def get_ctcf_vmamba_config(use_cascade: bool = True) -> ml_collections.ConfigDict:
    """VMambaMorph cascade preset. L3-SVF on by default."""
    c = _unified_cascade_base(
        backbone="vmamba",
        use_cascade=use_cascade,
        l3_svf=True,
    )
    c.vmamba_config = "VMambaMorph"
    return c


def get_ctcf_effm_config(
    config_key: str = "EfficientMorph_2x3_2_hires",
    use_cascade: bool = True,
) -> ml_collections.ConfigDict:
    """EfficientMorph cascade preset."""
    c = _unified_cascade_base(
        backbone="effm",
        use_cascade=use_cascade,
        l3_svf=False,
    )
    c.effm_config = config_key
    return c


CONFIGS = {
    "CTCF-CascadeA": get_ctcf_config(),
    "CTCF-CascadeA-VM": get_ctcf_vm_legacy_config(use_cascade=True),
    "CTCF-VM-solo": get_ctcf_vm_legacy_config(use_cascade=False),
    "CTCF-CascadeA-VM-Unified": get_ctcf_vm_unified_config(use_cascade=True),
    "CTCF-VM-Unified-solo": get_ctcf_vm_unified_config(use_cascade=False),
    "CTCF-CascadeA-LKU8": get_ctcf_lku_config("LKU-8", use_cascade=True),
    "CTCF-LKU8-solo": get_ctcf_lku_config("LKU-8", use_cascade=False),
    "CTCF-CascadeA-LKU32": get_ctcf_lku_config("LKU-32", use_cascade=True),
    "CTCF-LKU32-solo": get_ctcf_lku_config("LKU-32", use_cascade=False),
    "CTCF-CascadeA-Mamba": get_ctcf_mamba_config(use_cascade=True),
    "CTCF-Mamba-solo": get_ctcf_mamba_config(use_cascade=False),
    "CTCF-CascadeA-VMamba": get_ctcf_vmamba_config(use_cascade=True),
    "CTCF-VMamba-solo": get_ctcf_vmamba_config(use_cascade=False),
    "CTCF-CascadeA-EFFM": get_ctcf_effm_config("EfficientMorph_2x3_2_hires", use_cascade=True),
    "CTCF-EFFM-solo": get_ctcf_effm_config("EfficientMorph_2x3_2_hires", use_cascade=False),
}
