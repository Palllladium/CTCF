import ml_collections


def get_CTCF_config():
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

    c.use_level1 = True
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = True
    c.level3_base_ch = 64
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = False
    c.l3_full_res = False
    c.l3_svf = False

    return c


def get_CTCF_VM_config(*, use_cascade=True):
    """Legacy VxM cascade config — half-res L2, L3 base_ch=64.
    Preserved for backward-compatibility with P2_13 and P9_CASC_VXM_SVF_* ckpts."""
    c = ml_collections.ConfigDict()
    c.backbone = "vxm"
    c.img_size = (160, 192, 224)
    c.time_steps = 0  # unused by VxmDenseHalf (kept for adapter compatibility)

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

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = False
    c.l3_full_res = False
    c.l3_svf = False

    return c


def get_CTCF_VM_Unified_config(*, use_cascade=True):
    """Unified VxM cascade config matching LKU/Mamba/VMamba protocol:
    full-res L2, L3 base_ch=32. Used for Phase 10 longruns and beyond.
    Re-trained from scratch — not compatible with legacy CTCF-CascadeA-VM ckpts."""
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
    c.level3_base_ch = 32          # unified with other backbones
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True           # unified — full-res L2
    c.l3_full_res = False
    c.l3_svf = False               # toggle via --l3_svf 1 at training

    return c


def get_CTCF_LKU_config(config_key="LKU-8", *, use_cascade=True):
    c = ml_collections.ConfigDict()
    c.backbone = "lku"
    c.lku_config = str(config_key)
    c.img_size = (160, 192, 224)
    c.time_steps = 0

    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 32
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = False

    return c


def get_CTCF_Mamba_config(*, use_cascade=True):
    c = ml_collections.ConfigDict()
    c.backbone = "mamba"
    c.mamba_config = "MambaMorph"
    c.img_size = (160, 192, 224)
    c.time_steps = 0

    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 32
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = True

    return c


def get_CTCF_EFFM_config(config_key="EfficientMorph_2x3_2_hires", *, use_cascade=True):
    """EfficientMorph as L2 cascade backbone — full-res, L3 base_ch=32 (unified protocol)."""
    c = ml_collections.ConfigDict()
    c.backbone = "effm"
    c.effm_config = str(config_key)
    c.img_size = (160, 192, 224)
    c.time_steps = 0

    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 32
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = False  # toggle via --l3_svf 1 at training

    return c


def get_CTCF_VMamba_config(*, use_cascade=True):
    c = ml_collections.ConfigDict()
    c.backbone = "vmamba"
    c.vmamba_config = "VMambaMorph"
    c.img_size = (160, 192, 224)
    c.time_steps = 0

    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 32
    c.level3_error_mode = "ncc"
    c.level3_num_heads = 1

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = True

    return c


CONFIGS = {
    "CTCF-CascadeA": get_CTCF_config(),
    # VxM — legacy config (half-res L2, L3 base_ch=64) kept for P9/historical ckpts
    "CTCF-CascadeA-VM": get_CTCF_VM_config(use_cascade=True),
    "CTCF-VM-solo": get_CTCF_VM_config(use_cascade=False),
    # VxM — unified protocol (full-res L2, L3 base_ch=32) — for Phase 10+ runs
    "CTCF-CascadeA-VM-Unified": get_CTCF_VM_Unified_config(use_cascade=True),
    "CTCF-VM-Unified-solo": get_CTCF_VM_Unified_config(use_cascade=False),
    "CTCF-CascadeA-LKU8": get_CTCF_LKU_config("LKU-8", use_cascade=True),
    "CTCF-LKU8-solo": get_CTCF_LKU_config("LKU-8", use_cascade=False),
    "CTCF-CascadeA-LKU32": get_CTCF_LKU_config("LKU-32", use_cascade=True),
    "CTCF-LKU32-solo": get_CTCF_LKU_config("LKU-32", use_cascade=False),
    "CTCF-CascadeA-Mamba": get_CTCF_Mamba_config(use_cascade=True),
    "CTCF-Mamba-solo": get_CTCF_Mamba_config(use_cascade=False),
    "CTCF-CascadeA-VMamba": get_CTCF_VMamba_config(use_cascade=True),
    "CTCF-VMamba-solo": get_CTCF_VMamba_config(use_cascade=False),
    "CTCF-CascadeA-EFFM": get_CTCF_EFFM_config("EfficientMorph_2x3_2_hires", use_cascade=True),
    "CTCF-EFFM-solo": get_CTCF_EFFM_config("EfficientMorph_2x3_2_hires", use_cascade=False),
}
