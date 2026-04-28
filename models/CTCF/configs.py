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

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = False
    c.l3_full_res = False
    c.l3_svf = False

    return c


def get_CTCF_VM_config(*, use_cascade=True):
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

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = False
    c.l3_full_res = False
    c.l3_svf = False

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

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = True

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

    c.l3_iters = 1
    c.l3_unshared = False
    c.l1_half_res = False
    c.l2_full_res = True
    c.l3_full_res = False
    c.l3_svf = True

    return c


CONFIGS = {
    "CTCF-CascadeA": get_CTCF_config(),
    "CTCF-CascadeA-VM": get_CTCF_VM_config(use_cascade=True),
    "CTCF-VM-solo": get_CTCF_VM_config(use_cascade=False),
    "CTCF-CascadeA-LKU8": get_CTCF_LKU_config("LKU-8", use_cascade=True),
    "CTCF-LKU8-solo": get_CTCF_LKU_config("LKU-8", use_cascade=False),
    "CTCF-CascadeA-LKU32": get_CTCF_LKU_config("LKU-32", use_cascade=True),
    "CTCF-LKU32-solo": get_CTCF_LKU_config("LKU-32", use_cascade=False),
    "CTCF-CascadeA-Mamba": get_CTCF_Mamba_config(use_cascade=True),
    "CTCF-Mamba-solo": get_CTCF_Mamba_config(use_cascade=False),
    "CTCF-CascadeA-VMamba": get_CTCF_VMamba_config(use_cascade=True),
    "CTCF-VMamba-solo": get_CTCF_VMamba_config(use_cascade=False),
}
