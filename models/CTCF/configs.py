import ml_collections


def get_CTCF_DCA_SR_config():
    config = ml_collections.ConfigDict()

    config.if_transskip = True
    config.if_convskip = True

    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96

    config.dwin_kernel_size = (7, 5, 3)
    config.depths = (4, 4, 5)
    config.num_heads = (8, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False

    config.drop_rate = 0.0
    config.drop_path_rate = 0.3

    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)

    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)

    # === SR decoder toggles (used by your CTCF_DCA_SR_Cascade model) ===
    # Note: In your SR blocks you currently used scale_factor=1 everywhere.
    # Keep these in config so you can change behavior without editing model code.
    config.sr_enable = True
    config.sr_scale_factor = 1       # keep 1 to preserve TM-DCA spatial schedule
    config.sr_use_batchnorm = False  # match your SR implementation defaults

    return config


def get_CTCF_DCA_SR_debug_config():
    config = ml_collections.ConfigDict()

    config.if_transskip = True
    config.if_convskip = True

    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 48  # lighter

    # Keep 3-level structure, but shallow
    config.dwin_kernel_size = (7, 5, 3)
    config.depths = (1, 1, 1)
    config.num_heads = (4, 4, 4)

    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False

    config.drop_rate = 0.0
    config.drop_path_rate = 0.1

    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)

    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)

    config.sr_enable = True
    config.sr_scale_factor = 1
    config.sr_use_batchnorm = False

    return config