import ml_collections


def get_3DTransMorphDWin3Lvl_config():
    """
    Trainable params: 15,201,579
    """
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
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (80, 96, 112)  # half-res
    return config


CONFIGS = {
    "TransMorph-3-LVL": get_3DTransMorphDWin3Lvl_config(),
}
