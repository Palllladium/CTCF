import ml_collections


def get_CTCF_config():
    c = ml_collections.ConfigDict()
    c.if_transskip = True
    c.if_convskip = True
    c.in_chans = 1
    c.img_size = (80, 96, 112)  # HALF-res
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
    c.time_steps = 12

    # Cascade switches
    c.use_level1 = True
    c.level1_base_ch = 16
    c.use_level2 = True
    c.use_level3 = True
    c.level3_base_ch = 16

    return c


CONFIGS = {
    "CTCF-CascadeA": get_CTCF_config(),
}