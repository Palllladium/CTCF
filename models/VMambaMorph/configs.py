import ml_collections


def get_3DVMambaMorph_config():
    cfg = ml_collections.ConfigDict()
    cfg.if_transskip = True
    cfg.if_convskip = True
    cfg.patch_size = 4
    cfg.in_chans = 2
    cfg.embed_dim = 96
    cfg.depths = (2, 2, 2)
    cfg.drop_rate = 0.0
    cfg.drop_path_rate = 0.3
    cfg.ape = False
    cfg.spe = True
    cfg.rpe = False
    cfg.patch_norm = True
    cfg.use_checkpoint = False
    cfg.out_indices = (0, 1, 2)
    cfg.reg_head_chan = 16
    cfg.img_size = (160, 192, 224)
    cfg.d_state = 16
    cfg.d_conv = 4
    cfg.expand = 2
    return cfg


CONFIGS = {
    "VMambaMorph": get_3DVMambaMorph_config(),
}
