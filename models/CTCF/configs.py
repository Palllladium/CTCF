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

    # Cascade switches
    c.use_level1 = True
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = True
    c.level3_base_ch = 64
    c.level3_error_mode = "ncc"
    c.prealign_encoder = False

    # GEN2 enhancements (architectural)
    c.l3_iters = 1              # Iterative L3: number of refinement passes (1 = default)
    c.l3_full_res = False       # Run L3 at full-res (160x192x224) instead of half-res
    c.learned_upsample = False  # Learned flow upsampling instead of trilinear
    c.l2_l3_skip = False        # Pass L2 decoder features to L3 as skip connection
    c.l1_half_res = False       # Run L1 at half-res instead of quarter-res
    c.l1_l2_skip = False        # Pass L1 encoder features to L2 conv-skip path

    # GEN2.5 enhancements (capacity)
    c.l3_cab = False            # Channel attention (CAB) in L3 decoder
    c.l3_context_blocks = 0     # ResidualContext3D blocks in L3 bottleneck
    c.l3_gate = False           # RefineGate3D spatial gating on L3 delta
    c.l3_unshared = False       # Separate L3 weights per iteration (requires l3_iters>1)
    c.l1_cab = False            # Channel attention (CAB) in L1 decoder

    return c


def get_CTCF_VM_config(*, use_cascade=True):
    c = ml_collections.ConfigDict()
    c.backbone = "vxm"
    c.img_size = (160, 192, 224)
    c.time_steps = 0  # unused by VxmDenseHalf (kept for adapter compatibility)

    # VoxelMorph L2 params (standard VxmDense-2 diffeomorphic)
    c.vxm = ml_collections.ConfigDict()
    c.vxm.enc_nf = [16, 32, 32, 32]
    c.vxm.dec_nf = [32, 32, 32, 32, 32, 16, 16]
    c.vxm.int_steps = 7

    # Cascade switches
    c.use_level1 = use_cascade
    c.level1_base_ch = 32
    c.use_level2 = True
    c.use_level3 = use_cascade
    c.level3_base_ch = 64
    c.level3_error_mode = "ncc"
    c.prealign_encoder = False

    # GEN2 enhancements (architectural)
    c.l3_iters = 1
    c.l3_full_res = False
    c.learned_upsample = False
    c.l2_l3_skip = False
    c.l1_half_res = False
    c.l1_l2_skip = False

    # GEN2.5 enhancements (capacity)
    c.l3_cab = False
    c.l3_context_blocks = 0
    c.l3_gate = False
    c.l3_unshared = False
    c.l1_cab = False

    return c


CONFIGS = {
    "CTCF-CascadeA": get_CTCF_config(),
    "CTCF-CascadeA-VM": get_CTCF_VM_config(use_cascade=True),
    "CTCF-VM-solo": get_CTCF_VM_config(use_cascade=False),
}