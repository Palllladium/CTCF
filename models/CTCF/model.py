import torch
import torch.nn as nn
import torch.nn.functional as F

import models.CTCF.configs as configs
from models.TransMorph_DCA.model import (
    SwinTransformer, 
    Conv3dReLU, 
    RegistrationHead, 
    SpatialTransformer
)
from models.UTSRMorph.model import (
    PixelShuffle3d,
    ConvergeHead,
    SR
)


class CTCF_DCA_SR(nn.Module):
    """
    CTCF = TM-DCA Cascade (flow integration preserved) + UTSRMorph SR-style decoder blocks.
    Expected tensor layout everywhere: [B, C, D, H, W]
    config.img_size is (D, H, W) of the resolution at which the model operates.
    """

    def __init__(self, config, time_steps: int = 7):
        super().__init__()

        self.if_convskip = bool(getattr(config, "if_convskip", True))
        self.if_transskip = bool(getattr(config, "if_transskip", True))
        embed_dim = config.embed_dim
        self.time_steps = time_steps
        self.img_size = tuple(config.img_size)  # (D, H, W)
        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
            img_size=config.img_size,
            dwin_size=config.dwin_kernel_size,
        )

        # SR-style decoder backbone (replaces TM DecoderBlock up0/up1/up2)
        # These SR blocks must internally upsample by x2.
        self.up0 = SR(
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            skip_channels=(embed_dim * 2 if self.if_transskip else 0),
            use_batchnorm=False,
        )
        self.up1 = SR(
            in_channels=embed_dim * 2,
            out_channels=embed_dim,
            skip_channels=(embed_dim if self.if_transskip else 0),
            use_batchnorm=False,
        )
        self.up2 = SR(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            skip_channels=(embed_dim // 2 if self.if_transskip else 0),
            use_batchnorm=False,
        )

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, embed_dim // 2, kernel_size=3, stride=1, use_batchnorm=False)
        self.cs = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for _ in range(self.time_steps):
            self.cs.append(
                Conv3dReLU(2, embed_dim // 2, kernel_size=3, stride=1, use_batchnorm=False)
            )

            # SR block does the final x2 upsample before reg_head (replaces TM up3s[t] DecoderBlock)
            self.up3s.append(
                SR(
                    in_channels=embed_dim // 2,
                    out_channels=config.reg_head_chan,
                    skip_channels=(embed_dim // 2),
                    use_batchnorm=False,
                )
            )

            self.reg_heads.append(
                RegistrationHead(
                    in_channels=config.reg_head_chan,
                    out_channels=3,
                    kernel_size=3,
                )
            )
        self.spatial_trans = SpatialTransformer(self.img_size)

    def forward(self, inputs, return_all_flows: bool = False):
        mov, fix = inputs
        assert mov.dim() == 5 and fix.dim() == 5, "Expected [B,C,D,H,W]"
        assert tuple(mov.shape[2:]) == self.img_size, f"mov shape {tuple(mov.shape[2:])} != img_size {self.img_size}"
        assert tuple(fix.shape[2:]) == self.img_size, f"fix shape {tuple(fix.shape[2:])} != img_size {self.img_size}"

        x_cat = torch.cat((mov, fix), dim=1)
        x_s1 = self.avg_pool(x_cat)
        f3 = self.c1(x_s1) if self.if_convskip else None
        out_feats = self.transformer((mov, fix))
        if self.if_transskip:
            mov_f1, fix_f1 = out_feats[-2]
            f1 = (mov_f1 + fix_f1)
            mov_f2, fix_f2 = out_feats[-3]
            f2 = (mov_f2 + fix_f2)
        else:
            f1 = None
            f2 = None

        mov_f0, fix_f0 = out_feats[-1]
        f0 = mov_f0 + fix_f0
        x = self.up0(f0, f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)
        def_x = mov.clone()
        flow_prev = torch.zeros((mov.shape[0], 3, *self.img_size), 
                                device=mov.device, 
                                dtype=mov.dtype)
        flows = []

        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix), dim=1))
            x_t = self.up3s[t](xx, f_out)
            flow_step = self.reg_heads[t](x_t)
            flows.append(flow_step)
            flow_new = flow_prev + self.spatial_trans(flow_step, flow_prev)
            def_x = self.spatial_trans(mov, flow_new)
            flow_prev = flow_new

        if return_all_flows:
            return def_x, flow_prev, flows
        return def_x, flow_prev


CONFIGS = {
    'CTCF-DCA-SR': configs.get_CTCF_config(),
    'CTCF-DCA-SR-Debug': configs.get_CTCF_debug_config(),
}