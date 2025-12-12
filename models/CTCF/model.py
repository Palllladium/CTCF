import torch
import torch.nn as nn
import torch.nn.functional as F

import models.CTCF.configs as configs
from TransMorph_DCA.model import (
    SwinTransformer, 
    Conv3dReLU, 
    RegistrationHead, 
    SpatialTransformer
)


class PixelShuffle3d(nn.Module):
    """
    3D PixelShuffle used in UTSRMorph SR block.

    Rearranges channels to upsample spatially.
    Here we allow scale_factor=1 for "no-op" upsampling but keep the same API.
    """
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upscale_factor == 1:
            return x
        # x: (B, C*r^3, D, H, W)
        r = self.upscale_factor
        b, c, d, h, w = x.size()
        assert c % (r ** 3) == 0, "Channels must be divisible by upscale_factor^3."
        c //= r ** 3
        x = x.view(b, c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, c, d * r, h * r, w * r)
        return x


class ConvergeHead(nn.Module):
    """
    ConvergeHead from UTSRMorph: a small conv + PixelShuffle block
    that combines feature channels and (optionally) upsamples.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        scale_factor: int = 2,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        total_in = in_channels
        if skip_channels > 0:
            total_in += skip_channels

        layers = [nn.Conv3d(total_in, out_channels * (scale_factor ** 3), kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels * (scale_factor ** 3)))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.shuffle = PixelShuffle3d(scale_factor)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.shuffle(x)
        return x


class SR(nn.Module):
    """
    Super-resolution style decoder block from UTSRMorph.

    It wraps ConvergeHead and adds residual refinement.
    In this CTCF-DCA-SR variant we typically use scale_factor=1
    to keep the spatial resolution consistent with TM-DCA cascade,
    while still benefiting from the SR-style convergence block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_batchnorm: bool = False,
        scale_factor: int = 1,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = ConvergeHead(
            in_channels,
            out_channels,
            skip_channels=0,
            scale_factor=scale_factor,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: feature map from previous decoder stage
        skip: skip connection feature (can be None)
        """
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x, None)
        return x


class CTCF_DCA_SR_Cascade(nn.Module):
    """
    CTCF model built on top of TM-DCA TransMorphCascadeAd-style backbone,
    but using SR-based decoder blocks from UTSRMorph.

    Key ideas:
    - Encoder: SwinTransformer with deformable cross-attention (DCA),
      exactly as in TransMorph_DCA.
    - Decoder:
        * up0, up1: SR blocks aggregating high-level transformer features.
        * up2s[t]: SR blocks inside the cascade head for each time step.
    - Cascade:
        * Multi-step flow integration as in TransMorphCascadeAd.
        * Defining def_x by warping pooled moving image at each step.
    - Output:
        * Returns warped moving image and final flow at full resolution,
          for compatibility with TM/CTCF training code.
    """

    def __init__(self, config, time_steps: int = 7):
        super().__init__()
        self.time_steps = time_steps

        # Store some keys from config
        if_transskip = config.if_transskip
        if_convskip = config.if_convskip
        self.if_transskip = if_transskip
        self.if_convskip = if_convskip

        embed_dim = config.embed_dim

        # DCA Swin backbone (same as TM-DCA)
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
        )

        # SR-style decoder backbone (analogous to TransMorphCascadeAd up0/up1,
        # but using SR blocks instead of plain DecoderBlock).
        self.up0 = SR(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim if if_transskip else 0,
            use_batchnorm=False,
            scale_factor=1,  # keep resolution consistent with DCA cascade
        )
        self.up1 = SR(
            embed_dim,
            embed_dim // 2,
            skip_channels=embed_dim // 2 if if_transskip else 0,
            use_batchnorm=False,
            scale_factor=1,
        )

        # Convolutional skip from pooled concatenated (mov, fix)
        # same dimensionality as in TransMorph_DCA (embed_dim // 2)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        if if_convskip:
            self.c1 = Conv3dReLU(
                2,
                embed_dim // 2,
                kernel_size=3,
                stride=1,
                use_batchnorm=False,
            )
        else:
            self.c1 = None

        # Cascade-specific modules:
        #   - cs[t]: small conv that compresses concatenated (def_x, pooled_fix)
        #   - up2s[t]: SR decoder for each cascade step
        #   - reg_heads[t]: registration heads per step
        self.cs = nn.ModuleList()
        self.up2s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for _ in range(time_steps):
            self.cs.append(
                Conv3dReLU(
                    2,
                    embed_dim // 2,
                    kernel_size=3,
                    stride=1,
                    use_batchnorm=False,
                )
            )
            self.up2s.append(
                SR(
                    in_channels=embed_dim // 2,
                    out_channels=config.reg_head_chan,
                    skip_channels=embed_dim // 2 if if_convskip else 0,
                    use_batchnorm=False,
                    scale_factor=1,  # keep pooled resolution
                )
            )
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=config.reg_head_chan,
                    out_channels=3,
                    kernel_size=3,
                )
            )

        # Spatial transformers: one at pooled resolution (cascade),
        # one at full resolution (final output).
        self.spatial_trans_down = SpatialTransformer(
            [s // 2 for s in config.img_size]
        )
        self.spatial_trans_full = SpatialTransformer(config.img_size)

        # Upsampling of flow from pooled resolution to full resolution
        self.flow_upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )

    def forward(self, inputs):
        """
        inputs: tensor of shape (B, 2, D, H, W)
                inputs[:, 0:1] = moving image
                inputs[:, 1:2] = fixed image

        Returns:
            def_mov_full: warped moving image at full resolution
            flow_full: final integrated flow (full resolution)
        """
        mov = inputs[:, 0:1, ...]
        fix = inputs[:, 1:2, ...]

        # Pooled versions for cascade
        source_d = self.avg_pool(mov)
        x_cat = torch.cat((mov, fix), dim=1)
        x_s1 = self.avg_pool(x_cat)

        # Transformer backbone (DCA Swin)
        # For DCA Swin, moving and fixed are passed as a tuple.
        out_feats = self.transformer((mov, fix))

        if self.if_transskip:
            mov_f1, fix_f1 = out_feats[-2]
            f1 = mov_f1 + fix_f1

            mov_f2, fix_f2 = out_feats[-3]
            f2 = mov_f2 + fix_f2
        else:
            f1 = None
            f2 = None

        mov_f0, fix_f0 = out_feats[-1]
        f0 = mov_f0 + fix_f0

        # SR-based decoder backbone
        x = self.up0(f0, f1)
        xx = self.up1(x, f2)

        # Convolutional pooled skip from (mov, fix)
        if self.if_convskip and self.c1 is not None:
            f4 = self.c1(x_s1)
        else:
            f4 = None

        # Initial warped moving image at pooled resolution
        def_x = x_s1[:, 0:1, ...]
        flow_previous = torch.zeros(
            (
                source_d.shape[0],
                3,
                source_d.shape[2],
                source_d.shape[3],
                source_d.shape[4],
            ),
            device=source_d.device,
            dtype=source_d.dtype,
        )

        # Cascade integration
        for t in range(self.time_steps):
            # Combine current warped moving (def_x) with pooled fixed
            f_out = self.cs[t](
                torch.cat((def_x, x_s1[:, 1:2, ...]), dim=1)
            )

            # SR-style cascade decoder step: (xx, conv skip)
            x_dec = self.up2s[t](xx, f4)
            flow_step = self.reg_heads[t](x_dec)

            # Velocity-like integration at pooled resolution
            flow_new = flow_previous + self.spatial_trans_down(
                flow_step, flow_previous
            )
            def_x = self.spatial_trans_down(source_d, flow_new)
            flow_previous = flow_new

        flow_pooled = flow_previous

        # Upsample flow to full resolution and warp moving image
        flow_full = self.flow_upsample(flow_pooled) * 2.0
        def_mov_full = self.spatial_trans_full(mov, flow_full)

        return def_mov_full, flow_full


CONFIGS = {
    'CTCF-DCA-SR': configs.get_CTCF_DCA_SR_config(),
    'CTCF-DCA-SR-Debug': configs.get_CTCF_DCA_SR_debug_config(),
}