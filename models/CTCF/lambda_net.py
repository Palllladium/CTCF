import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class LambdaNet3D(nn.Module):
    """
    Predict spatially-varying lambda(x) on HALF-res grid.
    Input:  mov_half, fix_half  (B,1,D,H,W each) -> concat -> (B,2,D,H,W)
    Output: lambda_half (B,1,D,H,W) in [lambda_min, lambda_max]
    """
    def __init__(self, base_ch=16, lambda_min=0.3, lambda_max=1.0):
        super().__init__()
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)

        self.enc1 = ConvBlock(2, base_ch)
        self.pool1 = nn.AvgPool3d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.AvgPool3d(2)

        self.bot = ConvBlock(base_ch * 2, base_ch * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)

        self.out = nn.Conv3d(base_ch, 1, kernel_size=1, bias=True)

    def forward(self, mov, fix):
        x = torch.cat([mov, fix], dim=1)  # (B,2,D,H,W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bot(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        raw = self.out(d1)
        sig = torch.sigmoid(raw)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * sig
        return lam