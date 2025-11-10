
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention3D(nn.Module):
    """
    Lightweight cross-attention between feature maps of F and M at coarse scale.
    Inputs: feat_F, feat_M: [B,C,D,H,W]
    Output: fused_F, fused_M (refined features), both [B,C,D,H,W]
    """
    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj_F = nn.Linear(channels, channels)
        self.proj_M = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, feat_F, feat_M):
        B, C, D, H, W = feat_F.shape
        N = D*H*W
        fF = feat_F.view(B, C, N).transpose(1,2)  # [B,N,C]
        fM = feat_M.view(B, C, N).transpose(1,2)  # [B,N,C]

        qF = self.to_q(fF)
        kM = self.to_k(fM)
        vM = self.to_v(fM)

        qF = qF.view(B, N, self.heads, C//self.heads).transpose(1,2)  # [B,h,N,C/h]
        kM = kM.view(B, N, self.heads, C//self.heads).transpose(1,2)
        vM = vM.view(B, N, self.heads, C//self.heads).transpose(1,2)

        attn = (qF @ kM.transpose(-2,-1)) * self.scale  # [B,h,N,N]
        attn = attn.softmax(dim=-1)
        outF = attn @ vM  # [B,h,N,C/h]
        outF = outF.transpose(1,2).contiguous().view(B, N, C)
        outF = self.proj_F(outF).view(B, D, H, W, C).permute(0,4,1,2,3)  # [B,C,D,H,W]

        # symmetric path (M attends to F)
        qM = self.to_q(fM)
        kF = self.to_k(fF)
        vF = self.to_v(fF)
        qM = qM.view(B, N, self.heads, C//self.heads).transpose(1,2)
        kF = kF.view(B, N, self.heads, C//self.heads).transpose(1,2)
        vF = vF.view(B, N, self.heads, C//self.heads).transpose(1,2)
        attn2 = (qM @ kF.transpose(-2,-1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        outM = attn2 @ vF
        outM = outM.transpose(1,2).contiguous().view(B, N, C)
        outM = self.proj_M(outM).view(B, D, H, W, C).permute(0,4,1,2,3)

        # residual
        outF = outF + feat_F
        outM = outM + feat_M
        return outF, outM

class PlaneAttention3D(nn.Module):
    """
    Parameter-efficient plane-wise attention (XY, XZ, YZ) at the finest scale.
    Input: x [B,C,D,H,W] -> output [B,C,D,H,W]
    """
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Conv1d(channels, channels*3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

    def _attn1d(self, x):
        # x: [B,C,N]; apply MHSA along token dim N
        B, C, N = x.shape
        qkv = self.qkv(x)  # [B,3C,N]
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.heads, C//self.heads, N)
        k = k.view(B, self.heads, C//self.heads, N)
        v = v.view(B, self.heads, C//self.heads, N)
        attn = (q.transpose(-2,-1) @ k) * self.scale  # [B,h,N,N]
        attn = attn.softmax(dim=-1)
        out = attn @ v.transpose(-2,-1)               # [B,h,N,C/h]
        out = out.transpose(-2,-1).contiguous().view(B, C, N)
        return self.proj(out)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # XY planes: N = D*H*W, but we split by planes for efficiency
        xy = x.permute(0,1,3,4,2).contiguous().view(B, C, H*W, D)  # (B,C,HW,D)
        xy = self._attn1d(xy.view(B,C,-1)).view(B,C,H*W,D)
        xy = xy.view(B,C,H,W,D).permute(0,1,4,2,3)

        xz = x.permute(0,1,2,4,3).contiguous().view(B, C, D*W, H)
        xz = self._attn1d(xz.view(B,C,-1)).view(B,C,D*W,H)
        xz = xz.view(B,C,D,W,H).permute(0,1,2,4,3)

        yz = x.permute(0,1,2,3,4).contiguous().view(B, C, D*H, W)
        yz = self._attn1d(yz.view(B,C,-1)).view(B,C,D*H,W)
        yz = yz.view(B,C,D,H,W)

        out = (xy + xz + yz) / 3.0 + x  # residual
        return out
