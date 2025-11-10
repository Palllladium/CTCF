import torch
import torch.nn as nn
import TransMorph.models.TransMorph as TransMorph

class BaseRegModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = TransMorph.TransMorph(cfg)

    def forward(self, F_img, M_img):
        x_in = torch.cat((F_img, M_img), dim=1)  # [B,2,D,H,W]
        out, flow = self.net(x_in)               # out: warped M->F, flow: phi_ab
        return out, flow