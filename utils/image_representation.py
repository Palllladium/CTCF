import torch
import torch.nn.functional as F
from torch import nn


class Mind6(nn.Module):
    """Six-neighbour MIND input, exp(-D/mean(D)), with patch weights exp(-p^2/0.5^2).

    The separable five-tap kernel implements Eqs. (1)-(2) of arXiv:2510.23079.
    Replicated boundaries and a 1e-6 variance floor are explicit implementation choices.
    This is separate from the historical box-filter MIND loss.
    """

    def __init__(self):
        super().__init__()
        offsets = torch.arange(-2, 3, dtype=torch.float32)
        kernel = torch.exp(-(offsets / 0.5).square())
        self.register_buffer("kernel", kernel / kernel.sum())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("MIND input requires scalar [B,1,D,H,W] images")
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            padded = F.pad(image, (1, 1, 1, 1, 1, 1), mode="replicate")
            d, h, w = image.shape[2:]
            differences = torch.cat(
                [
                    (image - padded[:, :, 1 + z : 1 + z + d, 1 + y : 1 + y + h, 1 + x : 1 + x + w]).square()
                    for z, y, x in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
                ],
                dim=1,
            )
            for axis in range(3):
                shape = [1, 1, 1]
                shape[axis] = 5
                padding = [0] * 6
                padding[4 - 2 * axis : 6 - 2 * axis] = [2, 2]
                weight = self.kernel.view(1, 1, *shape).expand(6, 1, *shape)
                differences = F.conv3d(F.pad(differences, padding, mode="replicate"), weight, groups=6)
            return torch.exp(-differences / differences.mean(1, keepdim=True).clamp_min(1e-6))
