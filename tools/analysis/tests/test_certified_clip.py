from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from experiments.train_CTCF import certified_crop_warp
from utils.field import certified_local_clip, differentiable_local_clip, trilinear_cert_bound
from utils.spatial import SpatialTransformer


def smooth_field(size, scale, seed, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(1, 3, 3, 3, 3, generator=generator, dtype=dtype) * scale
    return F.interpolate(coarse, size=size, mode="trilinear", align_corners=True)


class DifferentiableLocalClipTest(unittest.TestCase):
    def test_matches_the_certified_wrapper_bit_for_bit(self):
        proposal = smooth_field((12, 14, 16), 3.0, seed=1)
        expected = certified_local_clip(torch.zeros_like(proposal), proposal, eps=0.001, sweeps=2)
        actual = differentiable_local_clip(proposal, eps=0.001, sweeps=2).float()
        self.assertTrue(torch.equal(expected, actual))

    def test_output_keeps_the_work_margin(self):
        proposal = smooth_field((12, 14, 16), 4.0, seed=2)
        self.assertLess(trilinear_cert_bound(proposal.float()), 0.0)
        clipped = differentiable_local_clip(proposal, eps=0.0011, sweeps=2)
        self.assertGreaterEqual(trilinear_cert_bound(clipped), 0.0011 - 1e-9)

    def test_exact_piecewise_derivative(self):
        proposal = smooth_field((4, 4, 4), 2.5, seed=3).requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda p: differentiable_local_clip(p, eps=0.001, sweeps=1),
                (proposal,),
                eps=1e-7,
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_crop_interior_equals_full_volume_clip(self):
        sweeps, size, origin = 1, 24, (2, 4, 6)
        margin = 8 * sweeps
        proposal = smooth_field((28, 30, 32), 3.0, seed=4)
        full = differentiable_local_clip(proposal, eps=0.001, sweeps=sweeps)
        crop = proposal[..., origin[0] : origin[0] + size, origin[1] : origin[1] + size, origin[2] : origin[2] + size]
        clipped = differentiable_local_clip(crop, eps=0.001, sweeps=sweeps)
        inner = (slice(None), slice(None), *(slice(margin, size - margin),) * 3)
        block = (slice(None), slice(None), *(slice(o + margin, o + size - margin) for o in origin))
        torch.testing.assert_close(clipped[inner], full[block], rtol=0, atol=1e-12)

    def test_crop_warp_samples_like_the_spatial_transformer(self):
        sweeps, size, origin = 1, 24, (2, 4, 6)
        flow = smooth_field((28, 30, 32), 3.0, seed=5, dtype=torch.float32)
        moving = torch.rand(1, 4, 28, 30, 32, generator=torch.Generator().manual_seed(6))
        warped, block = certified_crop_warp(flow, moving, sweeps, size, origin)
        full = SpatialTransformer((28, 30, 32))(moving, differentiable_local_clip(flow, eps=0.0011, sweeps=sweeps))
        torch.testing.assert_close(warped, full[(slice(None), slice(None), *block)], rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
