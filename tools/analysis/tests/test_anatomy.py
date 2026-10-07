from __future__ import annotations

import unittest

import numpy as np
import torch

from utils.anatomy import fit_tissue_gmm, tissue_posteriors
from utils.dice import soft_dice_loss


class TissueGMMTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        foreground = np.concatenate(
            [rng.normal(0.05, 0.01, 30_000), rng.normal(0.22, 0.02, 50_000), rng.normal(0.40, 0.015, 40_000)]
        ).clip(1e-4, None)
        volume = np.zeros(200_000)
        volume[: foreground.size] = foreground
        self.volume = volume
        self.gmm = fit_tissue_gmm([volume], samples_per_volume=foreground.size, iters=200)

    def test_recovers_sorted_components_from_foreground_only(self):
        np.testing.assert_allclose(self.gmm.means, [0.05, 0.22, 0.40], atol=0.005)
        np.testing.assert_allclose(np.sqrt(self.gmm.variances), [0.01, 0.02, 0.015], atol=0.003)
        np.testing.assert_allclose(self.gmm.weights, [0.25, 5 / 12, 1 / 3], atol=0.01)

    def test_posteriors_split_background_and_sum_to_one(self):
        image = torch.tensor([0.0, 0.05, 0.22, 0.40], dtype=torch.float32).view(1, 1, 1, 1, 4)
        post = tissue_posteriors(image, self.gmm)
        self.assertEqual(tuple(post.shape), (1, 4, 1, 1, 4))
        torch.testing.assert_close(post.sum(dim=1), torch.ones(1, 1, 1, 4))
        self.assertEqual(post[0, 0, 0, 0, 0].item(), 1.0)
        self.assertEqual(post[0, 0, 0, 0, 1:].abs().max().item(), 0.0)
        np.testing.assert_array_equal(post[0, 1:, 0, 0, 1:].argmax(dim=0).numpy(), [0, 1, 2])

    def test_aligned_images_score_better_than_shifted(self):
        # Soft targets give soft Dice a non-zero floor (sum p^2 / sum p < 1), so only the ordering
        # of aligned versus misaligned pairs is meaningful.
        generator = torch.Generator().manual_seed(0)
        image = 0.45 * torch.rand(1, 1, 12, 12, 12, generator=generator)
        post = tissue_posteriors(image, self.gmm)
        shifted = torch.roll(post, shifts=3, dims=4)
        self.assertLess(soft_dice_loss(post, post).item(), soft_dice_loss(shifted, post).item())


if __name__ == "__main__":
    unittest.main()
