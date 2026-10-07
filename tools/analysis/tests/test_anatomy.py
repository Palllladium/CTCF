from __future__ import annotations

import unittest

import numpy as np
import torch

from utils.anatomy import (
    LatentParcellation,
    augment_intensity,
    effective_classes,
    fit_tissue_gmm,
    information_loss,
    tissue_posteriors,
    with_background,
)
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


class LatentParcellationTest(unittest.TestCase):
    def setUp(self):
        self.image = torch.ones(1, 1, 4, 4, 8)
        self.image[..., :2] = 0.0  # a background slab

    def test_information_loss_rewards_confident_balanced_classes(self):
        k = 4
        uniform = torch.full((1, k, 4, 4, 8), 1.0 / k)
        one_class = torch.zeros(1, k, 4, 4, 8)
        one_class[:, 0] = 1.0
        balanced = torch.zeros(1, k, 4, 4, 8)
        for c in range(k):
            balanced[:, c, ..., 2 + c] = 1.0
        balanced[:, 0, ..., 6:] = 1.0  # the remaining foreground columns
        self.assertAlmostEqual(information_loss(uniform, self.image).item(), 0.0, places=5)
        self.assertAlmostEqual(information_loss(one_class, self.image).item(), 0.0, places=5)
        self.assertLess(information_loss(balanced, self.image).item(), -0.8)
        self.assertAlmostEqual(effective_classes(one_class, self.image), 1.0, places=4)

    def test_background_channel_and_shapes(self):
        net = LatentParcellation(classes=5, width=4)
        probs = torch.softmax(net(torch.rand(1, 1, 16, 16, 16)), dim=1)
        self.assertEqual(tuple(probs.shape), (1, 5, 16, 16, 16))
        classes = with_background(torch.softmax(torch.rand(1, 5, 4, 4, 8), dim=1), self.image)
        torch.testing.assert_close(classes.sum(dim=1), torch.ones(1, 4, 4, 8))
        self.assertTrue(torch.equal(classes[:, 0, ..., :2], torch.ones(1, 4, 4, 2)))

    def test_augmentation_leaves_background_untouched(self):
        image = torch.rand(1, 1, 6, 6, 6) * (torch.rand(1, 1, 6, 6, 6) > 0.4)
        augmented = augment_intensity(image, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(augmented[image == 0], image[image == 0]))
        self.assertFalse(torch.equal(augmented[image > 0], image[image > 0]))
        self.assertTrue(bool((augmented[image > 0] > 0).all()))


if __name__ == "__main__":
    unittest.main()
