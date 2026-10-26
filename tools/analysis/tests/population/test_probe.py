import torch

from experiments.population.probe import fit_label_velocity
from utils.field import _warp


def test_label_fit_reduces_translation_error_in_the_pull_direction():
    fixed = torch.zeros(1, 1, 13, 13, 13, dtype=torch.long)
    moving = torch.zeros_like(fixed)
    fixed[:, :, 4:9, 4:9, 4:8] = 1
    moving[:, :, 4:9, 4:9, 5:9] = 1
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        fit = fit_label_velocity(
            moving,
            fixed,
            steps=20,
            lr=0.15,
            regularization=0.01,
            collar=2,
            integration_steps=3,
            log_every=20,
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert fit.history[-1]["label_ssd"] < 0.1 * fit.history[0]["label_ssd"]
    assert fit.flow[0, 2, 4:9, 4:9, 4:8].mean() > 0.5
    warped = _warp(moving.float(), fit.flow, mode="nearest").long()
    overlap = 2 * ((warped == 1) & (fixed == 1)).sum() / ((warped == 1).sum() + (fixed == 1).sum())
    assert overlap > 0.95


def test_label_smoothing_captures_a_thin_structure_beyond_its_width():
    fixed = torch.zeros(1, 1, 20, 20, 20, dtype=torch.long)
    moving = torch.zeros_like(fixed)
    fixed[:, :, 6:14, 6:14, 6:8] = 1
    moving[:, :, 6:14, 6:14, 10:12] = 1
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        fit = fit_label_velocity(
            moving,
            fixed,
            steps=40,
            lr=0.15,
            regularization=0.01,
            collar=2,
            integration_steps=3,
            log_every=40,
            smoothing_sigmas=(2.0, 1.0, 0.0),
        )
    finally:
        torch.set_num_threads(previous_threads)
    warped = _warp(moving.float(), fit.flow, mode="nearest").long()
    overlap = 2 * ((warped == 1) & (fixed == 1)).sum() / ((warped == 1).sum() + (fixed == 1).sum())
    assert overlap > 0.90
    assert fit.flow[0, 2, 6:14, 6:14, 6:8].mean() > 3
