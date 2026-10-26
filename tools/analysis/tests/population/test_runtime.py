import argparse
import json
import pickle

import numpy as np
import pytest
import torch

from experiments.population.evaluate import evaluate_model, fit_subject
from experiments.population.observation import appearance_prior
from experiments.population.train import load_population_checkpoint, train_model


@pytest.fixture(autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _inputs(tmp_path):
    coordinates = np.stack(np.meshgrid(*([np.arange(9)] * 3), indexing="ij"))
    images = [
        np.exp(-((coordinates - np.array(center)[:, None, None, None]) ** 2).sum(0) / 5).astype(np.float32)
        for center in ((4, 4, 4), (4, 4, 5))
    ]
    paths = []
    for index, image in enumerate(images):
        path = tmp_path / f"subject_{index}.pkl"
        path.write_bytes(pickle.dumps((image, (image > 0.5).astype(np.int16))))
        paths.append(path)
    return paths, images


def _args(paths, output, epochs):
    return argparse.Namespace(
        subjects=paths,
        output=output,
        epochs=epochs,
        warmup_epochs=1,
        levels=1,
        collar=1,
        width=2,
        seed=8,
        lr=0.001,
        scale_min=0.001,
        scale_max=0.5,
        prior_std=[0.03],
        posterior_std=[0.03],
        conditional=1,
        integration_steps=2,
        atlas_precision=0.01,
        weight_precision=0.1,
        noise_shape=2.0,
        noise_rate=0.01,
        image_scale=1.0,
        noise_std=0.2,
        gain_mean=0.3,
        gain_std=0.2,
        offset_mean=0.1,
        offset_std=0.1,
        bias_std=0.02,
        gain_min=0.01,
        device="cpu",
        tf32=0,
        resume=False,
    )


def test_resume_matches_uninterrupted_cpu_training_and_map_keeps_globals(tmp_path):
    paths, images = _inputs(tmp_path)
    args = _args(paths, tmp_path / "resumed", 1)
    train_model(args)
    args.resume, args.epochs = True, 2
    train_model(args)
    train_model(_args(paths, tmp_path / "whole", 2))
    model, resumed = load_population_checkpoint(args.output / "last.pth", "cpu")
    _, uninterrupted = load_population_checkpoint(tmp_path / "whole/last.pth", "cpu")
    assert resumed["completed_epochs"] == 2
    assert len((args.output / "metrics.csv").read_text().splitlines()) == 3
    for name in resumed["model"]:
        torch.testing.assert_close(resumed["model"][name], uninterrupted["model"][name], rtol=0, atol=0)
    torch.testing.assert_close(resumed["log_noise_variance"], uninterrupted["log_noise_variance"], rtol=0, atol=0)
    observed = torch.from_numpy(images[0])[None, None]
    mean, covariance = appearance_prior(args, "cpu")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    _, flow, info = fit_subject(
        model,
        observed,
        resumed["log_noise_variance"].double().exp(),
        mean,
        covariance,
        args.gain_min,
        steps=3,
        lr=0.001,
    )
    assert info["best_objective_per_voxel"] < info["trace"][0]["objective_per_voxel"]
    assert torch.isfinite(flow).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_pair_evaluation_checks_saved_raw_and_repaired_bytes(tmp_path):
    paths, images = _inputs(tmp_path)
    trained = _args(paths, tmp_path / "trained", 1)
    train_model(trained)
    pair = tmp_path / "p_A_B.pkl"
    pair.write_bytes(pickle.dumps((*images, *((image > 0.5).astype(np.int16) for image in images))))
    args = argparse.Namespace(
        checkpoint=trained.output / "last.pth",
        output=tmp_path / "evaluation",
        pairs=[pair],
        subjects=None,
        map_steps=0,
        map_lr=0.001,
        repair=1,
        certify=1,
        seed=3,
        device="cpu",
        importance_samples=2,
    )
    evaluate_model(args)
    report = json.loads((args.output / "p_A_B/report.json").read_text())
    for mode in ("FF", "REP"):
        assert (args.output / mode / "per_case.csv").is_file()
        assert report[f"certificate_{mode}"]["status"] == "CERTIFIED"
        assert report[f"certificate_{mode}"]["global_homeomorphism_hypotheses_met"] is True
        assert (args.output / f"p_A_B/{mode}.npz").is_file()
    heldout = tmp_path / "heldout.pkl"
    heldout.write_bytes(pickle.dumps((images[0] * 0.9, None)))
    args.subjects, args.pairs = [heldout], None
    args.output, args.repair, args.certify = tmp_path / "predictive", 0, 0
    evaluate_model(args)
    predictive = json.loads((args.output / "importance.json").read_text())
    assert np.isfinite(predictive[0]["log_importance_estimate"]).all()
    assert 1 <= predictive[0]["effective_sample_size"][0] <= args.importance_samples
