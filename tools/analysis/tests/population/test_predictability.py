import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.population.geometry import HaarLevel, HaarPyramid, coefficient_bands, haar_forward, haar_inverse
from experiments.population.predictability import (
    evaluate_predictability,
    local_predictability,
    main,
    replace_coarse,
    ridge_weights,
    select_ridge,
    validate_fits,
)


def test_dual_ridge_recovers_fine_relation_on_unseen_subjects():
    train = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    heldout = torch.tensor([[-0.4], [0.6]], dtype=torch.float64)
    weights = ridge_weights(train, heldout, ridge=1e-8)
    predicted = weights @ (2 * train + 0.3)
    torch.testing.assert_close(predicted, 2 * heldout + 0.3, atol=1e-8, rtol=1e-8)
    assert (predicted - (2 * train + 0.3).mean()).square().mean() > 0.5
    assert ridge_weights(train + 4, heldout + 4, 0.3).sum(dim=1).tolist() == pytest.approx([1, 1])


def test_coarse_replacement_preserves_every_predicted_detail_band():
    source = torch.randn(1, 3, 5, 6, 7, dtype=torch.float64)
    other = torch.randn_like(source)
    pyramid = haar_forward(source, 2)
    coarse = haar_forward(other, 2).coarse
    result = haar_forward(replace_coarse(source, coarse, 2), 2)
    torch.testing.assert_close(result.coarse, coarse)
    for actual, expected in zip(coefficient_bands(result)[1:], coefficient_bands(pyramid)[1:], strict=True):
        torch.testing.assert_close(actual, expected)


def test_ridge_selection_matches_independent_subject_leave_one_out_refits():
    features = torch.tensor([[-1.0, 0.2], [-0.3, 0.1], [0.4, -0.2], [1.1, 0.3]], dtype=torch.float64)
    targets = torch.stack((2 * features[:, 0] + 0.1, features[:, 0] - 0.2 * features[:, 1]), dim=1)
    candidates = [0.01, 0.3, 10.0]
    report = select_ridge(features, targets @ targets.T, candidates, targets.shape[1])
    expected = {}
    for ridge in candidates:
        errors = []
        for index in range(len(features)):
            keep = torch.arange(len(features)) != index
            prediction = ridge_weights(features[keep], features[index : index + 1], ridge) @ targets[keep]
            errors.append((prediction[0] - targets[index]).square().mean())
        expected[ridge] = float(torch.stack(errors).mean())
    for entry in report["scores"]:
        assert entry["coefficient_mse"] == pytest.approx(expected[entry["ridge"]], abs=1e-12)
    assert report["selected"] == min(candidates, key=expected.get)
    assert report["heldout_used"] is False


def test_subject_split_rejects_template_leakage_and_duplicate_subjects():
    args = {"steps": 1, "lr": 0.1, "regularization": 0.01, "collar": 1, "levels": 1, "integration_steps": 1}
    records = [{"subject": Path(str(i)), "template": Path("0"), "arguments": args} for i in range(3)]
    validate_fits(records[:2], records[2:])
    with pytest.raises(ValueError, match="distinct"):
        validate_fits(records[:2], records[:1])
    with pytest.raises(ValueError, match="inner-Train"):
        validate_fits(records[1:], records[:1])


def test_readout_from_probe_folders_does_not_create_a_negative_gate(tmp_path, monkeypatch):
    directories = []
    template = tmp_path / "subject0.pkl"
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[2:6, 2:6, 2:6] = 1
    shape = (1, 3, 6, 6, 6)
    base = haar_forward(torch.zeros(shape), levels=1)
    for i, value in enumerate((0.0, -0.1, 0.1, 0.2, 0.25)):
        subject = tmp_path / f"subject{i}.pkl"
        with subject.open("wb") as handle:
            pickle.dump((np.linspace(0, 1, labels.size, dtype=np.float32).reshape(labels.shape), labels), handle)
        directory = tmp_path / f"fit{i}"
        directory.mkdir()
        details = tuple(torch.full_like(band, 2 * value) for band in base.levels[0].details)
        interior = haar_inverse(HaarPyramid(torch.full_like(base.coarse, value), (HaarLevel((6, 6, 6), details),)))
        velocity = torch.nn.functional.pad(interior, (1,) * 6)
        np.save(directory / "velocity.npy", velocity.numpy())
        arguments = {
            "fixed": str(subject),
            "moving": str(template),
            "steps": 1,
            "lr": 0.1,
            "regularization": 0.01,
            "collar": 1,
            "levels": 1,
            "integration_steps": 2,
        }
        (directory / "report.json").write_text(
            json.dumps({"kind": "SUPERVISED_DIAGNOSTIC", "direction": "fixed_to_moving", "arguments": arguments})
        )
        directories.append(directory)
    result = evaluate_predictability(directories[:4], directories[4:], ridge=1e-8)
    case = result["cases"][0]
    assert case["readouts"]["conditional"]["fine_mse"] < 1e-10
    assert case["readouts"]["unconditional"]["fine_mse"] > 0.005
    assert case["f"] is None
    assert result["certification"] == "NOT_RUN"
    assert str(template) not in result["train_subjects"]
    assert result["reference_qc"]["included_in_regression"] is False
    assert result["reference_qc"]["velocity_rms"] == 0
    result = evaluate_predictability(directories[:4], directories[4:], ridge=1e-8, f_denominator_min=0.1)
    assert result["cases"][0]["f"] is None
    selected = evaluate_predictability(directories[:4], directories[4:], ridge_grid=[0.001, 0.1, 10])
    assert selected["ridge_selection"]["heldout_used"] is False
    output = tmp_path / "readout.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "predictability",
            "--train",
            *map(str, directories[:4]),
            "--heldout",
            str(directories[4]),
            "--ridge",
            "1e-8",
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text())["kind"] == "SUPERVISED_DIAGNOSTIC"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for arm in ("A0", "A_lin", "A1"):
            result = local_predictability(
                directories[:4],
                directories[4:],
                dict(
                    levels=1,
                    collar=1,
                    width=2,
                    scale_min=0.001,
                    scale_max=0.5,
                    prior_std=0.03,
                    posterior_std=0.03,
                    integration_steps=2,
                    arm=arm,
                ),
                tmp_path / arm,
                epochs=1,
            )
            assert result["training_subjects"] == 3
            assert np.isfinite(result["cases"][0]["fine_nll_per_coefficient"])
            assert len(result["cases"][0]["stage_mse"]) == 1
    finally:
        torch.set_num_threads(previous_threads)
