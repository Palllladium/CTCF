from __future__ import annotations

import json
from argparse import Namespace

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from datasets.LUMIR import (
    LUMIRTrainDataset,
    LUMIRValDataset,
    labelled_t1_pairs,
    load_lumir_volume,
    lumir_disp_name,
    read_lumir_json,
)
from experiments.core import data_loaders
from experiments.core.path_profiles import resolve_lumir_paths
from experiments.export_lumir import evaluator_displacement, save_displacement
from utils.spatial import SpatialTransformer

SHAPE = (12, 16, 14)
VAL = "./imagesVal/LUMIRMRI_{}.nii.gz"


def _write(path, data, affine=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine), path)


def _write_json(path, training, pairs) -> None:
    payload = {
        "numTraining": len(training),
        "training": [{"image": image} for image in training],
        "numValidation": len(pairs),
        "validation": [{"fixed": fixed, "moving": moving} for fixed, moving in pairs],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _smooth_flow(amplitude: float) -> torch.Tensor:
    rng = np.random.default_rng(3)
    channels = [gaussian_filter(rng.normal(size=SHAPE), sigma=3.0) for _ in range(3)]
    flow = np.stack([c / np.abs(c).max() for c in channels])[None] * amplitude
    return torch.from_numpy(flow.astype(np.float32))


def test_disp_name_follows_the_evaluator_rule():
    assert lumir_disp_name(VAL.format("3455_0000"), VAL.format("3454_0000")) == "disp_3455_3454"
    assert lumir_disp_name(VAL.format("4024_0000"), VAL.format("4033_0001")) == "disp_4024_4033"
    assert lumir_disp_name(VAL.format("4033_0001"), VAL.format("4024_0000")) == "disp_4033_0001_4024_0000"
    with pytest.raises(ValueError):
        lumir_disp_name("./imagesVal/scan_3455.nii.gz", VAL.format("3454_0000"))


def test_json_counts_and_disjoint_splits_are_enforced(tmp_path):
    training = ["./imagesTr/LUMIRMRI_0000_0000.nii.gz", "./imagesTr/LUMIRMRI_0001_0000.nii.gz"]
    pairs = [(VAL.format("3455_0000"), VAL.format("3454_0000"))]
    _write_json(tmp_path / "ok.json", training, pairs)
    assert read_lumir_json(tmp_path / "ok.json") == (training, pairs)
    _write_json(tmp_path / "overlap.json", training, [(training[0], VAL.format("3454_0000"))])
    with pytest.raises(ValueError):
        read_lumir_json(tmp_path / "overlap.json")
    _write_json(
        tmp_path / "contrast_overlap.json", training, [(training[0].replace("0000.nii", "0001.nii"), pairs[0][0])]
    )
    with pytest.raises(ValueError, match="subjects"):
        read_lumir_json(tmp_path / "contrast_overlap.json")
    payload = json.loads((tmp_path / "ok.json").read_text(encoding="utf-8"))
    payload["numTraining"] = 3
    (tmp_path / "count.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_lumir_json(tmp_path / "count.json")


def test_volume_keeps_the_stored_axis_order(tmp_path):
    data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    _write(tmp_path / "LUMIRMRI_0000_0000.nii.gz", data, np.diag([-1.0, -1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(load_lumir_volume(tmp_path / "LUMIRMRI_0000_0000.nii.gz", SHAPE), data)
    with pytest.raises(ValueError):
        load_lumir_volume(tmp_path / "LUMIRMRI_0000_0000.nii.gz", (14, 16, 12))


def test_train_pairs_stay_in_the_training_list_and_scale_to_unit_range(tmp_path):
    training = [f"./imagesTr/LUMIRMRI_000{i}_0000.nii.gz" for i in range(3)]
    for i, image in enumerate(training):
        _write(tmp_path / image, np.full(SHAPE, 51.0 * (i + 1)))
    dataset = LUMIRTrainDataset(tmp_path, training, SHAPE)
    for index in range(len(training)):
        x, y = dataset[index]
        assert x.shape == y.shape == (1, *SHAPE) and x.dtype == torch.float32
        assert float(x[0, 0, 0, 0]) == pytest.approx(0.2 * (index + 1))
        others = [0.2 * (j + 1) for j in range(len(training)) if j != index]
        assert min(abs(float(y[0, 0, 0, 0]) - value) for value in others) < 1e-6


def test_validation_keeps_labelled_t1_pairs_as_moving_then_fixed(tmp_path):
    pairs = [
        (VAL.format("3455_0000"), VAL.format("3454_0000")),
        (VAL.format("4024_0000"), VAL.format("4033_0001")),
        (VAL.format("3485_0000"), VAL.format("3484_0000")),
    ]
    for value, name in enumerate(("3454_0000", "3455_0000", "4024_0000", "4033_0001", "3484_0000", "3485_0000")):
        _write(tmp_path / VAL.format(name), np.full(SHAPE, 25.5 * (value + 1)))
    for value, name in enumerate(("3454_0000", "3455_0000", "4024_0000", "4033_0001")):
        _write(tmp_path / "simplelabelsVal" / f"LUMIRMRI_{name}.nii.gz", np.full(SHAPE, value + 1))
    kept = labelled_t1_pairs(pairs, tmp_path / "simplelabelsVal")
    assert kept == pairs[:1]
    x, y, x_seg, y_seg = LUMIRValDataset(tmp_path, kept, tmp_path / "simplelabelsVal", SHAPE)[0]
    assert float(x.max()) == pytest.approx(0.1) and float(y.max()) == pytest.approx(0.2)
    assert int(x_seg.max()) == 1 and int(y_seg.max()) == 2 and x_seg.dtype == torch.int16
    with pytest.raises(FileNotFoundError):
        LUMIRValDataset(tmp_path, pairs[2:], tmp_path / "simplelabelsVal", SHAPE)


def test_loader_factory_wires_the_official_lists(tmp_path, monkeypatch):
    training = [f"./imagesTr/LUMIRMRI_000{i}_0000.nii.gz" for i in range(2)]
    pairs = [(VAL.format("3455_0000"), VAL.format("3454_0000")), (VAL.format("3485_0000"), VAL.format("3484_0000"))]
    _write_json(tmp_path / "train.json", training, [])
    _write_json(tmp_path / "val.json", [], pairs)
    for name in ("3454_0000", "3455_0000"):
        _write(tmp_path / "labels" / f"LUMIRMRI_{name}.nii.gz", np.ones(SHAPE))
    paths = {
        "train_root": tmp_path,
        "train_json": tmp_path / "train.json",
        "val_root": tmp_path,
        "val_json": tmp_path / "val.json",
        "label_dir": tmp_path / "labels",
    }
    monkeypatch.setattr(data_loaders, "resolve_lumir_paths", lambda args: paths)
    train_loader, val_loader = data_loaders.lumir_loaders(Namespace(batch_size=1, num_workers=0))
    assert train_loader.dataset.paths == training
    assert val_loader.dataset.pairs == pairs[:1]


def test_exported_displacement_reproduces_the_training_sampler():
    rng = np.random.default_rng(5)
    image = gaussian_filter(rng.normal(size=SHAPE), sigma=1.5).astype(np.float32)
    labels = (image > 0).astype(np.float32) + (image > 0.1)
    flow = _smooth_flow(amplitude=1.7)
    displacement = evaluator_displacement(flow)
    identity = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij"))
    coords = identity + displacement.transpose(3, 0, 1, 2)
    inner = (slice(2, -2),) * 3
    ours = SpatialTransformer(SHAPE)(torch.from_numpy(image)[None, None], flow)[0, 0].numpy()
    official = map_coordinates(image, coords, order=1, mode="constant")
    np.testing.assert_allclose(official[inner], ours[inner], atol=1e-5)
    ours_labels = SpatialTransformer(SHAPE, mode="nearest")(torch.from_numpy(labels)[None, None], flow)[0, 0].numpy()
    official_labels = map_coordinates(labels, coords, order=0, mode="constant")
    np.testing.assert_array_equal(official_labels[inner], ours_labels[inner])


def test_zero_network_flow_exports_the_half_voxel_stretch():
    displacement = evaluator_displacement(torch.zeros(1, 3, *SHAPE))
    for axis, n in enumerate(SHAPE):
        profile = displacement[..., axis].mean(axis=tuple(a for a in range(3) if a != axis))
        np.testing.assert_allclose(profile, np.arange(n) / (n - 1) - 0.5, atol=1e-5)


def test_export_rejects_nonfinite_or_batched_fields():
    with pytest.raises(ValueError, match="finite"):
        evaluator_displacement(torch.full((1, 3, *SHAPE), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        evaluator_displacement(torch.zeros(2, 3, *SHAPE))


def test_lumir24_export_paths_do_not_require_lumir25_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.core.path_profiles.get_dataset_paths",
        lambda *args: {
            "root": str(tmp_path),
            "train_json": "train.json",
            "val_root": "LUMIR25",
            "val_json": "val.json",
            "label_dir": "labels",
        },
    )
    (tmp_path / "train.json").write_text("{}", encoding="utf-8")
    paths = resolve_lumir_paths(Namespace(paths=3), required=("train_root", "train_json"))
    assert paths["train_json"].is_file() and not paths["label_dir"].exists()


def test_saved_displacement_reads_back_in_the_evaluator_layout(tmp_path):
    displacement = np.random.default_rng(0).normal(size=(*SHAPE, 3)).astype(np.float32)
    save_displacement(tmp_path / "disp_0001_0000.nii.gz", displacement)
    loaded = nib.load(tmp_path / "disp_0001_0000.nii.gz")
    np.testing.assert_array_equal(loaded.get_fdata(), displacement)
    np.testing.assert_array_equal(loaded.affine, np.eye(4))
    with pytest.raises(ValueError):
        save_displacement(tmp_path / "bad.nii.gz", displacement.transpose(3, 0, 1, 2))
