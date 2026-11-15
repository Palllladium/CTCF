import json
import struct
from collections import Counter
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from datasets.LPBA40 import LPBAPairDataset, read_lpba_manifest
from tools.analysis.prepare_external import (
    air_pull_matrix,
    balanced_pairs,
    extract_zip,
    resample,
    target_grid,
    unpack_archives,
)
from tools.analysis.run_artifacts import sha256_file


@pytest.mark.parametrize("endian", ["<", ">"])
def test_air_pull_uses_inverse_and_handles_anisotropic_native_voxels(endian):
    data = bytearray(720)
    matrix = np.asarray([[0.5, 0, 0, 1], [0, 0.5, 0, 2], [0, 0, 0.5, 3], [0, 0, 0, 1]])
    struct.pack_into(endian + "16d", data, 0, *matrix.T.ravel())
    struct.pack_into(endian + "4I3d", data, 256, 16, 21, 11, 21, 0.5, 1.0, 0.5)
    struct.pack_into(endian + "4I3d", data, 424, 16, 9, 9, 9, 1.0, 1.0, 1.0)
    pull, shape, spacing = air_pull_matrix(data)
    assert shape == (9, 9, 9) and spacing == (1, 1, 1)
    np.testing.assert_allclose(pull @ [3, 4, 5, 1], [4, 2, 4, 1])
    values = np.indices((21, 11, 21)).sum(axis=0).astype(np.float32)
    assert resample(values, pull, shape, order=1)[3, 4, 5] == pytest.approx(10)


def test_physical_grid_preserves_fov_and_tensor_laterality():
    pull, affine = target_grid((181, 217, 181), (1, 1, 1))
    assert nib.aff2axcodes(affine) == ("L", "I", "A")
    np.testing.assert_allclose(np.linalg.norm(affine[:3, :3], axis=0), 1.15)
    target_corners = np.asarray(list(np.ndindex(2, 2, 2))) * (np.asarray((160, 192, 224)) - 1)
    source_center = np.asarray([90, 108, 90])
    np.testing.assert_allclose(pull @ [79.5, 95.5, 111.5, 1], [*source_center, 1], atol=1e-12)
    source_corners = np.asarray(list(np.ndindex(2, 2, 2))) * np.asarray([180, 216, 180])
    model_coords = (np.linalg.inv(pull) @ np.c_[source_corners, np.ones(8)].T).T[:, :3]
    assert np.all(model_coords >= 0) and np.all(model_coords <= target_corners.max(axis=0))
    source_to_tensor = np.linalg.inv(pull)
    left = source_to_tensor @ [30, 108, 90, 1]
    right = source_to_tensor @ [150, 108, 90, 1]
    anterior = source_to_tensor @ [90, 180, 90, 1]
    posterior = source_to_tensor @ [90, 30, 90, 1]
    assert left[0] > right[0] and anterior[2] > posterior[2]
    with pytest.raises(ValueError, match="truncates"):
        target_grid((181, 217, 181), (1, 1, 1), spacing=1.0)


def test_balanced_graph_has_three_neighbors_and_reverse_pairs():
    ids = [f"S{i:02d}" for i in range(1, 41)]
    pairs = balanced_pairs(ids)
    assert len(pairs) == len(set(map(tuple, pairs))) == 120
    assert Counter(a for a, _ in pairs) == dict.fromkeys(ids, 3)
    assert Counter(b for _, b in pairs) == dict.fromkeys(ids, 3)
    assert all([b, a] in pairs and a != b for a, b in pairs)
    assert pairs == balanced_pairs(ids)


def test_extraction_rejects_traversal_before_writing_any_payload(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("normal.txt", b"ok")
        zip_file.writestr("../outside.txt", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip(archive, tmp_path / "raw")
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "raw/normal.txt").exists()


def test_archive_checksum_and_idempotent_extraction(tmp_path):
    root = tmp_path / "hcp"
    root.mkdir()
    archive = root / "100307_3T_Structural_unproc.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("100307/unprocessed/test.txt", b"payload")
    checksum = archive.with_suffix(".zip.md5")
    checksum.write_text("0" * 32 + "  " + archive.name)
    with pytest.raises(ValueError, match="MD5"):
        unpack_archives(root, move_archives=True)
    assert not (root / "raw").exists()
    (root / "archives" / checksum.name).unlink()
    first = unpack_archives(root)
    assert unpack_archives(root) == first
    payload = root / "raw/100307/unprocessed/test.txt"
    payload.write_bytes(b"altered")
    with pytest.raises(ValueError, match="Existing extracted bytes differ"):
        unpack_archives(root)
    assert payload.read_bytes() == b"altered"


def test_lpba_manifest_is_portable_and_preserves_dense_labels(tmp_path):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24
    labels = np.ones(values.shape, dtype=np.int16) * 56
    rows = []
    for subject in ("S01", "S02"):
        file = tmp_path / f"{subject}.npz"
        np.savez_compressed(file, image=values, labels=labels)
        rows.append({"id": subject, "file": file.name, "sha256": sha256_file(file)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"subjects": rows, "pairs": [["S01", "S02"]], "shape": [2, 3, 4]}))
    dataset = LPBAPairDataset(read_lpba_manifest(manifest, verify_bytes=True))
    x, y, x_seg, y_seg, case_id = dataset[0]
    assert x.shape == y.shape == x_seg.shape == y_seg.shape == (1, 2, 3, 4)
    assert x_seg.max() == 56 and case_id == "S01__S02"
    (tmp_path / "S01.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        read_lpba_manifest(manifest, verify_bytes=True)
