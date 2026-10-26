import json

import torch

from tools.analysis.certify_psi import check_collection
from utils.cert_exact import certify_flow_exact
from utils.field import legacy_sampling_displacement
from utils.field_io import save_flow_npz_atomic


def _inputs(tmp_path, materialized=True):
    phi = torch.zeros(1, 3, 4, 5, 6)
    psi = legacy_sampling_displacement(phi)
    root = tmp_path / "results"
    source = root / "psi_cert" / "test"
    source.mkdir(parents=True)
    cell = {"file": "/old/server/case.npz", **certify_flow_exact(psi, eps="0.001")}
    (source / "exact_psi.json").write_text(json.dumps([cell]))
    if materialized:
        save_flow_npz_atomic(source / "flows/case.npz", psi)
    else:
        save_flow_npz_atomic(root / "collar/canonical/flows/case.npz", phi)
    return root, source


def test_existing_psi_is_checked_without_a_second_materialization(tmp_path):
    root, _ = _inputs(tmp_path)
    report = check_collection(root, tmp_path / "out", "test", "canonical", 1)
    assert report["passed"] and report["fresh"]["status"] == "GLOBAL_HOMEOMORPHISM"
    assert report["rows"][0]["sha256"] == report["fresh"]["sha256"]


def test_missing_psi_can_be_reconstructed_only_when_its_bytes_match(tmp_path):
    root, _ = _inputs(tmp_path, materialized=False)
    report = check_collection(root, tmp_path / "out", "test", "canonical", 1)
    assert report["passed"]
    assert (tmp_path / "out/fields/test/case.npz").is_file()


def test_changed_bytes_and_missing_payload_cannot_pass(tmp_path):
    root, source = _inputs(tmp_path)
    path = source / "flows/case.npz"
    save_flow_npz_atomic(path, torch.zeros(1, 3, 4, 5, 6))
    report = check_collection(root, tmp_path / "out", "test", "canonical", 1)
    assert not report["passed"] and "different stored arrays" in report["rows"][0]["error"]
    path.unlink()
    report = check_collection(root, tmp_path / "out", "test", "canonical", 1)
    assert not report["passed"] and report["fresh"] is None
