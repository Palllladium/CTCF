from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import (
    certified_local_clip_candidate,
    geometry_mask,
    load_flow_npz,
    phi_to_psi_displacement,
    save_flow_npz_atomic,
)
from utils.cert_exact import certify_flow_exact
from utils.field import (
    boundary_nonzero_count,
    boundary_vertex_mask,
    digital_project,
    enforce_identity_boundary,
    identity_collar,
    trilinear_project,
)

CLAIM_EPS = "0.001"
WORK_EPS = 0.0011
COLLAR_WIDTH = 7
CLIP_SWEEPS = 1
BOOTSTRAP_POLICIES = ("collar_repair", "identity")


@dataclass(frozen=True, slots=True)
class InitialFieldArtifact:
    policy: str
    phi_path: Path
    psi_path: Path
    phi_sha256: str
    psi_sha256: str
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ControllerTransaction:
    status: str
    requested_path: Path
    candidate_path: Path
    returned_path: Path
    requested_sha256: str
    candidate_sha256: str
    returned_sha256: str
    requested_array_sha256: str
    candidate_array_sha256: str
    returned_array_sha256: str
    rollback_byte_identical: bool
    clip_report: dict[str, float | int | str]
    candidate_exact_report: dict[str, Any]
    returned_exact_report: dict[str, Any]


def _exact_passed(report: dict[str, Any]) -> bool:
    return report.get("status") == "CERTIFIED" and report.get("certified") is True


def _save_reload_exact(path: Path) -> tuple[torch.Tensor, str, dict[str, Any]]:
    stored = load_flow_npz(path)
    digest = sha256_file(path)
    report = certify_flow_exact(stored, eps=CLAIM_EPS)
    return stored, digest, report


def construct_initial_field(
    raw_phi: torch.Tensor,
    *,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the frozen bootstrap construction without persisting field bytes."""
    if policy not in BOOTSTRAP_POLICIES:
        raise ValueError(f"unsupported bootstrap policy: {policy}")
    if raw_phi.dim() != 5 or raw_phi.shape[0] != 1 or raw_phi.shape[1] != 3:
        raise ValueError("raw_phi must have shape [1,3,D,H,W]")
    if not raw_phi.is_floating_point() or not bool(torch.isfinite(raw_phi).all()):
        raise ValueError("raw_phi must be finite and floating point")

    raw_phi = raw_phi.detach().float().clone()
    if policy == "identity":
        phi = torch.zeros_like(raw_phi, dtype=torch.float32)
        report: dict[str, Any] = {
            "policy": policy,
            "scientific_degradation": "IDENTITY_BASELINE",
            "digital_residual_percent": 0.0,
            "digital_iterations": 0,
            "trilinear_repair": None,
        }
    else:
        collared = identity_collar(raw_phi.float(), width=COLLAR_WIDTH)
        fixed_mask = boundary_vertex_mask(collared)
        fixed_values = torch.zeros_like(collared)
        digital, residual, digital_iterations = digital_project(
            collared,
            eps=0.0,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )
        if residual != 0.0:
            raise RuntimeError(f"digital bootstrap preconditioner failed: residual={residual}")
        repaired, repair = trilinear_project(
            digital,
            eps=WORK_EPS,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )
        phi = enforce_identity_boundary(repaired.float())
        if not repair.certified or repair.cert_bound < WORK_EPS or boundary_nonzero_count(phi) != 0:
            raise RuntimeError(f"initial field repair failed: status={repair.status}, bound={repair.cert_bound}")
        report = {
            "policy": policy,
            "scientific_degradation": None,
            "digital_residual_percent": residual,
            "digital_iterations": digital_iterations,
            "trilinear_repair": asdict(repair),
        }

    psi = phi_to_psi_displacement(phi).float()
    return phi.float(), psi, report


def prepare_initial_field(raw_phi: torch.Tensor, root: Path, *, policy: str) -> InitialFieldArtifact:
    """Persist, reload, and exactly certify the shared bootstrap construction."""
    root.mkdir(parents=True, exist_ok=True)
    phi, _, report = construct_initial_field(raw_phi, policy=policy)

    phi_path = root / "initial_phi.npz"
    save_flow_npz_atomic(phi_path, phi)
    stored_phi, phi_sha, phi_exact = _save_reload_exact(phi_path)
    if not _exact_passed(phi_exact) or int(phi_exact.get("boundary_nonzero_count", -1)) != 0:
        raise RuntimeError("stored initial Phi failed exact certification")

    psi = phi_to_psi_displacement(stored_phi).float()
    psi_path = root / "initial_psi.npz"
    save_flow_npz_atomic(psi_path, psi)
    _, psi_sha, psi_exact = _save_reload_exact(psi_path)
    if not _exact_passed(psi_exact):
        raise RuntimeError("stored initial Psi failed exact certification")
    report.update(
        {
            "phi_exact": phi_exact,
            "phi_sha256": phi_sha,
            "psi_exact": psi_exact,
            "psi_sha256": psi_sha,
        }
    )
    return InitialFieldArtifact(
        policy=policy,
        phi_path=phi_path,
        psi_path=psi_path,
        phi_sha256=phi_sha,
        psi_sha256=psi_sha,
        report=report,
    )


def commit_controller_delta(
    initial_path: Path,
    requested_delta: torch.Tensor,
    output_root: Path,
) -> ControllerTransaction:
    output_root.mkdir(parents=True, exist_ok=True)
    initial_path = initial_path.resolve()
    initial = load_flow_npz(initial_path).to(requested_delta.device)
    # Fingerprint the source before the transaction writes anything: a rollback is only
    # byte-identical against the generation this call actually started from.
    initial_sha = sha256_file(initial_path)
    initial_array_sha = array_sha256(initial)
    if requested_delta.shape != initial.shape:
        raise ValueError("requested controller delta and initial field must share shape")
    mask = geometry_mask(tuple(initial.shape[-3:]), COLLAR_WIDTH, initial.device)

    requested = (initial + requested_delta.float()).float()
    requested_array_sha = array_sha256(requested)
    requested_path = output_root / "requested.npz"
    save_flow_npz_atomic(requested_path, requested)
    requested_reloaded = load_flow_npz(requested_path)
    if array_sha256(requested_reloaded) != requested_array_sha:
        raise RuntimeError("Stage5 requested field changed across save and reload")
    requested_sha = sha256_file(requested_path)

    candidate, clip_report = certified_local_clip_candidate(
        initial,
        requested_delta.float(),
        mask,
        work_eps=WORK_EPS,
        sweeps=CLIP_SWEEPS,
    )
    candidate_path = output_root / "post_safety_candidate.npz"
    candidate_array_sha = array_sha256(candidate)
    save_flow_npz_atomic(candidate_path, candidate.float())
    candidate_reloaded, candidate_sha, candidate_exact_report = _save_reload_exact(candidate_path)
    if array_sha256(candidate_reloaded) != candidate_array_sha:
        raise RuntimeError("Stage5 safety candidate changed across save and reload")

    accepted = _exact_passed(candidate_exact_report)
    returned_path = candidate_path if accepted else initial_path
    status = "ACCEPTED" if accepted else "ROLLED_BACK"
    returned_reloaded, returned_sha, returned_exact_report = _save_reload_exact(returned_path)
    returned_array_sha = array_sha256(returned_reloaded)
    if not _exact_passed(returned_exact_report):
        raise RuntimeError("Stage5 returned field failed exact certification")
    if accepted:
        rollback_identical = False
        if returned_array_sha != candidate_array_sha:
            raise RuntimeError("Stage5 accepted field differs from its certified candidate")
    else:
        # Measured against the entry fingerprint, not against a second read of the same file.
        rollback_identical = returned_sha == initial_sha and returned_array_sha == initial_array_sha
        if not rollback_identical:
            raise RuntimeError("Stage5 rolled-back field differs from the source this transaction started from")

    return ControllerTransaction(
        status=status,
        requested_path=requested_path,
        candidate_path=candidate_path,
        returned_path=returned_path,
        requested_sha256=requested_sha,
        candidate_sha256=candidate_sha,
        returned_sha256=returned_sha,
        requested_array_sha256=requested_array_sha,
        candidate_array_sha256=candidate_array_sha,
        returned_array_sha256=returned_array_sha,
        rollback_byte_identical=rollback_identical,
        clip_report=clip_report,
        candidate_exact_report=candidate_exact_report,
        returned_exact_report=returned_exact_report,
    )


__all__ = [
    "BOOTSTRAP_POLICIES",
    "CLAIM_EPS",
    "CLIP_SWEEPS",
    "COLLAR_WIDTH",
    "WORK_EPS",
    "ControllerTransaction",
    "InitialFieldArtifact",
    "commit_controller_delta",
    "construct_initial_field",
    "prepare_initial_field",
]
