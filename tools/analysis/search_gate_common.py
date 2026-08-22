from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.transactional_search import (
    ProposalResult,
    load_flow_npz,
    mind_distance_from_features,
    ncc_loss_from_normalized,
    phi_to_psi_displacement,
    sample_at_psi,
    save_flow_npz_atomic,
)
from utils import dice_per_label
from utils.cert_exact import certify_flow_exact
from utils.field import (
    boundary_nonzero_count,
    boundary_vertex_mask,
    digital_fold_percent,
    digital_project,
    enforce_identity_boundary,
    identity_collar,
    jacobian_nonpositive_percent,
    logdet_std_from_flow,
    trilinear_project,
)

PROTOCOL_SALT = "CTCF-GATE-C0-V1|"
SPLIT_PROTOCOL_ID = "CTCF-GATE-C0-V1-SALTED-IXI-VAL-58"
CLAIM_EPS = 0.001
WORK_EPS = 0.0011
COLLAR_WIDTH = 4
TIME_STEPS = 6
CONFIG_KEY = "CTCF-CascadeA-VM-Unified"
DEFAULT_CHECKPOINT = "results/P10_LONGRUN_VXM_UNIFIED_SVF_IXI/ckpt/best.pth"
UTILITY_RELATIVE_TOLERANCE = 1e-6
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The 19 IXI validation cases the salted C0 order puts first; C0 development and C1
# exploration run on exactly this set.
IXI_DEVELOPMENT_CASES = (
    "subject_344",
    "subject_136",
    "subject_165",
    "subject_475",
    "subject_131",
    "subject_389",
    "subject_485",
    "subject_153",
    "subject_252",
    "subject_509",
    "subject_126",
    "subject_459",
    "subject_222",
    "subject_474",
    "subject_144",
    "subject_85",
    "subject_248",
    "subject_151",
    "subject_295",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text_sha256(encoded)


def case_id_from_path(path: str) -> str:
    """File stem, with the OASIS `p_` pairing prefix removed."""
    stem = Path(path).stem
    return stem[2:] if stem.startswith("p_") else stem


def salted_case_hash(value: str) -> str:
    return hashlib.sha256((PROTOCOL_SALT + value).encode("utf-8")).hexdigest()


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def require_finite(values: dict[str, Any], label: str) -> None:
    invalid = sorted(key for key, value in values.items() if not is_finite_number(value))
    if invalid:
        raise RuntimeError(f"{label} contains non-finite or non-numeric values: {invalid}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(checkpoint: str, config: str, device: torch.device):
    adapter = get_model_adapter("ctcf")
    model = adapter.build(time_steps=TIME_STEPS, config_key=config, l3_svf=True).to(device)
    load_report = load_checkpoint_state(model, checkpoint, strict=True)
    model.eval()
    return adapter, model, load_report


def prepare_initial_state(flow: torch.Tensor, work_dir: Path) -> tuple[torch.Tensor, Path, dict[str, Any]]:
    collared = identity_collar(flow.float(), width=COLLAR_WIDTH)
    fixed_mask = boundary_vertex_mask(collared)
    fixed_values = torch.zeros_like(collared)
    digital, residual, digital_iterations = digital_project(
        collared,
        eps=0.0,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    if residual != 0.0:
        raise RuntimeError(f"Digital preconditioner failed closed: residual={residual}")
    repaired, repair = trilinear_project(
        digital,
        eps=WORK_EPS,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    phi = enforce_identity_boundary(repaired.float())
    if not repair.certified or repair.cert_bound < WORK_EPS or boundary_nonzero_count(phi) != 0:
        raise RuntimeError(f"Initial Phi repair failed closed: status={repair.status}, bound={repair.cert_bound}")

    work_dir.mkdir(parents=True, exist_ok=True)
    phi_path = work_dir / "initial_phi.npz"
    save_flow_npz_atomic(phi_path, phi)
    stored_phi = load_flow_npz(phi_path)
    phi_exact = certify_flow_exact(stored_phi, eps=str(CLAIM_EPS))
    if phi_exact["status"] != "CERTIFIED" or phi_exact["boundary_nonzero_count"] != 0:
        raise RuntimeError(f"Stored initial Phi failed exact certification: {phi_exact['status']}")

    psi = phi_to_psi_displacement(phi).float()
    psi_path = work_dir / "initial_psi.npz"
    save_flow_npz_atomic(psi_path, psi)
    stored_psi = load_flow_npz(psi_path)
    psi_exact = certify_flow_exact(stored_psi, eps=str(CLAIM_EPS))
    if psi_exact["status"] != "CERTIFIED":
        raise RuntimeError(f"Stored initial Psi failed exact certification: {psi_exact['status']}")
    report = {
        "digital_residual_percent": residual,
        "digital_iterations": digital_iterations,
        "trilinear_repair": asdict(repair),
        "phi_npz_sha256": sha256_file(phi_path),
        "phi_exact": phi_exact,
        "psi_npz_sha256": sha256_file(psi_path),
        "psi_exact": psi_exact,
    }
    return psi, psi_path, report


def dice_score(psi: torch.Tensor, moving_seg: torch.Tensor, fixed_seg: torch.Tensor, labels: tuple[int, ...]) -> float:
    warped = sample_at_psi(moving_seg.float(), psi, mode="nearest").long()
    return float(dice_per_label(warped, fixed_seg.long(), labels=labels).mean())


def proposal_statistics(proposal: ProposalResult, tensor: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    magnitude = tensor.square().sum(dim=1, keepdim=True).sqrt()
    return {
        "entropy_mean": float(proposal.entropy.masked_select(mask).double().mean().item()),
        "confidence_mean": float(proposal.confidence.masked_select(mask).double().mean().item()),
        "proposal_norm_mean": float(magnitude.masked_select(mask).double().mean().item()),
        "proposal_norm_max": float(magnitude.masked_select(mask).max().item()),
    }


def candidate_metrics(
    candidate: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    mask: torch.Tensor,
    support_weights: torch.Tensor,
) -> dict[str, float]:
    metrics = {
        "ncc9": ncc_loss_from_normalized(fixed_norm, moving_norm, candidate, mask, win=9),
        "ncc7": ncc_loss_from_normalized(fixed_norm, moving_norm, candidate, mask, win=7),
        "support_ncc9": ncc_loss_from_normalized(
            fixed_norm,
            moving_norm,
            candidate,
            mask,
            win=9,
            weights=support_weights,
        ),
        "mind": mind_distance_from_features(fixed_mind, moving_mind, candidate, mask),
    }
    require_finite(metrics, "candidate utility metrics")
    return metrics


def relative_improvement(baseline: float, candidate: float) -> tuple[float | None, float, bool]:
    tolerance = UTILITY_RELATIVE_TOLERANCE * max(abs(baseline), np.finfo(np.float64).tiny)
    if not (math.isfinite(baseline) and math.isfinite(candidate)):
        return None, tolerance, False
    improvement = baseline - candidate
    return improvement, tolerance, improvement >= tolerance


def deformation_quality_metrics(field: torch.Tensor, *, exact_certified: bool) -> dict[str, Any]:
    """Report paper-facing geometry without conflating a sampled count with the exact certificate."""
    metrics: dict[str, Any] = {
        "sdlogj": float(logdet_std_from_flow(field)),
        "j_leq0_central_percent": float(jacobian_nonpositive_percent(field, crop=1)),
        "j_leq0_digital10_percent": float(digital_fold_percent(field).item()),
        "trilinear_fold_percent_upper_bound": 0.0 if exact_certified else None,
        "trilinear_fold_status": "ZERO_BY_EXACT_CERTIFICATE" if exact_certified else "NOT_ESTABLISHED",
    }
    require_finite(
        {
            key: value
            for key, value in metrics.items()
            if key not in {"trilinear_fold_percent_upper_bound", "trilinear_fold_status"}
        },
        "deformation quality metrics",
    )
    if any(float(metrics[key]) < 0.0 for key in ("sdlogj", "j_leq0_central_percent", "j_leq0_digital10_percent")):
        raise RuntimeError("Deformation quality metrics must be non-negative")
    return metrics


def bootstrap_ci(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"method": "not_available", "low": None, "high": None, "replicates": 0, "seed": 0}
    generator = np.random.default_rng(0)
    samples = generator.choice(values, size=(10_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "method": "case-bootstrap percentile CI for the mean; diagnostic only",
        "low": float(low),
        "high": float(high),
        "replicates": 10_000,
        "seed": 0,
    }


def sign_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Sign summary requires a non-empty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "improved": int((values > 0.0).sum()),
        "worsened": int((values < 0.0).sum()),
        "unchanged": int((values == 0.0).sum()),
        "mean_ci95": bootstrap_ci(values),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Distribution summary requires a non-empty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
