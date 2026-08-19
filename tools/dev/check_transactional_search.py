from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.transactional_search import (
    OFFSETS,
    ZERO_OFFSET_INDEX,
    CandidateScreen,
    build_proposal,
    commit_exact_candidate,
    geometry_mask,
    mind_ssc,
    phi_to_psi_displacement,
    psi_to_phi_displacement,
    sample_at_psi,
    save_flow_npz_atomic,
)
from utils.cert_exact import certify_flow_exact

REFERENCE_COMMIT = "b229e52e44b114e2040a503334c92269750c16b2"


def _convexadam_reference_mind_ssc(image: torch.Tensor, radius: int = 1, dilation: int = 2) -> torch.Tensor:
    """Literal test-only transcription of ConvexAdam's pinned MINDSSC function."""
    six_neighbourhood = torch.tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]],
        dtype=torch.long,
        device=image.device,
    )
    points = six_neighbourhood.t().unsqueeze(0).float()
    squared = (points**2).sum(dim=1).unsqueeze(2)
    distances = (squared + squared.permute(0, 2, 1) - 2.0 * torch.bmm(points.permute(0, 2, 1), points)).clamp_min(0)
    row, col = torch.meshgrid(torch.arange(6, device=image.device), torch.arange(6, device=image.device), indexing="ij")
    mask = (row > col).reshape(-1) & (distances.squeeze(0) == 2).reshape(-1)
    shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).reshape(-1, 3)[mask]
    shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).reshape(-1, 3)[mask]
    kernel1 = image.new_zeros((12, 1, 3, 3, 3))
    kernel2 = image.new_zeros((12, 1, 3, 3, 3))
    indices = torch.arange(12, device=image.device)
    kernel1[indices, 0, shift1[:, 0], shift1[:, 1], shift1[:, 2]] = 1
    kernel2[indices, 0, shift2[:, 0], shift2[:, 1], shift2[:, 2]] = 1
    padded = F.pad(image, (dilation,) * 6, mode="replicate")
    ssd = F.avg_pool3d(
        F.pad(
            (F.conv3d(padded, kernel1, dilation=dilation) - F.conv3d(padded, kernel2, dilation=dilation)).square(),
            (radius,) * 6,
            mode="replicate",
        ),
        radius * 2 + 1,
        stride=1,
    )
    descriptor = ssd - torch.min(ssd, 1, keepdim=True)[0]
    variance = torch.mean(descriptor, 1, keepdim=True)
    variance = torch.clamp(variance, variance.mean().item() * 0.001, variance.mean().item() * 1000)
    descriptor = torch.exp(-descriptor / variance)
    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=image.device)
    return descriptor[:, order]


def run_checks() -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}

    checks["offset_contract"] = {
        "passed": len(OFFSETS) == 27
        and ZERO_OFFSET_INDEX == 13
        and OFFSETS[0] == (-1, -1, -1)
        and OFFSETS[-1] == (1, 1, 1),
        "zero_index": ZERO_OFFSET_INDEX,
        "first": OFFSETS[0],
        "last": OFFSETS[-1],
    }

    generator = torch.Generator(device="cpu").manual_seed(20260820)
    image = torch.randn((1, 1, 9, 10, 11), generator=generator, dtype=torch.float32)
    production = mind_ssc(image, radius=1, dilation=2)
    reference = _convexadam_reference_mind_ssc(image, radius=1, dilation=2)
    mind_error = float((production - reference).abs().max().item())
    checks["mind_reference"] = {
        "passed": torch.equal(production, reference) or torch.allclose(production, reference, atol=1e-7, rtol=1e-6),
        "reference_commit": REFERENCE_COMMIT,
        "max_abs_error": mind_error,
        "radius": 1,
        "dilation": 2,
        "channels": int(production.shape[1]),
    }

    shape = (9, 10, 11)
    field = torch.zeros((1, 3, *shape), dtype=torch.float32)
    ramp = torch.arange(shape[-1], dtype=torch.float32).view(1, 1, 1, 1, shape[-1]).expand(1, 1, *shape)
    shifted = sample_at_psi(ramp, field, offset=(0, 0, 1))
    sign_error = float((shifted[..., 2:-2] - (ramp[..., 2:-2] + 1.0)).abs().max().item())
    checks["coordinate_sign"] = {
        "passed": sign_error <= 1e-5,
        "statement": "positive dx samples the moving source at x+1",
        "max_abs_error": sign_error,
    }

    phi = torch.randn((1, 3, *shape), generator=generator, dtype=torch.float32) * 0.05
    recovered = psi_to_phi_displacement(phi_to_psi_displacement(phi))
    roundtrip_error = float((phi - recovered).abs().max().item())
    checks["phi_psi_roundtrip"] = {
        "passed": roundtrip_error <= 2e-6,
        "max_abs_error": roundtrip_error,
    }

    fixed = torch.randn((1, 1, *shape), generator=generator)
    mask = geometry_mask(shape, collar_width=2, device=fixed.device)
    proposal = build_proposal(
        fixed=fixed,
        moving=fixed.clone(),
        psi_displacement=field,
        mask=mask,
        feature="intensity",
        collar_width=2,
    )
    zero_error = float(proposal.displacement.abs().max().item())
    checks["zero_proposal"] = {
        "passed": zero_error <= 5e-4,
        "max_abs_value": zero_error,
        "tolerance": 5e-4,
    }

    moving_feature = torch.randn((1, 12, *shape), generator=generator)
    fixed_feature = sample_at_psi(moving_feature, field, offset=(0, 0, 1))
    signed_means: dict[str, list[float]] = {}
    for orientation in ("target_centered", "reversed"):
        signed = build_proposal(
            torch.zeros_like(fixed),
            torch.zeros_like(fixed),
            field,
            mask,
            feature="mind",
            orientation=orientation,
            collar_width=2,
            fixed_feature_override=fixed_feature,
            moving_feature_override=moving_feature,
        )
        means = signed.hard_displacement[:, :, 3:-3, 3:-3, 3:-3].mean(dim=(0, 2, 3, 4))
        signed_means[orientation] = [float(value) for value in means]
    expected_sign = [0.0, 0.0, 1.0]
    checks["reversed_additive_sign"] = {
        "passed": all(signed_means[name] == expected_sign for name in signed_means),
        "expected_additive_zyx": expected_sign,
        "observed": signed_means,
    }

    safe = torch.zeros((1, 3, 5, 6, 7), dtype=torch.float32)
    unsafe = safe.clone()
    unsafe[:, 0] = -2.0 * torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
    safe_report = certify_flow_exact(safe, eps="0.001")
    unsafe_report = certify_flow_exact(unsafe, eps="0.001")
    checks["synthetic_topology_barrier"] = {
        "passed": safe_report["status"] == "CERTIFIED" and unsafe_report["status"] == "NOT_CERTIFIED_BY_PREDICATE",
        "safe_status": safe_report["status"],
        "unsafe_status": unsafe_report["status"],
    }

    batch_guard = False
    try:
        sample_at_psi(torch.zeros((2, 1, *shape)), torch.zeros((2, 3, *shape)))
    except ValueError:
        batch_guard = True
    checks["batch_guard"] = {"passed": batch_guard}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        initial = root / "initial.npz"
        output = root / "output.npz"
        save_flow_npz_atomic(initial, safe)
        initial_sha = sha256_file(initial)
        outcome = commit_exact_candidate(initial, output, eligible=[])
        checks["byte_exact_rollback"] = {
            "passed": outcome.status == "ROLLED_BACK"
            and outcome.rollback_byte_identical
            and sha256_file(output) == initial_sha,
            "initial_sha256": initial_sha,
            "output_sha256": sha256_file(output),
        }
        inversion_output = root / "inversion_output.npz"
        synthetic_screen = CandidateScreen(
            coefficient=1.0,
            utility=-1.0,
            improvement=0.1,
            tolerance=1e-6,
            cert_bound=1.0,
            utility_passed=True,
            fast_certificate_passed=True,
        )
        inversion = commit_exact_candidate(
            initial,
            inversion_output,
            eligible=[synthetic_screen],
            initial_psi=safe,
            proposal=unsafe,
        )
        checks["inverting_proposal_rejected"] = {
            "passed": inversion.status == "ROLLED_BACK"
            and inversion.rollback_byte_identical
            and inversion.exact_report["status"] == "CERTIFIED",
            "transaction_status": inversion.status,
            "returned_exact_status": inversion.exact_report["status"],
        }

    passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "schema": "ctcf-search-c0-selfcheck-v1",
        "status": "PASS" if passed else "FAIL",
        "reference": {"name": "ConvexAdam MIND-SSC", "git_commit": REFERENCE_COMMIT},
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed C0.1 search/coordinate/topology self-check.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_checks()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
