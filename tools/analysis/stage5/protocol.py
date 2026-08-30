from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets.OASIS100 import load_stage5_runtime_contract
from experiments.stage5.config import (
    ControllerTrainingConfig,
    U0TrainingConfig,
    build_stage5_controller,
)
from experiments.stage5.features import (
    CENTRE_BETA,
    COLLAR_WIDTH,
    COST_STD_FLOOR,
    FLOW_CONTEXT_SCALE_VOXELS,
    IMAGE_STD_FLOOR,
    POSTERIOR_TEMPERATURE,
    SEARCH_STRIDES,
)
from experiments.stage5.safety import CLAIM_EPS, CLIP_SWEEPS, WORK_EPS
from models.CTCF.controller import (
    STAGE5_INPUT_CHANNEL_COUNT,
    STAGE5_INPUT_CHANNELS,
    STAGE5_RESERVED_HEAD,
    STAGE5_VARIANTS,
)
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.contracts import (
    build_protocol_contract,
    canonical_sha256,
    validate_protocol_contract,
    write_immutable_json,
)
from tools.analysis.stage5.evaluation import STAGE5_EVALUATION_METRIC_IDS

STAGE5_METRIC_IDS = STAGE5_EVALUATION_METRIC_IDS
PROTOCOL_BUNDLE_SCHEMA = "ctcf-stage5-protocol-bundle-v1"


def u0_training_contract(config: U0TrainingConfig) -> dict[str, Any]:
    return {
        "schema": "ctcf-stage5-u0-training-contract-v2",
        "dataset_split": "training",
        "labels_reachable": False,
        "pair_schedule": "epoch-specific deterministic directed derangement over all 294 training subjects",
        "architecture": "CTCF-CascadeA-Mamba",
        "fixed_endpoint_epoch": config.fixed_epoch,
        "selection_policy": "FIXED_EPOCH_NOT_LABEL_SELECTED",
        "config": asdict(config),
        "development_access_during_training": False,
        "training_metrics_are_diagnostic_only": True,
        "best_checkpoint_written": False,
        "amp_overflow_policy": "FAIL_CLOSED_NO_SKIPPED_OPTIMIZER_UPDATES",
    }


def controller_training_contract(config: ControllerTrainingConfig) -> dict[str, Any]:
    controller = build_stage5_controller(config)
    parameter_count = sum(parameter.numel() for parameter in controller.parameters())
    return {
        "schema": "ctcf-stage5-controller-training-contract-v2",
        "dataset_split": "training",
        "labels_reachable": False,
        "source_field_policy": "frozen_U0_no_grad_then_shared_frozen_bootstrap_construction_on_the_fly",
        "development_source_policy": "persisted_exactly_certified_U0_field",
        "pair_schedule": (
            "epoch-specific deterministic perfect matching over all 294 training subjects; "
            "both directions share one optimizer update"
        ),
        "fixed_endpoint_epoch": config.fixed_epoch,
        "selection_policy": "FIXED_EPOCH_NOT_LABEL_SELECTED",
        "variants": list(STAGE5_VARIANTS),
        "equal_parameter_count": parameter_count,
        "common_initial_state_within_seed": True,
        "config": asdict(config),
        "development_access_during_training": False,
        "training_metrics_are_diagnostic_only": True,
        "best_checkpoint_written": False,
        "amp_overflow_policy": "FAIL_CLOSED_NO_SKIPPED_OPTIMIZER_UPDATES",
    }


def search_contract() -> dict[str, Any]:
    return {
        "schema": "ctcf-stage5-search-feature-contract-v1",
        "input_channel_count": STAGE5_INPUT_CHANNEL_COUNT,
        "input_channels": list(STAGE5_INPUT_CHANNELS),
        "strides_voxels": list(SEARCH_STRIDES),
        "candidate_order": "lexicographic_zyx_27",
        "posterior_temperature": POSTERIOR_TEMPERATURE,
        "centre_beta": CENTRE_BETA,
        "image_std_floor": IMAGE_STD_FLOOR,
        "cost_std_floor": COST_STD_FLOOR,
        "flow_context_scale_voxels": FLOW_CONTEXT_SCALE_VOXELS,
        "vector_input_units": "stride-normalized",
        "physical_proposal_units": "full-resolution voxels",
        "common_support": f"all S2/S4 candidates valid inside collar {COLLAR_WIDTH}",
        "reserved_head_channel": STAGE5_RESERVED_HEAD,
        "reserved_head_semantics": "literal zero, unused",
        "safety": {
            "claim_epsilon_decimal": CLAIM_EPS,
            "work_epsilon": WORK_EPS,
            "clip_sweeps": CLIP_SWEEPS,
            "save_reload_before_exact_certificate": True,
            "rollback_is_source_byte_identity": True,
        },
    }


def bootstrap_parameters() -> dict[str, Any]:
    return {
        "collar_width": COLLAR_WIDTH,
        "work_epsilon_decimal": str(WORK_EPS),
        "claim_epsilon_decimal": CLAIM_EPS,
        "repair_operator_id": "CTCF_DIGITAL_THEN_TRILINEAR_COLLAR_REPAIR_V1",
        "repair_parameters": {
            "digital_epsilon": 0.0,
            "fixed_boundary_values": 0.0,
            "trilinear_work_epsilon": WORK_EPS,
            "phi_then_psi_conversion": True,
            "save_reload_exact_each_field": True,
        },
    }


def prepare_protocol_bundle(
    *,
    git_head: str,
    data_contract_path: Path,
    output_root: Path,
    u0_config: U0TrainingConfig | None = None,
    controller_config: ControllerTrainingConfig | None = None,
    bootstrap_policy: str = "collar_repair",
) -> dict[str, Any]:
    runtime = load_stage5_runtime_contract(data_contract_path)
    u0_config = u0_config or U0TrainingConfig()
    controller_config = controller_config or ControllerTrainingConfig()
    output_root.mkdir(parents=True, exist_ok=True)

    contracts = {
        "u0_training_contract.json": u0_training_contract(u0_config),
        "controller_training_contract.json": controller_training_contract(controller_config),
        "search_contract.json": search_contract(),
    }
    digests: dict[str, str] = {}
    for name, payload in contracts.items():
        path = output_root / name
        write_immutable_json(path, payload)
        digests[name] = sha256_file(path)

    protocol = build_protocol_contract(
        git_head=git_head,
        data_contract_sha256=runtime.contract_sha256,
        u0_training_contract_sha256=digests["u0_training_contract.json"],
        controller_training_contract_sha256=digests["controller_training_contract.json"],
        search_contract_sha256=digests["search_contract.json"],
        directed_case_ids=tuple(item["case_id"] for item in runtime.pairs["cases"]),
        metric_ids=STAGE5_METRIC_IDS,
        u0_fixed_epoch=u0_config.fixed_epoch,
        controller_fixed_epoch=controller_config.fixed_epoch,
        bootstrap_policy=bootstrap_policy,
        bootstrap_parameters=bootstrap_parameters() if bootstrap_policy == "collar_repair" else {},
    )
    validate_protocol_contract(protocol)
    protocol_path = output_root / "protocol.json"
    write_immutable_json(protocol_path, protocol)
    bundle = {
        "schema": PROTOCOL_BUNDLE_SCHEMA,
        "protocol_sha256": canonical_sha256(protocol),
        "data_contract_sha256": runtime.contract_sha256,
        "files": {
            "protocol": {"path": protocol_path.name, "sha256": sha256_file(protocol_path)},
            **{
                name.removesuffix(".json"): {"path": name, "sha256": digest} for name, digest in sorted(digests.items())
            },
        },
        "contains_dice_success_threshold": False,
    }
    write_immutable_json(output_root / "protocol_bundle.json", bundle)
    return bundle


__all__ = [
    "PROTOCOL_BUNDLE_SCHEMA",
    "STAGE5_METRIC_IDS",
    "bootstrap_parameters",
    "controller_training_contract",
    "prepare_protocol_bundle",
    "search_contract",
    "u0_training_contract",
]
