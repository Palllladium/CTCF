from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from models.CTCF.controller import (
    STAGE5_CHANNEL_GROUPS,
    STAGE5_INPUT_CHANNELS,
    STAGE5_VARIANT_GROUPS,
    STAGE5_VARIANTS,
)
from tools.analysis.stage5.primitives import (
    canonical_json_bytes,
    canonical_sha256,
    load_json_object,
    relative_posix,
    require_exact_fields,
    require_finite,
    require_git_sha,
    require_int,
    require_nonempty,
    require_positive_decimal,
    require_sha256,
    write_immutable_json,
)

PROTOCOL_ID = "CTCF-STAGE5-LEARNED-CONTROLLER-V1"
PROTOCOL_SCHEMA = "ctcf-stage5-protocol-v1"
CHECKPOINT_SCHEMA = "ctcf-stage5-checkpoint-v1"
TRAINING_BARRIER_SCHEMA = "ctcf-stage5-training-barrier-v1"
DECISION_RECORD_SCHEMA = "ctcf-stage5-decision-record-v1"
DECISION_BARRIER_SCHEMA = "ctcf-stage5-decision-barrier-v1"
EVALUATION_RECORD_SCHEMA = "ctcf-stage5-evaluation-record-v1"
EVALUATION_BARRIER_SCHEMA = "ctcf-stage5-evaluation-barrier-v1"

BASE_SEEDS = (0, 1, 2)
CHECKPOINT_SELECTION_POLICY = "FIXED_EPOCH_NOT_LABEL_SELECTED"
BOOTSTRAP_POLICIES = frozenset({"collar_repair", "identity"})
TRANSACTION_STATUSES = frozenset({"BASELINE_CERTIFIED", "ACCEPTED", "ROLLED_BACK", "CERTIFIED_DEGRADED_IDENTITY"})


def _channels(group: str) -> tuple[str, ...]:
    return tuple(STAGE5_INPUT_CHANNELS[STAGE5_CHANNEL_GROUPS[group]])


# The channel layout has exactly one owner, the controller module. Naming the groups
# here rather than re-listing their members keeps the frozen protocol and the tensor
# the controller actually consumes from drifting apart.
CONTEXT_CHANNELS = _channels("context")
SEARCH_CHANNELS = tuple(STAGE5_INPUT_CHANNELS[len(CONTEXT_CHANNELS) :])


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    role: str
    controller_family: str
    active_search_channels: tuple[str, ...]


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    variant_id: str
    reference_variant_id: str
    scientific_question: str


CONTROLLER_FAMILIES = {
    variant: "ATTENUATION" if variant.startswith("A") else "FREE_RESIDUAL" for variant in STAGE5_VARIANTS
}


def _active_search_channels(variant: str) -> tuple[str, ...]:
    """The search channels a variant may read, taken from the mask the controller applies."""
    return tuple(name for group in STAGE5_VARIANT_GROUPS[variant] if group != "context" for name in _channels(group))


VARIANT_SPECS = (
    VariantSpec("U0", "BASELINE", "NONE", ()),
    *(
        VariantSpec(
            variant,
            "CONTROL" if variant == "F0" else "MECHANISM",
            CONTROLLER_FAMILIES[variant],
            _active_search_channels(variant),
        )
        for variant in STAGE5_VARIANTS
    ),
)
VARIANT_BY_ID = {spec.variant_id: spec for spec in VARIANT_SPECS}
VARIANT_IDS = tuple(VARIANT_BY_ID)
CONTROLLER_VARIANT_IDS = tuple(value for value in VARIANT_IDS if value != "U0")

# These are the causal questions supported by the frozen matrix.  Reporting only
# controller-minus-U0 would confound generic refiner capacity with search information.
PLANNED_CONTRASTS = (
    ContrastSpec("F0_MINUS_U0", "F0", "U0", "generic learned refiner versus the fresh base model"),
    ContrastSpec("F2V_MINUS_F0", "F2V", "F0", "incremental value of the S2 proposal vector"),
    ContrastSpec("F2S_MINUS_F2V", "F2S", "F2V", "incremental value of compact S2 uncertainty summaries"),
    ContrastSpec("F2P_MINUS_F2V", "F2P", "F2V", "incremental value of the full S2 posterior"),
    ContrastSpec("F2P_MINUS_F2S", "F2P", "F2S", "full S2 posterior versus its compact summaries"),
    ContrastSpec("F4P_MINUS_F2P", "F4P", "F2P", "S4 reach versus S2 reach under a free residual"),
    ContrastSpec("F24P_MINUS_F2P", "F24P", "F2P", "incremental S4 information given S2"),
    ContrastSpec("F24P_MINUS_F4P", "F24P", "F4P", "incremental S2 information given S4"),
    ContrastSpec("A2P_MINUS_F2P", "A2P", "F2P", "attenuation versus free residual with S2 inputs"),
    ContrastSpec("A24P_MINUS_F24P", "A24P", "F24P", "attenuation versus free residual with S2 and S4 inputs"),
    ContrastSpec("A24P_MINUS_A2P", "A24P", "A2P", "incremental S4 information under attenuation"),
)


def _variant_payloads() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": spec.variant_id,
            "role": spec.role,
            "controller_family": spec.controller_family,
            "active_search_channels": list(spec.active_search_channels),
        }
        for spec in VARIANT_SPECS
    ]


def _contrast_payloads() -> list[dict[str, str]]:
    return [
        {
            "contrast_id": item.contrast_id,
            "variant_id": item.variant_id,
            "reference_variant_id": item.reference_variant_id,
            "scientific_question": item.scientific_question,
        }
        for item in PLANNED_CONTRASTS
    ]


ROOT_ROLES = {
    "training": frozenset({"image_root", "checkpoint_root", "source_field_root", "heavy_output_root"}),
    "decision": frozenset({"image_root", "checkpoint_root", "source_field_root", "decision_output_root"}),
    "evaluation": frozenset(
        {
            "image_root",
            "label_root",
            "source_field_root",
            "decision_output_root",
            "evaluation_output_root",
        }
    ),
}

FILE_RECORD_FIELDS = frozenset({"root_id", "relative_path", "bytes", "sha256"})
FIELD_RECORD_FIELDS = frozenset({*FILE_RECORD_FIELDS, "array_sha256"})
CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_id",
        "role",
        "variant_id",
        "seed",
        "fixed_epoch",
        "selection_policy",
        "git_head",
        "protocol_sha256",
        "data_contract_sha256",
        "training_contract_sha256",
        "checkpoint_file",
        "state_dict_sha256",
        "metrics_sha256",
        "base_checkpoint_sha256",
        "initial_controller_state_sha256",
        "source_contract_sha256",
        "controller_parameter_count",
    }
)
DECISION_RECORD_FIELDS = frozenset(
    {
        "schema",
        "decision_id",
        "case_id",
        "seed",
        "variant_id",
        "checkpoint_sha256",
        "certified_source_field",
        "requested_field",
        "candidate_field",
        "returned_field",
        "requested_save_reload",
        "candidate_save_reload",
        "returned_save_reload",
        "exact_report",
        "candidate_exact_status",
        "candidate_exact_certified",
        "returned_exact_status",
        "returned_certified",
        "transaction_status",
        "rollback_source_sha256_equal",
        "runtime_seconds",
        "peak_memory_bytes",
        "requested_delta_rms",
        "candidate_delta_rms",
        "returned_delta_rms",
        "candidate_retained_ratio",
        "returned_retained_ratio",
        "labels_loaded",
        "execution_sha256",
    }
)
SAVE_RELOAD_FIELDS = frozenset(
    {
        "file_sha256",
        "in_memory_array_sha256",
        "reloaded_array_sha256",
        "reloaded_from_persisted_bytes",
    }
)
EVALUATION_RECORD_FIELDS = frozenset(
    {
        "schema",
        "evaluation_id",
        "decision_id",
        "decision_record_sha256",
        "returned_field_sha256",
        "label_source_sha256",
        "metrics_file",
        "labels_loaded_after_decision_barrier",
        "heldout_test_accessed",
        "execution_sha256",
    }
)


def validate_file_record(payload: Mapping[str, Any], *, field: bool = False, label: str = "file") -> None:
    require_exact_fields(payload, FIELD_RECORD_FIELDS if field else FILE_RECORD_FIELDS, label)
    require_nonempty(payload["root_id"], f"{label}.root_id")
    relative_posix(payload["relative_path"], f"{label}.relative_path")
    require_int(payload["bytes"], f"{label}.bytes", minimum=1)
    require_sha256(payload["sha256"], f"{label}.sha256")
    if field:
        require_sha256(payload["array_sha256"], f"{label}.array_sha256")


def validate_save_reload_attestation(
    payload: Mapping[str, Any],
    field_record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Bind an in-memory field to the exact bytes saved and loaded for certification."""
    require_exact_fields(payload, SAVE_RELOAD_FIELDS, label)
    file_sha = require_sha256(payload["file_sha256"], f"{label}.file_sha256")
    before = require_sha256(payload["in_memory_array_sha256"], f"{label}.in_memory_array_sha256")
    after = require_sha256(payload["reloaded_array_sha256"], f"{label}.reloaded_array_sha256")
    if payload["reloaded_from_persisted_bytes"] is not True:
        raise RuntimeError(f"{label} must attest a reload from persisted bytes")
    if file_sha != field_record["sha256"]:
        raise RuntimeError(f"{label} is bound to another persisted file")
    if before != after or after != field_record["array_sha256"]:
        raise RuntimeError(f"{label} changed across save and reload")


def _same_field_bytes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["bytes"] == right["bytes"]
        and left["sha256"] == right["sha256"]
        and left["array_sha256"] == right["array_sha256"]
    )


def validate_root_access(phase: str, roots: Mapping[str, Any]) -> None:
    """Accept only the exact root roles assigned to a phase; labels fail closed elsewhere."""
    if phase not in ROOT_ROLES:
        raise RuntimeError(f"Unknown Stage 5 phase: {phase}")
    expected = ROOT_ROLES[phase]
    if set(roots) != expected:
        raise RuntimeError(
            f"{phase} root roles changed; missing={sorted(expected - set(roots))} extra={sorted(set(roots) - expected)}"
        )
    values = [require_nonempty(roots[key], f"{phase}.{key}") for key in sorted(expected)]
    if len(values) != len(set(values)):
        raise RuntimeError(f"{phase} root identifiers must be distinct")
    if phase != "evaluation" and "label_root" in roots:
        raise RuntimeError(f"{phase} must not receive a label root")


def _validate_bootstrap(policy: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if policy not in BOOTSTRAP_POLICIES:
        raise RuntimeError(f"bootstrap_policy must be one of {sorted(BOOTSTRAP_POLICIES)}")
    frozen = dict(parameters)
    if policy == "identity":
        if frozen:
            raise RuntimeError("identity bootstrap must not carry collar/repair parameters")
        return frozen

    required = {
        "collar_width",
        "work_epsilon_decimal",
        "claim_epsilon_decimal",
        "repair_operator_id",
        "repair_parameters",
    }
    if set(frozen) != required:
        raise RuntimeError("collar_repair bootstrap parameters must freeze exactly " + ", ".join(sorted(required)))
    require_int(frozen["collar_width"], "bootstrap collar_width", minimum=1)
    work = require_positive_decimal(frozen["work_epsilon_decimal"], "bootstrap work epsilon")
    claim = require_positive_decimal(frozen["claim_epsilon_decimal"], "bootstrap claim epsilon")
    if work < claim:
        raise RuntimeError("bootstrap work epsilon must be >= claim epsilon")
    require_nonempty(frozen["repair_operator_id"], "bootstrap repair_operator_id")
    repair = frozen["repair_parameters"]
    if not isinstance(repair, Mapping) or not repair:
        raise RuntimeError("collar_repair must freeze a non-empty repair_parameters object")
    canonical_json_bytes(repair)
    return frozen


def build_protocol_contract(
    *,
    git_head: str,
    data_contract_sha256: str,
    u0_training_contract_sha256: str,
    controller_training_contract_sha256: str,
    search_contract_sha256: str,
    directed_case_ids: Sequence[str],
    metric_ids: Sequence[str],
    u0_fixed_epoch: int,
    controller_fixed_epoch: int,
    bootstrap_policy: str,
    bootstrap_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen protocol without selecting or encoding a Dice success threshold."""
    cases = [require_nonempty(value, "directed_case_id") for value in directed_case_ids]
    if not cases or len(cases) != len(set(cases)):
        raise RuntimeError("directed_case_ids must be a non-empty unique ordered inventory")
    metrics = [require_nonempty(value, "metric_id") for value in metric_ids]
    if not metrics or len(metrics) != len(set(metrics)):
        raise RuntimeError("metric_ids must be a non-empty unique ordered inventory")
    bootstrap = _validate_bootstrap(bootstrap_policy, bootstrap_parameters)
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": require_git_sha(git_head, "protocol git_head"),
        "data_contract_sha256": require_sha256(data_contract_sha256, "data contract"),
        "u0_training_contract_sha256": require_sha256(u0_training_contract_sha256, "U0 training contract"),
        "controller_training_contract_sha256": require_sha256(
            controller_training_contract_sha256, "controller training contract"
        ),
        "search_contract_sha256": require_sha256(search_contract_sha256, "search contract"),
        "base_seeds": list(BASE_SEEDS),
        "directed_case_ids": cases,
        "checkpoint_selection_policy": CHECKPOINT_SELECTION_POLICY,
        "u0_fixed_epoch": require_int(u0_fixed_epoch, "U0 fixed epoch", minimum=1),
        "controller_fixed_epoch": require_int(controller_fixed_epoch, "controller fixed epoch", minimum=1),
        "bootstrap": {"policy": bootstrap_policy, "parameters": bootstrap},
        "context_channels": list(CONTEXT_CHANNELS),
        "search_channels": list(SEARCH_CHANNELS),
        "variants": _variant_payloads(),
        "planned_contrasts": _contrast_payloads(),
        "metric_ids": metrics,
        "label_policy": {
            "label_free_phases": ["training", "decision"],
            "evaluation_requires_decision_barrier": True,
            "heldout_test_authorized": False,
        },
        "expected_inventory": {
            "u0_checkpoints": len(BASE_SEEDS),
            "controller_checkpoints": len(BASE_SEEDS) * len(CONTROLLER_VARIANT_IDS),
            "decision_records": len(BASE_SEEDS) * len(VARIANT_IDS) * len(cases),
        },
    }
    validate_protocol_contract(payload)
    return payload


def validate_protocol_contract(payload: Mapping[str, Any]) -> None:
    fields = frozenset(
        {
            "schema",
            "protocol_id",
            "git_head",
            "data_contract_sha256",
            "u0_training_contract_sha256",
            "controller_training_contract_sha256",
            "search_contract_sha256",
            "base_seeds",
            "directed_case_ids",
            "checkpoint_selection_policy",
            "u0_fixed_epoch",
            "controller_fixed_epoch",
            "bootstrap",
            "context_channels",
            "search_channels",
            "variants",
            "planned_contrasts",
            "metric_ids",
            "label_policy",
            "expected_inventory",
        }
    )
    require_exact_fields(payload, fields, "Stage 5 protocol")
    if payload["schema"] != PROTOCOL_SCHEMA or payload["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Stage 5 protocol identity changed")
    require_git_sha(payload["git_head"], "protocol git_head")
    for key in (
        "data_contract_sha256",
        "u0_training_contract_sha256",
        "controller_training_contract_sha256",
        "search_contract_sha256",
    ):
        require_sha256(payload[key], key)
    if payload["base_seeds"] != list(BASE_SEEDS):
        raise RuntimeError("Stage 5 base seeds changed")
    cases = payload["directed_case_ids"]
    metrics = payload["metric_ids"]
    if not isinstance(cases, list) or not cases or len(cases) != len(set(cases)):
        raise RuntimeError("Stage 5 directed-case inventory changed")
    if not isinstance(metrics, list) or not metrics or len(metrics) != len(set(metrics)):
        raise RuntimeError("Stage 5 metric inventory changed")
    if payload["checkpoint_selection_policy"] != CHECKPOINT_SELECTION_POLICY:
        raise RuntimeError("Stage 5 checkpoint selection is not fixed-epoch")
    require_int(payload["u0_fixed_epoch"], "U0 fixed epoch", minimum=1)
    require_int(payload["controller_fixed_epoch"], "controller fixed epoch", minimum=1)
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {"policy", "parameters"}:
        raise RuntimeError("Stage 5 bootstrap contract changed")
    if not isinstance(bootstrap["parameters"], Mapping):
        raise RuntimeError("Stage 5 bootstrap parameters must be an object")
    _validate_bootstrap(str(bootstrap["policy"]), bootstrap["parameters"])
    if payload["context_channels"] != list(CONTEXT_CHANNELS) or payload["search_channels"] != list(SEARCH_CHANNELS):
        raise RuntimeError("Stage 5 channel layout changed")
    if payload["variants"] != _variant_payloads():
        raise RuntimeError("Stage 5 variant definitions changed")
    if payload["planned_contrasts"] != _contrast_payloads():
        raise RuntimeError("Stage 5 planned causal contrasts changed")
    expected_label_policy = {
        "label_free_phases": ["training", "decision"],
        "evaluation_requires_decision_barrier": True,
        "heldout_test_authorized": False,
    }
    if payload["label_policy"] != expected_label_policy:
        raise RuntimeError("Stage 5 label-access policy changed")
    expected_inventory = {
        "u0_checkpoints": len(BASE_SEEDS),
        "controller_checkpoints": len(BASE_SEEDS) * len(CONTROLLER_VARIANT_IDS),
        "decision_records": len(BASE_SEEDS) * len(VARIANT_IDS) * len(cases),
    }
    if payload["expected_inventory"] != expected_inventory:
        raise RuntimeError("Stage 5 expected inventory changed")
    canonical_json_bytes(payload)


def validate_checkpoint_metadata(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    validate_protocol_contract(protocol)
    require_exact_fields(payload, CHECKPOINT_FIELDS, "Stage 5 checkpoint metadata")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise RuntimeError("Stage 5 checkpoint schema changed")
    require_nonempty(payload["checkpoint_id"], "checkpoint_id")
    variant = require_nonempty(payload["variant_id"], "variant_id")
    if variant not in VARIANT_BY_ID:
        raise RuntimeError(f"Unknown Stage 5 variant: {variant}")
    seed = require_int(payload["seed"], "checkpoint seed")
    if seed not in BASE_SEEDS:
        raise RuntimeError("Checkpoint seed is outside the frozen inventory")
    role = "U0" if variant == "U0" else "CONTROLLER"
    if payload["role"] != role:
        raise RuntimeError("Checkpoint role and variant disagree")
    expected_epoch = protocol["u0_fixed_epoch"] if variant == "U0" else protocol["controller_fixed_epoch"]
    if payload["fixed_epoch"] != expected_epoch or payload["selection_policy"] != CHECKPOINT_SELECTION_POLICY:
        raise RuntimeError("Checkpoint was not selected at the frozen endpoint")
    if payload["git_head"] != protocol["git_head"]:
        raise RuntimeError("Checkpoint Git head differs from the protocol")
    if payload["protocol_sha256"] != canonical_sha256(protocol):
        raise RuntimeError("Checkpoint protocol SHA-256 differs")
    if payload["data_contract_sha256"] != protocol["data_contract_sha256"]:
        raise RuntimeError("Checkpoint data contract differs")
    training_key = "u0_training_contract_sha256" if variant == "U0" else "controller_training_contract_sha256"
    if payload["training_contract_sha256"] != protocol[training_key]:
        raise RuntimeError("Checkpoint training contract differs")
    validate_file_record(payload["checkpoint_file"], label="checkpoint_file")
    if payload["checkpoint_file"]["root_id"] != "checkpoint_root":
        raise RuntimeError("Checkpoint file has the wrong Stage 5 root role")
    require_sha256(payload["state_dict_sha256"], "state_dict_sha256")
    require_sha256(payload["metrics_sha256"], "metrics_sha256")
    if variant == "U0":
        if (
            payload["base_checkpoint_sha256"] is not None
            or payload["initial_controller_state_sha256"] is not None
            or payload["source_contract_sha256"] is not None
            or payload["controller_parameter_count"] != 0
        ):
            raise RuntimeError("U0 checkpoint must not claim controller state")
    else:
        require_sha256(payload["base_checkpoint_sha256"], "base_checkpoint_sha256")
        require_sha256(payload["initial_controller_state_sha256"], "initial_controller_state_sha256")
        require_sha256(payload["source_contract_sha256"], "source_contract_sha256")
        require_int(payload["controller_parameter_count"], "controller_parameter_count", minimum=1)


def _checkpoint_index(
    checkpoints: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[tuple[int, str], dict[str, Any]]:
    index: dict[tuple[int, str], dict[str, Any]] = {}
    ids: set[str] = set()
    for raw in checkpoints:
        validate_checkpoint_metadata(raw, protocol)
        item = dict(raw)
        key = (int(item["seed"]), str(item["variant_id"]))
        if key in index or item["checkpoint_id"] in ids:
            raise RuntimeError("Duplicate Stage 5 checkpoint metadata")
        index[key] = item
        ids.add(str(item["checkpoint_id"]))

    counts = {int(item["controller_parameter_count"]) for item in index.values() if item["variant_id"] != "U0"}
    if len(counts) > 1:
        raise RuntimeError("Controller variants do not have an identical parameter count")
    for seed in BASE_SEEDS:
        initial = {
            item["initial_controller_state_sha256"]
            for (item_seed, variant), item in index.items()
            if item_seed == seed and variant != "U0"
        }
        if len(initial) > 1:
            raise RuntimeError(f"Controller initial states differ within seed {seed}")
        sources = {
            item["source_contract_sha256"]
            for (item_seed, variant), item in index.items()
            if item_seed == seed and variant != "U0"
        }
        if len(sources) > 1:
            raise RuntimeError(f"Controller training-source contracts differ within seed {seed}")
        base = index.get((seed, "U0"))
        for (item_seed, variant), item in index.items():
            if (
                item_seed == seed
                and variant != "U0"
                and (base is None or item["base_checkpoint_sha256"] != base["checkpoint_file"]["sha256"])
            ):
                raise RuntimeError(f"Controller {variant}/seed {seed} is bound to another U0 checkpoint")
    return index


def build_training_barrier(
    protocol: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_protocol_contract(protocol)
    index = _checkpoint_index(checkpoints, protocol)
    expected = {(seed, variant) for seed in BASE_SEEDS for variant in VARIANT_IDS}
    observed = set(index)
    complete = observed == expected
    ordered = [index[key] for key in sorted(index)]
    return {
        "schema": TRAINING_BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": canonical_sha256(protocol),
        "root_roles": sorted(ROOT_ROLES["training"]),
        "labels_loaded": False,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "expected_checkpoints": len(expected),
        "observed_checkpoints": len(observed),
        "checkpoints": ordered,
        "checkpoint_metadata_sha256": {item["checkpoint_id"]: canonical_sha256(item) for item in ordered},
    }


def validate_training_barrier(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """Re-derive the barrier from its own inventory and demand byte-equal content.

    Every envelope field — schema, protocol binding, root roles, label flag, status and
    the digest map — is produced by :func:`build_training_barrier`, so the equality below
    subsumes a field-by-field pass; checking them separately first would only duplicate it.
    """
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise RuntimeError("Training barrier checkpoint inventory is invalid")
    if dict(payload) != build_training_barrier(protocol, checkpoints):
        raise RuntimeError("Training barrier content or computed status changed")
    if require_complete and payload["status"] != "COMPLETE":
        raise RuntimeError("Stage 5 training barrier is incomplete")


def _validate_decision_record(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    checkpoint_map: Mapping[tuple[int, str], Mapping[str, Any]],
) -> None:
    require_exact_fields(payload, DECISION_RECORD_FIELDS, "Stage 5 decision record")
    if payload["schema"] != DECISION_RECORD_SCHEMA:
        raise RuntimeError("Stage 5 decision record schema changed")
    case_id = require_nonempty(payload["case_id"], "decision case_id")
    seed = require_int(payload["seed"], "decision seed")
    variant = require_nonempty(payload["variant_id"], "decision variant")
    if case_id not in protocol["directed_case_ids"] or (seed, variant) not in checkpoint_map:
        raise RuntimeError("Decision record is outside the frozen inventory")
    expected_id = f"{case_id}__S{seed}__{variant}"
    if payload["decision_id"] != expected_id:
        raise RuntimeError("Decision identifier changed")
    if payload["checkpoint_sha256"] != checkpoint_map[(seed, variant)]["checkpoint_file"]["sha256"]:
        raise RuntimeError("Decision record is bound to another checkpoint")
    for key in ("certified_source_field", "requested_field", "candidate_field", "returned_field"):
        validate_file_record(payload[key], field=True, label=key)
    for key in ("requested", "candidate", "returned"):
        validate_save_reload_attestation(
            payload[f"{key}_save_reload"],
            payload[f"{key}_field"],
            label=f"{key}_save_reload",
        )
    validate_file_record(payload["exact_report"], label="exact_report")
    if payload["certified_source_field"]["root_id"] != "source_field_root":
        raise RuntimeError("Certified source field has the wrong Stage 5 root role")
    if payload["exact_report"]["root_id"] != "decision_output_root":
        raise RuntimeError("Decision exact report has the wrong Stage 5 root role")
    candidate_status = require_nonempty(payload["candidate_exact_status"], "candidate_exact_status")
    candidate_certified = payload["candidate_exact_certified"]
    if not isinstance(candidate_certified, bool):
        raise RuntimeError("candidate_exact_certified must be boolean")
    if (candidate_status == "CERTIFIED") != candidate_certified:
        raise RuntimeError("Candidate exact status and certified flag disagree")
    if payload["returned_exact_status"] != "CERTIFIED" or payload["returned_certified"] is not True:
        raise RuntimeError("Every returned Stage 5 field must be exactly certified after reload")
    status = payload["transaction_status"]
    if status not in TRANSACTION_STATUSES:
        raise RuntimeError("Unknown Stage 5 transaction status")
    source = payload["certified_source_field"]
    candidate = payload["candidate_field"]
    returned = payload["returned_field"]
    rollback_equal = payload["rollback_source_sha256_equal"]
    if not isinstance(rollback_equal, bool):
        raise RuntimeError("rollback_source_sha256_equal must be boolean")
    if status == "ACCEPTED":
        if (
            not candidate_certified
            or rollback_equal is not False
            or not _same_field_bytes(returned, candidate)
            or any(
                payload[key]["root_id"] != "decision_output_root"
                for key in ("requested_field", "candidate_field", "returned_field")
            )
        ):
            raise RuntimeError("Accepted Stage 5 transaction must return the exactly certified candidate bytes")
    elif status == "ROLLED_BACK":
        if (
            candidate_certified
            or rollback_equal is not True
            or not _same_field_bytes(returned, source)
            or payload["requested_field"]["root_id"] != "decision_output_root"
            or payload["candidate_field"]["root_id"] != "decision_output_root"
            or payload["returned_field"]["root_id"] != "source_field_root"
        ):
            raise RuntimeError("Rolled-back Stage 5 transaction must return source bytes with equal SHA-256")
    else:
        if (
            not candidate_certified
            or rollback_equal is not False
            or not _same_field_bytes(candidate, source)
            or not _same_field_bytes(returned, source)
            or any(
                payload[key]["root_id"] != "source_field_root"
                for key in ("requested_field", "candidate_field", "returned_field")
            )
        ):
            raise RuntimeError("Baseline/degraded identity record must return its exactly certified source bytes")
    if variant == "U0" and status not in {"BASELINE_CERTIFIED", "CERTIFIED_DEGRADED_IDENTITY"}:
        raise RuntimeError("U0 decision record has a controller transaction status")
    if variant != "U0" and status == "BASELINE_CERTIFIED":
        raise RuntimeError("Controller decision record cannot claim the U0 baseline status")
    if payload["labels_loaded"] is not False:
        raise RuntimeError("Decision record accessed labels before the barrier")
    require_finite(payload["runtime_seconds"], "runtime_seconds", minimum=0.0)
    require_int(payload["peak_memory_bytes"], "peak_memory_bytes")
    requested_rms = require_finite(payload["requested_delta_rms"], "requested_delta_rms", minimum=0.0)
    candidate_rms = require_finite(payload["candidate_delta_rms"], "candidate_delta_rms", minimum=0.0)
    returned_rms = require_finite(payload["returned_delta_rms"], "returned_delta_rms", minimum=0.0)
    candidate_ratio = require_finite(
        payload["candidate_retained_ratio"], "candidate_retained_ratio", minimum=0.0, allow_none=True
    )
    returned_ratio = require_finite(
        payload["returned_retained_ratio"], "returned_retained_ratio", minimum=0.0, allow_none=True
    )
    if requested_rms == 0.0:
        if any(value != 0.0 for value in (candidate_rms, returned_rms)) or any(
            value is not None for value in (candidate_ratio, returned_ratio)
        ):
            raise RuntimeError("Zero-request decision has inconsistent retention diagnostics")
    else:
        if candidate_ratio is None or returned_ratio is None:
            raise RuntimeError("Non-zero Stage 5 request requires explicit retention ratios")
        tolerance = 1e-6 * max(1.0, requested_rms, candidate_rms, returned_rms)
        if abs(candidate_rms / requested_rms - candidate_ratio) > tolerance:
            raise RuntimeError("Candidate retention ratio is arithmetically inconsistent")
        if abs(returned_rms / requested_rms - returned_ratio) > tolerance:
            raise RuntimeError("Returned retention ratio is arithmetically inconsistent")
        if status == "ROLLED_BACK" and (returned_rms != 0.0 or returned_ratio != 0.0):
            raise RuntimeError("Rolled-back Stage 5 request must retain zero applied amplitude")
    require_sha256(payload["execution_sha256"], "decision execution_sha256")


def _decision_index(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    training_barrier: Mapping[str, Any],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    validate_training_barrier(training_barrier, protocol, require_complete=True)
    checkpoints = _checkpoint_index(training_barrier["checkpoints"], protocol)
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw in records:
        _validate_decision_record(raw, protocol, checkpoints)
        item = dict(raw)
        key = (str(item["case_id"]), int(item["seed"]), str(item["variant_id"]))
        if key in index:
            raise RuntimeError("Duplicate Stage 5 decision record")
        index[key] = item

    for case_id in protocol["directed_case_ids"]:
        for seed in BASE_SEEDS:
            base = index.get((case_id, seed, "U0"))
            if base is None:
                continue
            source = base["certified_source_field"]
            degraded = base["transaction_status"] == "CERTIFIED_DEGRADED_IDENTITY"
            for variant in CONTROLLER_VARIANT_IDS:
                item = index.get((case_id, seed, variant))
                if item is None:
                    continue
                if item["certified_source_field"] != source:
                    raise RuntimeError("Variants of one seed/case do not share certified source bytes")
                if degraded and item["transaction_status"] != "CERTIFIED_DEGRADED_IDENTITY":
                    raise RuntimeError("A degraded U0 source must block every controller variant")
    return index


def build_decision_barrier(
    protocol: Mapping[str, Any],
    training_barrier: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = _decision_index(records, protocol, training_barrier)
    expected = {
        (case_id, seed, variant)
        for case_id in protocol["directed_case_ids"]
        for seed in BASE_SEEDS
        for variant in VARIANT_IDS
    }
    complete = set(index) == expected
    ordered = [index[key] for key in sorted(index)]
    return {
        "schema": DECISION_BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": canonical_sha256(protocol),
        "training_barrier_sha256": canonical_sha256(training_barrier),
        "root_roles": sorted(ROOT_ROLES["decision"]),
        "labels_loaded": False,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "expected_decisions": len(expected),
        "observed_decisions": len(index),
        "records": ordered,
        "decision_record_sha256": {item["decision_id"]: canonical_sha256(item) for item in ordered},
    }


def validate_decision_barrier(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    training_barrier: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """Re-derive the barrier from its own records; see :func:`validate_training_barrier`."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Decision barrier record inventory is invalid")
    if dict(payload) != build_decision_barrier(protocol, training_barrier, records):
        raise RuntimeError("Decision barrier content or computed status changed")
    if require_complete and payload["status"] != "COMPLETE":
        raise RuntimeError("Stage 5 decision barrier is incomplete")


def _validate_evaluation_record(payload: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    require_exact_fields(payload, EVALUATION_RECORD_FIELDS, "Stage 5 evaluation record")
    if payload["schema"] != EVALUATION_RECORD_SCHEMA:
        raise RuntimeError("Stage 5 evaluation record schema changed")
    decision_id = require_nonempty(payload["decision_id"], "evaluation decision_id")
    if payload["evaluation_id"] != decision_id:
        raise RuntimeError("Stage 5 evaluation identifier changed")
    if payload["decision_record_sha256"] != canonical_sha256(decision):
        raise RuntimeError("Evaluation record is bound to another decision record")
    if payload["returned_field_sha256"] != decision["returned_field"]["sha256"]:
        raise RuntimeError("Evaluation record used another returned field")
    require_sha256(payload["label_source_sha256"], "label_source_sha256")
    validate_file_record(payload["metrics_file"], label="metrics_file")
    if payload["metrics_file"]["root_id"] != "evaluation_output_root":
        raise RuntimeError("Evaluation metrics have the wrong Stage 5 root role")
    if payload["labels_loaded_after_decision_barrier"] is not True:
        raise RuntimeError("Evaluation did not attest post-barrier label access")
    if payload["heldout_test_accessed"] is not False:
        raise RuntimeError("Stage 5 evaluation accessed an unauthorized held-out test")
    require_sha256(payload["execution_sha256"], "evaluation execution_sha256")


def build_evaluation_barrier(
    protocol: Mapping[str, Any],
    training_barrier: Mapping[str, Any],
    decision_barrier: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_decision_barrier(decision_barrier, protocol, training_barrier, require_complete=True)
    decisions = {item["decision_id"]: item for item in decision_barrier["records"]}
    index: dict[str, dict[str, Any]] = {}
    for raw in records:
        item = dict(raw)
        decision_id = str(item.get("decision_id", ""))
        if decision_id not in decisions:
            raise RuntimeError("Evaluation record is outside the decision barrier")
        _validate_evaluation_record(item, decisions[decision_id])
        if decision_id in index:
            raise RuntimeError("Duplicate Stage 5 evaluation record")
        index[decision_id] = item
    complete = set(index) == set(decisions)
    ordered = [index[key] for key in sorted(index)]
    return {
        "schema": EVALUATION_BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": canonical_sha256(protocol),
        "decision_barrier_sha256": canonical_sha256(decision_barrier),
        "root_roles": sorted(ROOT_ROLES["evaluation"]),
        "labels_loaded_after_decision_barrier": True,
        "heldout_test_accessed": False,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "expected_evaluations": len(decisions),
        "observed_evaluations": len(index),
        "records": ordered,
        "evaluation_record_sha256": {item["evaluation_id"]: canonical_sha256(item) for item in ordered},
    }


def validate_evaluation_barrier(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    training_barrier: Mapping[str, Any],
    decision_barrier: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """Re-derive the barrier from its own records; see :func:`validate_training_barrier`."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Evaluation barrier record inventory is invalid")
    if dict(payload) != build_evaluation_barrier(protocol, training_barrier, decision_barrier, records):
        raise RuntimeError("Evaluation barrier content or computed status changed")
    if require_complete and payload["status"] != "COMPLETE":
        raise RuntimeError("Stage 5 evaluation barrier is incomplete")


__all__ = [
    "BASE_SEEDS",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SELECTION_POLICY",
    "CONTEXT_CHANNELS",
    "CONTROLLER_VARIANT_IDS",
    "DECISION_RECORD_SCHEMA",
    "EVALUATION_RECORD_SCHEMA",
    "PLANNED_CONTRASTS",
    "PROTOCOL_ID",
    "SEARCH_CHANNELS",
    "VARIANT_IDS",
    "VARIANT_SPECS",
    "ContrastSpec",
    "build_decision_barrier",
    "build_evaluation_barrier",
    "build_protocol_contract",
    "build_training_barrier",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_json_object",
    "validate_checkpoint_metadata",
    "validate_decision_barrier",
    "validate_evaluation_barrier",
    "validate_file_record",
    "validate_protocol_contract",
    "validate_root_access",
    "validate_save_reload_attestation",
    "validate_training_barrier",
    "write_immutable_json",
]
