"""Exercise the producer/decision/evaluation joins on real small tensor artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.stage5 import runtime
from experiments.stage5.config import ControllerTrainingConfig, build_stage5_controller
from tools.analysis import run_stage5
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5 import continuation, pipeline
from tools.analysis.stage5.artifacts import file_record, load_canonical_json
from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    VARIANT_IDS,
    build_evaluation_barrier,
    build_training_barrier,
    canonical_json_bytes,
    canonical_sha256,
    write_immutable_json,
)
from tools.analysis.stage5.decision_reuse import decision_digest, verify_reusable_decision
from tools.analysis.stage5.evaluation import (
    STAGE5_EVALUATION_METRIC_IDS,
    EvaluationContext,
    build_evaluation_record,
    evaluate_returned_decision,
    write_decision_metrics,
)
from tools.analysis.stage5.protocol import bootstrap_parameters
from tools.analysis.tests.stage5.test_contracts import _all_checkpoints, _protocol


class PostTrainingEndToEndTest(unittest.TestCase):
    def test_full_matrix_decisions_recovery_evaluation_and_products(self):
        protocol = _protocol(bootstrap_policy="collar_repair", bootstrap_parameters=bootstrap_parameters())
        protocol["metric_ids"] = list(STAGE5_EVALUATION_METRIC_IDS)
        cases = [
            {"case_id": "pair00_ab", "moving_subject_id": "A", "fixed_subject_id": "B"},
            {"case_id": "pair00_ba", "moving_subject_id": "B", "fixed_subject_id": "A"},
            {"case_id": "pair01_ab", "moving_subject_id": "C", "fixed_subject_id": "D"},
            {"case_id": "pair01_ba", "moving_subject_id": "D", "fixed_subject_id": "C"},
        ]
        protocol["directed_case_ids"] = [case["case_id"] for case in cases]
        protocol["expected_inventory"]["decision_records"] = len(BASE_SEEDS) * len(VARIANT_IDS) * len(cases)
        training = build_training_barrier(protocol, _all_checkpoints(protocol))
        pairs = [
            {"pair_id": "pair00", "case_ids": [case["case_id"] for case in cases[:2]]},
            {"pair_id": "pair01", "case_ids": [case["case_id"] for case in cases[2:]]},
        ]
        shape = (18, 18, 18)
        image = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
        contract = SimpleNamespace(
            pairs={"cases": cases, "pairs": pairs}, contract_sha256=protocol["data_contract_sha256"]
        )
        store = SimpleNamespace(runtime=contract, image_shape=shape, load_image=lambda _: image)
        raw = torch.zeros((1, 3, *shape))
        raw[:, 0, 8:10, 8:10, 8:10] = 0.1
        runner = SimpleNamespace(model=lambda moving, fixed, **kwargs: (moving, raw))
        controller = build_stage5_controller(ControllerTrainingConfig()).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol/protocol.json"
            training_path = root / "barriers/training_barrier.json"
            source_root, decision_root, evaluation_root = root / "sources", root / "decisions", root / "evaluation"
            write_immutable_json(protocol_path, protocol)
            write_immutable_json(training_path, training)
            for seed in BASE_SEEDS:
                u0_sha = next(
                    item["checkpoint_file"]["sha256"]
                    for item in training["checkpoints"]
                    if item["seed"] == seed and item["variant_id"] == "U0"
                )
                with (
                    patch.object(runtime, "_require_cuda"),
                    patch.object(runtime, "Stage5OasisImageStore", return_value=store),
                    patch.object(runtime, "_verify_checkpoint_sidecar", return_value=u0_sha),
                    patch.object(runtime, "load_frozen_u0", return_value=runner),
                ):
                    runtime.materialize_source_fields(
                        data_contract=root / "data",
                        image_root=root / "images",
                        output_root=source_root,
                        u0_checkpoint=root / "u0",
                        seed=seed,
                        device=torch.device("cpu"),
                        protocol_sha256=canonical_sha256(protocol),
                        u0_training_contract_sha256=protocol["u0_training_contract_sha256"],
                        u0_config=runtime.U0TrainingConfig(),
                        bootstrap_policy="collar_repair",
                        shard_index=0,
                        num_shards=1,
                    )
                for variant in VARIANT_IDS:
                    args = dict(
                        protocol_path=protocol_path,
                        training_barrier_path=training_path,
                        data_contract_path=root / "data",
                        image_root=root / "images",
                        checkpoint_root=root / "checkpoints",
                        source_root=source_root,
                        decision_root=decision_root,
                        seed=seed,
                        variant=variant,
                        shard_index=0,
                        num_shards=1,
                        device=torch.device("cpu"),
                        execution_git_head="c" * 40,
                        continuation_sha256="d" * 64,
                    )
                    with (
                        patch.object(pipeline, "Stage5OasisImageStore", return_value=store),
                        patch.object(pipeline, "_load_controller", return_value=controller),
                    ):
                        if seed == 0 and variant == "F0":
                            # Stop precisely between publishing the exact report and its record.
                            real_write = pipeline.write_immutable_json

                            def interrupted_write(path, payload, real_write=real_write):
                                if path.parent.name == "records":
                                    raise InterruptedError("synthetic interruption")
                                return real_write(path, payload)

                            with (
                                patch.object(pipeline, "write_immutable_json", side_effect=interrupted_write),
                                self.assertRaises(InterruptedError),
                            ):
                                pipeline.materialize_decisions(**args)
                            frozen_exact = next((decision_root / "exact_reports").glob("*__F0.json")).read_bytes()
                            # Convert the interrupted journal into an authentic legacy
                            # nominal result, then resume under another evaluation HEAD.
                            journal_path = next((decision_root / "commits").glob("*__F0.json"))
                            journal = load_canonical_json(journal_path)
                            record, exact = journal["record"], journal["exact_report"]
                            old_head = "b" * 40
                            parent_path = root / "continuations" / f"{old_head}.json"
                            write_immutable_json(parent_path, {"schema": "test-parent"})
                            exact["schema"] = "ctcf-stage5-decision-exact-report-v2"
                            exact["execution"].pop("decision_safety_policy")
                            exact["execution"]["execution_git_head"] = old_head
                            exact["execution"]["continuation_sha256"] = sha256_file(parent_path)
                            exact["clip_report"] = {
                                k: v for k, v in exact["clip_report"].items() if not k.startswith("margin_")
                            }
                            record["exact_report"]["bytes"] = len(canonical_json_bytes(exact))
                            record["exact_report"]["sha256"] = canonical_sha256(exact)
                            performance = {
                                k: record[k]
                                for k in (
                                    "runtime_seconds",
                                    "peak_memory_bytes",
                                    "requested_delta_rms",
                                    "candidate_delta_rms",
                                    "returned_delta_rms",
                                    "candidate_retained_ratio",
                                    "returned_retained_ratio",
                                )
                            }
                            record["execution_sha256"] = canonical_sha256(
                                {"environment": exact["execution"], "performance": performance}
                            )
                            journal_path.write_bytes(canonical_json_bytes(journal))
                            exact_path = decision_root / "exact_reports" / journal_path.name
                            exact_path.write_bytes(canonical_json_bytes(exact))
                            frozen_exact = exact_path.read_bytes()
                            roots = {"source_field_root": source_root, "decision_output_root": decision_root}
                            verify_reusable_decision(record, exact, roots=roots, protocol=protocol, training=training)
                            # Parent provenance is validated separately from heavy bytes.
                            with (
                                patch.object(
                                    continuation, "validate_continuation", return_value={"source_inventory": []}
                                ),
                                patch.object(continuation, "verify_code_compatibility"),
                            ):
                                approved = continuation._reuse_inventory(
                                    paths={
                                        "decision_root": decision_root,
                                        "source_root": source_root,
                                        "protocol": protocol_path,
                                    },
                                    output=root / "continuations" / ("c" * 40 + ".json"),
                                    repo_root=root,
                                    execution_head="c" * 40,
                                    protocol=protocol,
                                    training=training,
                                    sources=[],
                                )
                            self.assertEqual(approved[record["decision_id"]], decision_digest(record, exact))
                            args.update(execution_git_head="c" * 40, continuation_sha256="d" * 64)
                            with self.assertRaisesRegex(RuntimeError, "another execution"):
                                pipeline.materialize_decisions(**args)
                            args["reusable_decisions"] = approved
                        pipeline.materialize_decisions(**args)
                        if seed == 0 and variant == "F0":
                            self.assertEqual(
                                next((decision_root / "exact_reports").glob("*__F0.json")).read_bytes(), frozen_exact
                            )
            roots = {"source_field_root": source_root, "decision_output_root": decision_root}
            for path in (decision_root / "records").glob("*.json"):
                run_stage5._verify_decision_artifacts(load_canonical_json(path), roots)
            barrier_path = root / "barriers/decision_barrier.json"
            decision = pipeline.freeze_decision_barrier(
                protocol_path=protocol_path,
                training_barrier_path=training_path,
                decision_root=decision_root,
                output_path=barrier_path,
            )
            context = EvaluationContext.from_barriers(
                protocol, training, decision, run_stage5._evaluation_case_inventory(contract)
            )
            label = np.ones(shape, dtype=np.uint8)
            records = []
            for decision_id, row in context.decisions.items():
                paths = {
                    name: roots[row[name]["root_id"]] / row[name]["relative_path"]
                    for name in ("returned_field", "requested_field", "candidate_field")
                }
                evaluated = evaluate_returned_decision(
                    context,
                    decision_id,
                    paths["returned_field"],
                    label,
                    label,
                    requested_field_path=paths["requested_field"],
                    candidate_field_path=paths["candidate_field"],
                )
                metrics = evaluation_root / "metrics" / f"{decision_id}.json"
                write_decision_metrics(metrics, evaluated)
                record = build_evaluation_record(
                    evaluated, file_record("evaluation_output_root", evaluation_root, metrics)
                )
                write_immutable_json(evaluation_root / "records" / f"{decision_id}.json", record)
                records.append(record)
            evaluation_barrier = build_evaluation_barrier(protocol, training, decision, records)
            self.assertEqual(evaluation_barrier["status"], "COMPLETE")
            evaluation_path = root / "barriers/evaluation_barrier.json"
            write_immutable_json(evaluation_path, evaluation_barrier)
            args = SimpleNamespace(
                evaluation_barrier=evaluation_path,
                evaluation_root=evaluation_root,
                source_root=source_root,
                decision_root=decision_root,
                output_root=evaluation_root / "products",
                device="cpu",
            )
            with patch.object(run_stage5, "_evaluation_context", return_value=(context, contract)):
                self.assertEqual(run_stage5.command_aggregate(args), 0)
            with patch.object(run_stage5, "load_stage5_runtime_contract", return_value=contract):
                run_stage5._validate_complete_compact_run(root, protocol["git_head"])
            self.assertEqual(len(records), len(BASE_SEEDS) * len(VARIANT_IDS) * len(cases))
            for path in (decision_root / "exact_reports").glob("*__A24P.json"):
                self.assertIsNotNone(load_canonical_json(path)["controller_observations"])


if __name__ == "__main__":
    unittest.main()
