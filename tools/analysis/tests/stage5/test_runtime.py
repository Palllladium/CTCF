from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import experiments.stage5.config as stage5_config
import experiments.stage5.runtime as runtime
from experiments.stage5.losses import ControllerLossConfig
from experiments.stage5.safety import prepare_initial_field
from models.CTCF.controller import Stage5SpatialController
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.stage5.primitives import readable_json_bytes

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GIT_SHA = "d" * 40


class RuntimeConfigTest(unittest.TestCase):
    def test_frozen_horizons_and_mamba_contract(self) -> None:
        self.assertEqual(runtime.U0TrainingConfig().fixed_epoch, 500)
        self.assertEqual(runtime.ControllerTrainingConfig().fixed_epoch, 100)
        with self.assertRaisesRegex(ValueError, "500-epoch"):
            runtime.U0TrainingConfig(fixed_epoch=499)
        with self.assertRaisesRegex(ValueError, "Mamba"):
            runtime.U0TrainingConfig(config_key="CTCF-CascadeA-VM")
        with self.assertRaisesRegex(ValueError, "100-epoch"):
            runtime.ControllerTrainingConfig(fixed_epoch=99)
        with self.assertRaisesRegex(ValueError, "500-epoch"):
            runtime.U0TrainingConfig(fixed_epoch=500.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Mamba"):
            runtime.U0TrainingConfig(time_steps=6.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "100-epoch"):
            runtime.ControllerTrainingConfig(fixed_epoch=100.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "finite"):
            runtime.ControllerTrainingConfig(loss=ControllerLossConfig(ncc_weight=float("nan")))

    def test_runner_namespace_is_label_free_and_complete(self) -> None:
        args = runtime._u0_args(runtime.U0TrainingConfig(), seed=7)
        required = {
            "config",
            "dare_beta",
            "disable_l1",
            "disable_l3",
            "ds",
            "elastic_lam",
            "elastic_mu",
            "ema_decay",
            "ema_lambda",
            "icon_mode",
            "jac_mode",
            "l1_from_start",
            "log_tri_gradnorm",
            "lr",
            "max_epoch",
            "reg_mode",
            "schedule_max_epoch",
            "time_steps",
            "tri_pen_mode",
            "tri_pen_reduce",
            "use_checkpoint",
            "w_dice",
            "w_icon",
            "w_jac",
            "w_ncc",
            "w_reg",
        }
        self.assertTrue(required.issubset(vars(args)))
        self.assertEqual(args.ds, "OASIS")
        self.assertEqual(args.w_dice, 0.0)
        source = inspect.getsource(runtime)
        self.assertNotIn("Stage5OasisEvaluationLabelStore", source)
        self.assertNotIn("load_label", source)

    def test_execution_contract_does_not_promise_impossible_grid_sample_determinism(self) -> None:
        contract = runtime._execution_determinism_contract()
        self.assertIs(contract["deterministic_algorithms"], False)
        self.assertIn("grid_sample", contract["known_limitation"])

    def test_h100_smoke_requires_real_parameter_updates_and_post_step_delta(self) -> None:
        # The smoke run is split across an orchestrator and its per-variant step; the
        # contract below is a property of the pipeline, not of either function alone.
        source = inspect.getsource(runtime.smoke_stage5_runtime) + inspect.getsource(runtime._smoke_controller_step)
        self.assertIn("parameters_after == parameters_before", source)
        self.assertIn("u0_parameters_after == u0_parameters_before", source)
        self.assertIn("requested_delta_rms", source)
        self.assertIn('variant == "F24P" and requested_delta_rms <= 0.0', source)
        self.assertIn("commit_controller_delta", source)
        self.assertGreaterEqual(source.count("load_training_state"), 2)
        self.assertGreaterEqual(source.count("_write_checkpoint_with_sidecar"), 2)

    def test_cuda_entrypoints_fail_before_touching_data(self) -> None:
        common = {
            "data_contract": Path("missing-contract"),
            "image_root": Path("missing-images"),
            "device": torch.device("cpu"),
        }
        with self.assertRaisesRegex(RuntimeError, "visible CUDA GPU"):
            runtime.train_u0(
                **common,
                output_root=Path("unused"),
                seed=0,
                git_head=GIT_SHA,
                protocol_sha256=SHA_A,
                training_contract_sha256=SHA_B,
                config=runtime.U0TrainingConfig(),
            )
        with self.assertRaisesRegex(RuntimeError, "visible CUDA GPU"):
            runtime.materialize_source_fields(
                **common,
                output_root=Path("unused"),
                u0_checkpoint=Path("missing.pth"),
                seed=0,
                protocol_sha256=SHA_A,
                u0_training_contract_sha256=SHA_B,
                u0_config=runtime.U0TrainingConfig(),
                bootstrap_policy="collar_repair",
                shard_index=0,
                num_shards=1,
            )
        with self.assertRaisesRegex(RuntimeError, "visible CUDA GPU"):
            runtime.train_controller(
                **common,
                output_root=Path("unused"),
                base_checkpoint=Path("missing-u0.pth"),
                initial_controller=Path("missing-controller.pth"),
                variant="F0",
                seed=0,
                git_head=GIT_SHA,
                protocol_sha256=SHA_A,
                u0_training_contract_sha256=SHA_B,
                training_contract_sha256=SHA_C,
                bootstrap_policy="collar_repair",
                config=runtime.ControllerTrainingConfig(),
            )
        with self.assertRaisesRegex(RuntimeError, "visible CUDA GPU"):
            runtime.smoke_stage5_runtime(
                **common,
                output_root=Path("unused"),
                seed=0,
                git_head=GIT_SHA,
                protocol_sha256=SHA_A,
                u0_training_contract_sha256=SHA_B,
                controller_training_contract_sha256=SHA_C,
                bootstrap_policy="collar_repair",
                u0_config=runtime.U0TrainingConfig(),
                controller_config=runtime.ControllerTrainingConfig(),
            )


class PairScheduleTest(unittest.TestCase):
    def test_u0_schedule_is_deterministic_bijective_derangement(self) -> None:
        subjects = tuple(f"S{index:03d}" for index in range(294))
        first = runtime.epoch_pair_schedule(subjects, seed=0, epoch=0)
        self.assertEqual(first, runtime.epoch_pair_schedule(subjects, seed=0, epoch=0))
        self.assertNotEqual(first, runtime.epoch_pair_schedule(subjects, seed=0, epoch=1))
        self.assertEqual({moving for moving, _ in first}, set(subjects))
        self.assertEqual({fixed for _, fixed in first}, set(subjects))
        self.assertTrue(all(moving != fixed for moving, fixed in first))
        with self.assertRaisesRegex(ValueError, "seed"):
            runtime.epoch_pair_schedule(subjects, seed=0.0, epoch=0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "epoch"):
            runtime.epoch_pair_schedule(subjects, seed=0, epoch=0.0)  # type: ignore[arg-type]

    def test_controller_pairs_form_a_new_deterministic_perfect_matching_each_epoch(self) -> None:
        subjects = tuple(f"S{index:03d}" for index in range(294))
        pairs = runtime.controller_epoch_pairs(subjects, seed=0, epoch=0)
        self.assertEqual(len(pairs), 147)
        flattened = [subject for pair in pairs for subject in (pair["subject_a"], pair["subject_b"])]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(subjects))
        self.assertEqual(pairs, runtime.controller_epoch_pairs(tuple(reversed(subjects)), seed=0, epoch=0))
        self.assertNotEqual(pairs, runtime.controller_epoch_pairs(subjects, seed=0, epoch=1))
        self.assertNotEqual(pairs, runtime.controller_epoch_pairs(subjects, seed=1, epoch=0))


class RuntimeIntegrityTest(unittest.TestCase):
    def test_checkpoint_sidecar_detects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            digest = runtime._write_checkpoint_with_sidecar(path, {"value": torch.tensor([1.0])})
            self.assertEqual(runtime._verify_checkpoint_sidecar(path), digest)
            with path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runtime._verify_checkpoint_sidecar(path)

    def test_checkpoint_pair_recovers_a_complete_staged_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            runtime._write_checkpoint_with_sidecar(path, {"generation": 1})
            staging = runtime._checkpoint_generations(path)[1]
            digest = runtime.atomic_torch_save(staging, {"generation": 2})
            atomic_write_json(
                runtime._checkpoint_sidecar_path(staging),
                runtime._checkpoint_sidecar_record(path, staging, digest),
            )
            backup = runtime._checkpoint_generations(path)[2]
            os.replace(path, backup)
            os.replace(runtime._checkpoint_sidecar_path(path), runtime._checkpoint_sidecar_path(backup))
            self.assertEqual(runtime._verify_checkpoint_sidecar(path), digest)
            self.assertEqual(torch.load(path, weights_only=False)["generation"], 2)

    def test_common_controller_initial_state_is_equal_per_seed(self) -> None:
        config = runtime.ControllerTrainingConfig()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pth"
            second = root / "second.pth"
            first_sha = runtime.initialize_controller_state(seed=2, config=config, output_path=first)
            second_sha = runtime.initialize_controller_state(seed=2, config=config, output_path=second)
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(
                runtime.initialize_controller_state(seed=2, config=config, output_path=first),
                first_sha,
            )
            model = Stage5SpatialController(
                width=config.width,
                free_residual_limit_voxels=config.free_residual_limit_voxels,
            )
            self.assertEqual(
                runtime._load_initial_controller(first, model, seed=2, config=config),
                first_sha,
            )
            with first.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runtime._load_initial_controller(first, model, seed=2, config=config)

    def test_checkpoint_metadata_and_metrics_recovery_are_fail_closed(self) -> None:
        config = runtime.U0TrainingConfig()
        metrics = {
            "schema": "ctcf-stage5-u0-metrics-v1",
            "role": "U0",
            "variant": "U0",
            "seed": 0,
            "label_metrics_present": False,
            "selection_policy": "FIXED_EPOCH_NOT_LABEL_SELECTED",
            "epochs": [{"epoch": 1, "pair_schedule_sha256": SHA_A, "pairs": 294, "metrics": {"ncc": 1.0}}],
        }
        payload = {
            "fixed_epoch": 500,
            "git_head": GIT_SHA,
            "epoch_completed": 1,
            "pair_schedule_sha256": SHA_A,
            "metrics_sha256": __import__("hashlib").sha256(readable_json_bytes(metrics)).hexdigest(),
            "base_checkpoint_sha256": None,
            "initial_controller_state_sha256": None,
            "source_contract_sha256": None,
        }
        runtime._attach_runtime_checkpoint_metadata(payload, config=config, metrics_payload=metrics)
        observed = runtime._validate_runtime_checkpoint_metadata(
            payload,
            role="U0",
            variant="U0",
            seed=0,
            config=config,
            expected_git_head=GIT_SHA,
            expected_base_checkpoint_sha256=None,
            expected_initial_controller_state_sha256=None,
            expected_source_contract_sha256=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.json"
            metrics_path.write_text("stale\n", encoding="utf-8")
            runtime._restore_authoritative_metrics(metrics_path, payload, observed)
            self.assertEqual(sha256_file(metrics_path), payload["metrics_sha256"])
        payload["metrics_payload"]["epochs"][0]["metrics"]["ncc"] = 2.0
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            runtime._validate_runtime_checkpoint_metadata(
                payload,
                role="U0",
                variant="U0",
                seed=0,
                config=config,
                expected_git_head=GIT_SHA,
                expected_base_checkpoint_sha256=None,
                expected_initial_controller_state_sha256=None,
                expected_source_contract_sha256=None,
            )

    def test_controller_checkpoint_must_bind_training_source_contract(self) -> None:
        config = runtime.ControllerTrainingConfig()
        metrics = {
            "schema": "ctcf-stage5-controller-metrics-v1",
            "role": "CONTROLLER",
            "variant": "F0",
            "seed": 0,
            "label_metrics_present": False,
            "selection_policy": "FIXED_EPOCH_NOT_LABEL_SELECTED",
            "epochs": [],
        }
        payload = {
            "fixed_epoch": 100,
            "git_head": GIT_SHA,
            "epoch_completed": 0,
            "pair_schedule_sha256": SHA_A,
            "metrics_sha256": __import__("hashlib").sha256(readable_json_bytes(metrics)).hexdigest(),
            "base_checkpoint_sha256": SHA_A,
            "initial_controller_state_sha256": SHA_B,
            "source_contract_sha256": SHA_C,
        }
        runtime._attach_runtime_checkpoint_metadata(payload, config=config, metrics_payload=metrics)
        with self.assertRaisesRegex(RuntimeError, "training-source contract"):
            runtime._validate_runtime_checkpoint_metadata(
                payload,
                role="CONTROLLER",
                variant="F0",
                seed=0,
                config=config,
                expected_git_head=GIT_SHA,
                expected_base_checkpoint_sha256=SHA_A,
                expected_initial_controller_state_sha256=SHA_B,
                expected_source_contract_sha256="e" * 64,
            )


class CertifiedSourceTest(unittest.TestCase):
    @staticmethod
    def _create_source(root: Path, *, seed: int, case: dict[str, str], base_sha: str) -> Path:
        case_root = root / f"seed_{seed}" / case["case_id"]
        artifact = prepare_initial_field(torch.zeros(1, 3, 16, 16, 16), case_root, policy="identity")
        report_path = case_root / "initial_report.json"
        atomic_write_json(
            report_path,
            {
                "schema": "ctcf-stage5-certified-source-v1",
                "seed": seed,
                "case": case,
                "u0_checkpoint_sha256": base_sha,
                "bootstrap_policy": "identity",
                "phi_sha256": artifact.phi_sha256,
                "psi_sha256": artifact.psi_sha256,
                "report": artifact.report,
            },
        )
        return report_path

    def test_inventory_authenticates_and_detects_runtime_drift(self) -> None:
        case = {
            "case_id": "S5TRAIN-001-D0",
            "pair_id": "S5TRAIN-001",
            "moving_subject_id": "A",
            "fixed_subject_id": "B",
            "split": "training",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = self._create_source(root, seed=0, case=case, base_sha=SHA_A)
            store = runtime._CertifiedSourceStore(
                root,
                seed=0,
                u0_checkpoint_sha256=SHA_A,
                image_shape=(16, 16, 16),
                bootstrap_policy="identity",
            )
            digest = store.inventory_sha256((case,))
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            psi = store.load(case, torch.device("cpu"))
            self.assertEqual(psi.dtype, torch.float32)
            self.assertGreater(int(torch.count_nonzero(psi)), 0)
            report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                store.load(case, torch.device("cpu"))

    def test_inventory_rejects_field_byte_tampering(self) -> None:
        case = {
            "case_id": "S5TRAIN-001-D0",
            "pair_id": "S5TRAIN-001",
            "moving_subject_id": "A",
            "fixed_subject_id": "B",
            "split": "training",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_source(root, seed=0, case=case, base_sha=SHA_A)
            psi_path = root / "seed_0" / case["case_id"] / "initial_psi.npz"
            with psi_path.open("ab") as stream:
                stream.write(b"tamper")
            store = runtime._CertifiedSourceStore(
                root,
                seed=0,
                u0_checkpoint_sha256=SHA_A,
                image_shape=(16, 16, 16),
                bootstrap_policy="identity",
            )
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                store.verify_case(case)

    def test_identity_policy_rejects_an_arbitrary_exact_psi(self) -> None:
        from tools.analysis.search.transaction import save_flow_npz_atomic
        from utils.cert_exact import certify_flow_exact

        case = {
            "case_id": "S5TRAIN-001-D0",
            "pair_id": "S5TRAIN-001",
            "moving_subject_id": "A",
            "fixed_subject_id": "B",
            "split": "training",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = self._create_source(root, seed=0, case=case, base_sha=SHA_A)
            psi_path = report_path.with_name("initial_psi.npz")
            psi = torch.zeros(1, 3, 16, 16, 16)
            psi[:, :, 8, 8, 8] = 1e-4
            save_flow_npz_atomic(psi_path, psi)
            exact = certify_flow_exact(psi, eps="0.001")
            self.assertIs(exact["certified"], True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["psi_sha256"] = sha256_file(psi_path)
            report["report"]["psi_sha256"] = report["psi_sha256"]
            report["report"]["psi_exact"] = exact
            atomic_write_json(report_path, report)
            store = runtime._CertifiedSourceStore(
                root,
                seed=0,
                u0_checkpoint_sha256=SHA_A,
                image_shape=(16, 16, 16),
                bootstrap_policy="identity",
            )
            with self.assertRaisesRegex(RuntimeError, "authoritative identity"):
                store.verify_case(case)

    def test_inventory_rejects_another_frozen_bootstrap_policy(self) -> None:
        case = {
            "case_id": "S5TRAIN-001-D0",
            "pair_id": "S5TRAIN-001",
            "moving_subject_id": "A",
            "fixed_subject_id": "B",
            "split": "training",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_source(root, seed=0, case=case, base_sha=SHA_A)
            store = runtime._CertifiedSourceStore(
                root,
                seed=0,
                u0_checkpoint_sha256=SHA_A,
                image_shape=(16, 16, 16),
                bootstrap_policy="collar_repair",
            )
            with self.assertRaisesRegex(RuntimeError, "another frozen bootstrap"):
                store.verify_case(case)


class TrainingMechanicsTest(unittest.TestCase):
    def test_inference_features_become_ordinary_fp32_training_tensors(self) -> None:
        with torch.inference_mode():
            features = SimpleNamespace(
                controller_input=torch.zeros(1, 71, 4, 4, 4, dtype=torch.float16),
                s2=SimpleNamespace(proposal=torch.zeros(1, 3, 4, 4, 4, dtype=torch.float16)),
                s4=SimpleNamespace(proposal=torch.zeros(1, 3, 4, 4, 4, dtype=torch.float16)),
                fixed_normalized=torch.zeros(1, 1, 4, 4, 4, dtype=torch.float16),
                moving_normalized=torch.zeros(1, 1, 4, 4, 4, dtype=torch.float16),
            )
        tensors = runtime._controller_training_tensors(features)
        self.assertTrue(all(value.dtype == torch.float32 for value in tensors))
        self.assertTrue(all(not value.is_inference() for value in tensors))

    def test_u0_logs_drop_only_the_inert_dice_placeholder(self) -> None:
        self.assertEqual(runtime._sanitize_u0_logs({"all": 1.0, "dice": 0.0}), {"all": 1.0})
        with self.assertRaisesRegex(RuntimeError, "label-derived"):
            runtime._sanitize_u0_logs({"dice": 0.1})
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            runtime._sanitize_u0_logs({"all": float("nan")})

    def test_strict_scaler_detects_an_amp_skip(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler("cpu")
        scaler.scale(parameter.square()).backward()
        runtime._strict_scaler_step(scaler, optimizer, phase="test")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(parameter * torch.tensor(float("inf"))).backward()
        with self.assertRaisesRegex(FloatingPointError, "optimizer-step skip"):
            runtime._strict_scaler_step(scaler, optimizer, phase="test")

    def test_mocked_u0_run_resumes_from_checkpoint_authoritative_metrics(self) -> None:
        original_scaler = torch.amp.GradScaler

        class FakeStore:
            def __init__(self, *_args: object) -> None:
                self.runtime = SimpleNamespace(
                    contract_sha256=SHA_C,
                    split={"training": [{"subject_id": "A"}, {"subject_id": "B"}]},
                )

        class FakeRunner:
            def __init__(self, _args: object, _device: torch.device) -> None:
                self.model = torch.nn.Linear(1, 1, bias=False)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
                self.lr_policy = "ctcf"

            def train_step(self, _batch: object, _epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
                loss = self.model.weight.square().mean()
                return loss, {"all": float(loss.detach()), "ncc": 1.0, "dice": 0.0}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(stage5_config, "STAGE5_U0_FIXED_EPOCH", 2),
                mock.patch.object(runtime, "_require_cuda"),
                mock.patch.object(runtime, "Stage5OasisImageStore", FakeStore),
                mock.patch.object(runtime, "Runner", FakeRunner),
                mock.patch.object(
                    runtime,
                    "_tensor_image",
                    return_value=torch.zeros(1, 1, 2, 2, 2),
                ),
                mock.patch.object(
                    runtime.torch.amp,
                    "GradScaler",
                    side_effect=lambda *_args, **_kwargs: original_scaler("cpu"),
                ),
            ):
                config = runtime.U0TrainingConfig(fixed_epoch=2)
                checkpoint = runtime.train_u0(
                    data_contract=Path("unused"),
                    image_root=Path("unused"),
                    output_root=root,
                    seed=0,
                    device=torch.device("cuda"),
                    git_head=GIT_SHA,
                    protocol_sha256=SHA_A,
                    training_contract_sha256=SHA_B,
                    config=config,
                )
                metrics_path = root / "metrics.json"
                metrics_path.write_text("stale\n", encoding="utf-8")
                resumed = runtime.train_u0(
                    data_contract=Path("unused"),
                    image_root=Path("unused"),
                    output_root=root,
                    seed=0,
                    device=torch.device("cuda"),
                    git_head=GIT_SHA,
                    protocol_sha256=SHA_A,
                    training_contract_sha256=SHA_B,
                    config=config,
                    resume=checkpoint,
                )
                self.assertEqual(resumed, checkpoint)
                restored = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual([row["epoch"] for row in restored["epochs"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
