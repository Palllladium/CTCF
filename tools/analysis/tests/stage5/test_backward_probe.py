from __future__ import annotations

import copy
import json
import unittest
from unittest import mock

import torch

from tools.analysis.stage5 import backward_probe as diagnostic


class DiagnosticProbeTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)

    def loss_fn(self, fp32):
        output = self.model(torch.ones(1, 1))
        if not fp32:
            output = output.half().float()
        return (output * 2).sum(), {"value": 2.0}

    def test_scale_sensitive_backward_and_fp32_reference_without_updates(self):
        state = copy.deepcopy(self.model.state_dict())
        optimizer = copy.deepcopy(self.optimizer.state_dict())
        trials = []
        with mock.patch.object(self.optimizer, "step", side_effect=AssertionError("Must not update")):
            for scale, fp32 in ((65536.0, False), (1.0, False), (1.0, True)):
                trials.append(
                    diagnostic.probe(
                        self.model,
                        self.optimizer,
                        self.loss_fn,
                        device_type="cpu",
                        scale=scale,
                        fp32=fp32,
                    )
                )
        self.assertEqual([t["status"] for t in trials], ["NONFINITE_GRADIENTS", "FINITE", "FINITE"])
        self.assertIsNotNone(trials[0]["first_observed_nonfinite"])
        self.assertEqual(trials[0]["first_observed_nonfinite"]["stage"], "scaled_backward")
        self.assertTrue(torch.equal(self.model.weight, state["weight"]))
        self.assertEqual(self.optimizer.state_dict(), optimizer)
        self.assertEqual(len(self.model._forward_hooks), 0)
        self.assertEqual(diagnostic.classify_probes(trials), "SCALE_SENSITIVE_OVERFLOW_ON_THIS_PAIR")
        json.dumps(trials, allow_nan=False)

    def test_nonfinite_loss_is_not_misreported_as_gradient_overflow(self):
        result = diagnostic.probe(
            self.model,
            self.optimizer,
            lambda _fp32: (self.model(torch.ones(1, 1)).sum() * float("nan"), {"loss": float("nan")}),
            device_type="cpu",
            scale=1.0,
            fp32=False,
        )
        self.assertEqual(result["status"], "NONFINITE_LOSS")
        self.assertIsNone(result["loss"])
        json.dumps(result, allow_nan=False)

    def test_probe_exception_is_recorded_and_hooks_removed(self):
        def fail(_fp32):
            self.model(torch.ones(1, 1))
            raise FloatingPointError("bad head")

        result = diagnostic.probe(
            self.model,
            self.optimizer,
            fail,
            device_type="cpu",
            scale=1.0,
            fp32=False,
        )
        self.assertEqual(result["status"], "PROBE_EXCEPTION")
        self.assertIn("bad head", result["error"])
        self.assertEqual(len(self.model._forward_hooks), 0)

    def test_tensor_stats_are_json_safe(self):
        result = diagnostic.tensor_stats(torch.tensor([1.0, float("inf"), float("nan"), -3.0]))
        self.assertEqual(result["nonfinite"], 2)
        self.assertEqual(result["max_finite_abs"], 3)
        json.dumps(result, allow_nan=False)

    def test_classification_never_calls_one_finite_probe_a_training_success(self):
        trials = [{"status": "NONFINITE_GRADIENTS"}, {"status": "NONFINITE_GRADIENTS"}, {"status": "FINITE"}]
        self.assertEqual(diagnostic.classify_probes(trials), "FP16_PATH_FAILURE_NOT_RESOLVED_BY_TESTED_SCALES")
        trials[0]["status"] = "FINITE"
        self.assertEqual(diagnostic.classify_probes(trials), "FAILURE_NOT_REPRODUCED_IN_INSTRUMENTED_PROBE")
