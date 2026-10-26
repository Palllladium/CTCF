import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from experiments.core import helpers
from experiments.core.cli_ctcf import add_ctcf_model_args
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.precision import add_precision_arg, configure_precision, precision_autocast
from models.CTCF.configs import CONFIGS


@pytest.mark.parametrize("mode", ("fp16", "bf16", "fp32", "tf32"))
def test_precision_reaches_autocast_and_backend(mode):
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        with (
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            configure_precision(mode, torch.device("cuda"))
        assert torch.backends.cuda.matmul.allow_tf32 == (mode == "tf32")
        assert torch.backends.cudnn.allow_tf32 == (mode == "tf32")
        with patch("torch.autocast") as autocast:
            precision_autocast(mode, torch.device("cuda"))
            assert autocast.call_args.kwargs == dict(
                device_type="cuda",
                dtype=torch.bfloat16 if mode == "bf16" else torch.float16,
                enabled=mode in ("fp16", "bf16"),
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_unsupported_bf16_fails_instead_of_switching_mode():
    with (
        patch("torch.cuda.device", return_value=nullcontext()),
        patch("torch.cuda.is_bf16_supported", return_value=False),
        pytest.raises(RuntimeError, match="BF16"),
    ):
        configure_precision("bf16", torch.device("cuda"))


@pytest.mark.parametrize("config", CONFIGS)
def test_every_ctcf_preset_accepts_the_lumir_grid_without_replacing_its_levels(config, monkeypatch):
    from models.CTCF import model

    monkeypatch.setattr(model, "CTCFCascadeA", lambda cfg: SimpleNamespace(cfg=cfg))
    parser = argparse.ArgumentParser()
    add_ctcf_model_args(parser)
    add_precision_arg(parser)
    args = parser.parse_args(["--config", config, "--precision", "bf16"])
    shape = (160, 224, 192)
    built = get_model_adapter("ctcf").build(config_key=args.config, img_size=shape, dwin_size=volume_window_size(shape))
    assert built.cfg.img_size == shape
    assert built.cfg.use_level1 == CONFIGS[config].use_level1
    assert built.cfg.use_level3 == CONFIGS[config].use_level3
    assert built.cfg.l3_svf == CONFIGS[config].l3_svf


def test_checkpoint_precision_is_exported_and_a_different_resume_mode_is_rejected(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "model.pth"
    torch.save({"state_dict": model.state_dict(), "precision": "tf32"}, path)
    report = load_checkpoint_state(model, str(path), True)
    assert report["precision"] == "tf32" and report["precision_recorded"]
    monkeypatch.setattr(helpers, "load_checkpoint_if_exists", lambda *a, **k: {"epoch": 0, "precision": "tf32"})
    with pytest.raises(ValueError, match="resume precision"):
        helpers.resume_from_ckpt(
            SimpleNamespace(resume=str(path), precision="fp16"),
            SimpleNamespace(model=model, optimizer=None),
            Mock(),
            torch.device("cpu"),
            -1,
        )


def test_legacy_checkpoint_retains_fp16_policy(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "legacy.pth"
    torch.save({"state_dict": model.state_dict()}, path)
    report = load_checkpoint_state(model, str(path), True)
    assert report["precision"] == "fp16" and not report["precision_recorded"]
