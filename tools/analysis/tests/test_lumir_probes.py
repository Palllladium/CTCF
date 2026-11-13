import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments import lumir_probes
from experiments.lumir_probes import contrast_image


def test_intensity_interventions_preserve_background_and_do_not_permute_voxels():
    image = torch.tensor([0, 0.1, 0.5, 0.9]).reshape(1, 1, 1, 1, 4)
    for mode in ("original", "gamma_half", "gamma_two", "invert", "gain_bias"):
        result = contrast_image(image, mode)
        assert result.shape == image.shape and result.flatten()[0] == 0 and torch.isfinite(result).all()
        expected = {"original": 0.5, "gamma_half": 0.5**0.5, "gamma_two": 0.25, "invert": 0.5, "gain_bias": 0.5}[mode]
        torch.testing.assert_close(result.flatten()[2], torch.tensor(expected))


def test_check_releases_cyclic_runner_before_starting_export(tmp_path, monkeypatch):
    references = []

    class FakeRunner:
        def __init__(self, *_):
            self.model = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.forward_flow = self._forward_flow
            references.append(weakref.ref(self))

        def _forward_flow(self, *_):
            return None

        def train_step(self, *_):
            return self.model.weight.square().sum(), {}

    monkeypatch.setattr(lumir_probes, "Runner", FakeRunner)
    monkeypatch.setattr(
        lumir_probes, "resolve_lumir_paths", lambda *_: {"train_json": "unused", "train_root": tmp_path}
    )
    monkeypatch.setattr(lumir_probes, "read_lumir_json", lambda *_: (["a", "b"], []))
    monkeypatch.setattr(lumir_probes, "image", lambda *_: torch.ones(1, 1, 2, 2, 2))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)

    def empty_cache():
        assert references[0]() is None

    def export(command, **_):
        assert references[0]() is None
        directory = Path(command[command.index("--out") + 1])
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"certified": {"global_homeomorphism_pairs": 1}, "files": ["field"]})
        )

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(lumir_probes.subprocess, "run", export)
    previous = gc.isenabled()
    gc.disable()
    try:
        lumir_probes.check(SimpleNamespace(out=tmp_path, device=torch.device("cpu")))
    finally:
        if previous:
            gc.enable()
        gc.collect()
    assert (tmp_path / "check.json").exists()
