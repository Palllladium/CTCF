import json
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from datasets.LUMIR import LUMIRPairSampler
from experiments.core.distributed import accumulation_steps, finite_on_all, restore_training_state, training_state
from utils.runtime import PerfInfo


class PairData(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, pair):
        m, f = pair
        return torch.tensor([m / 7, f / 7], dtype=torch.float32)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.early = torch.nn.Linear(2, 1)
        self.late = torch.nn.Linear(2, 1)

    def forward(self, x, epoch):
        return checkpoint(self.early, x, use_reentrant=False) + (
            checkpoint(self.late, x, use_reentrant=False) if epoch else 0
        )


class SmallRunner:
    def __init__(self, stochastic):
        self.device = torch.device("cpu")
        self.model = SmallModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, amsgrad=True)
        self.stochastic = stochastic
        self.forward_flow = lambda x, y: x

    def train_step(self, batch, epoch):
        prediction = self.model(batch, epoch) - self.model(-batch, epoch)
        target = batch[:, :1] - 0.3
        if self.stochastic:
            target = target + 0.01 * (random.random() + np.random.random() + torch.rand_like(target))
        loss = (prediction - target).square().mean()
        return loss, {"all": loss.item()}


def _worker(rank, world, rendezvous, root, mode, stochastic):
    from experiments.core.train_runtime import run_train
    from experiments.train_CTCF import parse_args

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=rendezvous, timeout=timedelta(seconds=45))
    try:
        os.chdir(root)
        assert not finite_on_all(torch.tensor(float("nan") if rank == 0 else 1.0))
        torch.manual_seed(123)
        with patch.object(
            sys,
            "argv",
            [
                "train",
                "--ds",
                "LUMIR",
                "--precision",
                "fp32",
                "--global_batch_size",
                "4",
                "--max_epoch",
                "2",
                "--max_train_iters",
                "3",
                "--num_workers",
                "0",
                "--exp",
                "test",
            ],
        ):
            args = parse_args()
        args.lr = 0.01
        args.quiet = 1
        args.use_tb = 0
        if mode == "resume":
            args.resume = "results/test/ckpt/last.pth"
        runner = SmallRunner(stochastic)
        cpu_scaler = torch.amp.GradScaler
        if mode.startswith("amp_"):
            args.precision = "fp16"
        if mode in ("amp_once", "amp_persistent", "strict_gradient"):
            backward_calls = 0

            def overflow(gradient):
                nonlocal backward_calls
                inject = rank == 0 and (mode != "amp_once" or backward_calls < 4 // world)
                backward_calls += 1
                return torch.full_like(gradient, float("inf")) if inject else gradient

            runner.model.early.weight.register_hook(overflow)
        if mode == "forward_nan" and rank == 0:
            original_step = runner.train_step

            def invalid_forward(batch, epoch):
                loss, logs = original_step(batch, epoch)
                return loss * float("nan"), logs

            runner.train_step = invalid_forward

        def loaders(_):
            dataset = PairData()
            sampler = LUMIRPairSampler(dataset, global_batch=4, updates=3, rank=rank, world=world, seed=args.seed)
            return DataLoader(dataset, batch_size=1, sampler=sampler), []

        barriers = 0
        real_barrier = dist.barrier

        def pause_at_next_epoch():
            nonlocal barriers
            barriers += 1
            if mode == "pause" and barriers == 2:
                raise InterruptedError("test pause at epoch boundary")
            real_barrier()

        validation = SimpleNamespace(dsc=0.7, jac_nonpos_percent=0.0, ndv_percent=0.0, sdlogj=0.1, last_vis={})
        with (
            patch("experiments.core.train_runtime.torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("experiments.core.train_runtime.torch.cuda.synchronize"),
            patch("experiments.core.train_runtime.attach_stdout_logger"),
            patch("experiments.core.train_runtime.print_experiment_header"),
            patch("experiments.core.train_runtime.perf_epoch_start", side_effect=time.perf_counter),
            patch(
                "experiments.core.train_runtime.perf_epoch_end",
                side_effect=lambda start, **_: PerfInfo(time.perf_counter() - start, 1.0, None),
            ),
            patch("experiments.core.train_runtime.validate", return_value=validation),
            patch("torch.distributed.barrier", side_effect=pause_at_next_epoch),
            patch(
                "experiments.core.train_runtime.torch.amp.GradScaler",
                side_effect=lambda _, **kw: cpu_scaler("cpu", **kw),
            ),
        ):
            try:
                run_train(args, runner, loaders)
            except InterruptedError:
                if mode != "pause":
                    raise
            except (FloatingPointError, RuntimeError) as error:
                expected = {
                    "amp_persistent": "3 consecutive FP16 updates skipped",
                    "forward_nan": "non-finite loss",
                    "strict_gradient": "non-finite",
                }
                if mode not in expected or expected[mode] not in str(error):
                    raise
                Path(f"error_{rank}.txt").write_text(str(error))
        torch.save(runner.model.state_dict(), f"rank_{rank}.pth")
    finally:
        dist.destroy_process_group()


def _run(root: Path, world=2, mode="full", stochastic=False):
    root.mkdir(exist_ok=True)
    store = root / ("store_" + mode)
    mp.spawn(_worker, args=(world, store.as_uri(), str(root), mode, stochastic), nprocs=world, join=True)
    return torch.load(root / "rank_0.pth", weights_only=True)


def test_global_pairs_and_full_pass_are_independent_of_gpu_count():
    dataset = PairData()
    for updates in (0, 3):
        streams = []
        for world in (1, 2, 4):
            shards = [
                list(LUMIRPairSampler(dataset, global_batch=4, updates=updates, rank=r, world=world, seed=3))
                for r in range(world)
            ]
            streams.append([pair for step in zip(*shards, strict=True) for pair in step])
        assert streams[0] == streams[1] == streams[2]
        assert all(m != f for m, f in streams[0])
        if updates == 0:
            assert {m for m, _ in streams[0]} == set(range(len(dataset)))
    sampler = LUMIRPairSampler(dataset, global_batch=4, updates=3, rank=0, world=1, seed=3)
    first = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_accumulation_validates_the_global_batch():
    args = SimpleNamespace(global_batch_size=8, batch_size=1)
    assert [accumulation_steps(args, world) for world in (1, 2, 4, 8)] == [8, 4, 2, 1]
    with pytest.raises(ValueError, match="multiple"):
        accumulation_steps(args, 3)


def test_adjacent_seeds_do_not_reuse_neighboring_epoch_or_rank_streams():
    dataset = PairData()
    a = LUMIRPairSampler(dataset, global_batch=4, updates=3, rank=0, world=1, seed=0)
    b = LUMIRPairSampler(dataset, global_batch=4, updates=3, rank=0, world=1, seed=1)
    a.set_epoch(1)
    assert list(a) != list(b)
    streams = []
    for seed, rank in ((0, 1), (1, 0)):
        with patch("experiments.core.distributed.world_info", return_value=(rank, 2)):
            restore_training_state({}, SimpleNamespace(seed=seed), torch.device("cpu"), 0)
        streams.append(torch.rand(4))
    assert not torch.equal(*streams)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("mode", ("amp_once", "amp_persistent", "forward_nan", "strict_gradient"))
def test_distributed_numerical_policy_agrees_across_ranks(tmp_path, mode):
    _run(tmp_path, mode=mode)
    if mode == "amp_once":
        checkpoint = torch.load(tmp_path / "results/test/ckpt/last.pth", weights_only=True)
        assert checkpoint["scaler"]["scale"] == 32768
        assert checkpoint["training"]["amp_skip_streak"] == 0
        import csv

        with (tmp_path / "logs/test/metrics.csv").open() as handle:
            first = json.loads(next(csv.DictReader(handle))["loss_meters_json"])
        assert first["optimizer_updates"]["avg"] == 2
        assert first["amp_skipped"]["avg"] == pytest.approx(1 / 3)
    else:
        assert (tmp_path / "error_0.txt").is_file() and (tmp_path / "error_1.txt").is_file()
        assert not (tmp_path / "results/test/ckpt/last.pth").exists()
    state = torch.load(tmp_path / "rank_0.pth", weights_only=True)
    peer = torch.load(tmp_path / "rank_1.pth", weights_only=True)
    for key in state:
        torch.testing.assert_close(state[key], peer[key], rtol=0, atol=0)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_bidirectional_training_matches_accumulated_single_rank(tmp_path):
    single = _run(tmp_path / "single", world=1)
    parallel = _run(tmp_path / "parallel")
    for key in single:
        torch.testing.assert_close(single[key], parallel[key], rtol=1e-5, atol=1e-7)
    peer = torch.load(tmp_path / "parallel/rank_1.pth", weights_only=True)
    for key in peer:
        torch.testing.assert_close(parallel[key], peer[key], rtol=0, atol=0)
    checkpoint = torch.load(tmp_path / "parallel/results/test/ckpt/last.pth", weights_only=True)
    assert checkpoint["training"]["world_size"] == 2
    assert len(checkpoint["training"]["rng"]) == 2
    assert all(not key.startswith("module.") for key in checkpoint["state_dict"])


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_epoch_resume_restores_optimizer_and_each_ranks_random_streams(tmp_path):
    full = _run(tmp_path / "full", stochastic=True)
    _run(tmp_path / "resumed", mode="pause", stochastic=True)
    shutil.copytree(tmp_path / "resumed", tmp_path / "resized")
    resumed = _run(tmp_path / "resumed", mode="resume", stochastic=True)
    for key in full:
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
    import csv

    with (tmp_path / "resumed/logs/test/metrics.csv").open() as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [0, 1]
    _run(tmp_path / "resized", world=1, mode="resume", stochastic=True)
    resized = torch.load(tmp_path / "resized/results/test/ckpt/last.pth", weights_only=True)
    assert resized["training"]["world_size"] == 1 and resized["epoch"] == 1
    assert resized["training"]["global_batch_size"] == 4
    assert all(torch.isfinite(tensor).all() for tensor in resized["state_dict"].values())


def test_saved_rng_uses_weights_only_compatible_types(tmp_path):
    args = SimpleNamespace(seed=0, max_train_iters=3, global_batch_size=4)
    state = training_state(args, torch.device("cpu"))
    expected = random.random(), np.random.random(), torch.rand(3)
    path = tmp_path / "state.pth"
    torch.save(state, path)
    restore_training_state(torch.load(path, weights_only=True), args, torch.device("cpu"), 0)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_nonfinite_loss_is_detected_without_changing_it():
    loss = torch.tensor(float("nan"), requires_grad=True)
    assert not finite_on_all(loss) and torch.isnan(loss)


def test_failed_checkpoint_write_keeps_the_previous_last(tmp_path):
    from experiments.core.helpers import save_ckpt

    save_ckpt({"epoch": 0}, str(tmp_path), False)

    def interrupted(state, path):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated storage interruption")

    with patch("experiments.core.helpers.torch.save", side_effect=interrupted), pytest.raises(OSError):
        save_ckpt({"epoch": 1}, str(tmp_path), False)
    assert torch.load(tmp_path / "last.pth", weights_only=True)["epoch"] == 0
    assert not list(tmp_path.glob("*.tmp"))
