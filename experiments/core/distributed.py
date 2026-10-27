"""DDP support for the existing training loop, including its bidirectional loss and epoch checkpoints."""

import math
import random
from dataclasses import replace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def world_info() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def accumulation_steps(args, world: int) -> int:
    global_batch = getattr(args, "global_batch_size", 0)
    if not global_batch:
        if world != 1:
            raise ValueError("DDP requires an explicit --global_batch_size")
        return 1
    local = args.batch_size * world
    if args.batch_size <= 0 or global_batch < local or global_batch % local:
        raise ValueError("global_batch_size must be a positive multiple of batch_size * world_size")
    return global_batch // local


class TrainingObjective(torch.nn.Module):
    """One DDP forward includes both registration directions and every loss term."""

    def __init__(self, runner):
        super().__init__()
        self.model = runner.model
        self.runner = runner
        model_parameters = {id(p) for p in self.model.parameters()}
        if any(id(p) not in model_parameters for group in runner.optimizer.param_groups for p in group["params"]):
            raise ValueError("DDP requires trainable auxiliary parameters to belong to runner.model for checkpointing")

    def forward(self, batch, epoch):
        return self.runner.train_step(batch, epoch)


def training_objective(runner):
    if world_info()[1] == 1:
        return runner.train_step
    objective = TrainingObjective(runner)
    return DistributedDataParallel(
        objective,
        device_ids=[runner.device.index] if runner.device.type == "cuda" else None,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )


def finite_on_all(loss: torch.Tensor) -> bool:
    finite = torch.isfinite(loss.detach()).to(torch.int32)
    if world_info()[1] > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite)


def reduce_epoch(meters, perf, device, measured_wall):
    allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    if world_info()[1] == 1:
        return perf, measured_wall, allocated
    keys = sorted(meters)
    values = torch.tensor(
        [[meters[k].sum, meters[k].count, sum(v * v for v in meters[k].vals)] for k in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values)
    for key, (total, count, squares) in zip(keys, values.cpu().tolist(), strict=True):
        meter = meters[key]
        meter.sum, meter.count = total, int(count)
        meter.avg = total / max(count, 1)
        meter.std = math.sqrt(max(0.0, squares / max(count, 1) - meter.avg**2))
    timing = torch.tensor(
        [perf.epoch_time_sec, perf.mean_iter_time_ms, perf.peak_gpu_mem_gib or 0.0, measured_wall, allocated],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    wall, step_ms, peak, measured_wall, allocated = timing.cpu().tolist()
    return (
        replace(
            perf,
            epoch_time_sec=wall,
            mean_iter_time_ms=step_ms,
            peak_gpu_mem_gib=peak if perf.peak_gpu_mem_gib is not None else None,
        ),
        measured_wall,
        allocated,
    )


def training_state(args, device, sampler=None) -> dict:
    rank, world = world_info()
    np_state = np.random.get_state()
    local = {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[rank] = local
    return {
        "global_batch_size": getattr(args, "global_batch_size", 0),
        "updates_per_epoch": args.max_train_iters,
        "seed": args.seed,
        "world_size": world,
        "rng": states,
        "sampler_fingerprint": getattr(sampler, "fingerprint", None),
    }


def restore_training_state(state, args, device, epoch: int, sampler=None) -> None:
    rank, world = world_info()
    if state:
        for key, value in (
            ("global_batch_size", getattr(args, "global_batch_size", 0)),
            ("updates_per_epoch", args.max_train_iters),
            ("seed", args.seed),
            ("sampler_fingerprint", getattr(sampler, "fingerprint", None)),
        ):
            if state.get(key) != value:
                raise ValueError(f"resume {key} differs from checkpoint; start a new experiment")
        if state["world_size"] == world:
            local = state["rng"][rank]
            random.setstate(local["python"])
            name, keys, *tail = local["numpy"]
            np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *tail))
            torch.set_rng_state(local["torch"].cpu())
            if device.type == "cuda" and local["cuda"] is not None:
                torch.cuda.set_rng_state(local["cuda"].cpu(), device)
            return
        print(
            ">>> Resume world size changed: global pair budget retained; rank RNG streams restart (not bitwise resume)"
        )
    elif epoch:
        print(">>> Legacy checkpoint has no RNG/batch metadata; continuation is not bitwise")
    if world > 1 or getattr(args, "global_batch_size", 0):
        seed = int(np.random.SeedSequence(args.seed, spawn_key=(1, epoch, rank)).generate_state(1, dtype=np.uint64)[0])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
