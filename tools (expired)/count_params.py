import argparse

from experiments.core.model_adapters import (
    build_ctcf_model,
    build_tm_dca_model,
    build_utsr_model,
)
from utils import setup_device


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def build_model(name: str, time_steps: int):
    key = name.lower().strip()
    if key == "ctcf":
        return build_ctcf_model(time_steps=int(time_steps))
    if key in ("tm-dca", "tm_dca", "tmdca"):
        return build_tm_dca_model(time_steps=int(time_steps))
    if key in ("utsrmorph", "utsr"):
        return build_utsr_model()
    raise ValueError(f"Unknown model: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["ctcf", "tm-dca", "utsrmorph"])
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--time_steps", type=int, default=12)
    args = p.parse_args()

    device = setup_device(args.gpu, seed=0, deterministic=False)
    print(f"Using device: {device}")

    for name in args.models:
        model = build_model(name, args.time_steps).to(device)
        total, trainable, frozen = count_params(model)
        print(f"\n{name.upper()}")
        print(f"Total params:         {total:,}")
        print(f"Trainable params:     {trainable:,}")
        print(f"Non-trainable params: {frozen:,}")


if __name__ == "__main__":
    main()
