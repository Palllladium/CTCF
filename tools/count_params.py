import argparse

from experiments.core.model_adapters import get_model_adapter
from utils import setup_device


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def fmt(n):
    if n >= 1_000_000:
        return f"{n:,} ({n / 1_000_000:.2f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1_000:.1f}K)"
    return f"{n:,}"


def main():
    p = argparse.ArgumentParser(description="Compare parameter counts across models.")
    p.add_argument("--models", nargs="+", default=["ctcf", "tm-dca", "utsrmorph"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--ctcf_config", type=str, default="CTCF-CascadeA")
    p.add_argument("--tm_config", type=str, default="TransMorph-3-LVL")
    p.add_argument("--utsr_config", type=str, default="UTSRMorph-Large")
    args = p.parse_args()

    device = setup_device(args.gpu, seed=0, deterministic=False)
    print(f"Device: {device}\n")

    results = []
    for name in args.models:
        adapter = get_model_adapter(name)
        if name.lower() in ("ctcf",):
            model = adapter.build(time_steps=args.time_steps, config_key=args.ctcf_config)
        elif name.lower() in ("tm-dca", "tm_dca", "tmdca"):
            model = adapter.build(time_steps=args.time_steps, config_key=args.tm_config)
        elif name.lower() in ("utsrmorph", "utsr"):
            model = adapter.build(config_key=args.utsr_config)
        else:
            raise ValueError(f"Unknown model: {name}")

        model = model.to(device)
        total, trainable, frozen = count_params(model)
        results.append((name.upper(), total, trainable, frozen))

        print(f"{'=' * 50}")
        print(f"  {name.upper()} (time_steps={args.time_steps})")
        print(f"{'=' * 50}")
        print(f"  Total:     {fmt(total)}")
        print(f"  Trainable: {fmt(trainable)}")
        print(f"  Frozen:    {fmt(frozen)}")
        print()

    if len(results) > 1:
        print(f"{'=' * 50}")
        print("  SUMMARY")
        print(f"{'=' * 50}")
        print(f"  {'Model':<12} {'Total':>16} {'Trainable':>16}")
        print(f"  {'-'*12} {'-'*16} {'-'*16}")
        for name, total, trainable, _ in results:
            print(f"  {name:<12} {total:>16,} {trainable:>16,}")


if __name__ == "__main__":
    main()
