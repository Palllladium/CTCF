from __future__ import annotations

import argparse


def optional_bool(value: int | None) -> bool | None:
    """Map an optional 0/1 CLI flag to bool, leaving None untouched (use config default)."""
    return None if value is None else bool(value)


def _add_path_profile_args(p: argparse.ArgumentParser) -> None:
    p.set_defaults(paths=1)
    p.add_argument(
        "--1",
        dest="paths",
        action="store_const",
        const=1,
        help="Use path profile #1",
    )
    p.add_argument(
        "--2",
        dest="paths",
        action="store_const",
        const=2,
        help="Use path profile #2",
    )
    p.add_argument(
        "--3",
        dest="paths",
        action="store_const",
        const=3,
        help="Use path profile #3 (remote, uses CTCF_DATA_DIR env var)",
    )
    p.add_argument("--paths", type=int, help="Path profile id (1/2/...)")


def _add_io_args(p: argparse.ArgumentParser, ds_choices: list[str]) -> None:
    p.add_argument(
        "--ds",
        choices=ds_choices,
        default="OASIS",
        help="Dataset key to run.",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes.",
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exp",
        default="",
        help="Experiment name used for logs/results directories.",
    )
    p.add_argument(
        "--max_epoch",
        type=int,
        default=400,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    p.add_argument(
        "--img_size",
        type=int,
        nargs=3,
        default=(160, 192, 224),
        help="Input volume size D H W.",
    )
    p.add_argument(
        "--max_train_iters",
        type=int,
        default=0,
        help="Limit train iterations per epoch; <=0 disables limit.",
    )
    p.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
        help="Limit validation batches per epoch; <=0 disables limit.",
    )
    p.add_argument(
        "--resume",
        default="",
        help="Checkpoint path to resume training from.",
    )
    p.add_argument(
        "--save_ckpt",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable/disable checkpoint saving to disk.",
    )
    p.add_argument(
        "--use_tb",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable/disable TensorBoard logging.",
    )
    p.add_argument(
        "--tb_images_every",
        type=int,
        default=5,
        help="TensorBoard image logging period in epochs.",
    )
    p.add_argument(
        "--quiet",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, suppress per-iter logs from console (file only). Epoch summaries still shown.",
    )


def add_common_args(
    p: argparse.ArgumentParser,
    include_synth: bool = False,
    mode: str = "train",
) -> argparse.ArgumentParser:
    """Add shared CLI args for train/infer experiment scripts."""
    if mode not in ("train", "infer"):
        raise ValueError(f"Unsupported mode='{mode}'. Expected 'train' or 'infer'.")

    ds_choices = ["OASIS", "IXI", "SYNTH"] if include_synth else ["OASIS", "IXI"]
    _add_io_args(p, ds_choices)
    _add_path_profile_args(p)
    if mode == "train":
        _add_train_args(p)
    return p
