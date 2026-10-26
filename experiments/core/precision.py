"""Numerical policy shared by CTCF training, validation and export."""

import torch


def add_precision_arg(parser, default="fp16") -> None:
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16", "fp32", "tf32"),
        default=default,
        help="fp16 AMP+scaler, bf16 AMP without scaler, strict fp32, or fp32 with TF32 kernels.",
    )


def configure_precision(mode: str, device: torch.device) -> None:
    if mode not in ("fp16", "bf16", "fp32", "tf32"):
        raise ValueError(f"unsupported precision: {mode}")
    if device.type == "cuda" and mode == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 is not supported on the selected CUDA device; choose a different precision")
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = mode == "tf32"


def precision_autocast(mode: str, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if mode == "bf16" else torch.float16,
        enabled=device.type == "cuda" and mode in ("fp16", "bf16"),
    )
