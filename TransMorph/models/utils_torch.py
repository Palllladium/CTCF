import os
import torch
from contextlib import nullcontext

def setup_torch(precision: str = "tf32", alloc_conf: str = "expandable_segments:True,max_split_size_mb:256"):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", alloc_conf)
    if precision in ("tf32", "ieee", "none"):
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.conv.fp32_precision = "tf32" if precision == "tf32" else "ieee"

    torch.backends.cudnn.benchmark = True

def amp_context_and_scaler(use_amp: bool):
    if torch.cuda.is_available():
        autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        autocast_ctx = nullcontext()
        scaler = None
    return autocast_ctx, scaler