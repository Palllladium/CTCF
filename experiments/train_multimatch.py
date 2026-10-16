"""The E3/D3 correspondence matcher (H1) on OASIS without labels: profile, calibrate lambda, train or evaluate.

python -m experiments.train_multimatch --mode D3 --calibrate RUN/lambda.json --paths 3 --gpu 0
python -m experiments.train_multimatch --mode D3 --lambda_file RUN/lambda.json --exp H1_D3 --paths 3 --gpu 0
python -m experiments.train_multimatch --mode D3 --eval_ckpt results/H1_D3/ckpt/last.pth --out RUN/h1_D3 --paths 3
"""

from __future__ import annotations

import argparse

import torch

from experiments.core.matcher_training import add_arguments, run
from models.Im2grid.fetch import PORT_VERSION, REVISION
from models.MultiMatch.model import MODES, MultiMatch, level_mismatch, level_smoothness
from utils import setup_device


def build_model(args):
    return MultiMatch(mode=args.mode)


def losses(model, moving, fixed):
    _, states = model(moving, fixed)
    with torch.autocast(device_type=moving.device.type, enabled=False):
        return level_mismatch(states, fixed.float(), moving.float()), level_smoothness(states)


def predict(model, moving, fixed):
    return model(moving, fixed)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    method = {
        "name": f"multimatch_{args.mode}",
        "mode": args.mode,
        "version": 2,
        "revision": REVISION,
        "port_version": PORT_VERSION,
        "objective": "expected level MSE + selected-level diffusion",
    }
    run(args, setup_device(args.gpu, seed=args.seed, deterministic=False), build_model, losses, predict, method)


if __name__ == "__main__":
    main()
