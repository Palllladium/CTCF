"""im2grid-Lite on OASIS without labels: profile, calibrate lambda, train or evaluate (experiments.dispatch1.matcher_training).

python -m experiments.dispatch1.train_im2grid --calibrate RUN/lambda.json --paths 3 --gpu 0
python -m experiments.dispatch1.train_im2grid --lambda_file RUN/lambda.json --exp IM2GRID --paths 3 --gpu 0
python -m experiments.dispatch1.train_im2grid --eval_ckpt results/IM2GRID/ckpt/last.pth --out RUN/im2grid --paths 3 --gpu 0
"""

from __future__ import annotations

import argparse

from experiments.dispatch1.matcher_training import add_arguments, run
from models.Im2grid import METHOD, Im2gridLite, to_canonical
from utils import setup_device


def build_model(args):
    return Im2gridLite()


def losses(model, moving, fixed):
    return model.losses(moving, fixed)


def predict(model, moving, fixed):
    return to_canonical(model(moving, fixed)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args()
    run(args, setup_device(args.gpu, seed=args.seed, deterministic=False), build_model, losses, predict, METHOD)


if __name__ == "__main__":
    main()
