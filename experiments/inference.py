from __future__ import annotations

import argparse

from experiments.core.cli_common import add_common_args
from experiments.core.cli_inference import add_inference_args
from experiments.core.inference_runtime import InferRunner


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, mode="infer")
    add_inference_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    InferRunner(args).run()


if __name__ == "__main__":
    main()
