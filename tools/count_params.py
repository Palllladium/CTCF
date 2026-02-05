import argparse
import torch

from utils import setup_device


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def build_model(model_name: str, device):
    if model_name == "ctcf":
        from models.CTCF.model import CTCF_DCA_SR
        from models.CTCF.model import CONFIGS as CONFIG_CTCF

        config = CONFIG_CTCF["CTCF-DCA-SR"]
        model = CTCF_DCA_SR(config, 12).to(device)

    elif model_name == "tm-dca":
        from models.TransMorph_DCA.model import TransMorphCascadeAd
        from models.TransMorph_DCA.model import CONFIGS as CONFIGS_TM

        config = CONFIGS_TM["TransMorph-3-LVL"]
        model = TransMorphCascadeAd(config, 12).to(device)

    elif model_name == "utsrmorph":
        from models.UTSRMorph.model import UTSRMorph
        from models.UTSRMorph.model import CONFIGS as CONFIGS_UTSR

        config = CONFIGS_UTSR["UTSRMorph-Large"]
        model = UTSRMorph(config).to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ctcf", "tm-dca", "utsrmorph"],
        help="Models to count params for",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for CPU")
    args = parser.parse_args()

    dev = setup_device(args.gpu, seed=0, deterministic=False)
    device = dev.device

    print(f"Using device: {device}")

    for name in args.models:
        model = build_model(name, device)
        total, trainable, non_trainable = count_params(model)

        print(f"\n{name.upper()}")
        print(f"Total params:         {total:,}")
        print(f"Trainable params:     {trainable:,}")
        print(f"Non-trainable params: {non_trainable:,}")
