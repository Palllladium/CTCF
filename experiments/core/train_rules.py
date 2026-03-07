"""CTCF train defaults."""


def apply_ctcf_dataset_defaults(args):
    ds_key = str(args.ds).upper()
    if args.w_reg is None:
        args.w_reg = 4.0 if ds_key == "IXI" else 1.0
