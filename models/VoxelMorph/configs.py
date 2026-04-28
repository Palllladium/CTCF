import ml_collections


def get_VxmDense_config():
    c = ml_collections.ConfigDict()
    c.img_size = (160, 192, 224)
    c.enc_nf = (16, 32, 32, 32)
    c.dec_nf = (32, 32, 32, 32, 32, 16, 16)
    c.int_steps = 7
    return c


def get_VxmDense_nodiff_config():
    c = get_VxmDense_config()
    c.int_steps = 0
    return c


CONFIGS = {
    "VxmDense": get_VxmDense_config(),
    "VxmDense-nodiff": get_VxmDense_nodiff_config(),
}
