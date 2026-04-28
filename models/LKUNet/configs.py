import ml_collections


def get_LKU_config(start_channel):
    c = ml_collections.ConfigDict()
    c.img_size = (160, 192, 224)
    c.start_channel = int(start_channel)
    c.in_channel = 2
    c.n_classes = 3
    return c


CONFIGS = {
    "LKU-4": get_LKU_config(4),
    "LKU-8": get_LKU_config(8),
    "LKU-16": get_LKU_config(16),
    "LKU-32": get_LKU_config(32),
}
