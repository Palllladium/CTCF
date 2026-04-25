CONFIGS = {
    # LKU-Net main paper config (Jia et al., WBIR 2022 / arxiv 2208.04939).
    # start_channel=4 -> 0.52M params at full-resolution 160x192x224.
    # Matches the paper's headline claim "outperforms TransMorph with 1.12% of its
    # parameters" (TransMorph ~46M -> ~0.52M).
    "LKU-4": {
        "start_channel": 4,
        "in_channel": 2,
        "n_classes": 3,
    },
    # Mid-light config (~2M params); commonly cited "lightweight LKU baseline" and
    # comparable in scale to EfficientMorph (~2.8M) and MambaMorph (~3.6M).
    "LKU-8": {
        "start_channel": 8,
        "in_channel": 2,
        "n_classes": 3,
    },
    # Middle config (~8M params).
    "LKU-16": {
        "start_channel": 16,
        "in_channel": 2,
        "n_classes": 3,
    },
    # Heavy config used for the Learn2Reg 2021 OASIS Task 3 winning submission;
    # ~33M params, comparable in scale to TransMorph.
    "LKU-32": {
        "start_channel": 32,
        "in_channel": 2,
        "n_classes": 3,
    },
}
