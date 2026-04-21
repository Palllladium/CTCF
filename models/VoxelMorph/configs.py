CONFIGS = {
    # VoxelMorph-2 diffeomorphic (Dalca et al., TMI 2019)
    # Standard baseline used in TransMorph, iPEAR, RDN, DARE, etc.
    "VxmDense": {
        "enc_nf": [16, 32, 32, 32],
        "dec_nf": [32, 32, 32, 32, 32, 16, 16],
        "int_steps": 7,
    },
    # VoxelMorph-1 (Balakrishnan et al., CVPR 2018)
    # Direct displacement, no diffeomorphic integration
    "VxmDense-nodiff": {
        "enc_nf": [16, 32, 32, 32],
        "dec_nf": [32, 32, 32, 32, 32, 16, 16],
        "int_steps": 0,
    },
}
