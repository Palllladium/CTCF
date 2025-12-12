# OASIS-CTCF
Файлы для работы:
- `TransMorph/train_CTCF_v2.py`
- `TransMorph/train_TransMorph_v2.py`

## Установка
```bash
conda env create -f environment.yml -n oasis-ctcf
conda activate oasis-ctcf
```

ЛИБО лучше:

```bash
conda env create -f environment_clear.yml -n oasis-ctcf
conda activate oasis-ctcf
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

## Запуск

1. На Windows на моей машине:
Общее: `python -m experiments.OASIS.`*имя файла*` --train_dir "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/All" --val_dir "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test"`
- train_CTCF
- train_TM-DCA
- train_UTSRMorph


2. Для Linux на вашей машине:
Общее: `python -m experiments.OASIS.`*имя файла*` --train_dir "/home/roman/P/OASIS_L2R_2021_task03/All" --val_dir "/home/roman/P/OASIS_L2R_2021_task03/Test"`
- train_CTCF
- train_TM-DCA
- train_UTSRMorph