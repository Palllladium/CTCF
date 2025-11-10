# OASIS-CTCF
Файлы для работы:
- `TransMorph/train_CTCF.py`
- `TransMorph/train_TM_baseline.py`

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

## Настройка
В `TransMorph\models\configs_CTCF.py` закомментировать 53 строку и раскомментировать 52.
На моей видеокарте с референсным размером просто не запускалось из-за нехватки памяти.

## Запуск

1. На Windows на моей машине:
CTCF: `python -m TransMorph.train_CTCF --train_dir "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/All" --val_dir "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test" --exp CTCF_LargeGPU --cfg CTCF-LargeGPU --precision tf32`
TM: `python -m TransMorph.train_TM_baseline --train_dir "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/All" --val_dir "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test" --exp TM_Baseline_LargeGPU --cfg CTCF-LargeGPU --precision tf32`

2. Для Linux на вашей машине:
CTCF: `python -m TransMorph.train_CTCF --train_dir "/home/roman/P/OASIS_L2R_2021_task03/All" --val_dir "/home/roman/P/OASIS_L2R_2021_task03/Test" --exp CTCF_LargeGPU --cfg CTCF-LargeGPU --precision tf32`
TM: `python -m TransMorph.train_TM_baseline --train_dir "/home/roman/P/OASIS_L2R_2021_task03/All" --val_dir "/home/roman/P/OASIS_L2R_2021_task03/Test" --exp TM_Baseline_LargeGPU --cfg CTCF-LargeGPU --precision tf32`