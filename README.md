# CTCF

Проект для unsupervised deformable registration на OASIS/IXI с тремя основными моделями:
- `CTCF`
- `TransMorph-DCA`
- `UTSRMorph`


## Структура

- `experiments/`
  - `train_CTCF.py` — обучение CTCF (`OASIS`, `IXI`, `SYNTH`)
  - `train_TransMorphDCA.py` — обучение TM-DCA (`OASIS`, `IXI`)
  - `train_UTSRMorph.py` — обучение UTSRMorph (`OASIS`, `IXI`)
  - `inference.py` — канонический инференс/оценка
  - `engine.py` — общий train-loop, валидация, логи
- `models/` — архитектуры и конфиги
- `datasets/` — загрузчики OASIS/IXI/SYNTH
- `utils/` — базовые лоссы/метрики/warp/train-utils/аугментации
- `tools/` — вспомогательные утилиты агрегации/визуализации (без дублирующего inference)


## Быстрый старт

OASIS:

```powershell
python -m experiments.train_CTCF --ds OASIS --1
python -m experiments.train_TransMorphDCA --ds OASIS --1
python -m experiments.train_UTSRMorph --ds OASIS --1
```

IXI:

```powershell
python -m experiments.train_CTCF --ds IXI --2
python -m experiments.train_TransMorphDCA --ds IXI --2
python -m experiments.train_UTSRMorph --ds IXI --2
```

SYNTH (для CTCF):

```powershell
python -m experiments.train_CTCF --ds SYNTH --gpu 0 --max_epoch 80
```

Инференс:

```powershell
python -m experiments.inference `
  --model ctcf `
  --ckpt results/CTCF/ckpt/best.pth `
  --ds OASIS --1
```


## Notes

- `logs/` и `results/` не версионируются.
- Для корректной работы используйте окружение с PyTorch + torchvision + CUDA совместимых версий.
- Для `--hd95` в inference нужен внешний пакет `surface-distance`.
