# 🧠 OASIS-CTCF

**OASIS-CTCF** — исследовательский проект в области медицинской регистрации изображений,  
реализующий архитектуру **Cycle-TransMorph-CF (CTCF)** — каскадную трансформер-модель  
для *unsupervised deformable medical image registration*.

---

## 🔬 Концепция

Проект основан на **TransMorph** и объединяет идеи:

- **CycleMorph** — *cycle-consistency loss*  
- **ICON / GradICON** — *inverse-consistency flow regularization*  
- **Cascade registration** — многоуровневая *coarse-to-fine* архитектура  
- **Cross / Plane attention** — улучшенное сопоставление признаков  
- Балансировку лоссов *(L_sim, L_reg, L_jac, L_icon, L_cyc)*  
  для оптимального соотношения качества, гладкости и топологии

---

## 📂 Структура репозитория

| Файл / папка | Назначение |
|---------------|------------|
| `TransMorph/train_CTCF.py` | обучение каскадной модели **CTCF** |
| `TransMorph/train_TM_baseline.py` | обучение чистого **TransMorph-baseline** для сравнения |
| `TransMorph/models/` | архитектуры, лоссы, каскад, attention-модули, утилиты |
| `TransMorph/models/configs_CTCF.py` | GPU-профили *(Small / Mid / Large)* |
| `TransMorph/models/utils_torch.py`, `utils_train.py` | AMP, TF32, логгеры, валидация |
| `evaluation.py` | оффлайн-оценка и экспорт полей деформаций |

---

## ⚙️ Особенности

- Полная совместимость с **PyTorch 2.9 + CUDA 12.8**  
- Поддержка **Windows / Linux**  
- **AMP-ускорение** и **TF32-режим** для современных GPU  
- Автоматическое сохранение логов и чекпойнтов (`logs/`, `experiments/`)  
- Готов к воспроизводимым экспериментам и публикации результатов  
  *(ElCon Conference 2026, Biomedical Engineering track)*

Guide:
python -m experiments.train_CTCF --ds OASIS --1
python -m experiments.train_TransMorphDCA --ds OASIS --1
python -m experiments.train_UTSRMorph --ds OASIS --1

python -m experiments.train_CTCF --ds IXI --2
python -m experiments.train_TransMorphDCA --ds IXI --2
python -m experiments.train_UTSRMorph --ds IXI --2