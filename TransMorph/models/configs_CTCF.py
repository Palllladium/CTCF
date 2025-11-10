from dataclasses import dataclass
from ml_collections import ConfigDict

# ---------- Гиперпараметры обучения ----------
@dataclass
class TrainHP:
    epochs: int = 300
    batch_size: int = 1
    lr: float = 5e-5
    weight_decay: float = 1e-5
    val_every: int = 1
    use_amp: bool = True  # AMP включён по умолчанию

# ---------- Расписание весов лоссов ----------
def loss_schedule(epoch: int, T: float = 80.0):
    t = min(epoch / T, 1.0)
    return {
        'w_sim': 1.0,
        'w_reg': 0.2 + 0.3 * t,
        'w_icon': 0.0 + 0.05 * t,
        'w_jac' : 0.0 + 0.05 * t,
        'w_cyc' : 0.0 + 0.05 * t,
    }

def _base():
    C = ConfigDict()
    # каскад (coarse→fine)
    C.levels = 2
    # AMP / чекпойнтинг
    C.hp = TrainHP()
    C.grad_checkpointing = True
    # расписание лоссов
    C.loss_schedule = loss_schedule
    # бэкабон (можно переопределить в профиле)
    C.base_backbone = 'TransMorph'
    return C

def get_ctcf_smallgpu():
    C = _base()
    C.img_size = (96, 96, 128)
    C.base_backbone = 'TransMorph-Small'
    return C

def get_ctcf_midgpu():
    C = _base()
    C.img_size = (128, 128, 128)
    C.base_backbone = 'TransMorph'
    return C

def get_ctcf_largegpu():
    C = _base()
    # C.img_size = (160,192,224) # на 5070 не хватает памяти
    C.img_size = (128, 160, 192)
    C.base_backbone = 'TransMorph'
    return C

CONFIGS_CTCF = {
    'CTCF-SmallGPU': get_ctcf_smallgpu(),
    'CTCF-MidGPU':   get_ctcf_midgpu(),
    'CTCF-LargeGPU': get_ctcf_largegpu(),
}