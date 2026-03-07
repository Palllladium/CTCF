# Remote VM Workflow

Актуальный порядок работы со скриптами `tools/`.
Режим выполнения: **без checkpoint и без TensorBoard**. В результате формируются только текстовые логи и итоговый файл `ablation_results.txt`.

## Шаг 0. Генерация SSH-ключа (ПК, PowerShell, один раз)

```powershell
mkdir C:\Users\user\.ssh -Force
ssh-keygen -t ed25519 -f C:\Users\user\.ssh\id_ed25519
```

При создании ВМ используется содержимое `C:\Users\user\.ssh\id_ed25519.pub`.

## Шаг 1. Архивация OASIS (ПК, PowerShell)

```powershell
cd C:\Users\user\Documents\Education\MasterWork\datasets
tar -czf OASIS_L2R_2021_task03.tar.gz OASIS_L2R_2021_task03\
```

## Шаг 2. Создание ВМ у провайдера и получение IP

## Шаг 3. Загрузка датасета на ВМ (ПК, Git Bash)

```bash
bash tools/upload_data.sh user@<IP> --key ~/.ssh/id_ed25519
```

После распаковки директории данных:
- `/data/OASIS_L2R_2021_task03/All`
- `/data/OASIS_L2R_2021_task03/Test`

## Шаг 4. Передача setup-скрипта на ВМ (ПК, Git Bash)

```bash
scp -i ~/.ssh/id_ed25519 tools/remote_setup.sh user@<IP>:~/remote_setup.sh
```

## Шаг 5. Подключение к ВМ и настройка окружения

```bash
ssh -i ~/.ssh/id_ed25519 user@<IP>
bash ~/remote_setup.sh --data-dir /data
```

Настройка выполняет:
- установку Miniconda,
- клонирование репозитория в `~/CTCF`,
- создание/обновление `oasis-ctcf`,
- фиксацию CUDA-стека PyTorch,
- сохранение `DATA_DIR` в `~/CTCF/.env_data`.

## Шаг 6. Запуск абляций в detached tmux (ВМ)

```bash
cd ~/CTCF
bash tools/run_experiments.sh \
  --save-ckpt 0 \
  --use-tb 0 \
  --tmux-session ctcf_abl
```

Подключение к tmux-сессии:

```bash
tmux attach -t ctcf_abl
```

## Шаг 7. Мониторинг логов на ВМ (опционально)

```bash
tail -f ~/CTCF/logs/ABL_01_BASELINE/logfile.log
```

## Шаг 8. Скачивание результатов на ПК после завершения

```bash
bash tools/sync_results.sh user@<IP> --key ~/.ssh/id_ed25519
```

Содержимое выгрузки в `./remote_results/<timestamp>/`:
- `ablation_results.txt`
- `logs/<EXP>/logfile.log`

## Шаг 9. Анализ результатов (ПК)

Основные файлы анализа:
- `ablation_results.txt`
- `logfile.log` каждого прогона

Сравнение выполняется по `best`/`final` Dice на одинаковом горизонте эпох.

## Шаг 10. Финальный долгий прогон лучшей конфигурации (ВМ, без ckpt/TB)

```bash
cd ~/CTCF
python -m experiments.train_CTCF \
  --ds OASIS --3 --exp FINAL_500EP \
  --max_epoch 500 --time_steps 8 \
  --save_ckpt 0 --use_checkpoint 0 --use_tb 0 \
  --gpu 0 --num_workers 8 \
  <лучшие гиперпараметры из абляций>
```

## Шаг 11. Удаление ВМ у провайдера

Удаление выполняется после выгрузки логов и фиксации результатов.