# Команды для внешнего аудита

Запускать из `/home/roman/P/CTCF`. Сначала один OASIS-парный случай; только после строки `PASS` — все 19.

```bash
cd /home/roman/P/CTCF
git pull --ff-only
conda activate oasis-ctcf
test -d /home/roman/P/convexAdam || git clone https://github.com/multimodallearning/convexAdam /home/roman/P/convexAdam
git -C /home/roman/P/convexAdam checkout b229e52e44b114e2040a503334c92269750c16b2
python -m pip install -e /home/roman/P/convexAdam

LIMIT=1 PROFILE=2 GPU=0 bash tools/runners/eval/external_audit.sh convexadam
LIMIT=0 PROFILE=2 GPU=0 bash tools/runners/eval/external_audit.sh convexadam
```

Вернуть только:

```text
results/external/convexadam/manifest.json
results/external/convexadam/per_case.csv
results/external/convexadam/run.log
results/audit/convexadam/audit.csv
results/audit/convexadam/exact_original.json
results/audit/convexadam/exact_repaired.json
results/audit/convexadam_smoke1/repeatability.json
```

Сами `.npz` оставить на сервере: manifest хранит отдельно SHA-256 массива и SHA-256 файла. Полный запуск
использует `ConvexAdam MIND-SSC API-default preset` и ремонтирует поля с рабочим запасом 0.0011 перед точной
проверкой заявляемого порога 0.001. Это не воспроизведение semantic Task3 leaderboard pipeline ConvexAdam.

FireANTs пока не запускать — его адаптер и точный preset ещё не прошли обязательную проверку совпадения native
warp на одном случае.

Smoke-команда сама выполняет два независимых CUDA-запуска. Из-за недетерминированного backward 3-D
`grid_sample`/`AvgPool3d` побитовое совпадение ConvexAdam не обещается. Gate проверяет разные run ID, общие
данные/commit/CUDA stack, SHA-256 всех артефактов и заранее фиксированные допуски для поля, Dice и topology-
метрик; точные статусы тоже должны совпасть. Точный сертификат всегда относится к SHA-256 конкретного
сохранённого float32-поля.
