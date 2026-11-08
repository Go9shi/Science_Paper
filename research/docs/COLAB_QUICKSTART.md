# Актуальный запуск в Google Colab

Этот файл — короткая инструкция. История артефактов/ошибок находится в STAGE1/2/3_VALIDATION.md.
План новых запусков — [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md).
**Этап 4 пока не выполнен; новых passed tests/метрик не заявляем.**

## Какую версию открыть

После публикации коммита открыть `research/notebooks/04_pilot_experiments.ipynb` **из того же SHA**.
В Colab выбрать Runtime Version 2026.07 / Python 3.12 / CPU.
При переходе на другой checkout — Runtime → Disconnect and delete runtime, подключиться заново.
Повтор одной ячейки не очищает ранее импортированный src. Drive-файлы сохранятся.

В первой кодовой ячейке notebook 04:

```python
GIT_REF = "<полный опубликованный SHA этапа 4>"
DEVICE = "cpu"
RUN_TESTS = False
ENABLE_TRAINING = False
BATCH_ID = ""
RECOVER_INTERRUPTED = False
```

Вместо placeholder нужен настоящий SHA нового коммита, содержащего notebook 04.
`PROJECT_ROOT` задаётся автоматически после Git checkout. Подключить Drive интерфейсом Google,
затем выполнять ячейки сверху вниз. Исходные данные уже должны находиться в
`MyDrive/ppo_turnover_research/stage1/runs/20261107T154124Z-9f1f7c1e`.
Новый download котировок или повтор Stage 1 не нужен.

## Три последовательных режима

1. **Просмотр плана:** оба флага False. Проверяются зависимости/Drive, сохраняется план и provenance.
   Нет pytest, чтения реальных котировок или обучения. Ожидается 27 planned jobs / 442368 steps,
   completed=0 для новой batch. Сохранить напечатанный BATCH_ID.
2. **Проверка новой версии:** в новой попытке указать прежний BATCH_ID, RUN_TESTS=True,
   ENABLE_TRAINING=False. Полный pytest включает маленькие синтетические обучения; затем
   проверяются train/validation. Реальные pilot jobs не запускаются. Нужны zero failures/errors/skips.
3. **Реальный пилот:** тот же SHA/BATCH_ID, RUN_TESTS=True, ENABLE_TRAINING=True.
   Notebook снова создаст новую попытку с полным pytest; только после успеха начнутся jobs.
   **Это явное разрешение на 27 запусков по 16384 шага.** Fixed/adaptive — по 12 запусков.
   Не выбирать параметры по test, не останавливать семейство после «хорошего» результата.

Для новой попытки внутри того же runtime повторить секцию 4 и следующие, чтобы получить новый
ATTEMPT_DIR. Не перезаписывать существующий test_gate; для смены GIT_REF начать с новой сессии.
При несовпадении библиотек в памяти после pip installation перезапустить runtime и начать сверху.
Не обходить failures, checksums или новые dependency conflicts.

## Продолжение после отключения

Открыть тот же commit, выбрать тот же runtime/device, задать прежний BATCH_ID,
RUN_TESTS=True и ENABLE_TRAINING=True. Выполнить сверху вниз.
Завершённые jobs проверятся и будут пропущены; incomplete/failed job получит новый run ID
и полный исходный бюджет с тем же seed. Это перезапуск job, не битовое продолжение optimizer/RNG.

Если notebook сообщает о writer lock, сначала убедиться, что старый runtime остановлен,
затем явно задать RECOVER_INTERRUPTED=True и повторить запуск в новой попытке.
Не запускать одновременно два notebook для одной batch. Изменение кода/плана/совместимости
контекста требует новой batch; предыдущие результаты не удаляются.

## Где результаты

`MyDrive/ppo_turnover_research/stage4/batches/<BATCH_ID>/`:

- `batch_report.json`, sidecar, `resolved_plan.json`, `batch.json`;
- `run_summary.csv` — все jobs, включая pending/failed; `attempt_history.csv` — все retries;
- `candidate_summary.csv`, `selection.json` — только полный допустимый validation выбор;
- `attempts/<id>/` — источники, environment/pip reports/freeze, pytest log/JUnit,
  test_gate, attempt metadata/sidecar и batch_snapshot с историческим снимком сводок;
- `runs/<run_id>/` — model/normalization/checkpoints, metadata/config/hashes,
  финансовые CSV и метрики, training logs и same-runtime reload check.

Для приёмки прислать перечисленные отчёты/сводки и provenance последней попытки,
плюс metadata/sidecar и финансовые/тренировочные журналы всех runs. Models/checkpoints
оставить на Drive, прикладывать при необходимости проверки загрузки.
`preview_complete`, `checks_complete`, `diagnostic_complete` не означают обученный пилот.

## Старые модели этапа 3

Для них используется notebook **03**, не 04. В 03:
GIT_REF — SHA кода evaluator (c51f107 либо совместимая опубликованная версия),
MODEL_GIT_SHA — исходный training SHA a768abcadd295f492d35c6f023b5efcad86fb9cb,
EXECUTION_MODE="audit_only". Межсессионный exact=false сохраняется в отчёте.
Для новых jobs этапа 4 evaluator/training SHA совпадают с текущим опубликованным SHA;
Stage 1 snapshot имеет отдельную неизменяемую provenance. Повторно обучать старые smoke-модели не требуется.
