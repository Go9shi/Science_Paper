# Актуальный запуск в Google Colab

Этот файл — короткая инструкция. История артефактов/ошибок находится в STAGE1/2/3/4_VALIDATION.md.
План новых запусков — [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md).
**Пилот этапа 4 выполнен: 27/27 jobs, 442368 шагов; перед обучением прошли 112 тестов.**
Артефакты batch `20261115T100446Z-757eee8d` проверены; отчёт и ограничения —
в [STAGE4_VALIDATION.md](STAGE4_VALIDATION.md). Повторять её обучение не требуется.
Ниже сохранена инструкция запуска/продолжения; новые обучения не запускать автоматически.

## После пилота: notebooks 05 и 06

Подробный порядок — [POSTPILOT_PROTOCOL.md](POSTPILOT_PROTOCOL.md).
Новый workflow пока не исполнялся в Colab; старые 112 passed не подтверждают новую версию.

1. Опубликовать новый evaluator и открыть `05_validation_and_freeze.ipynb` из его полного SHA.
   В новой независимой CPU-сессии указать этот SHA в GIT_REF;
   RUN_TESTS=True, RUN_REPLAY=True, INDEPENDENT_SESSION=True,
   RUN_ANALYSIS=True, SEAL_DECISIONS=True. MODEL_GIT_SHA и BATCH_ID оставить из notebook.
2. Выполнить сверху вниз. При несовпадении replay `diagnostic_complete` сохраняется,
   lock не создаётся и test закрыт. Прислать всю напечатанную папку stage5 session.
3. После seal записать SESSION_ID и lock SHA256 вне Drive до открытия test.
   В новой сессии открыть `06_locked_test_evaluation.ipynb` из **того же evaluator SHA**,
   указать DECISION_SESSION_ID, LOCK_SHA256, RUN_TESTS=True и ENABLE_TEST=True.
   Прислать папку stage6 session и весь каталог test_evaluations/<lock SHA>.

В 05/06 нет реального обучения. Test использует девять замороженных final-моделей,
cash на старте и одинаковые комиссии. Технические повторы сохраняются с тем же lock;
основной результат — первая полная попытка, повторы не считаются независимыми экспериментами.

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
