# Этап 3: инфраструктура PPO и протокол технической проверки

## Статус и область

**Текущий рекомендуемый запуск:** обновлённый notebook 03 с revision
`ppo-infrastructure-v2-audit`, `EXECUTION_MODE="audit_only"`.
Параметр `GIT_REF` указывает на новый commit evaluator/notebook; `MODEL_GIT_SHA` остаётся
`a768abcadd295f492d35c6f023b5efcad86fb9cb` для уже сохранённых моделей.
Три run ID последней проверенной группы заполнены по умолчанию. Режим audit не запускает
обучение и pytest с синтетическими обучениями; preflight и checksums выполняются.
Новая интеграция notebook подготовлена и проверена только статически; ещё не выполнена в Colab.

Причина повторения старого AssertionError после первого исправляющего commit:
ноутбук v1 продолжал вызывать строгий replay_saved_run. Добавление отдельного скрипта
не переключало существующую ячейку; при старом GIT_REF загружался старый код.
В v2 аудит подключён непосредственно к ячейкам 7–9. SHA evaluator и обучения раздельно
записываются в session; оба участвуют в проверках привязки. Режим reload_only сохраняет
строгое exact-сравнение и ожидаемо остановится при исходном расхождении, audit_only собирает
доказательства без подмены exact-флагов. `diagnostic_complete` не является приёмкой exact replay.

Порядок после публикации нового commit:

1. Открыть notebook 03 **из нового commit**, выбрать новый Colab runtime 2026.07 / CPU.
2. В первой кодовой ячейке указать новый полный SHA в GIT_REF; оставить audit_only,
   прежний MODEL_GIT_SHA и заполненные run IDs. Старый notebook v1 не обновляется от
   изменения GIT_REF автоматически — нужно открыть сам новый .ipynb.
3. Выполнить сверху вниз. Сохраняются session и отдельный каталог replay_audits;
   финал выводит их пути, точные флаги и разные SHA. Даже при original_exact_replay=false
   диагностический сбор заканчивается без старого AssertionError; критические ошибки
   checksum/config/version по-прежнему останавливают выполнение.
4. Предоставить обе папки `stage3/sessions/<id>` и `stage3/replay_audits/<id>`.

Тест маршрутизации notebook подготовлен для audit_only и reload_only: разные evaluator/model
SHA разрешены при верной привязке моделей, неверный MODEL_GIT_SHA блокируется, обучение
не вызывается. Результата исполнения этого нового теста пока нет.

При разработке выполнены только статические проверки Python/JSON/notebook.
После разработки получен и 2026-11-08 проверен архив пользовательских Colab-запусков:
обучение smoke и повторная загрузка внутри той же сессии подтверждены в объёме ниже.
Второй пользовательский архив подтвердил 97 пройденных тестов и preflight каждой из трёх сессий.
Полная приёмка этапа остаётся открытой: свидетельства загрузки после перезапуска runtime
ещё не предоставлены. Старые 83 теста этапа 2 не подменяют проверку текущей версии.
Codex локально обучение или численные тесты не выполнял.

### Проверенные артефакты Colab

Архив `drive-download-20261107T190851Z-1-001.zip`, 261 файл,
SHA-256 `a9fb6cefafceaab28086dda1e049a0ca13d6053f09fa4cfe560cfc334ccff9be`.
Все девять runs относятся к commit `a768abcadd295f492d35c6f023b5efcad86fb9cb`,
run_kind=smoke, random_seed=42, CPU, budget=actual training_steps=4096.

| Session ID в provenance | none run | fixed run | adaptive run |
|---|---|---|---|
| 20261107T183925Z-adefbb98 | 20261107T184058Z-none-b074fec0 | 20261107T184121Z-fixed-7252353d | 20261107T184140Z-adaptive-f9bd922e |
| 20261107T185410Z-340cccff | 20261107T185625Z-none-d541e8e8 | 20261107T185647Z-fixed-fe27326b | 20261107T185706Z-adaptive-da7e72f3 |
| 20261107T190521Z-8377c857 | 20261107T190619Z-none-ad65b98b | 20261107T190633Z-fixed-d87313a4 | 20261107T190645Z-adaptive-6f734f42 |

Проверено чтением готовых файлов, без запуска PPO/финансовых расчётов:

- Все девять metadata sidecar совпали; проверены все 243 SHA-256 из run manifests
  (по 27 артефактов на run), отсутствующих файлов и несовпадений нет.
- Все 45 checkpoint manifests полные, их hashes согласованы; поле num_timesteps
  в JSON внутри model.zip совпадает с steps manifest. Pickle и Torch tensors не загружались.
- Каждый training_financials.csv содержит 4096 последовательных training_step, даты только
  2010-01-04…2018-12-31. Каждый validation_financials.csv — 757 уникальных дат
  2019-01-02…2021-12-31. data_as_of строго раньше даты решения во всех строках.
- Сохранённые числовые поля финансов/действий/весов конечны; каждый training_metrics.json
  содержит 17 записей, основные losses/KL присутствуют и конечны, null не найден.
  Это проверка сохранённых журналов; исходные массивы всех observations в архив не входят.
- Config.json совпадают с metadata; все конфигурации одинаковы за исключением reward_mode.
  Контракты данных и версии библиотек совпадают между девятью runs.
- reload_check.json всех runs: status=passed, same_runtime_reload=true,
  exact_actions_and_finances=true, normalization_frozen=true,
  fresh_session_reload=not_yet_verified. Это свидетельства Colab-процедуры save/load,
  не самостоятельное повторение Codex и не подтверждение новой сессии.
- 33 заявленных source hashes совпали с точными Git blobs указанного commit.
  Сам каталог provenance/source.bundle в первый архив не включён; получен во втором архиве ниже.

Общий Stage 1 источник `20261107T154124Z-9f1f7c1e`, metadata SHA-256
`adf21713ad9d63871e7faf37456b244cea390a967d88abb5cb5100d113c2dd64`,
snapshot `daily-v1`, manifest SHA-256
`c7ebbcb88bb04de4daa3e394258b7bd6472f55ec4f4ee4e892486aa5eb41cb19`,
sigma_ref `0.11142612683693373`. В provenance train цены ограничены до 2019,
validation — до 2022; features_test не входит в read_files, test_used=false.
Проверенный код загрузчика согласуется с этим ограничением; исходные данные здесь не перечитывались.

Записанное окружение: Colab tag `release-colab-external-images_20260715-060047_RC00`,
Linux x86_64/glibc 2.35, NumPy 2.0.2, pandas 2.2.2, Torch 2.11.0+cpu,
Gymnasium 1.2.0, SB3 2.7.0, PyArrow 20.0.0, CUDA отсутствует.
Файлов environment.json/pip-freeze/pip-check в первом архиве нет; они получены и проверены
во втором архиве ниже.

Первые две группы имеют побайтово одинаковые training/validation financial CSV,
training_metrics.json и final vecnormalize.pkl для соответствующих режимов.
Третья группа по этим файлам отличается при том же записанном seed/config/versions.
Например, сохранённый Net Sharpe none: `1.254365166776023` в первых двух группах
и `1.2543651916875798` в третьей. Это небольшое различие готовых результатов,
не установленная причина недетерминизма и не повторная загрузка прежней модели:
созданы новые runs с новым обучением. При совпавших source hashes, pip freeze и записанных
версиях причина не установлена; полная конфигурация CPU/backend и внутреннее состояние
runtime не записывались.
Нельзя объявлять все повторные обучения побитово одинаковыми или требовать их повторения
вместо отдельного reload_only. Эти три группы с одним seed не являются тремя независимыми seeds.

### Дополнительная проверка sessions и тестов

Получен архив `drive-download-20261107T191147Z-1-001.zip`, 150 файлов,
SHA-256 `76e9cf930fedd93dee1556ff181b4a16d9ce21590e3acf63113565b4375c371d`.
Включены все три sessions из таблицы выше, по 50 файлов на каждую.

- Все три session sidecar совпали; все 144 хеша артефактов из session manifests проверены,
  отсутствующих файлов и несовпадений нет. Проверены 33 provenance-файла каждой session;
  source.bundle совпал по хешу, его восстановление не выполнялось.
- Все JUnit содержат по **97 testcase, 0 failures, 0 errors, 0 skipped**.
  Session tests status и pytest.log согласованы с JUnit. Время из логов по порядку таблицы:
  32.77, 33.71 и 22.64 секунды. Два предупреждения Gymnasium относятся к бесконечным
  границам observation_space и ранее были документированы.
- Присутствуют и прошли Stage 2 Gymnasium/SB3 checkers, tiny synthetic training/save-load,
  финансовая независимость от shaping, fixed/adaptive alpha=0, запрет test и фильтрация дат.
- Проверены связи session → три run ID и обратно, Git SHA, source hashes, версии/Python,
  все пять session_input_sha256 из metadata каждого run. Validation summary согласована
  с готовыми validation_metrics.json всех девяти runs; финансовые метрики не пересчитывались.
- Preflight: project_imports=passed, dependency_closure=passed, problems=[],
  NumPy/Torch bridge=passed, drive_read_write=passed. Python 3.12.13,
  Torch 2.11.0+cpu, остальные версии соответствуют описанной выше baseline.
- Raw pip check exit code=1 во всех случаях: только IPython 7.34.0/missing Jedi и
  Ibis 9.5.0/Toolz 1.2.0. Policy passed_with_known_warnings, blocking_or_unreviewed=[].
  Нулевой общий pip check не заявляется. По install reports install=[]: именно эти
  вызовы pip ничего не устанавливали; это не утверждение обо всех предшествующих установках.
  Полные pip-freeze совпали по хешу между тремя sessions.
- Просмотрены готовые validation/training PNG последней session: NAV режимов визуально близки,
  коэффициенты none/fixed/adaptive различаются согласно конфигурации; value loss снижается
  на коротком бюджете. Графики не доказывают сходимость, оптимальность или преимущество метода.

Все три session.json имеют execution_mode=train, training_performed_in_this_session=true,
fresh_session_replay=not_yet_verified. Новые session IDs и повторное обучение не заменяют
проверку загрузки старого checkpoint после удаления runtime. В обоих архивах нет
replay_report.json или session с execution_mode=reload_only.

Для завершения приёмки остаётся отдельный runtime restart и режим reload_only с исходными
run IDs, затем новый session и три replay_report.json. Повторно обучать три политики не требуется.

Разницы пилотных Sharpe не интерпретируются как статистическое или экономическое превосходство.
Статус: **97 тестов, окружение, smoke и same-runtime save/load подтверждены по артефактам;
fresh-session reload ещё не подтверждён, полная приёмка этапа 3 не завершена**.

### Ошибка exact replay после перезапуска и подготовленное исправление

После получения двух архивов пользователь запустил reload_only и предоставил traceback:
проверка версий прошла, затем assert_frame_equal(check_exact=True) не прошёл по portfolio_value.
Например, первая строка исходного CSV `1009.534986593021`, новая оценка
`1009.5349865765468`. Это наблюдаемый отказ exact replay; он не превращается в passed
из-за малости разницы. DeprecationWarning jupyter_client об utcnow не является этим AssertionError.

В commit a768abc выявлена асимметрия: build_ppo вызывает torch.set_num_threads(1) и
torch.use_deterministic_algorithms(True), а load_saved_policy этих process-global настроек
не восстанавливает. В режиме train последующая оценка наследует их; в новой сессии
reload_only build_ppo не вызывается. Настройки inference поэтому не гарантированно одинаковы.
Это установленный дефект кода и вероятное объяснение traceback, но его причинная роль
должна быть проверена повторным Colab replay. Число потоков в сессии отказа не приложено.

Подготовлено исправление: общий configure_torch_runtime вызывается до построения PPO
и до загрузки политики. Число intra-op/inter-op threads, deterministic flag и MKLDNN flag
записываются в новые checkpoint/replay reports; в metadata обучения — после настройки.
Добавлен регрессионный тест: маленький synthetic run и exact replay в отдельном
Python-процессе, которому до загрузки намеренно заданы два потока и deterministic=False.
Загрузчик должен восстановить один поток/deterministic=True и пройти исходный exact replay.
Этот новый тест пока подготовлен, **в Colab не выполнен**; ранее подтверждённые 97 cases
относятся к a768abc и не доказывают прохождение новой проверки.

Для проверки уже существующих runs из a768abc **не нужно переобучать модели, менять SHA,
NAV, CSV, normalization или допуски**. В текущем старом notebook перед ячейкой 7 выполнить:

```python
import torch

runtime_before = {
    "threads": torch.get_num_threads(),
    "deterministic": torch.are_deterministic_algorithms_enabled(),
}
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
write_json(SESSION_DIR / "torch_runtime_fix.json", {
    "git_sha": SOURCE_COMMIT,
    "before": runtime_before,
    "after": {"threads": torch.get_num_threads(),
              "deterministic": torch.are_deterministic_algorithms_enabled()},
    "previous_session_error": session.get("error"),
    "reason": "Match the existing build_ppo runtime settings before reload_only inference",
})
print("Torch runtime:", runtime_before, "->", torch.get_num_threads(),
      torch.are_deterministic_algorithms_enabled())
```

Затем повторить ячейку 7 и остальные итоговые ячейки. Файл runtime_fix входит в итоговый
session manifest. При новом отказе сохранить traceback и session, не ослаблять assert автоматически.
Этот runtime-only обход явно фиксируется как дополнение к исходному notebook; версии модели
и исходный provenance не переписываются. Новый код загрузчика вступит в силу только в будущем
опубликованном commit; для проверки старых моделей notebook сохраняет требование исходного SHA.
Даже после этой настройки точное равенство между разными CPU/backend/platform не обещается.

Обоснование API: [PyTorch 2.11 set_num_threads](https://docs.pytorch.org/docs/2.11/generated/torch.set_num_threads.html)
предписывает задавать потоки до eager/JIT/autograd-операций;
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
ограничивает гарантии между окружениями. Локально проверен только синтаксис исправления.

### Повторный отказ после настройки потоков

Пользователь предоставил повторный AssertionError с теми же исходным/новым NAV.
Устранение отказа настройкой потоков не подтверждено; считать гипотезу о потоках
доказанной причиной нельзя. Дефект асимметричной настройки исправлен в подготовленном
коде, но может не быть единственным источником разницы. Первый original NAV из traceback
совпадает с run `20261107T190619Z-none-ad65b98b` из последней группы архива.

Следующий диагностический шаг разделяет два источника: действие сохранённой политики
и исполнение уже сохранённого действия. Подготовлен first_step_replay_diagnostic:
сравнивает первое действие точно, затем подаёт исходное action_json новой среде и сравнивает
первый NAV. Сохраняет CPU capability, Torch build configuration, matmul precision, dtype,
runtime settings и начальное нормализованное observation. Совпадение первого шага не
доказывает равенство полного эпизода. В новой версии replay_saved_run при AssertionError
сохраняет новую траекторию, метрики и failed report вне исходного run, затем повторно
поднимает исключение. Проверка exact не ослабляется и failed не объявляется passed.

Для старого открытого notebook, без публикации новой версии кода, выполнить отдельную ячейку:

```python
import json
import numpy as np
import pandas as pd
import torch
from src.ppo_training import load_saved_policy

root = OUTPUT_ROOT / "runs" / RELOAD_RUN_IDS["none"]
row = pd.read_csv(root / "validation_financials.csv", float_precision="round_trip").iloc[0]
model, vector, _, _ = load_saved_policy(root, validation)
try:
    obs = vector.reset()
    predicted, _ = model.predict(obs, deterministic=True)
    saved = np.asarray(json.loads(row["action_json"]), dtype=model.action_space.dtype)
    _, _, _, infos = vector.step(saved.reshape(1, -1))
    diagnostic = {
        "run": root.name,
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cpu": getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "first_action_exact": bool(np.array_equal(saved, predicted[0])),
        "saved_action": saved.tolist(),
        "new_action": predicted[0].tolist(),
        "saved_nav": float(row["portfolio_value"]),
        "nav_using_saved_action": float(infos[0]["portfolio_value"]),
        "saved_action_execution_exact": bool(infos[0]["portfolio_value"] == float(row["portfolio_value"])),
    }
finally:
    vector.close()
write_json(SESSION_DIR / "replay_first_step_diagnostic.json", diagnostic)
print(json.dumps(diagnostic, indent=2))
```

При first_action_exact=false и saved_action_execution_exact=true расхождение
локализуется до нейросетевого вывода или его входов, а одинаковое исходное действие
воспроизводит финансовый первый шаг. Это не доказывает конкретную причину (CPU/backend,
precision или observation) и не подтверждает весь эпизод. Если saved_action_execution_exact=false,
нужно дополнительно сравнить конкретные цены, параметры и бухгалтерию исполнения до изменения PPO.
Флаги exact/failed сохраняют исходный смысл; допуски не выбираются по размеру полученного отклонения.
Выполнение этой диагностики ожидается в Colab. Локально только синтаксические проверки.

### Подтверждённая локализация первого шага и аудит всего validation

Пользователь предоставил JSON диагностики run `20261107T190619Z-none-ad65b98b`:
threads=1, deterministic=true, actions_equal=false. Сохранённое действие, поданное в новую
среду, дало **точно** исходный первый NAV `1009.534986593021`.
Это подтверждает первый финансовый переход при исходном действии, но не весь эпизод.
Новые значения отдельных компонентов действия отличаются в младших разрядах float32.
Конкретная причина различия политики/её входов между runtime пока не установлена.
Не приписывать это автоматически случайному seed, повреждению checkpoint или конкретной модели CPU.

Подготовлен `src/replay_audit.py`, который может выполняться как отдельный загружаемый
скрипт в текущем Colab с исходным checkout a768abc. Он использует существующие src modules,
проверяет хеши/версии, запрещает test и новое обучение; исходные runs не модифицирует.
Новый evaluator имеет собственный SHA-256 в отчёте, без подмены source Git SHA модели.

Для каждого из трёх matched runs выполняются:

1. Воспроизведение **всех** исходных действий на validation (teacher forcing). Полный CSV,
   включая NAV, комиссии, оборот, reward и веса, сравнивается с исходным точно.
2. Предсказания политики на состояниях этой исходной траектории. Сохраняются действия,
   нормализованные observation, точное равенство и максимальная абсолютная разница.
   Это отделяет начальную ошибку вывода от последующего feedback через NAV/веса.
3. Два независимых load и обычных closed-loop evaluation внутри текущего runtime.
   Действия, финансы и метрики между ними сравниваются точно; отличие от старого runtime
   сохраняется отдельно. Сохраняются новая финансовая траектория и метрики, без настройки параметров.

Status `diagnostic_complete` означает только завершение аудита. `closed_loop_original.exact=false`
не превращается в passed; допусков для действий/NAV/метрик не задаётся. Float32 epsilon_at_one
записывается только для масштаба диагностики, не является формальной границей погрешности нейросети.
Одно machine epsilon не равно ULP для всех значений действия.

Для запуска без нового обучения или изменения старого GIT_REF:

1. Через левую панель Files Google Colab загрузить локальный `research/src/replay_audit.py`
   как `/content/replay_audit.py`.
2. В текущем notebook, где validation уже загружен, выполнить отдельную ячейку:

```python
import runpy

audit = runpy.run_path("/content/replay_audit.py")["audit_saved_policies"]
audit_dir = audit(
    validation,
    {mode: OUTPUT_ROOT / "runs" / RELOAD_RUN_IDS[mode] for mode in REWARD_MODES},
    OUTPUT_ROOT / "replay_audits",
)
```

3. Предоставить созданный каталог `stage3/replay_audits/<UTC-UUID>` с audit_report.json,
   sidecar и тремя подпапками. Старую ячейку 7 с exact replay пока повторять не нужно.
   Скрипт выполняет только технические validation replay, PPO learn не вызывается.

Подготовлены синтетические тесты полного аудита, сохранности metadata и отличия exact
от расхождения на один representable шаг. Они пока не выполнялись в Colab.
Фактически подтверждённым результатом остаётся только предоставленная диагностика первого шага;
весь новый аудит и строгий fresh-session replay ещё не подтверждены.

Первичная документация: [PyTorch 2.11 Numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
не гарантирует побитовое совпадение вычислений между платформами;
[NumPy finfo](https://numpy.org/doc/2.0/reference/generated/numpy.finfo.html) определяет epsilon как
расстояние от единицы до следующего representable значения, а не допуск для портфельной траектории.

Цель — техническая готовность общей процедуры, а не сравнение эффективности методов.
Гипотеза будущего исследования: адаптация штрафа может повысить Net Sharpe относительно
оптимально настроенного fixed. Текущие lambda/alpha пилотные; fixed не объявляется оптимальным.
Ни поиск гиперпараметров, ни заключительное независимое исследование здесь не проводятся.

## Данные и причинность

Источник — неизменяемый Stage 1 run `20261107T154124Z-9f1f7c1e`, snapshot `daily-v1`.
Данные не скачиваются заново и не перезаписываются. Базовая методология этапов 1–2 сохраняется.

`src/training_data.py` разрешает только train (2010–2018) и validation (2019–2021).
`features_test.parquet` не читается. На общие ценовые таблицы применяется Parquet predicate
по индексу дат: `<2019-01-01` для train, `<2022-01-01` для validation.
Хеширование всего ценового файла проверяет сжатые байты, не используется как доступ
к тестовым наблюдениям. Footer/schema и Stage 1 manifest могут содержать границы test;
значения, доходности и метрики test не материализуются и не используются.
Если физический row group Parquet пересекает границу, библиотека может декодировать его
для фильтрации; в DataFrame/расчёты поступают только разрешённые строки. Это логическая
изоляция исследовательских данных, а не отдельное физическое хранилище test.
Для строгого запрета даже чтения общих файлов потребуется заранее отдельный development-export.

Сохранённые признаки сверяются с `build_features` на разрешённой истории. Решение после
Close[t−1], исполнение на синтетическом adjusted Open[t], оценка на adjusted Close[t].
Наблюдение не содержит Open[t] или Close[t]. Sigma[t] — годовой нецентрированный RMS
20 завершённых дневных лог-доходностей SPY до t−1 включительно; sigma_ref — медиана
sigma только по train, заново сверяемая с Stage 1. Ни scaler, ни коэффициент не fit на validation.

Исполнение остаётся моделью `synthetic_adjusted_open_close_v1`, а не номинальной бухгалтерией
акций и денежных дивидендов. Ретроспективные adjusted prices и survivorship bias современного
universe остаются ограничениями; абсолютные скорректированные цены не входят в observation.

## Одинаковые условия

Используется стандартный `stable_baselines3.PPO` версии 2.7.0, `MlpPolicy`.
Loss, optimizer, GAE и clipping SB3 не изменяются. Во всех трёх режимах одинаковы:

- признаки (включая SPY sigma), пространства наблюдений и действий;
- сети actor/critic: отдельные MLP 64–64, Tanh;
- цены, даты, начальный капитал 1000, начальные позиции 100% cash;
- long-only, дробные синтетические позиции, cash без процентов, без плеча;
- комиссия 5 bps на фактические покупки и продажи рискованных активов;
- преобразование действий и точная самофинансируемая ребалансировка этапа 2;
- алгоритм, seed, бюджет и процедура оценки final checkpoint.

Наблюдения нормализуются `VecNormalize` с одинаковой процедурой и clip_obs=10.
Статистики обновляются только во время train и сохраняются рядом с каждой моделью.
На validation они заморожены (`training=False`), `norm_reward=False` всегда.
Статистики, зависящие от траектории портфеля, могут различаться между обученными политиками;
это следствие разных действий при общей процедуре, не дополнительный признак режима.
`VecCheckNan` и callback останавливают запуск при NaN/Inf. Terminal observations могут выходить
за нормированный clip после обёрток SB3; финансы читаются из `info`, а не из terminal padding.

## Reward и новая конфигурируемая адаптация

Базовый reward среды этапа 2 сохраняется как **net log return**:
`r_net_log[t] = log(NAV[t]/NAV[t−1])`.
Простые net return сохраняются отдельно и используются для Sharpe/volatility.
`reward[t] = r_net_log[t] − lambda[t] * turnover[t]`.
Комиссия уже входит в NAV; регуляризатор не списывается из NAV повторно.

- none: lambda[t]=0;
- fixed: lambda[t]=lambda_0;
- adaptive: q=sigma[t]/sigma_ref, multiplier=min(q**alpha, ratio_cap),
  lambda[t]=clip(lambda_0*multiplier, lambda_min, lambda_max).

При alpha<0 q предварительно ограничивается снизу `ratio_floor=1e-6`, чтобы sigma=0
не приводила к делению на ноль. При alpha>0 и sigma=0 multiplier=0.
При alpha=0 helper возвращает ровно lambda_0, включая sigma=0, без неоднозначности 0**0.
Конфигурация требует lambda_min ≤ lambda_0 ≤ lambda_max, поэтому alpha=0 точно эквивалентен fixed.
`ratio_cap` ограничивает **множитель после степени**, а не q до степени.
Формула вычисляется через логарифмы с защитой от переполнения. Положительная/отрицательная
alpha допускают обе направленности чувствительности; предпочтительное направление не постулируется.

Изменение протокола относительно этапа 2 явно ограничено параметризацией reward:
старый EnvConfig по умолчанию оставляет alpha=1, lambda_min=0, lambda_max=None и прежнее
lambda_base*min(q,5), включая нулевую sigma. Финансовая механика не меняется.
В конфигурациях этапа 3 alpha=0.5, lambda_0=0.001, ratio_cap=5, lambda_max=0.01.
Это заранее заданная техническая проверка; отрицательная alpha требует отдельной заранее
зафиксированной конфигурации/коммита и отдельной группы, без просмотра test.

## PPO-конфигурации

| Параметр | Smoke | Pilot |
|---|---:|---:|
| total_timesteps на режим | 4096 | 65536 |
| checkpoint_steps | 1024 | 8192 |
| learning_rate | 0.0003 | 0.0003 |
| n_steps / batch_size / n_epochs | 256 / 64 / 5 | 256 / 64 / 5 |
| gamma / gae_lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| clip_range / ent_coef / vf_coef | 0.2 / 0 / 0.5 | 0.2 / 0 / 0.5 |
| random_seed | 42 | 42 |
| device по умолчанию | cpu | cpu |

Один DummyVecEnv. Стандартные SB3 defaults: normalize_advantage=True,
max_grad_norm=0.5, use_sde=False, target_kl=None; Adam policy defaults не меняются.
Все конфигурации вынесены в `configs/ppo_smoke.json`, `configs/ppo_pilot.json` и валидируются
`TrainingConfig`: полные rollouts, делимые minibatches, ограниченный бюджет, только validation для оценки.
Во всех трёх режимах меняется только reward_mode. Smoke всей группы — 12288 реальных train-шагов;
pilot — 196608. Pilot запускается только явным выбором пользователя в Colab.
Seed задаётся Python, NumPy, Torch и PPO/среде; CPU threads=1,
torch deterministic algorithms=True. CUDA необязательна, CUBLAS_WORKSPACE_CONFIG задаётся до Torch.
Равенство между разными CUDA/драйверами/устройствами не обещается; GPU пока не проверен.

## Процедура оценки и метрики

Оценивается только final, без выбора лучшего checkpoint по validation. Один полный validation-эпизод,
общие даты и капитал, 100% cash на старте, без переноса train-позиций. Первый overnight gap
при нулевых позициях не создаёт дохода. Первоначальное инвестирование включено в оборот и комиссии.
Финальная ликвидация не проводится. Train может циклически проходить только свой период;
на terminal сбрасывается cash, bootstrap через границу train/validation отсутствует.

`predict(..., deterministic=True)`, замороженные веса и normalization. По каждой дате сохраняются
действие, веса, NAV, gross/net simple и log returns, комиссии, traded notional, turnover,
sigma, lambda, penalty, reward и cash. Метрики вычисляет только `src/financial_metrics.py`:

- Net Sharpe = sqrt(252) * mean(r_net − rf_daily) / std(r_net − rf_daily, ddof=1).
  rf_daily=(1+rf_annual)**(1/252)−1, rf_annual=0 в текущем протоколе.
- CAGR = (NAV_end/NAV_initial)**(252/N)−1, N — число дневных интервалов.
- Maximum Drawdown = max(1−NAV/running_max_NAV), с initial NAV в траектории; положительная величина.
- Annualized Volatility = std(simple net returns, ddof=1)*sqrt(252).
- Turnover = sum(abs(actual risky trades))/NAV_before_trade: двусторонний рискованный notional,
  без фиктивной cash-сделки. Сохраняются средний дневной, сумма и annualized=mean*252.
- Cumulative Net Return = NAV_end/NAV_initial−1, сверяется с произведением (1+r_net).

Нулевая дисперсия (≤1e−15) или менее двух наблюдений дают Sharpe=null с явным статусом;
это не NaN и не условный успешный score. Volatility при менее двух наблюдений тоже null.
Undefined train/explained_variance кодируется null отдельно; основные PPO losses/KL должны быть конечны.
Training reward не используется ни в одной финансовой метрике. Никаких p-values/CI или
утверждений о превосходстве на этом этапе. В будущем нужны одинаковые бюджеты настройки fixed/adaptive,
несколько общих seeds, учёт зависимости доходностей и множественного выбора, заранее фиксированный
критерий модели и независимый test после фиксации протокола.

## Артефакты и отключение Colab

Все пути относительно `/content/drive/MyDrive/ppo_turnover_research/stage3`:

- `preflight/<id>`: установки/версии до установки, imports, environment, raw pip check;
- `sessions/<id>`: исходники/скиллы/документы/конфиги, source.bundle, hashes, environment,
  pip freeze, pytest log/JUnit, common config, сводка и графики, session.json/sidecar;
- `runs/<timestamp>-<mode>-<uuid>`: config, metadata/sidecar, data contract, seed, версии,
  budget/actual steps, training_financials.csv, training_metrics.json, logs/progress.csv,
  validation_financials.csv, validation_metrics.json, reload_check.json, checkpoints;
- `replays/<session>/<mode>`: доказательства отдельной повторной оценки без обучения.

Новые пути создаются с exist_ok=False; завершённые runs не дополняются графиками.
Metadata ссылается на общий session и хеши его входных provenance-файлов.
Чекпойнт включает model.zip, vecnormalize.pkl, config и manifest контрольных сумм.
Manifest записывается последним; без него checkpoint считается незавершённым.
Промежуточные checkpoint во время rollout сделаны **до** обновления соответствующего rollout;
их steps — число собранных переходов, а не число уже использованных optimizer-переходов.
Final сохраняется после всех PPO updates. CSV финансов сбрасывается на Drive при checkpoint.

При штатной ошибке metadata получает failed; если Colab убит, status может остаться started.
Существующие полные checkpoints можно загрузить через load_saved_policy с явным checkpoint-путём
для технической validation; они не подменяют final и не выбираются по лучшему результату.
Replay завершённого run сверяет хеши, все детерминированные действия, финансовые строки и метрики.
Точное продолжение прерванного обучения с RNG/rollout state не реализовано: новая попытка обучения
начинается заново с тем же seed в новой директории. Не выдавать checkpoint за доказательство
завершения бюджета или восстановления битовой траектории обучения.

## Запуск в Google Colab

1. Сохранить изменения проекта в Git, опубликовать новый commit на GitHub. Старый SHA этапа 2
   не содержит инфраструктуру этапа 3. Открыть `research/notebooks/03_ppo_training.ipynb` из нового commit.
2. Выбрать управляемый Colab Runtime Version 2026.07, Python 3.12, CPU. Новая сессия рекомендуется.
   По исходным этапам подтверждён Torch 2.11.0+cpu; для этой версии обучения подтверждения ещё нет.
3. В **первой кодовой ячейке** задать GIT_REF=полный новый SHA, SOURCE_RUN_ID указанного Stage 1,
   RUN_KIND="smoke", EXECUTION_MODE="train", DEVICE="cpu". Остальные режимы оставить общими.
4. Выполнять сверху вниз: Git checkout, Drive mount/read-back, pinned install без обновления Torch,
   imports/NumPy–Torch bridge/pip check, config/provenance, **весь pytest**, затем данные и обучение.
   Ошибку теста или новый dependency conflict не обходить. До обучения unit tests включают
   только крошечное синтетическое обучение, без реальных исследовательских результатов.
5. При успешном smoke сохранить вывод последней ячейки, session ID и все три run ID.
   Success печатается только после реального learn, save/load, детерминированной оценки и проверки хешей.
6. Для новой сессии: Runtime → Disconnect and delete runtime; подключиться заново, открыть тот же notebook.
   Оставить тот же GIT_REF/SOURCE_RUN_ID, указать EXECUTION_MODE="reload_only" и RELOAD_RUN_IDS
   из вывода. Все ячейки сверху; learn не вызывается. Требуются исходные Python/версии библиотек
   и устройство, записанные в run. Новая среда не подтверждается одним повторным запуском ячейки.
7. Pilot (RUN_KIND="pilot") — отдельная группа с новым ID; smoke не доказывает её завершение.
   Масштабный поиск или test запрещены на этом этапе.

Пример первой ячейки для повторной загрузки (значения ID взять из фактического вывода):

```python
EXECUTION_MODE = "reload_only"
RELOAD_RUN_IDS = {
    "none": "<actual-none-run-id>",
    "fixed": "<actual-fixed-run-id>",
    "adaptive": "<actual-adaptive-run-id>",
}
```

Не исполнять этот пример с placeholders. Не загружать pickle/model от неизвестных источников.

## Критерии приёмки

- Preflight подтверждён на Colab, активные зависимости согласованы; только ранее классифицированные
  системные предупреждения допустимы и сохранены в raw pip check.
- Полный pytest: zero failures/errors/skips, включая Stage 2 checkers и новые PPO cases.
- Все режимы имеют одинаковый исходный policy state при общем seed, одинаковые spaces и входы.
- Alpha=0 точно эквивалентен fixed; оба знака alpha и нулевая sigma проверены синтетически.
- Smoke каждого режима реально завершил ровно 4096 шагов; finite observations/rewards/core losses;
  реальные комиссии/оборот/штраф сохранены раздельно.
- Policy+normalization сохранены, два независимых load дают одинаковые действия/финансы в runtime.
  Отдельный Colab restart/reload_only повторяет их без learn.
- Финансовые формулы проходят известные синтетические примеры, изменение shaping не меняет
  метрики при фиксированной политике/действиях; validation не обновляет normalization.
- Хеши модели/конфигов/журналов/provenance проверены read-back; test_used=False,
  нет чтения features_test или использования 2022+ в DataFrame/расчётах.

Предоставить session directory (включая pytest.xml/log, environment.json, pip-check.txt/freeze,
session.json/sha256 и диагностики); по трём runs — metadata/sidecar, config,
training_metrics, validation_metrics, reload_check и финансовые CSV; после перезапуска —
новый session и replay_report.json каждого режима. Для проверки моделей нужны также
checkpoint manifest, model.zip и vecnormalize.pkl. Не публиковать эти объёмные артефакты в Git.

## Фактические проверки и открытые вопросы

Локально успешно: AST-разбор 20 Python-файлов src/tests, компиляция 10 кодовых ячеек
без исполнения, структурные проверки двух JSON-конфигов и notebook, `git diff --check`.
Все execution_count=null, outputs пусты. Это статические проверки, не результаты pytest.
Число собранных pytest cases устанавливается по JUnit Colab,
не вычисляется из названий функций. Фактически полученные позже результаты обучения
проверены по пользовательскому архиву и отдельно описаны в начале документа.

Остаются: fresh-session reload,
объяснение различий третьего повторного обучения, продолжительность сохранения Drive, GPU-детерминизм,
будущий бюджет настройки и seeds, чувствительность к периоду/издержкам/universe,
survivorship и point-in-time ограничения источника. Современный список и adjusted execution
ограничивают перенос результатов на реальное исполнение. Одного seed/короткого pilot недостаточно
для научного вывода; заключительный test остаётся закрытым.

## Использованные первичные источники

- [SB3 PPO 2.7.0](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html).
- [SB3 VecNormalize и VecEnv](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/vec_envs.html).
- [SB3 callbacks и момент сохранения](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/callbacks.html).
