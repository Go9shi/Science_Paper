# Google Colab: окружение env-v2

Статус на 2026-11-07: **техническая проверка этапа 1 и CPU-профиля env-v2 завершена в пределах полученных свидетельств**. JUnit: 52 passed; активные зависимости и NumPy↔Torch smoke прошли. Пользователь предоставил вывод точного сравнения всех 13 prepared-таблиц: значения, индексы и типы совпали с baseline. Глобальный pip check сохраняет два известных предупреждения. Старые успешные запуски остаются неизменными.

## Фактически проверенный env-v2

RUN_ID `20261107T154124Z-9f1f7c1e`, commit `52fba3a13afdfd68a8bdfdfc24524344dbdc3ea5`. Получены metadata, validation_report, JUnit, environment, pip-check и pip-install-report.

- Python 3.12.13; runtime `release-colab-external-images_20260715-060047_RC00`.
- numpy 2.0.2; pandas 2.2.2; yfinance 0.2.65; pandas_market_calendars 5.1.1; matplotlib 3.10.3; pyarrow 20.0.0; pytest 8.4.1.
- gymnasium 1.2.0; stable-baselines3 2.7.0; torch **2.11.0+cpu**. Torch до установки имел ту же версию; torchvision 0.26.0+cpu и torchaudio 2.11.0+cpu записаны в before-install.
- RL preflight: imports/PPO import/CPU NumPy bridge passed; model_created=false, training_performed=false, project_environment_created=false. CUDA build=null, cuda_available=false: GPU/CUDA совместимость этим запуском не подтверждается.
- dependency_closure.status=passed, problems пуст. Среди транзитивных версий: exchange-calendars 4.13.2, packaging 26.2, cloudpickle 3.1.2. Остальные фактические версии перечислены в environment.json; это observation, не новый автоматически применяемый lock-файл.
- JUnit содержит 52 testcase без failures/errors/skipped; metadata/validation_report сообщают complete/verified.
- Все пять предоставленных файлов, кроме самого metadata, совпали с их SHA-256 в metadata. Полученный metadata имеет SHA-256 `adf21713ad9d63871e7faf37456b244cea390a967d88abb5cb5100d113c2dd64`; sidecar этого запуска не приложен.
- pip check: отсутствуют прежние конфликты NumPy/Numba и pandas/google-colab. Остались только missing Jedi для IPython и Toolz/Ibis; blocking_or_unreviewed пуст.
- Before-install уже содержит numpy 2.0.2, pandas 2.2.2, missing Jedi и toolz 1.2.0. Список install в предоставленном pip-install-report пуст: именно этот вызов pip ничего не устанавливал. Это не доказывает отсутствие установок на более ранних шагах/сеансах. Jedi/Ibis предупреждения существовали до данного вызова pip.

Окружение проверено для CPU-импортов и текущего pipeline; это не доказательство корректности обучения, которого не было. Численные операции выполнены пользователем в Colab, не локально Codex.

### Результат сравнения данных после смены pandas/NumPy

Snapshot daily-v1 reused=true, его manifest SHA-256 остался `c7ebbcb88bb04de4daa3e394258b7bd6472f55ec4f4ee4e892486aa5eb41cb19`; размеры и границы splits, sigma_ref `0.11142612683693373` прежние. Однако все 13 hashes prepared Parquet отличаются от старой среды. Изменение метаданных сериализации pandas может объяснить различие, но без чтения обеих версий нельзя считать это установленной причиной или утверждать идентичность всех значений.

Пользователь впоследствии выполнил в Colab `pd.testing.assert_frame_equal(..., check_exact=True, check_freq=False)` для всех 13 файлов между baseline `20261107T145440Z-00c5be42` и env-v2 `20261107T154124Z-9f1f7c1e`. Получен stdout с отдельным «Совпадает» для каждой таблицы и итоговым сообщением без traceback. Все сравнения значений, индексов/колонок и типов прошли. Частота индекса этим вызовом намеренно не сравнивалась; бинарные метаданные Parquet не проверялись на идентичность.

Основание приёмки сравнения — предоставленный вывод пользовательской ячейки, не самостоятельное выполнение Codex и не JSON-отчёт сравнения. Проверенные табличные данные совпадают, несмотря на разные файловые SHA; точная бинарная причина различия (metadata/encoding/compression) отдельно не исследовалась. Для текущего этапа этот вопрос закрыт на уровне точного содержимого таблиц. Дополнительный повторный запуск или новое обучение для этой проверки не требуются.

Для последующих изменений окружения можно использовать следующую расширенную ячейку **в Colab после подключения Drive**. Она дополнительно сверяет физические хеши с manifest и сохраняет отдельный отчёт вне каталогов запусков. Именно этот расширенный вариант с JSON-отчётом пока не предоставлен; он сохранён как повторно используемая процедура. Ни исходные, ни prepared-данные не меняются. Это техническое сравнение, без подбора параметров по test.

```python
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd

runs = Path('/content/drive/MyDrive/ppo_turnover_research/stage1/runs')
old = runs / '20261107T145440Z-00c5be42'
new = runs / '20261107T154124Z-9f1f7c1e'
meta_old = json.loads((old / 'metadata.json').read_text())
meta_new = json.loads((new / 'metadata.json').read_text())
assert meta_old['config'] == meta_new['config'], 'Scientific configurations differ'
assert meta_old['snapshot']['manifest_sha256'] == meta_new['snapshot']['manifest_sha256']
names_old = {n for n in meta_old['files_sha256'] if n.endswith('.parquet')}
names_new = {n for n in meta_new['files_sha256'] if n.endswith('.parquet')}
assert names_old == names_new and len(names_old) == 13
checks = {}
for name in sorted(names_old):
    for directory, meta in ((old, meta_old), (new, meta_new)):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == meta['files_sha256'][name], f'Checksum mismatch: {directory / name}'
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(old / name), pd.read_parquet(new / name),
                                      check_exact=True, check_freq=False)
        checks[name] = {'exact_values_equal': True}
    except AssertionError as error:
        checks[name] = {'exact_values_equal': False, 'detail': str(error)}
report = {'baseline_run': old.name, 'candidate_run': new.name, 'files': checks,
          'status': 'equal' if all(x['exact_values_equal'] for x in checks.values()) else 'different'}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output = runs.parent / f'env-v2-data-comparison-{stamp}.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
print('Comparison report:', output)
assert report['status'] == 'equal', 'Tables differ: inspect report; do not overwrite baseline or loosen tests'
```

При different в будущих проверках сначала изучить конкретные столбцы/различия, не ослаблять проверку автоматически. Для текущей пары запусков точное равенство всех prepared-таблиц уже подтверждено предоставленным stdout описанной выше короткой ячейки.

## Фактически проверенная исходная среда

В запуске `20261107T145440Z-00c5be42`, commit `da1231de2f3009b3457ef91b024a22af65b6c75e`, изучены JUnit (48 passed), лог, environment и pip check. Второй запуск `20261107T151132Z-f58bf8e6` согласован по metadata/report и хешам prepared-данных.

- Runtime tag: `release-colab-external-images_20260715-060047_RC00`.
- Python: `3.12.13`, Linux x86_64, glibc 2.35.
- Прямые версии: numpy 2.2.6, pandas 2.2.3, yfinance 0.2.65, pandas_market_calendars 5.1.1, matplotlib 3.10.3, pyarrow 20.0.0, pytest 8.4.1.
- `pip check` завершился с кодом 1 и четырьмя сообщениями ниже.
- Фактическая версия PyTorch в первоначальном environment.json не записывалась. Нельзя приписывать старому запуску успешную проверку Torch/SB3/Gymnasium.

## Источники и серьёзность конфликтов

| Сообщение | Источник ограничения | Влияние и решение |
|---|---|---|
| google-colab 1.0.0 требует pandas==2.2.2, установлен 2.2.3 | Наш старый прямой pin pandas заменил версию, требуемую Colab | Исправить: прямое нарушение контракта хоста. Вернуть pandas 2.2.2; google-colab не обновлять |
| numba 0.60.0 требует numpy>=1.22,<2.1, установлен 2.2.6 | Наш старый pin numpy нарушил ограничение предустановленного Numba | Текущий pipeline не использует Numba и прошёл, но оставлять несовместимую общую численную среду не следует. Вернуть базовый numpy 2.0.2; Numba не обновлять |
| ipython 7.34.0 требует jedi, пакет отсутствует | Несогласованность установленного IPython с его dependency metadata; наш requirements Jedi не задаёт и код не удаляет его | Известное предупреждение для нашего pipeline: Jedi обслуживает интеллектуальное дополнение/анализ в IPython и не входит в подготовку данных или base RL-stack. Автоматически не обновлять IPython и не устанавливать Jedi ради общего зелёного pip check |
| ibis-framework 9.5.0 требует toolz>=0.11,<1, установлен 1.2.0 | Ограничение неиспользуемого Ibis против установленного Toolz; наш requirements Toolz не фиксирует | Известное предупреждение, пока Ibis не используется. Не понижать глобальный Toolz без необходимости: он может использоваться другими библиотеками |

По старым отчётам нельзя достоверно установить, какой именно шаг установки/состояние образа привёл к отсутствию Jedi и версии Toolz 1.2.0. Это не следует объявлять доказанным дефектом базового образа. Новая диагностика сохраняет версии **до** установки и `pip-install-report.json`, чтобы установить происхождение изменений при следующем запуске.

Все четыре сообщения — реальные несоответствия metadata; «предупреждение» здесь означает ограниченное влияние на используемый проект, а не отсутствие проблемы для соответствующей сторонней библиотеки. Добавление Ibis в проект отменит эту классификацию. Новые неизвестные сообщения, конфликты NumPy/pandas и нарушения активного dependency tree блокируют подготовку.

## Выбранная конфигурация env-v2

Выбрать **Colab Runtime Version 2026.07, CPU, Python 3.12**. [Официальный список runtime](https://research.google.com/colaboratory/runtime-version-faq.html) указывает для неё Python 3.12.13, NumPy 2.0.2 и PyTorch 2.11.0. Вариант точной Torch-сборки, включая CUDA/CPU suffix, определяется реальной distribution metadata и сохраняется; он не угадывается по названию runtime.

| Компонент | env-v2 |
|---|---|
| NumPy | 2.0.2 (базовая версия runtime) |
| pandas | 2.2.2 (требование google-colab) |
| yfinance | 0.2.65 |
| pandas_market_calendars | 5.1.1 |
| matplotlib | 3.10.3 |
| pyarrow | 20.0.0 |
| pytest | 8.4.1 |
| Gymnasium, только совместимость | 1.2.0, base install |
| Stable-Baselines3, только совместимость | 2.7.0, base install |
| PyTorch | существующая сборка 2.11.0 с сохранением точного suffix; автоматически не заменяется |

Первые семь версий заданы в `requirements-colab.txt`, две RL-библиотеки — отдельно в `requirements-rl-check.txt`. По умолчанию `CHECK_RL_COMPATIBILITY=True`. Для простого запуска data-only допустимо False, но такой запуск **не подтверждает** совместимость будущего RL-stack.

Не используются apt, обновление pip/Colab/IPython/Numba, `--upgrade`, `--force-reinstall`, `--no-deps` или удаление предустановленных библиотек. `pip install` может устанавливать недостающие необходимые транзитивные зависимости; фактические изменения отражаются в install-report/freeze. Существующий PyTorch закрепляется временным constraint-файлом. При его отсутствии или другой baseline-версии проверка останавливается, а не скачивает другую Torch/CUDA сборку.

## Совместимость будущих библиотек

По [install_requires SB3 2.7.0](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/setup.py): Gymnasium >=0.29.1,<1.3, NumPy >=1.20,<3 и Torch >=2.3,<3. Выбранные версии удовлетворяют этим ограничениям; pandas/matplotlib также входят в зависимости SB3.

По [metadata Gymnasium 1.2.0](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.0/pyproject.toml) base install поддерживает Python >=3.10 и NumPy >=1.21. Важно: extras `[torch]`, `[jax]`, `[array-api]`/`[all]` добавляют NumPy >=2.1. Их не устанавливаем; SB3 использует обычный Gymnasium и самостоятельный Torch. Это позволяет сохранить NumPy 2.0.2, совместимый с установленным Numba.

Совпадение declared ranges не доказывает binary/runtime совместимость. Поэтому notebook выполняет в новом Python-процессе:

1. Импорты всех используемых модулей и транзитивную проверку активных base requirements.
2. Импорт `gymnasium`, `stable_baselines3`, `PPO` и `check_env` без создания модели или среды.
3. Малую CPU-проверку `numpy array → torch.from_numpy → tensor operation → numpy`.
4. Запись torch version, CUDA build и cuda_available. Последнее — диагностика доступности, **не** проверка GPU-обучения.

Torch-операции и тесты выполняются только в Colab. PPO `.learn`, создание агента и торговая среда в коде проверки отсутствуют.

## Конкретный порядок проверки

1. Опубликовать новые файлы и notebook в GitHub, взять **новый** полный SHA. Старый commit сохраняет старые requirements и не применяет env-v2.
2. Открыть обновлённый notebook, выбрать Runtime Version 2026.07 и новую сессию. Задать новый GIT_REF, оставить `SNAPSHOT_ID="daily-v1"` и `CHECK_RL_COMPATIBILITY=True`. Notebook revision новой версии — `stage1-colab-env-v3`.
3. Выполнить ячейки сверху. Если установленные и уже загруженные версии NumPy/pandas различаются, перезапустить сеанс и начать заново. Нельзя продолжать с библиотеками старой версии в памяти.
4. Диагностика должна показать `dependency_closure.status=passed`, `future_stack.status=passed`, global policy `passed` либо `passed_with_known_warnings`. Вторая означает только сохранённые Jedi/Ibis-предупреждения, а не `pip check` exit 0.
5. Выполнить полный pytest до реальных данных. После добавления четырёх проверок dependency policy ожидается 52 cases вместо исходных 48, но результат определяется настоящим JUnit, не этим ожиданием.
6. Использовать сохранённый snapshot, завершить read-back проверку. Сопоставить с baseline границы, sigma_ref и подготовленные значения. Старые данные не перезаписываются: новый RUN_ID создаёт отдельный каталог. При отличии Parquet SHA после изменения NumPy/pandas не считать сразу данные эквивалентными или испорченными: сравнить значения в Colab и объяснить различия сериализации/численной арифметики; не подбирать формулу по test.

Для ручной проверки, если первые ячейки notebook уже определили PROJECT_ROOT и PREFLIGHT_DIR:

```python
import importlib.metadata as md
import subprocess
import sys

torch_build = md.version("torch")  # Остановится, если Torch отсутствует.
constraint = PREFLIGHT_DIR / "runtime-constraints.txt"
constraint.write_text(f"torch=={torch_build}\n", encoding="utf-8")
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "-r", str(PROJECT_ROOT / "requirements-colab.txt"),
    "-r", str(PROJECT_ROOT / "requirements-rl-check.txt"),
    "-c", str(constraint),
], check=True)
```

После проверки/перезапуска библиотек команда общего контроля:

```python
result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
print(result.stdout, result.stderr)
```

Полные ячейки notebook также сохраняют установку, dependency tree, классификацию, actual versions, NumPy/Torch smoke и raw pip check. Не заменять их одной ручной командой при финальной приёмке.

## Что предоставить для приёмки env-v2

Для последующих запусков сохранять `environment.json`, `versions-before-install.json`, `pip-install-report.json`, `pip-check.txt`, `pip-freeze.txt`, `imports.log`, `pytest.xml`, metadata/sidecar и validation_report. При отказе — полный traceback и файлы preflight даже без RUN_DIR. Фактический CPU-запуск env-v2 и точное равенство prepared-таблиц со старой средой подтверждены в указанном объёме; GPU не проверен. Научные формулы, universe, интервалы, sigma_ref rule и snapshot в этой правке не изменены.
