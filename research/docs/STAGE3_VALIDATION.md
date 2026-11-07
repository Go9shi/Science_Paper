# Этап 3: инфраструктура PPO и протокол технической проверки

## Статус и область

Подготовлено для Google Colab. При разработке выполнены только статические проверки
Python/JSON/notebook. Новые численные тесты, обучение, smoke test и повторная загрузка
после перезапуска **ещё не выполнены**. Старые 83 пройденных теста этапа 2 не являются
результатом проверки этой версии. Численные результаты появятся только в артефактах Colab.

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
Число собранных pytest cases устанавливается в Colab,
не вычисляется из названий функций. Результатов обучения и новых финансовых метрик пока нет.

Остаются: фактический Colab smoke/restart, продолжительность сохранения Drive, GPU-детерминизм,
будущий бюджет настройки и seeds, чувствительность к периоду/издержкам/universe,
survivorship и point-in-time ограничения источника. Современный список и adjusted execution
ограничивают перенос результатов на реальное исполнение. Одного seed/короткого pilot недостаточно
для научного вывода; заключительный test остаётся закрытым.

## Использованные первичные источники

- [SB3 PPO 2.7.0](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html).
- [SB3 VecNormalize и VecEnv](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/vec_envs.html).
- [SB3 callbacks и момент сохранения](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/callbacks.html).
