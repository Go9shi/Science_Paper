# Этап 4 — протокол ограниченного пилота PPO

## Статус и область вывода

Protocol ID: `ppo-turnover-pilot-v1`, машинный план: `configs/pilot_plan.json`.
Этот этап **подготовлен, но ещё не выполнен в Colab**. Нет новых результатов тестов,
обучения, сравнительных метрик, p-values или доверительных интервалов.

До составления плана уже просмотрены smoke-результаты 4096 шагов / seed 42 и validation
диагностики этапа 3. Поэтому это заранее фиксируемый план **следующих разведочных запусков**,
а не слепая preregistration и не независимое подтверждение гипотезы.
Исходный вопрос: может ли адаптивная регуляризация оборота улучшить Net Sharpe относительно
хорошо настроенного fixed при одинаковых прочих условиях? Пилот проверяет инфраструктуру,
масштаб штрафа и устойчивость ограниченного поиска; окончательный ответ потребует другого этапа.

Подтверждённые ранее 97 tests и smoke относятся к a768abc. Audit c51f107 подтвердил
полное точное исполнение сохранённых действий и same-runtime load, но не межсессионное
побитовое совпадение политики. Все исторические свидетельства остаются в STAGE1/2/3_VALIDATION.md.

## Данные, исполнение и одинаковые условия

Неизменяемый источник Stage 1: `20261107T154124Z-9f1f7c1e`, snapshot `daily-v1`.
12 исходных акций и SPY; новый сетевой download котировок не нужен.
Используется `load_development_market`: train 2010–2018, validation 2019–2021;
`features_test.parquet` не читается, ценовые DataFrame ограничены датами до 2019/2022 соответственно.
Test 2022–2025 запрещён для настройки, выбора seed/модели, метрики и исследовательских выводов.
Хеширование общих Parquet проверяет сжатые байты; физические ограничения из METHODOLOGY сохраняются.

Во всех jobs:

- Одна и та же стандартная реализация SB3 PPO 2.7.0 / MlpPolicy; отдельные actor/critic 64–64, Tanh.
- Те же признаки, включая рыночную sigma SPY, портфельные веса/NAV/остаток горизонта;
  одинаковые observation/action spaces. Нормализация observation fit только на train,
  frozen на validation, clip_obs=10; reward normalization отключена.
- Начальный капитал 1000, 100% cash, long-only, дробные синтетические единицы, без плеча.
  Cash без процентов. Комиссия 5 bps на фактические покупки и продажи рискованных активов.
  Проскальзывание/impact не добавляются, финальная ликвидация не выполняется.
- Решение после Close[t−1], исполнение synthetic adjusted Open[t], оценка adjusted Close[t].
  Sigma — годовой нецентрированный RMS 20 завершённых log returns SPY до t−1,
  sigma_ref — только train median. Все исходные финансовые формулы остаются прежними.
- Единая функция `train_one_run`; initial state/seed задаются одинаково. Train может циклически
  повторяться внутри своего периода. Validation — полный одинаковый эпизод с 100% cash,
  без переноса train-позиций. Оценивается только final, без выбора checkpoint по validation.

Финансовый NAV содержит комиссию ровно один раз. Reward = net log return − lambda_t*turnover.
Штраф не списывается из cash/NAV. Основная финансовая оценка использует simple net returns.
Существующая модель `synthetic_adjusted_open_close_v1` не заменяет номинальную бухгалтерию
дивидендов/splits/payment dates/налогов. Современный universe сохраняет survivorship bias;
источник Yahoo не является point-in-time архивом. Эти ограничения относятся ко всему сравнению.

## Общая PPO-конфигурация и объём

Основа — существующий `configs/ppo_pilot.json`; план явно переопределяет только
total_timesteps=16384 и checkpoint_steps=4096. Seeds: **11, 23, 47**, одинаковые у всех кандидатов.
Seed 42 из уже просмотренного smoke не выбирается как «удачный» и не входит в новую матрицу.

| Параметр | Значение |
|---|---:|
| learning_rate | 0.0003 |
| n_steps / batch_size / n_epochs | 256 / 64 / 5 |
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_range / ent_coef / vf_coef | 0.2 / 0 / 0.5 |
| policy actor/critic | 64–64, Tanh |
| total_timesteps на job | 16384, ровно 64 rollouts |
| checkpoint interval | 4096 |
| environments / device | 1 DummyVecEnv / CPU по умолчанию |
| max_grad_norm / normalize_advantage | 0.5 / True (SB3 defaults) |
| use_sde / target_kl | False / None (SB3 defaults) |

Optimizer/loss/GAE/clipping алгоритма не изменяются. GPU — только явный device=cuda,
с сохранением установленного Torch build; если CUDA недоступна, выполнение останавливается.
В одной batch нельзя менять device, версии, Torch build или CPU capability. GPU ещё не проверен.
Сохранённые runtime fingerprints не гарантируют идентичность всех аппаратных деталей.

| Семейство | Кандидатов | Seeds на кандидат | Jobs | Train-шагов |
|---|---:|---:|---:|---:|
| none | 1 | 3 | 3 | 49152 |
| fixed | 4 | 3 | 12 | 196608 |
| adaptive | 4 | 3 | 12 | 196608 |
| Всего | 9 | — | **27** | **442368** |

Бюджет относится к успешно завершённым исследовательским jobs. Синтетические тесты,
validation episodes и повторные попытки учитываются отдельно. Каждый job выполняет два
независимых validation load в текущем runtime для проверки, а не для настройки.
Время выполнения/достаточность сходимости 16384 шагов заранее не обещаются.
Hard caps инфраструктуры: не более 30 jobs и 524288 train-шагов в малом плане.
Масштабный поиск требует отдельного протокола/версии, не расширяется молча.

## Ограниченный поиск штрафа

| ID | Режим | lambda_0 | alpha |
|---|---|---:|---:|
| baseline | none | не используется | не используется |
| f01 | fixed | 0.0001 | не используется |
| f02 | fixed | 0.0003 | не используется |
| f03 | fixed | 0.001 | не используется |
| f04 | fixed | 0.003 | не используется |
| a01 | adaptive | 0.0003 | −0.5 |
| a02 | adaptive | 0.0003 | +0.5 |
| a03 | adaptive | 0.001 | −0.5 |
| a04 | adaptive | 0.001 | +0.5 |

Шкала lambda — пилотная логарифмическая сетка вокруг уже реализованного масштаба 0.001,
а не оценка оптимума. При адаптации q=sigma/sigma_ref,
multiplier=min(q**alpha,5), lambda=clip(lambda_0*multiplier,0,0.01).
При отрицательной alpha q ограничивается снизу 1e−6; нулевая sigma обработана существующим helper.
Alpha=0 точно совпадает с fixed при допустимых bounds и проверяется unit tests;
он не расходует search-budget отдельным дублирующим кандидатом.

Оба знака alpha исследуются симметрично без предположения полезного направления.
Fixed имеет 4 значения lambda, adaptive — 2 lambda × 2 alpha. Это равный бюджет числа
конфигураций/seeds/шагов, но **разные конечные пространства поиска**, что ограничивает вывод.
Глобальная оптимальность или полнота адаптивной сетки не заявляются.

Порядок jobs фиксирован до новых результатов: seed blocks, внутри каждого сначала baseline,
затем f01/a01, f02/a02, f03/a03, f04/a04. Он одинаков при resume, никакого обучения
«до хорошего результата» или дополнительного бюджета лучшему методу нет.

## Метрики и выбор на validation

Primary: Net Sharpe из `src/financial_metrics.py`, annualization=252, risk_free_annual=0,
std(ddof=1). Secondary: CAGR, Maximum Drawdown, Annualized Volatility,
mean/cumulative/annualized actual turnover, Cumulative Net Return, total commissions.
Training reward и losses используются для диагностики, не для финансового ранжирования.
Формулы не дублируются в notebook. Для undefined Sharpe остаётся null с исходным статусом.

Выбор разрешён только после завершения **всей** матрицы 27 jobs, без failure/pending/interrupted
вместо нужных результатов. Кандидат должен иметь все три seeds и конечный определённый Sharpe
каждого; null не заменяется нулём. Среди допустимых кандидатов каждого штрафного семейства
берётся максимальное арифметическое среднее трёх validation Net Sharpe.
При **точном** равенстве среднего — лексикографически меньший ID, без вторичного отбора по CAGR/seed.
Если нет допустимого кандидата одного семейства, selection.status=no_defined_candidate, winners пусты.
Baseline не настраивается. Все seeds, конфигурации и неудачи остаются в отчёте.

Название результата: **лучший внутри указанной сетки по validation**. Не «оптимальный fixed»
и не доказательство превосходства adaptive. Эти же validation данные использованы для выбора:
оценка оптимистична после selection. Малые различия средних не объявляются устойчивым эффектом.
Нет независимой проверки на test, публикации p-value или выбора лучшего seed в этом этапе.

## Политика численной воспроизводимости v1

План фиксируется до новых pilot runs; старые exact-аудиты не переписываются.

1. **Целостность:** точные SHA source/data/metadata/checkpoints/configs, фиксированный plan digest,
   размеры/даты/schema. Любой checksum/version/config mismatch блокирует job/resume.
2. **Финансы при сохранённых действиях:** точный replay действий и всего CSV бухгалтерии.
   Это отдельный критерий, не сравнение двух заново вычисленных политик. Для этапа 3 он подтверждён;
   при аудите будущих pilot runs указывается фактический статус именно этих runs.
3. **Один runtime:** два независимых load final model+VecNormalize, frozen normalization,
   deterministic predict; exact actions/finances/metrics обязательны в train_one_run.
   NaN/Inf observations/rewards/core losses останавливают job. Undefined explained_variance/Sharpe
   кодируются null с пояснением, не фальшивым finite score.
4. **Между runtime:** exact-флаги сохраняются отдельно. Численная близость описывается
   абсолютными/относительными различиями действий, NAV и финансовых метрик, а не глобальным passed.
   Сравнение policy на teacher-forced исходной траектории отделяет inference от feedback NAV/весов.
   Original exact=false остаётся false даже при небольшом отклонении.

**Допуски approximate-pass на этом этапе не вводятся.** Machine epsilon не является доказанной
границей ошибки многослойной политики или накопленного NAV. Уже наблюдавшиеся дельты этапа 3
не используются для настройки допуска. Если будущий протокол введёт допуски, нужна отдельная
версия, обоснование единиц/масштаба/dtype/накопления и фиксация до новой проверочной серии.
Нет обещания побитовой воспроизводимости между CPU/GPU/platforms. Diagnostic_complete означает
завершение диагностики. Изменение веса модели/округление NAV ради exact pass запрещено.

Три выбранных pilot winner/control имеют разные lambda/alpha и не должны молча передаваться
в старый matched-three-mode audit, требующий одинаковой config кроме reward_mode. Отдельный
межсессионный аудит каждого resolved job — последующая техническая проверка с собственным
evaluator SHA; он не должен смешиваться с ранжированием validation. В пилоте автоматически
проверяется пункт 3, а пункты 2/4 в новых runs не объявляются выполненными без артефактов.

## Артефакты, ошибки и продолжение Colab

`stage4/batches/<batch_id>` содержит resolved_plan.json, batch.json с встроенной контрольной
суммой состояния, runs/, attempts/, полные run_summary.csv, attempt_history.csv,
candidate_summary.csv, selection.json и batch_report.json/sha256.
Каждый attempt имеет новые ID, provenance, source.bundle, preflight, pip freeze,
pytest log/JUnit, неизменяемый test_gate.json и attempt.json/sha256.
Финальная ячейка сохраняет batch_snapshot/ с отчётами/состоянием текущей batch внутри attempt,
чтобы последующее обновление общих сводок не уничтожало историческое свидетельство этой попытки.
Code/evaluator/training SHA совпадают для новых jobs и записаны явно. Stage 1 source имеет
собственный старый SHA/metadata digest; его не подменяют SHA новых моделей.

До реальных данных/обучения — весь pytest, zero failures/errors/skips, ключевые pilot/PPO/checker cases.
В режиме preview обе опции RUN_TESTS и ENABLE_TRAINING False: tests/data/training не выполняются.
RUN_TESTS=True при ENABLE_TRAINING=False явно разрешает только синтетические тесты
и чтение/проверку разрешённых данных. ENABLE_TRAINING=True дополнительно разрешает весь пилот
и требует успешного test_gate текущей попытки. Gate содержит SHA JUnit и не меняется при
финализации attempt; след обучения ссылается на неизменяемые тестовые свидетельства.

run_pilot_batch использует общую train_one_run. Единственное расширение общей функции —
on_run_created callback, чтобы сохранить фактический путь нового run до начала learn,
и явный metadata stage=4 (по умолчанию stage=3 сохраняет совместимость старого notebook).
PPO и финансовые формулы не меняются. У каждого run уникальный путь; к checkpoint model.zip
прилагаются normalization/config/manifest. Final после updates, periodic во время rollout.

Resume требует тот же полный Git SHA, resolved plan, device/data contract, Python/library
versions/Torch build/CPU capability. Завершённые runs проверяются и пропускаются без повторного
learn. Если run успел завершиться, но batch запись прервана, completion manifest проверяется
и результат восстанавливается в ledger. Неполный job запускается целиком заново с тем же seed;
optimizer/RNG/rollout buffer прерванного learn не восстанавливаются побитово.

Failed/interrupted попытки не удаляются и не используются как скрытый дополнительный поиск.
Известные фактические steps учитываются отдельно от планового бюджета; для жёстко прерванного
runtime стоимость может быть unknown, последний полный checkpoint даёт только lower bound.
Дополнительные затраты retries могут нарушить равенство **фактической** общей стоимости семейств:
это раскрывается историей attempts, а равный плановый бюджет не выдаётся за равный фактический.
Ошибку нельзя обходить удалением seed/кандидата или выбором победителя неполной матрицы.

Одна batch — один writer. После потери runtime lock может остаться: остановить старый runtime,
явно включить RECOVER_INTERRUPTED=True, сохранить прежний BATCH_ID. Старый lock архивируется
в новой попытке. Не применять recovery к параллельно работающему ноутбуку. При смене кода/плана/
несовместимой среды — новая batch, старые результаты и ошибки остаются доступными.

## Статистический план, без вычисления значимости

1. Для каждого кандидата показать все три seed-метрики, среднее и sample std. Для двух выбранных
   семейных кандидатов сравнивать одинаковые seeds и даты. Seed uncertainty и market uncertainty
   описывать отдельно: три seed — очень малая выборка, не источник надёжной оценки tail/p-value.
2. Для будущей описательной оценки временной неопределённости — парный stationary block bootstrap
   дневных net returns. Общие временные индексы блоков используются у всех методов и seeds,
   чтобы сохранить рыночное сопоставление; «день × seed» не считается IID выборкой.
3. Предварительный дизайн последующего анализа: 2000 resamples, отдельный bootstrap seed=20261108,
   ожидаемая длина блока 20 торговых дней; заранее предусмотренная sensitivity 10 и 40.
   Месячная длина — рабочее допущение, не оценённый оптимум и не следствие sigma window=20.
   Выбор/обоснование на development history фиксируются до независимого исследования, без test.
4. В каждом resample заново вычислять разность Sharpe; для CAGR/drawdown строить путь из
   ресэмплированных net returns от общего initial NAV. Комиссии уже внутри net returns.
   Такой bootstrap условен на наблюдавшихся trained policies: он не моделирует заново действия
   агента на искусственном рынке и не устраняет selection bias/структурные сдвиги.
5. Любые интервалы выбранных на этой же validation конфигураций будут **разведочными**.
   Для подтверждающего вывода нужен отдельно зафиксированный протокол настройки/выбора,
   независимый test и учёт множественного поиска; простые несмещённые p-values по выбранному
   максимуму validation не заявляются. Holm/SPA либо другие корректировки требуют собственной
   постановки и не включаются автоматически вместо сохранения независимого test.

На этапе 4 нет bootstrap-кода, новых statistical dependencies, CI/p-value или заключения
о значимости. План описан для следующего этапа; не считать перечисление методов выполненной оценкой.

## Проверки разработки и приёмка

Подготовлены tests/test_pilot.py: равенство бюджетов/общих параметров, запрет test,
невалидные grids/seeds, отключённое обучение, неполная selection, undefined Sharpe,
gate/lock, failed attempt → resume → пропуск complete, контроль SHA/плана и состояния.
Ledger-тесты используют явно маркированные mock artifacts/scores, а не создают исследовательские результаты.
Numerical tests выполняются только в Colab. Локально — синтаксис/AST, JSON и notebook structure,
компиляция ячеек без исполнения, ссылки/поля/статические бюджеты, git diff --check.
Локально успешно проверены AST/синтаксис 24 Python-файлов, кодовые ячейки всех четырёх
notebook без исполнения, встроенные Python-сниппеты, структура/required paths нового notebook,
отключённые defaults, JSON-план и статическая арифметика 27/442368, ссылки и git diff --check.
Исследовательские модули не импортировались, pytest и численные проверки не запускались.

Приёмка тестовой стадии: полный реальный JUnit текущего SHA, preflight без blocking conflicts,
сохранённая проверка train/validation и отсутствие реального обучения при ENABLE_TRAINING=False.
Приёмка пилота: 27 complete jobs, все config/budget/seed/source/data hashes согласованы,
same-runtime load passed, полная история attempts, отсутствие test use, selection только после всей сетки.
Сходимость/превосходство не являются автоматически достигнутыми от статуса complete.

Предоставить batch.json, resolved_plan.json, batch_report.json/sha256, все summary/history/selection,
attempt.json/sha256, test_gate/JUnit/log, environment/pip reports/freeze/provenance и metadata/sidecar,
training_metrics, validation_metrics/financial CSV каждого run. Для проверки загрузки нужны
также final checkpoint manifest/model.zip/vecnormalize.pkl/config. Большие артефакты хранятся на Drive.

## Первичные источники

- [SB3 PPO 2.7.0](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html): стандартный PPO и CPU для MLP.
- [PyTorch numerical accuracy 2.11](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html): ограничения битового совпадения.
- [arch time-series bootstraps](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html): stationary/circular block bootstrap.
