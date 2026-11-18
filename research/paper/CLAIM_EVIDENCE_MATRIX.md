# Матрица утверждений и оснований

Пути ниже относительны к research/. Статусы: **А** — значение прочитано из доступного артефакта, хеш CSV проверен; **К** — проверено статически по коду/конфигурации; **О** — подтверждается существующим отчётом, первичный полный архив повторно не изучен; **Л** — первичная литература; **Н** — не проверено. Это не повторное исполнение расчётов.

| Утверждение | Источник | Точное место | Статус |
|---|---|---|---|
| 12 акций и SPY, Yahoo Finance, train/validation/test | docs/METHODOLOGY.md | разделы 1–3 и 5; таблица тикеров | К |
| Сдвиг признаков, RMS по 20 завершённым доходностям | src/data.py; src/volatility.py | build_features; historical_volatility | К |
| σ_ref=0,11142612683693373 | docs/STAGE6_VALIDATION.md; docs/decisions/20261115T123119Z-840594f6/decision_lock.json | раздел высокой/низкой волатильности; train-reference lock | О/А |
| Наблюдение: признаки, веса, NAV, горизонт | src/portfolio_env.py | PortfolioEnv._observation; __init__ | К |
| Действие нормируется из (clip(a)+1)/2, all−1→cash | src/portfolio.py | action_to_weights | К |
| Self-financing комиссия и оборот, формула (2) | src/portfolio.py | rebalance, уравнение в docstring и Execution | К |
| Ночь старым позициям, день новым, reward (3) | src/portfolio_env.py | PortfolioEnv.step | К |
| Адаптивная формула и bounds | src/regularization.py | penalty_coefficient | К |
| Synthetic adjusted исполнение, отсутствие повторных дивидендов | src/environment_data.py; docs/PILOT_PROTOCOL.md | PRICE_MODEL; раздел одинаковых условий | К |
| Архитектура и PPO-параметры | configs/ppo_pilot.json; docs/PILOT_PROTOCOL.md | общая конфигурация, переопределение total_timesteps | К |
| 27 runs, 442368 steps, 135 checkpoints | docs/STAGE4_VALIDATION.md | завершённый пилот; проверенная согласованность, пункт о checkpoint | О |
| Это не 135 независимых опытов | docs/STAGE4_VALIDATION.md | четыре промежуточных и один final на run | О |
| f01/a04, λ₀=0,0001/0,001 и α=0,5 | configs/evaluation_protocol.json; configs/pilot_plan.json | selected; кандидаты f01/a04 | К/А |
| Выбор по среднему validation, tie по ID, только final | docs/PILOT_PROTOCOL.md; configs/evaluation_protocol.json | метрики и выбор; selection_rule | К |
| Test 1003 даты и девять моделей | docs/results/test-20261115T141338Z-85cdf83a/metrics_by_seed.csv | девять строк, observations=1003 | А; границы дат — О |
| Все числа таблицы 2 и таблицы 5 | тот же metrics_by_seed.csv | строки mode×seed, одноимённые колонки | А |
| Все seed-разности таблицы 3 | docs/results/test-20261115T141338Z-85cdf83a/paired_metrics.csv | comparison=adaptive_minus_fixed, metric=net_sharpe/cagr/maximum_drawdown | А |
| Средние парные разности | docs/results/test-20261115T141338Z-85cdf83a/seed_stability.csv; docs/STAGE6_VALIDATION.md | adaptive_minus_fixed; основная таблица | А/О |
| Все интервалы таблицы 4 | docs/results/test-20261115T141338Z-85cdf83a/bootstrap_intervals.csv | adaptive_minus_fixed, overall/high/low, L=20/10/40; turnover interaction | А |
| Sharpe adaptive−none и его интервал | тот же bootstrap_intervals.csv | adaptive_minus_none, overall, net_sharpe, L=20 | А |
| Нет превосходства по основной оценке, но нет доказательства эквивалентности | bootstrap_intervals.csv; docs/POSTPILOT_PROTOCOL.md | основной интервал пересекает 0; estimand и ограничения | А/К, интерпретация |
| Оборот/комиссии adaptive ниже в каждом seed | paired_metrics.csv | adaptive_minus_fixed, turnover и total_commissions, все seeds | А |
| High 741 / low 262 | docs/results/test-20261115T141338Z-85cdf83a/regime_metrics.csv | observations по режимам | А |
| High−low взаимодействие не подтверждено | bootstrap_intervals.csv | adaptive_minus_fixed, взаимодействие, L=20 | А, интерпретация |
| 2000 повторов, общие индексы, seeds фиксированы | configs/evaluation_protocol.json; src/paired_analysis.py | bootstrap configuration; stationary bootstrap и парный анализ | К; исполнение — О |
| 126 предварительных тестов | docs/STAGE6_VALIDATION.md | таблица завершённой оценки | О; JUnit в текущем subset отсутствует |
| Lock создан до test; без нового обучения | decision_lock.json; docs/STAGE6_VALIDATION.md | sealed решения; время открытия и training_performed=False | А/О, без независимой подписи времени |
| Межсессионный exact replay всех 27 | docs/STAGE5_VALIDATION.md | воспроизводимость и окружение | О; не переносить на любой runtime |
| Содержание обзора [1]–[4] | REFERENCES.md | указанные первичные страницы и разделы | Л |
| Доказанный изолированный эффект адаптации | отсутствует | λ₀ выбранных конфигураций различается | Н; в статье не заявляется |
| Сходимость, новый рынок, point-in-time качество | отсутствует | ограничения раздела 6 | Н; в статье не заявляется |

Хеши шести CSV совпали с SOURCE.json. Это подтверждает целостность текущих копий относительно сохранённого манифеста, но не заменяет проверку истинности первичных данных и повторное вычисление метрик.
