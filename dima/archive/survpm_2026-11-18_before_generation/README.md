# survpm

Процессы с правой цензурой по журналам событий: сеть состояний, время переходов,
генерация, прогноз и метрики качества. Ядро — numpy, pandas, scipy; PM4Py и graphviz
подключаются только для сети Петри, рисунков и метрик уровней 1–3.

Модель v2: из каждого состояния независимо разыгрываются экспоненциальные времена по
исходящим рёбрам, побеждает минимум. `rate` — интенсивность ребра (1/день),
`p = rate / sum(rate)` — вероятность выхода по ребру, `1 / sum(rate)` — среднее ожидание.
Незавершённое ожидание (цензура) даёт экспозицию всем выходам состояния.

## Установка

```bash
pip install -e .                 # ядро
pip install -e ".[pm4py]"        # + сеть Петри и рисунки (graphviz; на macOS ещё brew install graphviz)
pip install -e ".[metrics]"      # + метрики уровней 1–3 и кросс-валидация
pip install --ignore-requires-python log-distance-measures jellyfish pulp   # метрики уровня 3
pip install matplotlib           # рисунки matplotlib
```

## Минимальный сценарий

```python
import survpm

rules = {"starts": ["Registration"], "terminals": ["Discharge"]}
log = survpm.datasets.load_sepsis_log("Sepsis Cases - Event Log_grouped.csv").prepare(rules)
train, test = log.split(seed=1)                       # целые объекты, равная доля цензуры

model = survpm.ProcessModel().fit(train)              # сеть = DFG по train, интенсивности с цензурой
model.edges_table()                                   # rate, p, mean_days по рёбрам

stream = survpm.ObjectStream.from_log(train, prefix="syn-")   # те же объекты, те же моменты старта
synthetic = survpm.Simulator(model, seed=1).run(stream, horizon=60)
survpm.compare_networks(model.network, survpm.discover(synthetic))   # восстановление сети
survpm.ProcessModel(model.network).fit(synthetic).edges_table()      # восстановление rate

case = test.case_ids[0]
as_of = test.events.loc[test.events.case_id == case, "timestamp"].iloc[1]
survpm.Forecaster(model, n_paths=1000, seed=1).survival(test, as_of, [1, 7, 30], case_id=case)

survpm.evaluate(model, test, "all")                   # все метрики: level, model, metric, value
model.plot()                                          # сеть с p и rate (graphviz)
model.plot("petri")                                   # сеть Петри с местом CENS (PM4Py)
model.save("sepsis.pkl"); survpm.ProcessModel.load("sepsis.pkl")
```

Полные примеры — `dima/sepsis_minimal.ipynb` (весь цикл и все метрики),
`dima/petri_net_time_estimation.ipynb` (синтетика с известной истиной),
`dima/petri_net_time_estimation_sepsis.ipynb` (Sepsis, признаки, повторы).

## Журнал: `EventLog`

Четыре таблицы: `events` (`case_id, activity, timestamp`), `cases` (статические признаки),
`measurements` (`case_id, timestamp, feature, value`), `observation` (`case_id, end, event, reason`).

| Действие | Вызов |
|---|---|
| CSV / DataFrame | `survpm.read_log(path, "case_id", "activity", "timestamp")`, `EventLog.from_frame(df, mapping)` |
| Правила трасс | `log.prepare({"starts": [...], "terminals": [...], "repeats": "simultaneous", "min_duration": 1/1440})` |
| Интервалы | `log.intervals()` — `case_id, src, dst, duration, observed, reason, enter` |
| Выборка объектов | `log.select(ids)`, `log.case_ids` |
| Разбиение | `log.split(seed=1, ratio=.8)`; без seed — первые объекты; `log.split("temporal", cutoff=...)` |
| Фолды | `log.folds(k=5, seed=2026)` — список `(train, test)`, доля цензуры в фолдах одинакова |
| Признаки на момент | `log.features_at(case_id, t, cutoff=as_of)`, `survpm.features.at_entry(intervals, log)` |

Правила, заданные в `prepare`, сохраняются в `log.rules`. Начальные и терминальные состояния
дальше **не нужно повторять**: их берут `discover`, `DirectlyFollowsBackend`, `Pm4pyBackend`,
`Network.discover`, `ProcessModel()` без сети, `CensoredStateNet()` и `CrossValidator()`.
Явно заданные аргументы важнее правил; `survpm.process_bounds(log)` возвращает `(starts, terminals)`.

`repeats`: `"simultaneous"` (по умолчанию) склеивает одинаковые состояния в один момент,
поздний повтор остаётся петлёй `s → s`; `"collapse"` склеивает всю серию; `"keep"` — ничего.

## Структура: `Network` и discovery

```python
network = survpm.discover(train)                                    # DFG без внешних зависимостей
network = survpm.discover(train, survpm.Pm4pyBackend("dfg"))         # DFG PM4Py
network = survpm.Network.from_edges([("A", "B"), ("B", "D")], starts=["A"], terminals=["D"])
network.validate()                                                  # недостижимые, тупики
survpm.compare_networks(reference, estimated)                       # precision, recall, f1, missing, extra
```

## Модель: `ProcessModel`

```python
model = survpm.ProcessModel(network).fit(train)                     # или ProcessModel() — DFG по train
naive = survpm.ProcessModel(network).fit(train, use_censoring=False)
truth = survpm.ProcessModel(network, survpm.ExponentialTransitionModel(rates={("A", "B"): .3, ...}))

labs = ["CRP", "Leucocytes", "LacticAcid"]
featured = survpm.ProcessModel(network, survpm.ExponentialTransitionModel(log_features=labs)).fit(
    train, features=["Age", *labs, "history_visits", "history_elapsed_days"])
```

`fit` возвращает новый объект. С признаками оценивается регрессия логарифма интенсивности
по X на входе в состояние (L2 = `l2`). `FeatureEncoder`: медиана + индикатор пропуска, one-hot,
стандартизация по train; `clip=True` (по умолчанию) обрезает значения до диапазона train,
`log_features` — `log1p` для скошенных признаков. Признак, которого ещё нет у объекта
(анализ не делали), считается пропуском. История: `history_visits`, `history_elapsed_days`.

## Генерация, применение, прогноз

```python
stream = survpm.ObjectStream.start(100, "A", "2025-01-01", X={"Age": 70})   # n новых объектов
stream = survpm.ObjectStream.from_log(train, prefix="syn-")                  # как объекты журнала
log = survpm.Simulator(model, seed=1).run(stream, horizon=60)               # EventLog; поток не меняется
path = survpm.Router(model, seed=1).run(test, case_id, as_of, horizon=30)   # один путь от as_of
f = survpm.Forecaster(model, n_paths=1000, seed=1)
f.survival(test, as_of, times, case_id=case_id)          # S(t) до терминала, t — дни от as_of
f.terminal_probs(...), f.hazard(...), f.expected_time(test, as_of, horizon=30, case_id=case_id)
```

Прогноз видит только историю до `as_of`. `updater(X, state, t, rng)` в `ObjectStream`
меняет X при входе в состояние; `future_features` в `Router`/`Forecaster` — сценарий будущих X.

## Метрики

Одна точка входа и один формат — таблица `level, model, metric, value`:

```python
survpm.evaluate(model, test, "nll")                  # NLL с цензурой (v2)
survpm.evaluate(model, test, ["level1", "km_sup"])   # уровни и отдельные метрики
survpm.evaluate(model, test, "all")                  # nll + уровни 1–4
survpm.evaluate(truth, test, "all", train=train)     # для модели, которая не обучалась fit
survpm.compare({"base": model, "naive": naive}, test, "all")
survpm.available_metrics()                           # список имён по уровням
survpm.register_metric("median_days", func)         # своя метрика (model, log) → число или таблица
```

| Уровень | Метрики | Зависимости |
|---|---|---|
| `nll` | `nll_per_interval`, `nll_per_trace` (неизвестный переход → inf) | ядро |
| `level1` | fitness, precision (token replay), generalization, simplicity | PM4Py |
| `level2` | entropic relevance, uEMSC, EMD путей | Ebi, POT |
| `level3` | temporal profile, CFLD, n-gram, cycle-time / event EMD, KS по рёбрам, km_sup, km_l1, log-rank, IBS | PM4Py, lifelines, scikit-survival, log-distance-measures |
| `level4` | `unseen_share`, NLL без трасс с неизвестными переходами | ядро |

Уровни 1–4 считаются через `model.legacy()` — эквивалентный `CensoredStateNet` с теми же
интенсивностями (без переобучения). Для модели с признаками пока доступна только `nll`.

Кросс-валидация (naive против censor-aware, сеть против Inductive Miner):

```python
from survpm.metrics.crossval import CrossValidator, summarize
cv = CrossValidator(k=5).add_metrics("all").run(log)     # start/terminal — из log.rules
survpm.long_format(cv)                                    # та же длинная таблица + fold
summarize(cv["level4"])                                   # среднее и std по фолдам
```

## Рисунки

```python
viz = survpm.GraphvizVisualizer()
viz.plot_dfg(train)            # DFG: число переходов и посещений
viz.plot_network(model)        # сеть с p, rate и средним ожиданием (= model.plot())
viz.plot_petri(model)          # сеть Петри с местом CENS (= model.plot("petri"))
viz.plot_censoring(train)      # наблюдаемые доли выходов и цензуры
network.plot()                 # matplotlib, состояния по слоям от начального
survpm.MatplotlibVisualizer().plot_edge_curves(model, ("A", "B"))   # P(выход B, ожидание ≤ t)
```

## Точки расширения

| Что меняется | Реализуйте | Пример |
|---|---|---|
| Структура процесса | `PetriBackend.discover(log) → Network` | `DirectlyFollowsBackend`, `Pm4pyBackend` |
| Время и направление переходов | `TransitionModel` (`fit`, `transition_parameters`, `next_transition`, `log_likelihood`) | `ExponentialTransitionModel`; `register_transition_model(name, factory)` |
| Признаки из истории | `HistoryFeatures` (`fit`, `transform`, `at_entry`) | `DefaultHistoryFeatures` |
| Изменение X при входе | `FeatureUpdater(X, state, t, rng)` | функция |
| Метрика | `(model, log) → число или таблица metric/value` | `register_metric` |
| Рисунок | `Visualizer` | `GraphvizVisualizer`, `MatplotlibVisualizer` |

## Устройство

| Модуль | Содержимое |
|---|---|
| `eventlog` | `EventLog`, `process_bounds` |
| `features` | чтение журналов, признаки на моменты (`values_at`, `at_entry`, `split`) |
| `log` | подготовка трасс и интервалы на уровне таблиц |
| `network`, `discovery` | `Network`, `discover`, `DirectlyFollowsBackend` |
| `models` | `TransitionModel`, `ExponentialTransitionModel`, `HistoryFeatures`, `FeatureUpdater` |
| `encoding` | `FeatureEncoder` |
| `process` | `ProcessModel` |
| `engine` | `ObjectStream`, `Simulator`, `Router`, `Forecaster`, `Evaluator` |
| `metrics` | `evaluate`, `compare`, уровни 1–4, `crossval` |
| `viz` | `GraphvizVisualizer`, `MatplotlibVisualizer`, `state_graph` |
| `adapters` | PM4Py (сеть Петри, DFG), Ebi (SDFA) — по требованию |
| `datasets` | Sepsis, синтетика A–D |
| `legacy` | прежний API: `CensoredStateNet`, семейства по AIC, кривые на сетке |

Ядро не импортирует PM4Py (AGPL v3), поэтому лицензию библиотеки можно выбрать свободно.
Прежнее описание (`choice_then_time`, семейства распределений, признаки в `CensoredStateNet`) —
`docs/legacy.md`; архитектура v2 — `docs/architecture_survpmv2.md`.

## Тесты

```bash
pip install -e ".[test]" && pytest tests
```
