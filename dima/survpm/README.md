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

start = survpm.EventLog.start(1000, "Registration", "2025-01-01")    # 1000 новых объектов
synthetic = survpm.ModelGenerator(model, seed=1).generate(start, horizon=60)
survpm.compare_networks(model.network, survpm.discover(synthetic))   # восстановление сети
survpm.ProcessModel(model.network).fit(synthetic).edges_table()      # восстановление rate

case = test.case_ids[0]
t = test.events.loc[test.events.case_id == case, "timestamp"].iloc[1]
now = test.cut(t).select([case])                     # журнал на момент t: «сейчас» = end
survpm.MarkovForecaster(model).survival(now, [1, 7, 30])                       # точно, X фиксированы
survpm.SimulationForecaster(model, n_paths=1000, seed=1).survival(now, [1, 7, 30])   # Монте-Карло

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
| Признаки на момент | `log.features_at(case_id, t)`, `DefaultStreamData().at(log.cut(t))`, `survpm.features.at_entry(intervals, log)` |

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

Вход и выход всех компонентов — `EventLog` с колонкой `end` в `observation` («наблюдали до» —
момент «сейчас» объекта). Общий цикл генерации записан в базе и не меняется:

```python
X    = data.at(log)                       # StreamData: где объект и какие признаки
step = model.predict(X)                   # ProcessModel: куда и через сколько
move = selector.select(X, step, rng)      # Selector: какой переход
log  = log.append(move)                   # новое событие, end += dt
```

```python
start = survpm.EventLog.start(100, "A", "2025-01-01", X={"Age": 70})   # n новых объектов
same = train.starts(prefix="syn-")         # те же объекты с начала: первое событие каждого
now = test.cut(t)                          # журнал на момент t (только для проверки на истории)
X = survpm.DefaultStreamData().at(now)     # таблица: case_id, state, elapsed, time, признаки
model.predict(X)                           # закон шага для каждой строки (StepDistribution)
log = survpm.ModelGenerator(model, seed=1).generate(start, horizon=60)       # EventLog
log = log.censor(survpm.StudyEnd(days=90) + survpm.RandomLoss(mean_days=60))  # цензура поверх
log.traces()                               # [(состояния, дни от старта)] того же прогона
survpm.ModelGenerator(model, survpm.BestSelector("Discharge")).generate(now)  # продолжение по правилу
f = survpm.MarkovForecaster(model)         # или GridForecaster, SimulationForecaster
f.survival(now, [1, 7, 30]); f.terminal_probs(now, [30]); f.hazard(now, [1, 7]); f.expected_time(now, 30)
```

`times` — дни от `end` объекта. Ответ — таблица с колонкой `case_id` (по строке на объект и горизонт).

* **`StreamData`** — признаки к моменту «сейчас», один метод `at(log) -> X`. `DefaultStreamData`:
  последнее состояние, `elapsed` (дни в нём), статические признаки, последние замеры на входе,
  `history_visits`, `history_elapsed_days`. Своя динамика признаков — свой `at`; признаки, которые
  меняются при генерации, пишутся в журнал замерами.
* **`model.predict(X)`** — список `StepDistribution` по строкам X: `dst`, `p`, `sample(rng)`,
  `sample_time(j, rng)`, `mean_time(j)`, `time_cdf(t)`. Реализации: `ExponentialStep`
  (`ExponentialTransitionModel`), `ChoiceThenTimeStep` (`ChoiceThenTimeModel(edges)` — заданная истина,
  выход по `p`, время из `dist`), `SampledStep` (модель только с `next_transition`).
* **`Selector`** — правило выбора, один метод `weights(X, steps) -> case_id, dst, weight`; `select`
  (розыгрыш по весам, время из закона) — в базе. Готовые: `SampleSelector` (по закону, по умолчанию),
  `MostLikelySelector`, `BestSelector(target)`, `WorstSelector(target)` (по вероятности дойти до
  `target`, дальше — по закону). Selector принимают и генератор, и прогноз.
* **`Censoring`** — `StudyEnd(days)`, `RandomLoss(mean_days)`, сумма правил `a + b`; свой — метод `times(log, rng)`.

### Свой генератор: `LogGenerator`

```python
class Coin(survpm.LogGenerator):
    terminals = {"Done"}

    def step(self, X, rng, **kw):              # по строке на объект: case_id, dst, dt (+ reason)
        return pd.DataFrame(dict(case_id=X.case_id,
                                 dst=["Done" if r.random() < .5 else "Start" for r in rng],
                                 dt=[r.exponential(1.) for r in rng]))

log = Coin(seed=1).generate(survpm.EventLog.start(200, "Start"), horizon=30)
```

`rng` — поток случайных чисел каждой строки (по `case_id`, результат не зависит от порядка объектов).
`dst=None, dt=inf` — выхода нет (`dead_end`); `dst=None, dt, reason="lost"` — обрыв наблюдения.
`generate(log, max_steps=1000, horizon=None)`: причины остановки в `observation.reason` — `""`
(терминал), `horizon`, `max_steps`, `dead_end`, `lost`. Готовый — `ModelGenerator(model, selector, data, seed)`.

### Свой прогноз: `Forecaster`

Один обязательный метод `cumulative_incidence(X, times)`: F_k(t) = P(T ≤ t, исход = k | журнал до end),
таблица `horizon, outcome, cdf` (+ `case_id`, если строк несколько). База строит X (`data.at(log)`),
проверяет ответ (0 ≤ F_k ≤ 1, не убывает, Σ_k F_k ≤ 1) и выводит `survival` (S = 1 − Σ F_k),
`terminal_probs`, `hazard` ((F_k(b) − F_k(a)) / S(a) / (b − a)), `expected_time` (∫₀ʰ S dt).

```python
class AlwaysExp(survpm.Forecaster):
    def cumulative_incidence(self, X, times, **kw):
        return pd.DataFrame(dict(horizon=times, outcome="D", cdf=1 - np.exp(-.5 * np.asarray(times))))

AlwaysExp(terminals=["D"]).survival(test.cut(t), [1, 7, 30])
```

| Реализация | Метод | Точность |
|---|---|---|
| `MarkovForecaster(model, selector)` | F_k(t) = [exp(Q t)]_{s,k}, Q = Λ·w | точно для экспоненциального закона при X, фиксированных после end |
| `GridForecaster(model, selector, n_grid)` | уравнение марковского восстановления на сетке, любой закон с `time_cdf` | O(h²) |
| `SimulationForecaster(model, n_paths, seed, selector)` | доля из n путей `ModelGenerator` (признаки пересчитываются на каждом шаге) | колонка `se` = √(F(1−F)/n) |

### Прежний интерфейс

Удалён 26.11: `ObjectStream`, `Simulator`, `Router`, `FeatureUpdater`, вызовы с `as_of` и движок на
объектах `Case` (модули `legacy_generation`, `legacy_forecasting`) лежат в `dima/archive/`. Замена:
`ObjectStream.start(n, s, t)` → `EventLog.start(n, s, t)`, `ObjectStream.from_log(log)` → `log.starts()`,
`Simulator(model).run(stream)` → `ModelGenerator(model).generate(log)`,
`Router(model).run(log, case_id, as_of)` → `ModelGenerator(model).generate(log.cut(as_of).select([case_id]))`,
`Forecaster(model).survival(log, as_of, times)` → `SimulationForecaster(model).survival(log.cut(as_of), times)`
(или `MarkovForecaster`), `updater` → своя реализация `StreamData.at`.

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
| Признаки к моменту «сейчас» | подкласс `StreamData` (`at`) | `DefaultStreamData` |
| Закон шага | `predict(X) → [StepDistribution]` у модели переходов | `ExponentialStep`, `ChoiceThenTimeModel` |
| Правило выбора перехода | подкласс `Selector` (`weights`) | `SampleSelector`, `MostLikelySelector`, `BestSelector`, `WorstSelector` |
| Генерация журнала | подкласс `LogGenerator` (`step`) | `ModelGenerator` |
| Цензура | подкласс `Censoring` (`times`) | `StudyEnd`, `RandomLoss` |
| Прогноз | подкласс `Forecaster` (`cumulative_incidence`) | `MarkovForecaster`, `GridForecaster`, `SimulationForecaster` |
| Метрика | `(model, log) → число или таблица metric/value` | `register_metric` |
| Рисунок | `Visualizer` | `GraphvizVisualizer`, `MatplotlibVisualizer` |

## Устройство

| Модуль | Содержимое |
|---|---|
| `eventlog` | `EventLog`, `process_bounds` |
| `features` | чтение журналов, признаки на моменты (`values_at`, `at_entry`, `split`) |
| `log` | подготовка трасс и интервалы на уровне таблиц |
| `network`, `discovery` | `Network`, `discover`, `DirectlyFollowsBackend` |
| `models` | `TransitionModel`, `ExponentialTransitionModel`, `HistoryFeatures` |
| `encoding` | `FeatureEncoder` |
| `process` | `ProcessModel` |
| `streamdata` | `StreamData`, `DefaultStreamData` |
| `steps` | `StepDistribution`, `ExponentialStep`, `ChoiceThenTimeStep`, `SampledStep`, `ChoiceThenTimeModel` |
| `selection` | `Selector`, `SampleSelector`, `MostLikelySelector`, `BestSelector`, `WorstSelector` |
| `censoring` | `Censoring`, `StudyEnd`, `RandomLoss` |
| `generation` | `LogGenerator`, `ModelGenerator` |
| `forecasting` | `Forecaster`, `MarkovForecaster`, `GridForecaster`, `SimulationForecaster` |
| `engine` | `Evaluator` (NLL на журнале) |
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
