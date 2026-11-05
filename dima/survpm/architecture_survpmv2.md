# Архитектура survpm (v2)

Библиотека соединяет Process Mining и Survival Analysis: сеть Петри задаёт структуру процесса, survival-модели — вероятности и распределения времени на рёбрах, движки — генерацию, применение и прогноз. Поддерживаемые данные: time-varying признаки, повторные события (циклы), конкурирующие риски (несколько терминальных состояний, причины цензуры).

Принцип: ядро задаёт структуры данных и протоколы (`typing.Protocol`); конкретные алгоритмы (PM4Py, семейства распределений, регрессионные модели, рендереры) — подключаемые реализации. Для каждой компоненты указана минимальная достаточная структура (атрибуты и методы); студенты реализуют варианты моделей в рамках этих контрактов.

## Каркас

```mermaid
classDiagram
    class EventLog
    class Network
    class ProcessModel {
        network
        transitions
        times
        fit(log)
    }
    class Simulator
    class Router
    class Forecaster
    class Evaluator

    ProcessModel --> Network
    ProcessModel --> EventLog : обучение
    Simulator --> ProcessModel : генерация
    Router --> ProcessModel : применение
    Forecaster --> ProcessModel : прогноз S(t), hazard
    Evaluator --> ProcessModel : метрики
    Simulator ..> EventLog : результат
    Router ..> EventLog : результат
```

Цикл «выбор перехода → выбор времени → обновление признаков → запись» принадлежит движкам. Модель отвечает на один запрос: вероятности и распределения времени исходящих переходов из заданного состояния.

Направление зависимостей (одностороннее):

```
log, network  ←  models  ←  model  ←  engine  ←  viz, metrics
adapters  →  network, log   (pm4py, graphviz — extras)
```

`EventLog` и `Network` ни от чего не зависят; ядро использует только numpy/pandas/scipy.

## Модули

| Модуль | Содержимое |
|---|---|
| `log` | `EventLog`, парсинг, подготовка трасс, `intervals()`, `features_at()` |
| `network` | `Network` (структура), валидация, экспорт |
| `discovery` | протокол `PetriBackend`, функция `discover(log, backend)` |
| `adapters/pm4py` | единственное место зависимости от PM4Py (AGPL), extras |
| `models/time` | протокол `TimeModel`, семейства (exponential, weibull, lognormal, gamma), KM, выбор по AIC |
| `models/choice` | протокол `TransitionModel` (вероятности выбора направления) |
| `models/features` | протокол `FeatureUpdater` |
| `model` | `ProcessModel` — композиция Network + TransitionModel + TimeModel |
| `engine` | `Simulator`, `Router`, `Forecaster`, `ObjectStream`, `TransitionPicker` |
| `viz` | протокол `Visualizer`, бэкенды graphviz/matplotlib |
| `metrics` | протокол `Metric`, реестр, `compare(models, log)`, структурные метрики сетей |

Реализации протоколов регистрируются по имени (`register_time_model("weibull_aft", WeibullAFT)`); эксперименты собираются из конфигурации без импорта реализаций в ядро.

## EventLog — данные

Пассивное хранилище наблюдений (данные в покое). Состояния START/TERMINAL/CENS — конфигурация, не константы.

```python
class EventLog:
    events: DataFrame        # case_id, activity, timestamp, ...
    cases: DataFrame         # статические признаки объектов
    measurements: DataFrame  # case_id, timestamp, feature, value (time-varying)
    observation: DataFrame   # конец наблюдения, индикатор исхода, причина цензуры

    @classmethod
    def from_frame(cls, data, mapping, **parse_options) -> "EventLog": ...
    def prepare(self, rules) -> "EventLog": ...      # сортировка, обрезка до TERMINAL,
                                                     # склейка повторов, min_duration
    def intervals(self) -> DataFrame: ...            # интервалы состояний + индикатор цензуры
    def features_at(self, case_id, t, mode="train") -> dict: ...   # as-of признаки
    def split(self, method, **options) -> tuple["EventLog", "EventLog"]: ...
    def to_frame(self) -> dict[str, DataFrame]: ...
```

Правила `prepare` заменяемы; трасса без START отбрасывается; незавершённая трасса — цензурированный интервал; `mode="forecast"` в `features_at` — защита от lookahead при прогнозе. Варианты реализации: правила подготовки трасс, парсеры источников данных.

## Network — структура процесса

Только структура: состояния, рёбра, начальные и терминальные состояния. Без вероятностей и без привязки к движку discovery; циклы допустимы (повторные события). Базовый вариант — граф состояний: одно активное состояние на объект (маркированные сети — см. «Расширения»).

```python
class Network:
    states: list[str]
    edges: DataFrame         # src, dst (+ веса)
    starts: list[str]
    terminals: list[str]

    @classmethod
    def from_edges(cls, edges, starts, terminals) -> "Network": ...
    def outgoing(self, state) -> list[str]: ...
    def validate(self) -> DataFrame: ...   # недостижимые состояния, тупики

class PetriBackend(Protocol):              # движок построения сети
    def discover(self, log: EventLog) -> Network: ...
```

```python
network = discover(train_log, backend=Pm4pyBackend(method="inductive"))
network = Network.from_edges(edges, starts=["START"], terminals=["TERMINAL"])
```

PM4Py импортируется только внутри адаптера (изоляция AGPL-зависимости от ядра). Варианты реализации: майнеры PM4Py (Inductive / Heuristics / DFG), собственные алгоритмы discovery.

## Протоколы моделей — точки расширения

Минимальные контракты; экземпляр `TimeModel` обучается на каждом ребре. Студенты реализуют варианты (параметрические семейства, CoxPH, деревья выживания, KDE; регрессии выбора; модели динамики признаков), не изменяя ядро.

```python
class TimeModel(Protocol):                      # модель времени перехода
    def fit(self, durations, events, X=None) -> Self: ...   # правое цензурирование
    def sample(self, X, rng) -> float: ...
    def sf(self, t, X) -> float: ...            # + cdf, hazard, pdf
    def logsf(self, t, X) -> float: ...         # + logpdf — для правдоподобия
    def quantile(self, q, X) -> float: ...      # + mean, median, aic, params

class TransitionModel(Protocol):                # выбор направления
    def fit(self, log, network) -> Self: ...
    def probs(self, state, X, history) -> dict[str, float]: ...

class FeatureUpdater(Protocol):                 # X -> X' при входе в состояние
    def __call__(self, X, state, t, rng) -> dict: ...
    def loglik(self, X, X_new, state) -> float: ...         # опционально, для валидации

class TransitionPicker(Protocol):               # совместный выбор (куда, когда)
    def pick(self, probs, time_models, X, rng) -> tuple[str, float]: ...
```

Встроенные реализации: семейства exponential / weibull / lognormal / gamma с выбором по AIC; `TransitionModel` нормировкой весов; тождественный `FeatureUpdater`; `ChoiceThenTime` (направление по p, затем время).

**Инварианты реализаций:**

- **Batch-семантика:** везде, где принимается `X`, допустим батч (несколько объектов); скалярный выход становится вектором.
- **`fit` возвращает новый экземпляр** (`Self`); обученная модель неизменяема.
- **Чистые функции от аргументов и `rng`:** без скрытого состояния и скрытой случайности.

Генерация, прогноз и оценка обязаны использовать один `TransitionPicker` — он фиксируется в конфигурации модели (альтернативные стратегии — см. «Расширения»).

## ProcessModel — композиция

```python
class ProcessModel:
    network: Network
    transitions: TransitionModel
    times: TimeModel             # прототип; обученный экземпляр на каждом ребре
    picker: TransitionPicker

    def fit(self, log, features=None, use_censoring=True) -> "ProcessModel": ...
    def transition_parameters(self, state, X, history) -> dict[str, tuple[float, TimeModel]]: ...
    def next_transition(self, state, X, history, rng) -> tuple[str, float]: ...
    def edges_table(self) -> DataFrame: ...      # диагностика обучения по рёбрам
```

Запросный интерфейс для движков: `next_transition` (стохастический шаг = `transition_parameters` + `picker.pick`) и `transition_parameters` (аналитический доступ к распределениям — для точного прогноза, диагностики, метрик). `picker` остаётся частью конфигурации модели, чтобы генерация, прогноз и оценка использовали одну совместную семантику. Модель не содержит `simulate/predict/evaluate` (это обязанности движков) и не хранит состояние объектов. Для генерации ground truth параметры задаются константными реализациями тех же протоколов.

## Движки

Движки взаимодействуют с моделью только через `next_transition` (стохастический шаг) и `transition_parameters` (аналитический доступ), а также публичную структуру сети (`outgoing`, `terminals`); внутренние параметры модели им недоступны. Все используемые движком функции и модели (признаки, выбор, исполнение) передаются параметрами при инициализации, а не внутри методов.

### Simulator — генерация

```python
class Simulator:
    def __init__(self, model: ProcessModel, seed=None, executor=None): ...
    def run(self, stream: ObjectStream, horizon=None, max_steps=None) -> EventLog: ...
```

Шаг цикла на активном объекте:

```python
state, X, t = stream.current(obj)
dst, dt = model.next_transition(state, X, history, rng)  # единственный запрос к модели
X_new = stream.updater(X, dst, t + dt, rng)              # объект FeatureUpdater (задан при init)
log.write(obj, dst, t + dt, X_new)                       # скрытые узлы не пишутся
```

Результат: журнал (`case_id, state, timestamp, X', переход, dt`) и таблица цензуры (`end, reason`). Остановка: TERMINAL / max_steps / горизонт; список скрытых узлов — конфигурация.

### Router — применение

Тот же цикл, но признаки не вычисляются, а читаются из журнала: `log.features_at(case_id, t, mode="forecast")`. Результат — цепочка состояний со временем для объекта.

```python
class Router:
    def __init__(self, model: ProcessModel): ...
    def run(self, log: EventLog, case_id, horizon=None) -> DataFrame: ...
```

### Forecaster — прогноз

```python
class Forecaster:
    def __init__(self, model, method="simulation", n_paths=1000, seed=None): ...
    def survival(self, state, X, times) -> DataFrame: ...        # S_k(times) до каждого терминала k
    def hazard(self, state, X, times) -> DataFrame: ...          # hazard_k(times)
    def terminal_probs(self, state, X) -> dict[str, float]: ...  # вероятности терминалов
    def expected_time(self, state, X) -> dict[str, float]: ...   # ожидаемое время до терминалов
```

`method="simulation"`: `n_paths` прогонов цикла из точки `(state, X, t0)`, агрегация по путям. `method="exact"`: композиция переходных ядер на сетке времени. Оба метода используют один и тот же `transition_parameters`. Цензура исключается из выбора перехода, но входит в оценку функций; время, уже проведённое в состоянии, задаётся параметром `elapsed`.

### ObjectStream — состояние исполнения

```python
class ObjectStream:
    objects: DataFrame             # case_id, state, t, X, история
    updater: FeatureUpdater        # модель обновления признаков (может содержать прогнозную модель)

    def __init__(self, initial, updater: FeatureUpdater, seed=None): ...
    def current(self, case_id) -> tuple[str, dict, float]: ...
```

`EventLog` — данные в покое; `ObjectStream` — изменяемое состояние исполнения, которым владеет движок. Движок записывает события потока в новый журнал, поэтому результат `Simulator` — полноценный `EventLog`.

## Визуализация и метрики

```python
class Visualizer(Protocol):                      # бэкенды: graphviz / matplotlib / web
    def plot_network(self, model) -> Figure: ...            # граф состояний с CENS
    def plot_edge_curves(self, model, edge) -> Figure: ...  # S(t), hazard, p, квантили
    def compare_km(self, model, log, edge) -> Figure: ...   # сравнение с Капланом–Майером

class Metric(Protocol):
    def __call__(self, model, log) -> DataFrame: ...
```

```python
scores = evaluate(model, test_log, metrics=["nll", "ibs"])
table  = compare([model_a, model_b], test_log, metrics=["nll"])
diff   = compare_networks(net_true, net_recovered)  # рёбра (precision/recall/F1), вероятности
```

Методы `plot` в классах ядра — делегаты к модулю `viz`. Варианты реализации: новые бэкенды рендеринга, новые метрики (регистрация по имени).

## Восстановление структуры (round-trip)

Сценарий «обучение на внешних данных → генерация → восстановление сети → сравнение» — замкнутый конвейер из существующих шагов, потому что результат `Simulator` — полноценный `EventLog`, пригодный для discovery:

```python
model = ProcessModel(network_true, ...).fit(real_log)          # обучение на внешних данных
syn_log = Simulator(model).run(stream, seed=42)                # генерация новых данных
network_rec = discover(syn_log, backend=Pm4pyBackend())        # восстановление сети
diff = compare_networks(network_true, network_rec)             # сравнение структур
score = evaluate(ProcessModel(network_rec, ...).fit(syn_log), test_log, ...)  # и поведения
```

## Перенос на новый домен

Каркас фиксирует **инварианты** (формат журнала, цикл движков, протоколы), домен задаёт **только точки расширения**:

| Что меняется в домене | Точка расширения | Что НЕ меняется |
|---|---|---|
| Источник и структура данных | `mapping` и правила `prepare` в `EventLog` | формат журнала |
| Структура процесса | `PetriBackend` или `Network.from_edges` | `Network`, движки |
| Время перехода (повторные события, конкурирующие риски, …) | реализация `TimeModel` | контракт `fit/sf/hazard/sample` |
| Вероятности выбора направления | реализация `TransitionModel` | `transition_parameters` |
| Динамика признаков | реализация `FeatureUpdater` | цикл движков |
| Совместный выбор (направление, время) | реализация `TransitionPicker` | согласованность расчёта и симуляции |
| Скрытое состояние | реализация `LatentModel` | поток, журнал, движки |
| Критерии качества | реализация `Metric` | `evaluate` / `compare` |

Студент в новом домене пишет 1–3 реализации протоколов и регистрирует их по имени; движки, журнал, визуализация и метрики работают без изменений. Пример: повторные события — `TimeModel` с признаками истории (счётчики, давность) и циклы в `Network`.

## Сквозной пример

```python
log = EventLog.from_frame(df, mapping={...}).prepare()
train, test = log.split(method="by_case", ratio=0.8)

network = discover(train, backend=Pm4pyBackend())
model = ProcessModel(network, times=AicTimeModel()).fit(train)

syn_log = Simulator(model).run(ObjectStream(objects, updater), seed=42)   # генерация
path    = Router(model).run(real_log, case_id=7)                          # применение
S       = Forecaster(model).survival("A", X0, times=[1, 7, 30])           # прогноз
scores  = evaluate(model, test, metrics=["nll", "ibs"])                   # качество
```

## Воспроизводимость

Единственный источник случайности — `numpy.random.Generator`, создаваемый движком из `seed` и передаваемый во все протоколы (`sample`, `pick`, `update_features`, `LatentModel`). Дочерние генераторы на объект порождаются через `SeedSequence.spawn`, поэтому результат не зависит от порядка исполнения и числа воркеров. Реализации, использующие собственные источники случайности, нарушают контракт.

## Расширения

### Латентные состояния Z

```python
class LatentModel(Protocol):                   # скрытое состояние объекта
    def init(self, case_id, rng) -> Any: ...        # Z0 при создании объекта
    def update(self, Z, state, t, rng) -> Any: ...  # Z' при входе в состояние
    def infer(self, history) -> Any: ...            # оценка Z по наблюдаемой истории
```

Z хранится в `ObjectStream` и не пишется в наблюдаемый журнал. При подключённой `LatentModel` протоколы получают дополнительный опциональный аргумент `Z=None`; движки передают его без интерпретации, модель не хранит Z. На реальных данных Z оценивается через `infer(history)`. Ядро и движки не изменяются: задача со скрытым состоянием сводится к одной реализации `LatentModel`.

### Конкурирующие риски

Несколько терминальных состояний и причины цензуры задаются конфигурацией `EventLog`. Совместное распределение (dst, dt) определяет стратегия `TransitionPicker`: `ChoiceThenTime` (направление по p, затем время) или `CompetingRisks` (минимум по сэмплированным временам; направление рождается из сравнения времён, модель выбора может не запускаться). Для пользователя стратегия — категориальный параметр конфигурации (имя из реестра); новые совместные семантики (зависимые риски, копулы) — новые реализации протокола.

### Маркированные сети

Базовый `Network` — граф состояний (одно активное состояние на объект). Параллелизм процесса (AND-split/join, общие ресурсы) — расширение `MarkedNetwork`: маркировка, разрешённые переходы, явная семантика конфликтов. Базовые движки от него не зависят.

### Ускорение и параллелизация

Инструменты не фиксируются; каркас фиксирует инварианты, делающие ускорение возможным без смены API:

1. Batch-семантика протоколов — векторизация шага по множеству объектов.
2. Независимость траекторий объектов — параллелизм по объектам.
3. Неизменяемость обученных моделей — разделение между воркерами без копирования.
4. Дочерние генераторы случайности на объект — детерминизм при любом числе воркеров.
5. Параметр `executor` в движках — смена стратегии исполнения без изменения моделей.
6. Кэширование `transition_parameters` по ключу (состояние, хэш X).
7. Колоночное представление журнала; векторные операции на горячих путях (интервалы, as-of признаки).

## Ключевые решения

| Решение | Причина |
|---|---|
| Структура (Network) отделена от параметров (ProcessModel) | Одна сеть — много профилей поведения; ground truth |
| Цикл принадлежит движкам, не модели | Модель остаётся компактной и тестируемой; политика обхода меняется отдельно |
| Генерация и применение — разные движки | Разные источники признаков (модель vs журнал), разная семантика |
| Discovery только через адаптер | Ядро не зависит от лицензии PM4Py (AGPL) |
| Протоколы вместо базовых классов | Любая прогнозная модель подключается без наследования |
| Единый `rng` во всех протоколах + `spawn` на объект | Полная воспроизводимость по seed, в т.ч. при параллелизме |
| Синтетический журнал = полноценный EventLog | Замкнутый round-trip: генерация → discovery → сравнение сетей |
| Домен = реализации протоколов + конфигурация | Перенос на новый домен без изменений ядра и движков |
| Расширения (Z, конкурирующие риски, маркировка, ускорение) — отдельный слой | Основной каркас остаётся минимальным |
