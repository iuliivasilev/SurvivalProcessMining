# survpm

## Первая итерация архитектуры v2

`CensoredStateNet` по умолчанию использует `transition_mode="min_time"`:
независимые экспоненциальные времена по исходящим рёбрам, переход по минимуму.
Интенсивности обучаются с учётом цензуры; при признаках используется регрессия
логарифма интенсивности. `coefficients_table()` возвращает колонку `rate`.
Другие семейства в этом режиме пока отклоняются явно.

```python
import pandas as pd
from survpm import (EventLog, Network, ProcessModel, ObjectStream, Simulator,
                    discover, DirectlyFollowsBackend, Forecaster,
                    evaluate, compare, compare_networks)

origin = pd.Timestamp("2025-01-01", tz="UTC")
events = pd.DataFrame([
    ("a", "A", origin), ("a", "D", origin + pd.Timedelta(days=2)),
    ("b", "A", origin), ("b", "D", origin + pd.Timedelta(days=4)),
    ("c", "A", origin),  # ещё три дня экспозиции без исхода
], columns=["case_id", "activity", "timestamp"])
train = EventLog(events, pd.DataFrame(index=["a", "b", "c"]),
                 pd.DataFrame(columns=["case_id", "timestamp", "feature", "value"]),
                 last_seen=pd.Series({"c": origin + pd.Timedelta(days=3)}))
network = Network.from_edges([("A", "D")], starts=["A"], terminals=["D"])
model = ProcessModel(network).fit(train)  # новый объект, rate = 2/9 в день
stream = ObjectStream([dict(case_id=f"sim-{i}", state="A", t=origin, X={})
                       for i in range(30)])
generated = Simulator(model, seed=42).run(stream, horizon=30)
found = discover(generated, DirectlyFollowsBackend(starts=["A"], terminals=["D"]))
print(compare_networks(network, found))
print(Forecaster(model, n_paths=100, seed=42).survival(
    generated, origin, [1., 7., 30.], case_id="sim-0"))
print(evaluate(model, train, metrics=["nll"])["nll"])
print(compare({"fitted": model}, train, metrics=["nll"]))
```

`ObjectStream` содержит один или несколько объектов. `updater(X, state, t, rng)`
обновляет X только при входе в состояние. События, значения X и конец наблюдения
возвращаются в `EventLog`; таблица причин остановки доступна в `log.observation`.
`Router` и `Forecaster` читают признаки на входе в текущее состояние и ограничивают
реальные наблюдения моментом `as_of`. Будущие X задаются `future_features`;
по умолчанию значения сохраняются. Остаточное ожидание в первой итерации
поддерживается только для экспоненциальных переходов.

`Network`, `ProcessModel` и `TransitionModel` разделены; смена модели не требует
переписывать `Simulator`. `fit` у `ProcessModel` и `ExponentialTransitionModel`
возвращает новый объект; у совместимого `CensoredStateNet.fit` остаётся прежняя
семантика изменения экземпляра. `ExponentialTransitionModel.update` переобучается
на объединённых исходных и новых интервалах. `Evaluator` поддерживает NLL;
другие существующие метрики доступны через `CensoredStateNet.evaluate`.

В `CensoredStateNet.edges_` и `edges_table()` хранится наблюдаемое ядро:
вероятность направления `lambda / sum(lambda)` и время до победившего события
со средним `1 / sum(lambda)`. Времена отдельных конкурирующих причин имеют
средние `1 / lambda`; интенсивности доступны в `model.transitions_.rates`.
Поэтому существующие grid-кривые и адаптеры используют согласованное ядро.

Старый алгоритм доступен явно через `transition_mode="choice_then_time"`.
Ниже описаны его семейства и softmax/AFT; для воспроизведения старых экспериментов
укажите этот режим. Синтетический `datasets.generate_ad` также использует старую
семантику. LatentModel, точный прогноз движком Forecaster и прочие расширения
архитектуры пока не реализованы.


### Публичный API v2: возможности первой итерации

Ядру примера нужны только numpy, pandas и scipy; Python ≥ 3.10.
`evaluate(model, log, metrics=["nll"])` возвращает словарь таблиц `metric/value`:
NLL содержит `nll_per_interval` и `nll_per_trace`. Цензурированные интервалы дают
экспозицию, невозможный наблюдаемый переход — бесконечный NLL.
`compare({name: model}, log, metrics)` объединяет результаты в `model/metric/value`.
Также можно передать список или итератор: `compare([model], log, ["nll"])`;
имена назначаются по порядку: `model_0`, `model_1`, … .
`register_metric(name, func)` добавляет функцию `(model, log)` со скалярным
результатом либо таблицей `metric/value`; повторное имя отклоняется. Этот реестр
отделён от существующего реестра fold-метрик кросс-валидации.
Оценка, сравнение и визуализация не обучают модели и признаки.

`compare_networks(reference, estimated, state_mapping=None)` принимает `Network`
или `ProcessModel` и сравнивает только направленные рёбра: precision, recall, f1
и таблицы missing/extra. Отображение задаётся из имён estimated в имена reference;
неуказанные имена сохраняются. Для пустого estimated precision = 1, для пустого
reference recall = 1. Вероятности и параметры модели эта функция не сравнивает;
у чистой `Network` вероятностей нет.

| Компонент | Реализовано | Ограничение / заглушка |
|---|---|---|
| Данные | EventLog, prepare, intervals, prefix | Одна активная вершина на объект |
| Структура | Network; DirectlyFollowsBackend без optional зависимостей | Общие Petri nets и параллелизм не поддержаны |
| PM4Py | Pm4pyBackend(method="dfg") с ленивым импортом | inductive/heuristics явно NotImplementedError |
| Переходы | Независимые exponential, минимум времён; log-rate с признаками, L2, update | Другие семейства и nonlinear-модели v2 не реализованы; старый choice_then_time сохраняется |
| Признаки | DefaultHistoryFeatures, HistoryFeatures и FeatureUpdater protocols | LatentModel — только protocol; скрытое состояние не исполняется |
| Исполнение | ObjectStream, Simulator, Router; фиксированные X или updater при входе | Marked point process и executor не реализованы |
| Прогноз | Forecaster Monte Carlo: survival, terminal_probs, hazard по бинам, restricted mean | method="exact" явно NotImplementedError; expected_time требует конечный horizon |
| Метрики v2 | NLL с цензурой; пользовательский registry; сравнение топологии | IBS явно NotImplementedError: требуется train-only IPCW; legacy cross-validation сохраняется |
| Визуализация | Visualizer, MatplotlibVisualizer.plot_network и plot_edge_curves | compare_km и latent явно NotImplementedError; edge curves только exponential |

Для рисунков отдельно установите `matplotlib`. `network.plot()` делегирует
`MatplotlibVisualizer.plot_network(network)`; можно передать свой `Visualizer`.
`MatplotlibVisualizer().plot_edge_curves(model, ("A", "D"), [0., 1., 7.])` рисует
совместную CDF `P(следующий выход D, ожидание ≤ t)`: для exponential это
`lambda_AD / sum(lambda_A*) * (1 - exp(-sum(lambda_A*) * t))`. Это кривая одного
ожидания, а прогноз всего пути до исхода даёт `Forecaster`.
`plot_network(model)` принимает также ProcessModel и рисует его network.
Без times вызов `plot_edge_curves(model, ("A", "D"))` выбирает 101 точку
от нуля до пяти средних времён ожидания выхода; при нулевой суммарной
интенсивности — от нуля до пяти дней.
`network` и `edge_curves` сохраняются как совместимые имена методов visualizer.
`compare_km(model, log, edge)` пока явно отклоняется: реализация сравнения KM не включена.
`survpm.log.EventLog` — совместимый ленивый импорт класса из `survpm.features`.


Стохастические сети состояний с правым цензурированием, оцениваемые по журналам событий.
Модель: из каждого состояния выход выбирается с вероятностью `p`, время перехода берётся из
своего распределения (exponential, lognormal, gamma, weibull), а трассы, не дошедшие до
исхода, учитываются как цензурированные. По модели строятся сеть Петри с местом `CENS`,
симуляция и время до исхода.

## Установка

```bash
pip install -e .                 # ядро: numpy, pandas, scipy
pip install -e ".[pm4py]"        # + сеть Петри PM4Py, схемы 4.1 и 4.2 (graphviz)
pip install -e ".[metrics]"      # + метрики качества и кросс-валидация
# log-distance-measures формально требует Python < 3.13, но работает и на 3.13:
pip install --ignore-requires-python log-distance-measures jellyfish pulp
```

Для схем нужна программа graphviz (`brew install graphviz` на macOS).

## Быстрый старт

```python
import survpm
from survpm import datasets

raw = datasets.load_sepsis("../Med_Eremina/Sepsis Cases - Event Log_grouped.csv")
model = survpm.CensoredStateNet("Registration", "Discharge", transition_mode="choice_then_time").fit(raw)

model.edges_table()                  # p, распределение, среднее, медиана, 90% квантиль
model.survival([1, 7, 30])           # P(не выписан к дню t), симуляция
model.survival([1, 7, 30], method="grid")      # то же свёрткой на сетке, без случайности
model.state_curves(60)               # S_s(t) из каждого состояния до Discharge
model.residual_survival("Admission", 2, [1, 7])  # остаток, если уже 2 дня в Admission
model.simulate(1000, seed=0)         # трассы [(состояния, времена), ...]
net, im, fm = model.to_pm4py()       # сеть Петри PM4Py с местом CENS
```

Свой журнал — любая таблица с колонками `case_id, activity, timestamp`
(или `survpm.read_csv(path, case_id=..., activity=..., timestamp=...)`).
Если известен момент окончания наблюдения незавершённых трасс, передайте его:
`fit(raw, observation_end=series_by_case_id)`. Без него цензура наступает в момент
последней записи трассы, и на Sepsis это почти всегда ноль: учёт цензуры там ничего не меняет.

Наивная оценка только по завершённым интервалам: `CensoredStateNet(..., use_censoring=False)`.

Повторы состояния подряд (`repeats`, есть у `prepare`, `CensoredStateNet` и `CrossValidator`):
`"simultaneous"` (по умолчанию) склеивает только записи в один момент, поздние повторы дают петлю
`s → s`; `"collapse"` склеивает всю серию, как до 28.10.2026; `"keep"` не склеивает ничего.
Вероятность повтора каждого состояния — `model.repeat_table()` (`p_immediate`, `p_return`,
`expected_visits`).

## Признаки

Журнал с признаками — `survpm.EventLog` из трёх таблиц: `events` (`case_id, activity, timestamp`,
по ним строится сеть), `cases` (статические признаки, индекс `case_id`) и `measurements`
(`case_id, timestamp, feature, value` — значения, меняющиеся во времени). Формат не привязан к
медицине: подходит любая широкая таблица событий, где признаки — колонки.

```python
log = survpm.read_log(path, case_id="case", activity="act", timestamp="time")
# measurement_activities=["Lab"] — строки Lab станут только замерами, не состояниями (по умолчанию трассы полные)
log.static, log.dynamic             # разобраны автоматически; можно задать static=[...], dynamic=[...]
model = survpm.CensoredStateNet("A", "D", transition_mode="choice_then_time").fit(log)    # цензура не раньше log.last_seen

X = survpm.features.at_entry(model.intervals_, log, window="1D")  # признаки на вход в состояние
pieces = survpm.features.split(model.intervals_, log)            # куски между замерами (counting process)
```

Статический признак — колонка, у которой в каждой трассе не больше одного значения; значение берётся
первое непустое. Динамический — каждое непустое значение становится замером в момент события.
`at_entry` даёт последнее значение не позже входа (`f`), его давность в днях (`f_age`), число замеров
(`f_n`) и, с `window`, число замеров за окно. Значения из будущего не используются.
Строки `да/нет`, `true/false` становятся 1/0, числа — числами, остальное — категориями.

Sepsis с признаками: `datasets.load_sepsis_log(path)` — трассы полные, Examination остаётся состоянием,
значения анализов дополнительно в `measurements`; `measurement_activities=datasets.SEPSIS_MEASUREMENTS`
делает Examination прозрачной. Проверка — `features_sepsis.ipynb`.

### Обучение с признаками

Признаки включаются явно; без `feature_columns` сохраняется прежняя модель.

```python
model = survpm.CensoredStateNet(
    "Registration", "Discharge", transition_mode="choice_then_time",
    feature_columns=["Age", "CRP", "CRP_age", "history_visits"],
    l2=1.0,
).fit(train_log)                       # EventLog, отдельный от test_log
model.coefficients_table()             # коэффициенты для стандартизованных признаков
model.evaluate(test_log)["level4"]     # индивидуальный likelihood с цензурой

profile = model.for_features({"Age": 65, "CRP": 30, "CRP_age": 0.5, "history_visits": 1})
profile.edges_table()
profile.survival([1, 7, 30], method="grid")
profile.simulate(100, seed=0)
```

`features.at_entry` берёт значения на входе. Дополнительно доступны `history_visits`
(номер посещения текущего состояния, начиная с 1) и `history_elapsed_days` (дни от начала трассы).
`feature_window="1D"` включает оконные признаки, например `CRP_n_1D`. Можно передать готовые
признаки: `fit(intervals=intervals, X=frame)`, `nll(test_intervals, X=test_frame)`;
индексы X и интервалов должны совпадать.

Оценка совместная: `p_sd(X) = softmax(alpha_sd + beta_sd·X)`,
`T_sd(X) = exp(gamma_sd·X) T_sd(0)`. У цензуры суммируется `p_sd(X) S_sd(c | X)`
по скрытым выходам. Поддерживаются exponential, lognormal, gamma и weibull, без uniform.
Медианы пропусков, индикаторы, one-hot и стандартизация оцениваются только на обучении;
неизвестная категория при прогнозе кодируется как базовая. Постоянные колонки удаляются.
Стандартизованные значения по умолчанию обрезаются до диапазона обучения (`clip=True`):
выброс анализа в тесте кодируется как крайнее значение train, и `exp(β·z)` не экстраполируется
(на Sepsis без обрезки NLL на тесте 2.67 против 1.00 у модели без признаков, с обрезкой 0.89).
Для скошенных положительных признаков есть `log1p`:
`ProcessModel(net, transitions=ExponentialTransitionModel(log_features=["CRP", "Leucocytes"]))`
(на Sepsis 0.86). `clip=False` возвращает прежнее поведение. Признаки на входе для
`ProcessModel.fit(..., features=...)` и `evaluate` считаются одним векторным проходом
(`DefaultHistoryFeatures.transform_intervals`); свой `HistoryFeatures` считается по интервалам.
`l2` штрафует коэффициенты признаков, но не базовые параметры. В таблице коэффициентов
положительный `time` означает растяжение времени; `choice` задан относительно последнего
по алфавиту выхода состояния. Базовые `edges_` относятся к нулю закодированных признаков.

`for_features` возвращает копию с **фиксированным сценарием во всех будущих состояниях**.
Будущие анализы, история и признаки не обновляются автоматически. Время ожидания внутри
состояния учитывается через `profile.residual_survival(...)`, но новые измерения внутри
ожидания пока не входят в обучение. Без профиля вероятностные прогнозы и экспорт отклоняются.

CV с признаками: `CrossValidator(..., feature_columns=[...]).add_metrics("level4").run(log)`.
Он принимает EventLog и обучает кодирование отдельно в каждом фолде. Для модели с признаками
сейчас реализована индивидуальная оценка level4; остальные уровни требуют явного сценария:
`profile.evaluate(test_log, metrics=...)`. Это оценка фиксированного сценария, а не
персонализированная оценка всей тестовой популяции.

## Метрики и кросс-валидация

### Метрики одной обученной модели

`evaluate` оценивает модель на переданном тестовом журнале без переобучения:

```python
train, train_censoring = datasets.generate_ad(500, seed=1)
test, test_censoring = datasets.generate_ad(200, seed=2)
model = survpm.CensoredStateNet("A", "D", transition_mode="choice_then_time").fit(train, train_censoring["end"])

results = model.evaluate(test, observation_end=test_censoring["end"])
results["level4"]  # unseen_share, nll_per_interval, nll_per_trace

# Отдельные метрики, уровни или "all"; для уровней 1–3 нужны дополнительные зависимости.
results = model.evaluate(test, metrics=["fitness_token", "km_sup", "level4"],
                         observation_end=test_censoring["end"], n_sim=5000, seed=2026)
```

Возвращается словарь таблиц по уровням с колонкой `model="model"`, без колонки `fold`.
`temporal_deviation_share` дополнительно возвращает базовую линию `train_log`.
Уровень 1 оценивает только сеть этой модели; сравнение с Inductive Miner выполняется в CV.
Принимаются DataFrame и `EventLog`; используются те же настройки повторов и минимальной
длительности, что при обучении. У `EventLog` время наблюдения учитывает также измерения.
По умолчанию считается только `level4`, которому достаточно зависимостей ядра. Зависимости
других уровней загружаются при вычислении выбранных метрик.
Если все тестовые трассы неизвестны модели, `unseen_share=1`, а значения NLL — `NaN`.

При `fit(raw)` модель сохраняет подготовленный обучающий журнал в `log_` и таблицу трасс
в `cases_`. Они нужны для `ibs` (обучающая оценка цензуры) и `temporal_deviation_share`
(обучающий временной профиль). После `fit(intervals=...)` эти две метрики недоступны;
остальные можно считать. Тестовые данные не используются для восстановления обучения.

### Кросс-валидация

У `CrossValidator` по умолчанию не считается ничего: метрики добавляются явно через `add_metrics` —
по одной, уровнем или все сразу.

```python
from survpm.metrics import crossval

raw, censoring = datasets.generate_ad(1000, seed=2026)
cv = crossval.CrossValidator("A", "D", k=5)
cv.add_metrics("all")                       # все уровни
# cv.add_metrics("level1", "level4")        # уровнями
# cv.add_metrics("uemsc", "km_sup", "ibs")  # по одной
results = cv.run(raw, censoring["end"], censoring["reason"])
crossval.summarize(results["level3"])                                # среднее и std по фолдам
crossval.paired_differences(results["level4"], "nll_per_interval")   # censor_aware − naive по фолдам
```

`run` возвращает по таблице на уровень, в ней только выбранные метрики этого уровня.
`remove_metrics` убирает метрики так же: по имени, уровнем или `"all"`. Короткая форма:
`crossval.cross_validate(raw, "A", "D", ..., metrics=["level3", "uemsc"])` (по умолчанию `"all"`).
Имена всех встроенных метрик — `crossval.METRICS`, по уровням — `crossval.LEVELS`:

| Уровень | Имена |
|---|---|
| `level1` | `fitness_token`, `precision_token`, `generalization`, `simplicity` |
| `level2` | `entropic_relevance`, `uemsc`, `emd` |
| `level3` | `temporal_deviation_share`, `cfld`, `ngram_3`, `cycle_time_emd`, `absolute_event_emd`, `edge_ks`, `km_sup`, `km_l1_days`, `logrank` (колонки `logrank_statistic`, `logrank_p`), `ibs` |
| `level4` | `unseen_share`, `nll_per_interval`, `nll_per_trace` |

Метрики одного фолда делят общие вычисления (симуляции, сети, журналы PM4Py), поэтому
уровень целиком не дороже, чем его метрики по отдельности.

### Свои метрики

Своя метрика добавляется тем же `add_metrics` под своим именем и попадает в отдельную таблицу.
Это функция от фолда `fold`, которая возвращает строку (`dict`) или список строк.
В `fold` есть обученные модели `fold.models` (`naive`, `censor_aware`), обучающие и тестовые
журналы, трассы и интервалы (`fold.train_log`, `fold.test_cases`, `fold.test_intervals` …),
`fold.seed` и `fold.n_sim`.

```python
import numpy as np

def median_days(model, fold):               # одно число на модель
    return float(np.median(model.completion_times(2000, fold.seed)))

cv = crossval.CrossValidator("A", "D", k=5).add_metrics(
    "level3",
    median=crossval.per_model(median_days),  # per_model: по строке на каждую модель фолда
    censored=lambda fold: dict(share=1 - fold.test_cases["event"].mean()),
)
results = cv.run(raw, censoring["end"], censoring["reason"])
crossval.summarize(results["median"])
```

Чтобы сравнивать другие модели, переопределите `CrossValidator.fit_models`.

| Уровень | Метрики | Модуль |
|---|---|---|
| 1. Process mining | fitness и precision (token replay), generalization, simplicity; сеть состояний против Inductive Miner | `metrics.process` |
| 2. Вероятности путей | entropic relevance, uEMSC (Ebi), EMD языков (POT) | `metrics.stochastic` |
| 3. Время | temporal profile, CFLD, 3-граммы, EMD длительности трасс и времени событий, KS по рёбрам, расстояние до Каплана–Майера, log-rank, integrated Brier score | `metrics.temporal` |
| 4. Обобщение | −log L тестовых интервалов с цензурой, доля трасс с невиданными переходами | `metrics.likelihood` |

Метрики считаются на всех тестовых трассах, цензурированная трасса считается
префиксом. Фолды делятся целыми трассами и равны по доле цензурированных трасс.

## Устройство

| Модуль | Что делает | Зависимости |
|---|---|---|
| `log` | журнал → трассы, интервалы, вероятности цензуры | ядро |
| `distributions` | семейства, выбор по AIC | ядро |
| `estimation` | совместная MLE `p` и параметров с цензурой | ядро |
| `simulation` | симуляция модели, время до исхода | ядро |
| `model` | `CensoredStateNet` | ядро |
| `datasets` | синтетика A–D с известной истиной, загрузка Sepsis | ядро |
| `adapters` | PM4Py (сеть Петри, схема 4.1), Ebi (SDFA) | PM4Py по требованию |
| `viz` | схема 4.2 | graphviz |
| `metrics.*` | метрики, кросс-валидация | extras `metrics` |

Ядро не импортирует PM4Py: PM4Py распространяется под AGPL v3, и так лицензию самой
библиотеки можно выбрать свободно.

## Тесты

```bash
pip install -e ".[test]" && pytest tests
```
