# survpm

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
model = survpm.CensoredStateNet("Registration", "Discharge").fit(raw)

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
model = survpm.CensoredStateNet("A", "D").fit(log)    # цензура не раньше log.last_seen

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
    "Registration", "Discharge",
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
model = survpm.CensoredStateNet("A", "D").fit(train, train_censoring["end"])

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
