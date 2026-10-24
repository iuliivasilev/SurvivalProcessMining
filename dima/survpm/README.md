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
model.survival([1, 7, 30])           # P(не выписан к дню t)
model.simulate(1000, seed=0)         # трассы [(состояния, времена), ...]
net, im, fm = model.to_pm4py()       # сеть Петри PM4Py с местом CENS
```

Свой журнал — любая таблица с колонками `case_id, activity, timestamp`
(или `survpm.read_csv(path, case_id=..., activity=..., timestamp=...)`).
Если известен момент окончания наблюдения незавершённых трасс, передайте его:
`fit(raw, observation_end=series_by_case_id)`. Без него цензура наступает в момент
последней записи трассы, и на Sepsis это почти всегда ноль: учёт цензуры там ничего не меняет.

Наивная оценка только по завершённым интервалам: `CensoredStateNet(..., use_censoring=False)`.

## Метрики и кросс-валидация

```python
from survpm.metrics import crossval

raw, censoring = datasets.generate_ad(1000, seed=2026)
results = crossval.cross_validate(raw, "A", "D", censoring["end"], censoring["reason"], k=5)
crossval.summarize(results["level3"])                          # среднее и std по фолдам
crossval.paired_differences(results["level4"], "nll_per_interval")   # censor_aware − naive по фолдам
```

### Свои метрики: `set_metrics`

Метрика — функция от фолда `fold`, которая возвращает строку (`dict`) или список строк.
В `fold` есть обученные модели `fold.models` (`naive`, `censor_aware`), обучающие и тестовые
журналы, трассы и интервалы (`fold.train_log`, `fold.test_cases`, `fold.test_intervals` …),
`fold.seed` и `fold.n_sim`. `set_metrics` добавляет метрику под новым именем, заменяет под
существующим (`level1` … `level4`) и убирает при значении `None`.

```python
import numpy as np

def median_days(model, fold):               # одно число на модель
    return float(np.median(model.completion_times(2000, fold.seed)))

cv = crossval.CrossValidator("A", "D", k=5).set_metrics(
    median=crossval.per_model(median_days),  # per_model: по строке на каждую модель фолда
    censored=lambda fold: dict(share=1 - fold.test_cases["event"].mean()),
    level2=None,                             # не считать уровень 2
)
results = cv.run(raw, censoring["end"], censoring["reason"])
crossval.summarize(results["median"])
```

То же короче: `crossval.cross_validate(raw, "A", "D", ..., metrics=dict(median=crossval.per_model(median_days)))`.
Чтобы сравнивать другие модели, переопределите `CrossValidator.fit_models`.

| Уровень | Метрики | Модуль |
|---|---|---|
| 1. Process mining | fitness и precision (token replay), generalization, simplicity; сеть состояний против Inductive Miner | `metrics.process` |
| 2. Вероятности путей | entropic relevance, uEMSC (Ebi), EMD языков (POT) | `metrics.stochastic` |
| 3. Время | temporal profile, CFLD, 3-граммы, EMD длительности трасс и времени событий, KS по рёбрам, расстояние до Каплана–Майера, log-rank, integrated Brier score | `metrics.temporal` |
| 4. Обобщение | −log L тестовых интервалов с цензурой, доля трасс с невиданными переходами | `CensoredStateNet.nll` |

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
