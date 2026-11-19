# Базовые классы LogGenerator и Forecaster в survpm

## LogGenerator: генерация журнала

`LogGenerator` превращает правило "откуда стартует объект и какой у него следующий переход" в журнал событий с правой цензурой. Подкласс определяет два метода, цикл по шагам и сборку журнала делает база.

```python
class LogGenerator(ABC):
    terminals: set                      # завершающие состояния

    def __init__(self, seed=None, horizon=None, max_steps=1000, rng="per_case"): ...

    # обязательно
    @abstractmethod
    def start_cases(self, n, rng) -> Iterable[Case]: ...     # начальные объекты
    @abstractmethod
    def next_step(self, case, rng) -> Step | None: ...      # переход из case.state

    # необязательно
    def on_enter(self, case, rng) -> dict: ...              # признаки X на входе в состояние

    # база
    def generate(self, n=None) -> EventLog: ...            # журнал + таблица наблюдения
    def traces(self, n=None) -> list: ...                  # [(состояния, дни от старта)]
```

Объект - это `Case(case_id, state, t, X, history)`. Шаг - это `Step(dst, dt)`: переход в `dst` через `dt` дней. `Step(None, dt, "lost")` означает обрыв наблюдения без события, а `None` - что из состояния нет выхода.

База отвечает за:

- цикл по шагам до терминала, горизонта наблюдения или `max_steps`;
- воспроизводимость: у каждого объекта свой поток случайных чисел по `case_id`, поэтому результат не зависит от порядка объектов;
- проверку шагов (конечное неотрицательное `dt`, уникальные `case_id`);
- сборку `EventLog`: события, признаки на каждом входе и таблица наблюдения `end, event, reason`, где reason - это причина цензуры (`horizon`, `lost`, `max_steps`, `dead_end`).

## Forecaster: прогноз для наблюдаемого объекта

`Forecaster` отвечает на один вопрос: с какой вероятностью объект с историей до момента as\_of завершится исходом k за время t. Единственный обязательный метод возвращает функции кумулятивной инцидентности конкурирующих исходов:

```latex
F_k(t) = P\left(T \le t,\ \text{исход} = k \mid \mathcal{H}_{\text{as\_of}}\right)
```

Здесь T - время от as\_of до терминала, а H - наблюдаемая история объекта. Как считать F\_k, решает подкласс.

```python
class Forecaster(ABC):
    def __init__(self, terminals, history_features=None): ...

    # обязательно: математика
    @abstractmethod
    def cumulative_incidence(self, case, times) -> pd.DataFrame: ...  # horizon, outcome, cdf

    # база: данные и проверки
    def context(self, log, case_id, as_of) -> Case: ...   # срез истории до as_of

    # база: производные величины (можно переопределить)
    def terminal_probs(self, log, as_of, times, case_id=None): ...
    def survival(self, log, as_of, times, case_id=None): ...
    def hazard(self, log, as_of, times, case_id=None): ...
    def expected_time(self, log, as_of, horizon, case_id=None): ...
```

База делает только то, что не зависит от метода расчёта:

1. **Данные.** Обрезает историю объекта на as\_of (прогноз не видит будущего) и строит `Case`: текущее состояние, признаки X и время, уже проведённое в состоянии.
2. **Проверка ответа любой реализации:** 0 ≤ F\_k(t) ≤ 1, F\_k не убывает по t, Σ\_k F\_k(t) ≤ 1.
3. **Производные величины по определению.** Это не оценки, а тождества конкурирующих рисков, верные для любого закона:

```latex
S(t) = 1 - \sum_k F_k(t), \qquad
h_k(a,b] = \frac{F_k(b) - F_k(a)}{S(a)\,(b-a)}, \qquad
\mathrm{RMST}(h) = \int_0^h S(t)\,dt
```

Если у подкласса есть своя формула для какой-то из них (например, аналитическая интенсивность), он переопределяет метод. Все методы принимают список `case_id` и возвращают длинную таблицу по всем объектам.
