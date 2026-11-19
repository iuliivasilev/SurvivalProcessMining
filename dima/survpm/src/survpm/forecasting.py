"""Прогноз для наблюдаемого объекта: базовый класс :class:`Forecaster` и реализации.

Прогноз — функции кумулятивной инцидентности конкурирующих исходов

    F_k(t) = P(T ≤ t, исход = k | история объекта до as_of),

где T — время от as_of до терминала. Базовый класс задаёт только этот интерфейс,
готовит данные (срез истории до as_of) и проверяет ответ; как считать F_k, решает подкласс.
Остальные величины — тождества конкурирующих рисков, верные для любого закона::

    S(t) = 1 − Σ_k F_k(t)
    h_k(a, b] = (F_k(b) − F_k(a)) / S(a) / (b − a)
    RMST(h) = ∫₀ʰ S(t) dt

Свой прогноз — подкласс с одним методом::

    class AlwaysExp(survpm.Forecaster):
        def cumulative_incidence(self, case, times):
            return pd.DataFrame(dict(horizon=times, outcome="D", cdf=1 - np.exp(-.5 * np.asarray(times))))

    AlwaysExp(terminals=["D"]).survival(log, as_of, [1, 7, 30], case_id="c1")

Реализации:

* :class:`SimulationForecaster` — Монте-Карло по :class:`~survpm.generation.LogGenerator`
  (по умолчанию :class:`~survpm.generation.ModelGenerator`); оценка со стандартной ошибкой;
* :class:`MarkovForecaster` — точное решение для экспоненциальных конкурирующих времён:
  марковская цепь с поглощающими терминалами, F_k(t) = [exp(Q t)]_{s,k}.

``survpm.Forecaster(model, n_paths=..., seed=...)`` по-прежнему работает и создаёт
:class:`SimulationForecaster` (``method="markov"`` — :class:`MarkovForecaster`).
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .eventlog import EventLog
from .generation import DAY, Case, ModelGenerator, timestamp
from .models.features import DefaultHistoryFeatures

INCIDENCE_COLUMNS = ['horizon', 'outcome', 'cdf']


def observed_history(log, case_id, as_of):
    """Журнал одного объекта, обрезанный на ``as_of``: прогноз не видит будущего."""
    cutoff = timestamp(as_of)
    events = log.events.loc[(log.events.case_id == case_id) & (log.events.timestamp <= cutoff)].copy()
    measures = log.measurements.loc[(log.measurements.case_id == case_id) & (log.measurements.timestamp <= cutoff)].copy()
    return EventLog(events, log.cases.reindex([case_id]).copy(), measures, pd.Series({case_id: cutoff}))


def _times(times):
    times = np.asarray(times, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('Задайте неотрицательные конечные горизонты')
    return times


class Forecaster(ABC):
    """Базовый прогноз. Обязательный метод — :meth:`cumulative_incidence`.

    ``terminals`` — завершающие состояния (исходы k). ``history_features`` строит контекст
    объекта по его истории до as_of (по умолчанию :class:`~survpm.DefaultHistoryFeatures`).
    Методы :meth:`terminal_probs`, :meth:`survival`, :meth:`hazard`, :meth:`expected_time`
    реализованы в базовом классе через F_k; подкласс может переопределить любой из них.
    ``case_id`` — один объект или список (тогда в таблицах есть колонка ``case_id``).
    """

    def __new__(cls, *args, **kwargs):
        # Совместимость: Forecaster(model, ...) — прежний прогноз по модели.
        if cls is Forecaster and args and hasattr(args[0], 'network'):
            method = kwargs.get('method', 'simulation')
            impl = {'simulation': SimulationForecaster, 'markov': MarkovForecaster}.get(method)
            if impl is None:
                raise NotImplementedError(f'Forecaster method={method!r} is not implemented; '
                                          f'use "simulation" or "markov" (exact for exponential models)')
            return super().__new__(impl)
        return super().__new__(cls)

    def __init__(self, terminals, history_features=None):
        self.terminals = list([terminals] if isinstance(terminals, str) else terminals)
        if not self.terminals:
            raise ValueError('Нужен хотя бы один терминал')
        self.history_features = history_features or DefaultHistoryFeatures()

    # --- обязательно: математика --------------------------------------------------------------
    @abstractmethod
    def cumulative_incidence(self, case, times):
        """F_k(t) для объекта ``case`` (:class:`~survpm.Case`: ``state``, ``t`` = as_of, ``X``,
        ``history``, ``sojourn`` — дни в текущем состоянии). ``times`` — дни от ``case.t``,
        неотрицательные, по возрастанию. Таблица ``horizon, outcome, cdf`` (+ свои колонки)."""

    # --- база: данные и проверки ---------------------------------------------------------------
    def context(self, log, case_id, as_of):
        """:class:`~survpm.Case` объекта на ``as_of`` по истории, обрезанной на ``as_of``."""
        ctx = self.history_features.transform(observed_history(log, case_id, as_of), case_id, as_of)
        return Case(ctx['case_id'], ctx['state'], ctx['t'], ctx.get('X', {}), ctx.get('history', []),
                    sojourn=ctx.get('elapsed', 0.), initialized=True)

    def _ids(self, log, case_id):
        if case_id is None:
            ids = log.events.case_id.unique()
            if len(ids) != 1:
                raise ValueError('Для нескольких объектов укажите case_id')
            return [ids[0]], False
        if isinstance(case_id, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            return list(case_id), True
        return [case_id], False

    def incidence(self, case, times):
        """:meth:`cumulative_incidence` с проверкой ответа: 0 ≤ F_k ≤ 1, F_k не убывает, Σ_k F_k ≤ 1."""
        times = _times(times)
        grid = np.unique(times)
        table = pd.DataFrame(self.cumulative_incidence(case, grid))
        missing = set(INCIDENCE_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f'cumulative_incidence должен вернуть колонки {INCIDENCE_COLUMNS}')
        cdf = (table.pivot_table(index='horizon', columns='outcome', values='cdf', aggfunc='first')
               .reindex(index=grid, columns=self.terminals))
        if cdf.isna().any().any():
            raise ValueError('cumulative_incidence должен дать cdf для каждого horizon и каждого терминала')
        values = cdf.to_numpy(float)
        eps = 1e-9
        if (values < -eps).any() or (values > 1 + eps).any():
            raise ValueError('F_k(t) вне [0, 1]')
        if (np.diff(values, axis=0) < -eps).any():
            raise ValueError('F_k(t) убывает по t')
        if (values.sum(axis=1) > 1 + eps).any():
            raise ValueError('Сумма F_k(t) по исходам больше 1')
        table = table.set_index(['horizon', 'outcome']).reindex(
            pd.MultiIndex.from_product([times, self.terminals], names=['horizon', 'outcome'])).reset_index()
        return table

    def _per_case(self, log, as_of, case_id, compute):
        ids, many = self._ids(log, case_id)
        tables = []
        for cid in ids:
            table = compute(self.context(log, cid, as_of))
            tables.append(table.assign(case_id=cid)[['case_id', *table.columns]] if many else table)
        return pd.concat(tables, ignore_index=True) if many else tables[0]

    # --- база: производные величины (можно переопределить) -------------------------------------
    def terminal_probs(self, log, as_of, times, case_id=None):
        """F_k(t): таблица ``horizon, outcome, probability``."""
        return self._per_case(log, as_of, case_id, lambda case: self.incidence(case, times)
                              .rename(columns={'cdf': 'probability'}))

    def _survival(self, case, times):
        table = self.incidence(case, times)
        cdf = table.groupby('horizon', sort=False).cdf.sum().reindex(_times(times)).to_numpy()
        cdf = np.minimum(cdf, 1.)
        return pd.DataFrame(dict(horizon=_times(times), survival=1 - cdf, cdf=cdf))

    def survival(self, log, as_of, times, case_id=None):
        """S(t) = 1 − Σ_k F_k(t): таблица ``horizon, survival, cdf``."""
        return self._per_case(log, as_of, case_id, lambda case: self._survival(case, times))

    def hazard(self, log, as_of, times, case_id=None):
        """Интенсивность исхода k на интервалах (a, b]: (F_k(b) − F_k(a)) / S(a) / (b − a).

        ``times`` — правые края, строго по возрастанию; первый интервал начинается в 0.
        Если S(a) = 0, интенсивность не определена (NaN). Таблица ``horizon, outcome, hazard``."""
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(np.r_[0., times]) <= 0).any():
            raise ValueError('hazard times must be strictly increasing and positive')
        edges = np.r_[0., times]

        def compute(case):
            F = self.incidence(case, edges).pivot(index='horizon', columns='outcome', values='cdf') \
                .reindex(index=edges, columns=self.terminals).to_numpy(float)
            S = 1 - F.sum(axis=1)
            rows = []
            for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                for j, outcome in enumerate(self.terminals):
                    value = (F[i + 1, j] - F[i, j]) / S[i] / (right - left) if S[i] > 0 else np.nan
                    rows.append(dict(horizon=right, outcome=outcome, hazard=value))
            return pd.DataFrame(rows, columns=['horizon', 'outcome', 'hazard'])
        return self._per_case(log, as_of, case_id, compute)

    def restricted_mean(self, case, horizon, n_grid=2000):
        """``(RMST, P(не завершён к horizon))``; RMST = ∫₀ʰ S(t) dt по сетке (трапеции)."""
        grid = np.linspace(0., horizon, n_grid + 1)
        S = self._survival(case, grid).survival.to_numpy()
        return float(np.trapezoid(S, grid) if hasattr(np, 'trapezoid') else np.trapz(S, grid)), float(S[-1])

    def expected_time(self, log, as_of, horizon, case_id=None):
        """Ограниченное среднее время до любого терминала и доля незавершённых к ``horizon``."""
        horizon = float(_times([horizon])[0])

        def compute(case):
            rmst, unfinished = self.restricted_mean(case, horizon)
            return pd.DataFrame([dict(restricted_mean_time=rmst, probability_unfinished=unfinished)])
        result = self._per_case(log, as_of, case_id, compute)
        return result if 'case_id' in result else result.iloc[0].to_dict()


# --- Монте-Карло ------------------------------------------------------------------------------

class SimulationForecaster(Forecaster):
    """Монте-Карло: F_k(t) — доля из ``n_paths`` путей, завершившихся в k не позже t.

    Оценка несмещённая, стандартная ошибка √(F(1−F)/n) — колонка ``se``. Пути разыгрывает
    :class:`~survpm.generation.LogGenerator` из :meth:`generator` (по умолчанию
    :class:`~survpm.generation.ModelGenerator` по ``model``); у каждого пути свой поток
    случайных чисел, поэтому при заданном ``seed`` все методы видят одни и те же пути.
    ``future_features`` — сценарий изменения X после as_of (updater потока).
    """
    def __init__(self, model, history_features=None, method='simulation', n_paths=1000, seed=None,
                 future_features=None):
        if method != 'simulation':
            raise ValueError('SimulationForecaster: method="simulation"')
        if n_paths < 1:
            raise ValueError('n_paths должен быть положительным')
        super().__init__(model.network.terminals, history_features or model.history_features)
        self.model, self.n_paths, self.seed, self.future_features = model, n_paths, seed, future_features

    def generator(self, case, horizon):
        """Генератор ``n_paths`` путей из контекста ``case`` до ``horizon`` дней."""
        from .engine import ObjectStream
        start = dict(state=case.state, t=case.t, X=case.X, history=case.history,
                     elapsed=case.sojourn, initialized=True)
        stream = ObjectStream([dict(start, case_id=i) for i in range(self.n_paths)], self.future_features)
        return ModelGenerator(self.model, stream, self.seed, horizon=horizon, max_steps=None)

    def paths(self, case, horizon):
        """``(duration, outcome)`` путей: дни от as_of до конца пути и терминал (или None)."""
        paths = list(self.generator(case, horizon)._paths(None))
        terminals = set(self.terminals)
        duration = np.array([p.end for p in paths], float)
        outcome = np.array([p.states[-1] if p.states[-1] in terminals else None for p in paths], object)
        return duration, outcome

    def cumulative_incidence(self, case, times):
        times = np.asarray(times, float)
        duration, outcome = self.paths(case, float(times.max()))
        rows = []
        for t in times:
            for k in self.terminals:
                F = float(np.mean((outcome == k) & (duration <= t)))
                rows.append(dict(horizon=t, outcome=k, cdf=F, se=float(np.sqrt(F*(1-F)/len(duration)))))
        return pd.DataFrame(rows)

    def restricted_mean(self, case, horizon, n_grid=None):
        # По путям точно: среднее min(T, h), у незавершённых — h.
        duration, outcome = self.paths(case, horizon)
        completed = (outcome != None) & (duration <= horizon)  # noqa: E711
        return (float(np.mean(np.where(completed, duration, horizon))), float(np.mean(~completed)))


# --- точное решение для экспоненциальных времён ------------------------------------------------

class MarkovForecaster(Forecaster):
    """Точный прогноз для экспоненциальных конкурирующих времён (марковская цепь).

    Q — матрица интенсивностей переходов по ``model.transition_parameters(state, X)`` при
    X, зафиксированных на as_of; терминалы и тупики поглощающие. Тогда
    F_k(t) = [exp(Q t)]_{s,k}, а RMST = Σ_{j ∉ терминалов} [∫₀ʰ exp(Q t) dt]_{s,j}.
    Время в текущем состоянии не влияет (отсутствие памяти). Формула точна, если
    интенсивности не зависят от признаков, меняющихся после as_of.
    """
    def __init__(self, model, history_features=None, method='markov', **_):
        super().__init__(model.network.terminals, history_features or model.history_features)
        self.model = model

    def generator_matrix(self, case):
        """``(состояния, Q)``: Q[s, d] = интенсивность s → d, диагональ — минус сумма строки."""
        states = list(self.model.network.states)
        if case.state not in states:
            raise ValueError(f'Состояния {case.state!r} нет в сети модели')
        index = {s: i for i, s in enumerate(states)}
        Q = np.zeros((len(states), len(states)))
        for s in states:
            if s in self.terminals:
                continue
            rates = self.model.transition_parameters(s, case.X, case.history) or {}
            for dst, rate in rates.items():
                if not np.isfinite(rate) or rate < 0:
                    raise ValueError('MarkovForecaster нужны конечные неотрицательные интенсивности')
                Q[index[s], index[dst]] += rate
            Q[index[s], index[s]] = -Q[index[s]].sum()
        return states, Q

    def cumulative_incidence(self, case, times):
        states, Q = self.generator_matrix(case)
        s = states.index(case.state)
        rows = []
        for t in np.asarray(times, float):
            P = expm(Q * t)[s]
            rows.extend(dict(horizon=t, outcome=k, cdf=float(np.clip(P[states.index(k)], 0, 1)))
                        for k in self.terminals)
        return pd.DataFrame(rows)

    def restricted_mean(self, case, horizon, n_grid=None):
        states, Q = self.generator_matrix(case)
        n, s = len(states), states.index(case.state)
        block = np.zeros((2*n, 2*n))
        block[:n, :n], block[:n, n:] = Q, np.eye(n)
        integral = expm(block * horizon)[:n, n:][s]           # ∫₀ʰ exp(Q t) dt, строка s
        alive = [i for i, st in enumerate(states) if st not in self.terminals]
        P = expm(Q * horizon)[s]
        return float(integral[alive].sum()), float(np.clip(P[alive].sum(), 0, 1))
