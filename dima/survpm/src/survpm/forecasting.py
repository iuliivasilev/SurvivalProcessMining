"""Прогноз: базовый класс :class:`Forecaster` и реализации.

Прогноз — функции кумулятивной инцидентности конкурирующих исходов

    F_k(t) = P(T ≤ t, исход = k | журнал объекта до его end),

где T — время от «сейчас» (``end`` объекта; его ставит :meth:`EventLog.cut`) до терминала.
Базовый класс задаёт только этот интерфейс, превращает журнал в таблицу X
(``StreamData.at``) и проверяет ответ; как считать F_k, решает подкласс. Остальное —
тождества конкурирующих рисков, верные для любого закона::

    S(t) = 1 − Σ_k F_k(t)
    h_k(a, b] = (F_k(b) − F_k(a)) / S(a) / (b − a)
    RMST(h) = ∫₀ʰ S(t) dt

Свой прогноз — подкласс с одним методом::

    class AlwaysExp(survpm.Forecaster):
        def cumulative_incidence(self, X, times, **kw):     # X — одна или несколько строк
            return pd.DataFrame(dict(horizon=times, outcome="D", cdf=1 - np.exp(-.5 * np.asarray(times))))

    AlwaysExp(terminals=["D"]).survival(log.cut(t), [1, 7, 30])

Реализации: :class:`MarkovForecaster` (точно для экспоненциального закона), :class:`GridForecaster`
(уравнение марковского восстановления на сетке, любой закон с ``time_cdf``),
:class:`SimulationForecaster` (Монте-Карло по :class:`~survpm.ModelGenerator`). Все принимают
``selector``: прогноз при заданном правиле выбора переходов (по умолчанию — по закону модели).
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .eventlog import EventLog
from .selection import SampleSelector
from .streamdata import DefaultStreamData, features_of, frame

DAY = pd.Timedelta(days=1)
INCIDENCE_COLUMNS = ['horizon', 'outcome', 'cdf']


def _times(times):
    times = np.asarray(times, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('Задайте неотрицательные конечные горизонты')
    return times


def _many(case_id):
    return isinstance(case_id, (list, tuple, np.ndarray, pd.Index, pd.Series))


class Forecaster(ABC):
    """Базовый прогноз. Обязательный метод — :meth:`cumulative_incidence`.

    ``model`` — модель шага (нужна реализациям, которые её используют), ``selector`` — правило
    выбора (по умолчанию :class:`~survpm.SampleSelector`), ``data`` — признаки
    (:class:`~survpm.StreamData`), ``terminals`` — исходы (по умолчанию терминалы сети модели).
    Методы :meth:`terminal_probs`, :meth:`survival`, :meth:`hazard`, :meth:`expected_time`
    реализованы в базе через F_k. Вход — журнал (каждый объект от его ``end``) или готовая
    таблица X. Ответ — таблица с колонкой ``case_id``.
    """

    def __init__(self, model=None, selector=None, data=None, terminals=None, **kw):
        if terminals is None and model is not None:
            terminals = model.network.terminals
        if terminals is None:
            raise ValueError('Задайте terminals или model')
        self.terminals = list([terminals] if isinstance(terminals, str) else terminals)
        if not self.terminals:
            raise ValueError('Нужен хотя бы один терминал')
        self.model = model
        self.selector = selector if selector is not None else SampleSelector()
        self.data = data if data is not None else DefaultStreamData(getattr(model, 'history_features', None))

    # --- обязательно: математика --------------------------------------------------------------
    @abstractmethod
    def cumulative_incidence(self, X, times, **kw):
        """F_k(t) для строк X (``case_id, state, elapsed, time`` и признаки). ``times`` — дни
        от «сейчас», неотрицательные, по возрастанию. Таблица ``horizon, outcome, cdf``
        (+ ``case_id``, если строк несколько; + свои колонки). ``kw["log"]`` — журнал, если
        прогноз строился по журналу."""

    # --- база: вход и проверки ----------------------------------------------------------------
    def features(self, log, case_id=None):
        """Таблица X по журналу (каждый объект на его ``end``) или сама X."""
        if isinstance(log, EventLog):
            if case_id is not None:
                log = log.select(list(case_id) if _many(case_id) else [case_id])
            return self.data.at(log), log
        X = frame(log)
        if case_id is not None:
            X = X.loc[X.case_id.isin(list(case_id) if _many(case_id) else [case_id])].reset_index(drop=True)
        return X, None

    def incidence(self, X, times, log=None):
        """:meth:`cumulative_incidence` с проверкой: 0 ≤ F_k ≤ 1, F_k не убывает, Σ_k F_k ≤ 1."""
        times = _times(times)
        grid = np.unique(times)
        table = pd.DataFrame(self.cumulative_incidence(X, grid, log=log))
        if 'case_id' not in table:
            if len(X) == 1:
                table.insert(0, 'case_id', X.case_id.iloc[0])
            else:
                parts = [pd.DataFrame(self.cumulative_incidence(X.iloc[[i]].reset_index(drop=True), grid, log=log))
                         .assign(case_id=X.case_id.iloc[i]) for i in range(len(X))]
                table = pd.concat(parts, ignore_index=True)
        if set(INCIDENCE_COLUMNS) - set(table.columns):
            raise ValueError(f'cumulative_incidence должен вернуть колонки {INCIDENCE_COLUMNS}')
        out = []
        for cid in X.case_id:
            part = table.loc[table.case_id == cid]
            cdf = (part.pivot_table(index='horizon', columns='outcome', values='cdf', aggfunc='first')
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
            index = pd.MultiIndex.from_product([times, self.terminals], names=['horizon', 'outcome'])
            part = part.drop(columns='case_id').drop_duplicates(['horizon', 'outcome']) \
                .set_index(['horizon', 'outcome']).reindex(index).reset_index()
            out.append(part.assign(case_id=cid))
        result = pd.concat(out, ignore_index=True)
        return result[['case_id', *[c for c in result.columns if c != 'case_id']]]

    # --- база: производные величины (можно переопределить) -------------------------------------
    def terminal_probs(self, log, times, case_id=None):
        """F_k(t): таблица ``case_id, horizon, outcome, probability``."""
        X, source = self.features(log, case_id)
        return self.incidence(X, times, source).rename(columns={'cdf': 'probability'})

    def _survival(self, X, times, log=None):
        table = self.incidence(X, times, log)
        times = _times(times)
        rows = []
        for cid in X.case_id:
            cdf = table.loc[table.case_id == cid].groupby('horizon', sort=False).cdf.sum().reindex(times).to_numpy()
            cdf = np.minimum(cdf, 1.)
            rows.append(pd.DataFrame(dict(case_id=cid, horizon=times, survival=1 - cdf, cdf=cdf)))
        return pd.concat(rows, ignore_index=True)

    def survival(self, log, times, case_id=None):
        """S(t) = 1 − Σ_k F_k(t): таблица ``case_id, horizon, survival, cdf``."""
        X, source = self.features(log, case_id)
        return self._survival(X, times, source)

    def hazard(self, log, times, case_id=None):
        """Интенсивность исхода k на интервалах (a, b]: (F_k(b) − F_k(a)) / S(a) / (b − a).

        ``times`` — правые края, строго по возрастанию; первый интервал начинается в 0.
        Если S(a) = 0, интенсивность не определена (NaN). Таблица ``case_id, horizon, outcome, hazard``."""
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(np.r_[0., times]) <= 0).any():
            raise ValueError('hazard times must be strictly increasing and positive')
        edges = np.r_[0., times]
        X, source = self.features(log, case_id)
        table = self.incidence(X, edges, source)
        rows = []
        for cid in X.case_id:
            F = table.loc[table.case_id == cid].pivot(index='horizon', columns='outcome', values='cdf') \
                .reindex(index=edges, columns=self.terminals).to_numpy(float)
            S = 1 - F.sum(axis=1)
            for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                for j, outcome in enumerate(self.terminals):
                    value = (F[i + 1, j] - F[i, j]) / S[i] / (right - left) if S[i] > 0 else np.nan
                    rows.append(dict(case_id=cid, horizon=right, outcome=outcome, hazard=value))
        return pd.DataFrame(rows, columns=['case_id', 'horizon', 'outcome', 'hazard'])

    def restricted_mean(self, X, horizon, log=None, n_grid=2000):
        """``(RMST, P(не завершён к horizon))`` для одной строки X; RMST = ∫₀ʰ S(t) dt (трапеции)."""
        grid = np.linspace(0., horizon, n_grid + 1)
        S = self._survival(X, grid, log).survival.to_numpy()
        return float(np.trapezoid(S, grid) if hasattr(np, 'trapezoid') else np.trapz(S, grid)), float(S[-1])

    def expected_time(self, log, horizon, case_id=None):
        """Ограниченное среднее время до любого терминала и доля незавершённых к ``horizon``:
        таблица ``case_id, restricted_mean_time, probability_unfinished``."""
        horizon = float(_times([horizon])[0])
        X, source = self.features(log, case_id)
        rows = []
        for i in range(len(X)):
            rmst, unfinished = self.restricted_mean(X.iloc[[i]].reset_index(drop=True), horizon, source)
            rows.append(dict(case_id=X.case_id.iloc[i], restricted_mean_time=rmst, probability_unfinished=unfinished))
        return pd.DataFrame(rows, columns=['case_id', 'restricted_mean_time', 'probability_unfinished'])


# --- общий помощник: шаги всех состояний при признаках строки ---------------------------------

def state_steps(model, selector, row, states):
    """Шаги и нормированные веса Selector для ``states`` при признаках строки ``row`` (elapsed = 0)."""
    X = pd.DataFrame([dict(features_of(row), case_id=s, state=s, elapsed=0.) for s in states])
    if not len(X):
        return [], []
    steps = model.predict(X)
    return steps, selector.weight_matrix(X, steps, model=model)


# --- точное решение для экспоненциальных времён ------------------------------------------------

class MarkovForecaster(Forecaster):
    """Точный прогноз для экспоненциальных конкурирующих времён (марковская цепь).

    Q[s, d] = Λ_s · w_sd, где Λ_s — общая интенсивность выхода из s (из ``model.predict``),
    w_sd — веса Selector (по умолчанию p_sd, тогда Q — интенсивности модели); петли s → s
    в Q не входят (при отсутствии памяти не меняют состояние). Признаки
    зафиксированы на «сейчас», терминалы и тупики поглощающие. F_k(t) = [exp(Q t)]_{s,k},
    RMST = Σ_{j ∉ терминалов} [∫₀ʰ exp(Q t) dt]_{s,j}. Время в текущем состоянии не влияет.
    """
    def __init__(self, model, selector=None, data=None, **kw):
        super().__init__(model, selector, data, **kw)


    def generator_matrix(self, row):
        """``(состояния, Q)`` при признаках строки ``row``."""
        states = list(self.model.network.states)
        if row['state'] not in states:
            raise ValueError(f"Состояния {row['state']!r} нет в сети модели")
        index = {s: i for i, s in enumerate(states)}
        inner = [s for s in states if s not in self.terminals]
        Q = np.zeros((len(states), len(states)))
        for s, step, w in zip(inner, *state_steps(self.model, self.selector, row, inner)):
            if step.empty:
                continue
            if step.rates is None:
                raise ValueError('MarkovForecaster нужен экспоненциальный закон шага; '
                                 'для других — GridForecaster или SimulationForecaster')
            if not np.isfinite(step.rates).all() or (step.rates < 0).any():
                raise ValueError('MarkovForecaster нужны конечные неотрицательные интенсивности')
            for d, weight in zip(step.dst, w):
                if d != s:          # петля s → s не меняет состояние (нет памяти): в Q её нет
                    Q[index[s], index[d]] += step.total * weight
            Q[index[s], index[s]] = -Q[index[s]].sum()
        return states, Q

    def cumulative_incidence(self, X, times, **kw):
        rows = []
        for _, row in X.iterrows():
            states, Q = self.generator_matrix(row)
            s = states.index(row['state'])
            for t in np.asarray(times, float):
                P = expm(Q * t)[s]
                rows.extend(dict(case_id=row['case_id'], horizon=t, outcome=k,
                                 cdf=float(np.clip(P[states.index(k)], 0, 1))) for k in self.terminals)
        return pd.DataFrame(rows)

    def restricted_mean(self, X, horizon, log=None, n_grid=None):
        row = X.iloc[0]
        states, Q = self.generator_matrix(row)
        n, s = len(states), states.index(row['state'])
        block = np.zeros((2*n, 2*n))
        block[:n, :n], block[:n, n:] = Q, np.eye(n)
        integral = expm(block * horizon)[:n, n:][s]           # ∫₀ʰ exp(Q t) dt, строка s
        alive = [i for i, st in enumerate(states) if st not in self.terminals]
        P = expm(Q * horizon)[s]
        return float(integral[alive].sum()), float(np.clip(P[alive].sum(), 0, 1))


# --- уравнение марковского восстановления на сетке ---------------------------------------------

class GridForecaster(Forecaster):
    """Полумарковский прогноз на сетке для любого закона шага с ``time_cdf``.

    F_s(t) = Σ_d ∫₀ᵗ F_d(t − u) dG_sd(u), G_sd(u) = w_sd · P(T ≤ u | выход d), F_k = 1 в терминале k.
    На равномерной сетке из ``n_grid`` шагов до max(times): массы G по ячейкам точные, F_d
    внутри ячейки — линейно между её концами в средней точке ухода (её позиция считается по
    G внутри ячейки, поэтому быстрые переходы не портят ответ). Ошибка O(h²). Первое
    ожидание — при условии ``elapsed``, следующие — с нуля. Признаки зафиксированы на «сейчас».
    """
    def __init__(self, model, selector=None, data=None, n_grid=1000, **kw):
        super().__init__(model, selector, data, **kw)
        self.n_grid = int(n_grid)

    # Точки внутри ячейки (доли h): сгущаются у начала, чтобы быстрые переходы считались верно.
    _SUB = np.r_[0., np.geomspace(1e-7, .1, 13), np.linspace(.2, 1., 9)]

    def _masses(self, steps, weights, grid, columns):
        """``(P, B)``: P[m, i, c] — вероятность уйти из i в c в ячейке m (m = 1..N, P[0] = 0),
        B[m, i, c] — средняя позиция ухода внутри ячейки в долях h (0 — начало, 1 — конец)."""
        h = grid[1] - grid[0]
        N = len(grid) - 1
        points = (grid[:-1, None] + h * self._SUB[None, :]).ravel()
        P = np.zeros((N + 1, len(steps), len(columns)))
        M = np.zeros_like(P)
        for i, (step, w) in enumerate(zip(steps, weights)):
            if step.empty:
                continue
            G = (w[:, None] * step.time_cdf(points)).reshape(len(step.dst), N, len(self._SUB))
            mass = G[:, :, -1] - G[:, :, 0]
            # ∫ (u − a)/h dG по ячейке [a, a + h] = G(b) − (1/h)∫ G du  (по частям)
            moment = G[:, :, -1] - np.trapezoid(G, self._SUB, axis=2)
            for d, m_, mo in zip(step.dst, mass, moment):
                P[1:, i, columns[d]] += m_
                M[1:, i, columns[d]] += mo
        B = np.divide(M, P, out=np.full_like(P, .5), where=P > 1e-300)
        return P, np.clip(B, 0., 1.)

    def cumulative_incidence(self, X, times, **kw):
        times = np.asarray(times, float)
        T = float(times.max())
        rows = []
        for _, row in X.iterrows():
            if row['state'] in self.terminals or T == 0:
                F = np.array([[float(row['state'] == k) for k in self.terminals]]*len(times))
            else:
                F = self._solve(row, T)
                grid = np.linspace(0., T, self.n_grid + 1)
                F = np.column_stack([np.interp(times, grid, F[:, j]) for j in range(len(self.terminals))])
            for t, values in zip(times, np.maximum.accumulate(np.clip(F, 0, 1), axis=0)):
                rows.extend(dict(case_id=row['case_id'], horizon=t, outcome=k, cdf=float(v))
                            for k, v in zip(self.terminals, values))
        return pd.DataFrame(rows)

    def _solve(self, row, T):
        # F(t_n) = CT(t_n) + Σ_m Pa[m] F(t_{n−m+1}) + Pb[m] F(t_{n−m}):
        # уход в ячейке m, F_d между концами ячейки — линейно по средней позиции ухода B.
        grid = np.linspace(0., T, self.n_grid + 1)
        inner = [s for s in self.model.network.states if s not in self.terminals]
        I, K, N = len(inner), len(self.terminals), self.n_grid
        columns = {s: i for i, s in enumerate(inner)}
        columns.update({k: I + j for j, k in enumerate(self.terminals)})
        steps, weights = state_steps(self.model, self.selector, row, inner)
        P, B = self._masses(steps, weights, grid, columns)
        Pa, Pb = (P * (1 - B))[:, :, :I], (P * B)[:, :, :I]
        CT = np.cumsum(P[:, :, I:], axis=0)
        F = np.zeros((N + 1, I, K))
        A = np.eye(I) - Pa[1]
        for n in range(1, N + 1):
            rhs = CT[n] + Pb[1] @ F[n - 1]
            if n >= 2:
                rhs = rhs + np.einsum('mij,mjk->ik', Pa[2:n + 1], F[n - 1:0:-1][:n - 1]) \
                          + np.einsum('mij,mjk->ik', Pb[2:n + 1], F[n - 2::-1][:n - 1])
            F[n] = np.linalg.solve(A, rhs)
        # первое ожидание — из самой строки (с её elapsed)
        start_x = pd.DataFrame([row])
        start_step = self.model.predict(start_x)
        start_w = self.selector.weight_matrix(start_x, start_step, model=self.model)
        P0, B0 = self._masses(start_step, start_w, grid, columns)
        Pa0, Pb0 = (P0 * (1 - B0))[:, 0, :I], (P0 * B0)[:, 0, :I]
        CT0 = np.cumsum(P0[:, 0, I:], axis=0)
        out = np.zeros((N + 1, K))
        for n in range(1, N + 1):
            out[n] = CT0[n] + np.einsum('mi,mik->k', Pa0[1:n + 1], F[n::-1][:n]) \
                            + np.einsum('mi,mik->k', Pb0[1:n + 1], F[n - 1::-1][:n])
        return out


# --- Монте-Карло ------------------------------------------------------------------------------

class SimulationForecaster(Forecaster):
    """Монте-Карло: F_k(t) — доля из ``n_paths`` путей, завершившихся в k не позже t.

    Пути строит :class:`~survpm.ModelGenerator` с тем же ``selector`` и ``data`` (признаки
    пересчитываются на каждом шаге пути). Оценка несмещённая, стандартная ошибка
    √(F(1−F)/n) — колонка ``se``. Нужен журнал объекта, а не только X.
    """
    def __init__(self, model, n_paths=1000, seed=None, selector=None, data=None, **kw):
        if n_paths < 1:
            raise ValueError('n_paths должен быть положительным')
        super().__init__(model, selector, data, **kw)
        self.n_paths, self.seed = n_paths, seed
        self._cache = {}


    def generator(self):
        from .generation import ModelGenerator
        return ModelGenerator(self.model, self.selector, self.data, self.seed)

    def paths(self, case_id, log, horizon):
        """``(duration, outcome)`` путей объекта: дни от «сейчас» до конца пути и терминал (или None)."""
        if log is None:
            raise ValueError('SimulationForecaster: передайте журнал (log.cut(t)), а не только X')
        key = (case_id, float(horizon), id(log))
        if key in self._cache:
            return self._cache[key]
        one = log.select([case_id])
        now = one.now.iloc[0]
        ids = list(range(self.n_paths))
        rep = lambda table: pd.concat([table.assign(case_id=i) for i in ids], ignore_index=True)
        cases = pd.concat([one.cases]*self.n_paths)
        cases.index = pd.Index(ids, name='case_id')
        events = one.events.loc[one.events.timestamp <= now]
        paths = EventLog(rep(events), cases, rep(one.measurements), pd.Series(now, index=ids),
                         pd.DataFrame(dict(case_id=ids, end=now, event=0, reason='')), one.rules)
        out = self.generator().generate(paths, max_steps=None, horizon=horizon)
        obs = out.observation.set_index('case_id').loc[ids]
        last = out.events.groupby('case_id').activity.last().reindex(ids)
        duration = ((obs.end - now) / DAY).to_numpy(float)
        outcome = np.where(obs.event.to_numpy() == 1, last.to_numpy(), None)
        self._cache = {key: (duration, outcome)}
        return duration, outcome

    def cumulative_incidence(self, X, times, log=None, **kw):
        times = np.asarray(times, float)
        rows = []
        for cid in X.case_id:
            duration, outcome = self.paths(cid, log, float(times.max()))
            for t in times:
                for k in self.terminals:
                    F = float(np.mean((outcome == k) & (duration <= t)))
                    rows.append(dict(case_id=cid, horizon=t, outcome=k, cdf=F,
                                     se=float(np.sqrt(F*(1-F)/len(duration)))))
        return pd.DataFrame(rows)

    def restricted_mean(self, X, horizon, log=None, n_grid=None):
        # По путям точно: среднее min(T, h), у незавершённых — h.
        duration, outcome = self.paths(X.case_id.iloc[0], log, horizon)
        completed = (outcome != None) & (duration <= horizon)  # noqa: E711
        return (float(np.mean(np.where(completed, duration, horizon))), float(np.mean(~completed)))
