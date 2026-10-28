"""Модель: сеть состояний с вероятностями выбора, временами переходов и цензурой."""
import numpy as np
import pandas as pd

from . import adapters
from .curves import completion_cdf, curves_table, residual_cdf
from .distributions import CANDIDATE_FAMILIES, dist_str, mean_time, scipy_args, select_families
from .estimation import estimate_network, source_nll
from .features import EventLog
from .log import censoring_summary, intervals as make_intervals, prepare
from .simulation import completion_times, simulate


class CensoredStateNet:
    """Стохастическая сеть состояний, оцениваемая по журналу с правой цензурой.

    ``use_censoring=False`` — наивная оценка только по завершённым интервалам.
    ``families`` — словарь ``(src, dst) → семейство``; если не задан, выбирается по AIC
    из ``candidates``. ``repeats`` — что делать с повторами состояния подряд, см. :func:`prepare`
    (``"collapse"`` воспроизводит результаты до 28.10.2026).

    Пример::

        model = CensoredStateNet("Registration", "Discharge").fit(raw)
        model.edges_table()
        model.survival([1, 7, 30])
    """

    def __init__(self, start, terminal, use_censoring=True, families=None, candidates=CANDIDATE_FAMILIES,
                 min_duration=1 / 1440, repeats="simultaneous"):
        self.start, self.terminal, self.repeats = start, terminal, repeats
        self.use_censoring, self.families = use_censoring, families
        self.candidates, self.min_duration = candidates, min_duration

    def fit(self, raw=None, observation_end=None, reason="lost", *, intervals=None):
        """Оценка по журналу ``raw`` (``case_id, activity, timestamp`` или :class:`EventLog`)
        или по готовым интервалам. У ``EventLog`` по умолчанию ``observation_end = raw.last_seen``.

        ``observation_end`` — момент окончания наблюдения незавершённых трасс (Series по ``case_id``);
        без него цензура наступает в момент последней записи трассы. ``reason`` — причина цензуры
        (строка или Series по ``case_id``).
        """
        if isinstance(raw, EventLog):
            seen = raw.last_seen if observation_end is None else \
                pd.concat([raw.last_seen, observation_end], axis=1).max(axis=1)
            raw, observation_end = raw.events, seen
        if intervals is None:
            log, cases = prepare(raw, self.start, self.terminal, observation_end, reason, self.repeats)
            intervals = make_intervals(log, cases, self.min_duration)
        self.intervals_ = intervals
        self.families_, self.aic_table_ = select_families(intervals, self.candidates)
        if self.families:
            self.families_.update(self.families)
        self.edges_, self.diagnostics_ = estimate_network(intervals, self.families_, self.use_censoring)
        self.stop_ = censoring_summary(intervals).set_index("src")["q_censored"].to_dict()
        return self

    def edges_table(self):
        """Оценённые переходы: вероятность выбора, распределение, среднее, медиана, 90% квантиль."""
        rows = []
        for edge in self.edges_:
            rv, args, kwargs = scipy_args(edge["dist"])
            rows.append(dict(transition=edge["name"], src=edge["src"], dst=edge["dst"], p=edge["p"],
                             distribution=dist_str(edge["dist"]), n_completed=edge["n_completed"],
                             mean_days=mean_time(edge["dist"]), median_days=rv.median(*args, **kwargs),
                             q90_days=rv.ppf(.9, *args, **kwargs)))
        return pd.DataFrame(rows)

    def repeat_table(self):
        """Вероятность повтора для каждого состояния (по вероятностям выбора, без времени и цензуры).

        ``p_immediate`` — следующее состояние то же самое (петля ``s → s``);
        ``p_return`` — состояние ещё встретится до ``terminal`` (сразу или через другие);
        ``expected_visits`` — среднее число заходов при одном заходе на старте, ``1 / (1 − p_return)``.
        """
        states = sorted({e["src"] for e in self.edges_} | {e["dst"] for e in self.edges_})
        index = {s: i for i, s in enumerate(states)}
        P = np.zeros((len(states), len(states)))
        for e in self.edges_:
            if e["src"] != self.terminal:
                P[index[e["src"]], index[e["dst"]]] += e["p"]
        P /= np.where(P.sum(1, keepdims=True) > 0, P.sum(1, keepdims=True), 1)
        rows = []
        for s in states:
            if s == self.terminal:
                continue
            i = index[s]
            # h[d] — вероятность из d попасть в s раньше, чем в terminal или тупик
            free = [j for j in range(len(states)) if j != i and states[j] != self.terminal]
            A = np.eye(len(free)) - P[np.ix_(free, free)]
            h = np.zeros(len(states))
            h[i] = 1.
            h[free] = np.linalg.solve(A, P[free, i])
            p_return = float(P[i] @ h)
            rows.append(dict(state=s, p_immediate=P[i, i], p_return=p_return,
                             expected_visits=1 / (1 - p_return) if p_return < 1 else np.inf))
        return pd.DataFrame(rows)

    def simulate(self, n, seed=None, censoring=False, horizon=None):
        """Трассы ``[(states, times), ...]``; ``censoring=True`` — обрыв с наблюдаемой вероятностью ``q``."""
        return simulate(self.edges_, self.start, self.terminal, n, seed,
                        stop=self.stop_ if censoring else None, horizon=horizon)

    def completion_times(self, n=10_000, seed=0):
        """Время от ``start`` до ``terminal`` по симуляции без цензуры; ``inf`` — не дошла."""
        return completion_times(self.simulate(n, seed), self.terminal)

    def survival(self, t, n=10_000, seed=0, method="sim", step=None):
        """``S(t) = P(T > t)`` для времени до ``terminal``.

        ``method="sim"`` — доля из ``n`` симулированных трасс; ``method="grid"`` — свёртка со
        сдвигом на сетке с шагом ``step`` дней (по умолчанию ``max(t) / 2000``), без случайности.
        """
        t = np.asarray(t, dtype=float)
        if method == "grid":
            times, states, F = self._grid(t.max(), step)
            return 1 - np.interp(t, times, F[states.index(self.start)])
        if method != "sim":
            raise ValueError(method)
        times = np.sort(self.completion_times(n, seed))
        return 1 - np.searchsorted(times, t, side="right") / len(times)

    def _grid(self, horizon, step=None):
        return completion_cdf(self.edges_, self.terminal, horizon, step or horizon / 2000)

    def state_curves(self, horizon, step=None):
        """Кривые «из текущего места»: для каждого состояния ``s`` — ``F_s(t)`` и ``S_s(t)``,
        время от входа в ``s`` до ``terminal`` (таблица ``state, time_days, cdf, survival``)."""
        return curves_table(*self._grid(horizon, step))

    def residual_survival(self, state, elapsed, t, step=None):
        """``S(t | s, a)`` — остаток времени до ``terminal`` для идущего случая, который уже
        ``elapsed`` дней находится в ``state``; ``t`` отсчитывается от текущего момента."""
        t = np.asarray(t, dtype=float)
        times, states, F = self._grid(t.max(), step)
        return 1 - np.interp(t, times, residual_cdf(self.edges_, state, elapsed, times, states, F))

    def nll(self, intervals):
        """−log L интервалов с цензурой. Трассы с переходом или состоянием, которых нет в модели,
        исключаются; их доля — ``unseen_share``."""
        edges = {(e["src"], e["dst"]): e for e in self.edges_}
        sources = {e["src"] for e in self.edges_}
        observed = intervals["observed"]
        known = pd.Series([(s, d) in edges for s, d in zip(intervals["src"], intervals["dst"])], index=intervals.index)
        unseen = (observed & ~known) | (~observed & ~intervals["src"].isin(sources))
        unseen_cases = set(intervals.loc[unseen, "case_id"])
        kept = intervals.loc[~intervals["case_id"].isin(unseen_cases)]
        nll = 0.
        for src, group in kept.groupby("src"):
            out = [edge for edge in edges if edge[0] == src]
            nll += source_nll(group, out, [edges[e]["p"] for e in out], [edges[e]["dist"] for e in out])
        return dict(unseen_share=len(unseen_cases) / intervals["case_id"].nunique(),
                    nll_per_interval=nll / len(kept), nll_per_trace=nll / kept["case_id"].nunique())

    def score(self, intervals):
        """Средний log L на интервал (больше — лучше), как ``score`` в sklearn."""
        return -self.nll(intervals)["nll_per_interval"]

    def to_pm4py(self, censoring=True):
        """``(net, initial_marking, final_marking)`` PM4Py; ``censoring=False`` — без места ``CENS``."""
        intervals = self.intervals_ if censoring else self.intervals_.loc[self.intervals_["observed"]]
        return adapters.state_petri_net(intervals, self.start, self.terminal, self.edges_)

    def to_sdfa(self, censoring=False, digits=3):
        """Автомат для Ebi; ``censoring=True`` — остановка в состоянии с вероятностью ``q``."""
        return adapters.to_sdfa(self.edges_, self.start, self.stop_ if censoring else None, digits)
