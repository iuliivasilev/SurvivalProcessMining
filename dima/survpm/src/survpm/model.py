"""Модель: сеть состояний с вероятностями выбора, временами переходов и цензурой."""
import numpy as np
import pandas as pd

from . import adapters
from .curves import completion_cdf, curves_table, residual_cdf
from .distributions import CANDIDATE_FAMILIES, dist_str, mean_time, scipy_args, select_families
from .estimation import estimate_network, source_nll
from .log import censoring_summary, intervals as make_intervals, prepare
from .simulation import completion_times, simulate


class CensoredStateNet:
    """Стохастическая сеть состояний, оцениваемая по журналу с правой цензурой.

    ``use_censoring=False`` — наивная оценка только по завершённым интервалам.
    ``families`` — словарь ``(src, dst) → семейство``; если не задан, выбирается по AIC
    из ``candidates``.

    Пример::

        model = CensoredStateNet("Registration", "Discharge").fit(raw)
        model.edges_table()
        model.survival([1, 7, 30])
    """

    def __init__(self, start, terminal, use_censoring=True, families=None, candidates=CANDIDATE_FAMILIES,
                 min_duration=1 / 1440):
        self.start, self.terminal = start, terminal
        self.use_censoring, self.families = use_censoring, families
        self.candidates, self.min_duration = candidates, min_duration

    def fit(self, raw=None, observation_end=None, reason="lost", *, intervals=None):
        """Оценка по журналу ``raw`` (``case_id, activity, timestamp``) или по готовым интервалам.

        ``observation_end`` — момент окончания наблюдения незавершённых трасс (Series по ``case_id``);
        без него цензура наступает в момент последней записи трассы. ``reason`` — причина цензуры
        (строка или Series по ``case_id``).
        """
        if intervals is None:
            log, cases = prepare(raw, self.start, self.terminal, observation_end, reason)
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
