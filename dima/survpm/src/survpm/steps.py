"""Закон одного шага: :class:`StepDistribution` и реализации.

``model.predict(X)`` возвращает список шагов, по одному на строку X. Шаг из состояния при
признаках X и уже прошедшем времени ``elapsed`` описывает:

* ``dst`` — возможные выходы, ``p`` — их вероятности (при условии, что за ``elapsed`` выхода не было);
* :meth:`StepDistribution.sample` — розыгрыш ``(dst, dt)`` по закону;
* :meth:`StepDistribution.sample_time`, :meth:`StepDistribution.mean_time` — время при выходе ``j``;
* :meth:`StepDistribution.time_cdf` — P(T ≤ t | выход j) (нужен формульным прогнозам).

Реализации: :class:`ExponentialStep` (конкурирующие экспоненты), :class:`ChoiceThenTimeStep`
(выход по ``p``, затем время из распределения выхода), :class:`SampledStep` (модель, которая
умеет только разыгрывать ``next_transition``). Своя модель — свой подкласс.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats

from .network import Network
from .models.features import DefaultHistoryFeatures
from .streamdata import frame, features_of


def sample_time(dist, rng):
    """Время перехода из ``(семейство, параметры)``: exponential(mean), lognormal(median, sigma),
    gamma(shape, scale), weibull(shape, scale), uniform(low, high); или ``callable(rng)``."""
    if callable(dist):
        return float(dist(rng))
    kind, par = dist
    if kind == 'exponential':
        return rng.exponential(par['mean'])
    if kind == 'lognormal':
        return rng.lognormal(np.log(par['median']), par['sigma'])
    if kind == 'gamma':
        return rng.gamma(par['shape'], par['scale'])
    if kind == 'weibull':
        return par['scale'] * rng.weibull(par['shape'])
    if kind == 'uniform':
        return rng.uniform(par['low'], par['high'])
    raise ValueError(kind)


class StepDistribution(ABC):
    """Шаг из одного состояния. Обязательные методы — :meth:`sample_time` и :meth:`time_cdf`."""
    dst = ()
    p = np.zeros(0)
    rates = None        # интенсивности выходов, если закон экспоненциальный (MarkovForecaster)

    @property
    def empty(self):
        """Выходов нет (терминал или тупик)."""
        return len(self.dst) == 0 or float(np.sum(self.p)) <= 0

    def sample(self, rng):
        """``(dst, dt)`` по закону; ``(None, inf)``, если выходов нет."""
        if self.empty:
            return None, np.inf
        j = rng.choice(len(self.p), p=self.p)
        return self.dst[j], self.sample_time(j, rng)

    @abstractmethod
    def sample_time(self, j, rng):
        """Время до выхода ``j`` (дни от «сейчас»), разыгранное при условии выхода ``j``."""

    @abstractmethod
    def time_cdf(self, t):
        """P(T ≤ t | выход j): массив ``(len(dst), len(t))``, t — дни от «сейчас»."""

    def mean_time(self, j, upper=None):
        """Среднее время при выходе ``j`` (по умолчанию численно по :meth:`time_cdf`)."""
        upper = upper or 1e4
        grid = np.linspace(0, upper, 200_001)
        return float(np.trapezoid(1 - self.time_cdf(grid)[j], grid))

    def exit_cdf(self, t):
        """P(выход j, T ≤ t): ``p_j · time_cdf_j(t)``."""
        return self.p[:, None] * self.time_cdf(t)

    @property
    def table(self):
        return pd.DataFrame(dict(dst=list(self.dst), p=self.p))


class EmptyStep(StepDistribution):
    """Выходов нет."""
    def sample_time(self, j, rng):
        raise ValueError("Выходов нет")

    def time_cdf(self, t):
        return np.zeros((0, len(np.atleast_1d(t))))


class ExponentialStep(StepDistribution):
    """Конкурирующие экспоненты: из каждого ребра «тикают» часы с интенсивностью λ_d.

    p_d = λ_d / Λ, время ~ Exp(Λ) при любом выходе, ``elapsed`` не влияет (нет памяти).
    :meth:`sample` разыгрывает все часы и берёт первый (как ``ExponentialTransitionModel``),
    поэтому при том же seed числа совпадают с прежним движком.
    """
    def __init__(self, rates, picker=None):
        rates = {d: float(r) for d, r in rates.items()}
        self.dst = list(rates)
        self.rates = np.array(list(rates.values()), float)
        self.total = float(self.rates.sum())
        self.p = self.rates / self.total if self.total > 0 else np.zeros(len(self.rates))
        self.picker = picker

    def sample(self, rng):
        positive = [(d, r) for d, r in zip(self.dst, self.rates) if r > 0]
        if not positive:
            return None, np.inf
        times = rng.exponential(1 / np.array([r for _, r in positive]))
        draws = dict(zip([d for d, _ in positive], times))
        if self.picker is not None:
            return self.picker.pick(draws, rng)
        dst = min(draws, key=draws.get)
        return dst, float(draws[dst])

    def sample_time(self, j, rng):
        return float(rng.exponential(1 / self.total))

    def mean_time(self, j, upper=None):
        return 1 / self.total

    def time_cdf(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return np.tile(1 - np.exp(-self.total * t), (len(self.dst), 1))


def _scipy(dist):
    kind, par = dist
    if kind == "exponential":
        return stats.expon(scale=par["mean"])
    if kind == "lognormal":
        return stats.lognorm(s=par["sigma"], scale=par["median"])
    if kind == "gamma":
        return stats.gamma(a=par["shape"], scale=par["scale"])
    if kind == "weibull":
        return stats.weibull_min(c=par["shape"], scale=par["scale"])
    if kind == "uniform":
        return stats.uniform(loc=par["low"], scale=par["high"] - par["low"])
    raise ValueError(kind)


class ChoiceThenTimeStep(StepDistribution):
    """Сначала выход по ``p``, затем время из распределения этого выхода.

    ``dists`` — ``(семейство, параметры)`` как в :func:`sample_time`
    или ``callable(rng)``. При ``elapsed > 0`` вероятности и время условны: за ``elapsed``
    выхода не было (для ``callable`` — отбором; формульные прогнозы его не поддерживают).
    """
    def __init__(self, dst, p, dists, elapsed=0.):
        self._draw = sample_time
        self.dst, self.dists, self.elapsed = list(dst), list(dists), float(elapsed)
        p = np.asarray(p, float)
        self.prior = p / p.sum() if p.sum() > 0 else p
        if self.elapsed > 0 and all(not callable(d) for d in self.dists):
            alive = np.array([_scipy(d).sf(self.elapsed) for d in self.dists])
            w = self.prior * alive
            self.p = w / w.sum() if w.sum() > 0 else w
        else:
            self.p = self.prior

    def sample(self, rng):
        if self.empty:
            return None, np.inf
        if self.elapsed == 0:
            j = rng.choice(len(self.prior), p=self.prior)
            return self.dst[j], self._draw(self.dists[j], rng)
        for _ in range(100_000):    # совместное (выход, время) при условии T > elapsed
            j = rng.choice(len(self.prior), p=self.prior)
            t = self._draw(self.dists[j], rng)
            if t > self.elapsed:
                return self.dst[j], t - self.elapsed
        raise ValueError("Не удалось разыграть время при условии elapsed")

    def sample_time(self, j, rng):
        for _ in range(100_000):
            t = self._draw(self.dists[j], rng)
            if t > self.elapsed:
                return t - self.elapsed
        raise ValueError("Не удалось разыграть время при условии elapsed")

    def mean_time(self, j, upper=None):
        if callable(self.dists[j]):
            raise NotImplementedError("Среднее для callable-распределения не определено")
        d = _scipy(self.dists[j])
        if self.elapsed == 0:
            return float(d.mean())
        return float(d.expect(lambda x: x - self.elapsed, lb=self.elapsed, conditional=True))

    def time_cdf(self, t):
        if any(callable(d) for d in self.dists):
            raise NotImplementedError("time_cdf для callable-распределения не определена")
        t = np.atleast_1d(np.asarray(t, float))
        rows = []
        for d in self.dists:
            d = _scipy(d)
            alive = d.sf(self.elapsed)
            rows.append((d.cdf(self.elapsed + t) - d.cdf(self.elapsed)) / alive if alive > 0 else np.ones_like(t))
        return np.array(rows)


class SampledStep(StepDistribution):
    """Шаг модели, которая умеет только разыгрывать ``(dst, dt)``: вероятности неизвестны.

    Работает с :class:`~survpm.SampleSelector` (генерация и ``SimulationForecaster``).
    """
    def __init__(self, draw, dst=()):
        self._sample, self.dst = draw, list(dst)
        self.p = np.full(len(self.dst), np.nan)

    @property
    def empty(self):
        return len(self.dst) == 0

    def sample(self, rng):
        return self._sample(rng)

    def sample_time(self, j, rng):
        raise NotImplementedError("Модель умеет только разыгрывать шаг целиком; используйте SampleSelector")

    def time_cdf(self, t):
        raise NotImplementedError("Модель умеет только разыгрывать шаг; формульный прогноз недоступен")


class ChoiceThenTimeModel:
    """Заданная истина: выход по ``p``, время из ``dist`` выбранного ребра.

    ``edges`` — словари ``src, dst, p, dist``. Сеть строится по рёбрам: ``starts`` — состояния
    без входов, ``terminals`` — без выходов (или заданные явно). Признаки не используются.
    Пример — синтетика A–D: ``ModelGenerator(ChoiceThenTimeModel(edges)).generate(EventLog.start(...))``.
    """
    def __init__(self, edges, starts=None, terminals=None):
        self.edges = [dict(e) for e in edges]
        src = {e["src"] for e in self.edges}
        dst = {e["dst"] for e in self.edges}
        starts = list(starts or sorted(src - dst) or [self.edges[0]["src"]])
        terminals = list(terminals or sorted(dst - src))
        self.network = Network.from_edges([(e["src"], e["dst"]) for e in self.edges], starts, terminals)
        self.history_features = DefaultHistoryFeatures()
        self._out = {}
        for e in self.edges:
            self._out.setdefault(e["src"], []).append(e)

    def predict(self, X, **kw):
        X = frame(X)
        steps = []
        for state, elapsed in zip(X.state, X.elapsed):
            out = self._out.get(state, [])
            if not out or state in self.network.terminals:
                steps.append(EmptyStep())
                continue
            steps.append(ChoiceThenTimeStep([e["dst"] for e in out], [e["p"] for e in out],
                                            [e["dist"] for e in out], elapsed))
        return steps
