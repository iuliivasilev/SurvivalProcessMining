"""Семейства распределений времени перехода и выбор семейства по AIC.

Распределение задаётся парой ``(семейство, параметры)``, например
``("weibull", {"shape": 1.4, "scale": 3.7})``; время — в днях.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

PARAMETERS = {
    "exponential": ("mean",), "lognormal": ("median", "sigma"),
    "gamma": ("shape", "scale"), "weibull": ("shape", "scale"),
    "uniform": ("low", "high"),
}
CANDIDATE_FAMILIES = ("exponential", "lognormal", "gamma", "weibull")


def scipy_args(dist):
    """``(семейство, параметры)`` → ``(rv, args, kwargs)`` для ``scipy.stats``."""
    kind, par = dist
    if kind == "exponential":
        return stats.expon, (), dict(scale=par["mean"])
    if kind == "lognormal":
        return stats.lognorm, (par["sigma"],), dict(scale=par["median"])
    if kind == "gamma":
        return stats.gamma, (par["shape"],), dict(scale=par["scale"])
    if kind == "weibull":
        return stats.weibull_min, (par["shape"],), dict(scale=par["scale"])
    if kind == "uniform":
        return stats.uniform, (), dict(loc=par["low"], scale=par["high"] - par["low"])
    raise ValueError(kind)


def mean_time(dist):
    """Математическое ожидание времени перехода."""
    kind, par = dist
    if kind == "exponential":
        return par["mean"]
    if kind == "lognormal":
        return par["median"] * math.exp(par["sigma"] ** 2 / 2)
    if kind == "gamma":
        return par["shape"] * par["scale"]
    if kind == "weibull":
        return par["scale"] * math.gamma(1 + 1 / par["shape"])
    if kind == "uniform":
        return (par["low"] + par["high"]) / 2
    raise ValueError(kind)


def initial_dist(kind, x):
    """MLE семейства по завершённым длительностям ``x`` без учёта цензуры."""
    if kind == "exponential":
        par = dict(mean=x.mean())
    elif kind == "lognormal":
        par = dict(median=np.exp(np.log(x).mean()), sigma=max(np.log(x).std(), .05))
    elif kind == "gamma":
        shape, _, scale = stats.gamma.fit(x, floc=0)
        par = dict(shape=shape, scale=scale)
    elif kind == "weibull":
        shape, _, scale = stats.weibull_min.fit(x, floc=0)
        par = dict(shape=shape, scale=scale)
    elif kind == "uniform":
        par = dict(low=x.min(), high=x.max())
    else:
        raise ValueError(kind)
    return kind, par


def select_families(intervals, candidates=CANDIDATE_FAMILIES):
    """Семейство каждого ребра по минимальному AIC на завершённых интервалах.

    Ребро с менее чем тремя завершениями получает exponential.
    Возвращает ``(families, aic_table)``: словарь ``(src, dst) → семейство`` и таблицу AIC.
    """
    rows = []
    for (src, dst), group in intervals.loc[intervals.observed].groupby(["src", "dst"]):
        x = group.duration.to_numpy()
        if len(x) < 3:
            rows.append(dict(src=src, dst=dst, n_completed=len(x), family="exponential", aic=np.nan))
            continue
        for kind in candidates:
            rv, args, kwargs = scipy_args(initial_dist(kind, x))
            aic = 2 * len(PARAMETERS[kind]) - 2 * rv.logpdf(x, *args, **kwargs).sum()
            rows.append(dict(src=src, dst=dst, n_completed=len(x), family=kind, aic=aic))
    table = pd.DataFrame(rows)
    table["delta_aic"] = table.aic - table.groupby(["src", "dst"]).aic.transform("min")
    best = table.sort_values("aic", na_position="last").groupby(["src", "dst"]).head(1)
    return {(row.src, row.dst): row.family for row in best.itertuples()}, table


def dist_str(dist):
    """``("weibull", {"shape": 1.4, "scale": 3.7})`` → ``"weibull(shape=1.4, scale=3.7)"``."""
    kind, par = dist
    return kind + "(" + ", ".join(f"{k}={v:g}" for k, v in par.items()) + ")"
