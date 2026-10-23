"""Совместная MLE вероятностей выбора и параметров времени с учётом правой цензуры.

Для каждого состояния ``s`` отдельно, по всем его выходам: завершённый интервал ``s→d`` за ``t``
даёт множитель ``p_sd · f_sd(t)``, цензурированный через ``c`` — ``Σ_d p_sd · S_sd(c)``.
Оптимизатор L-BFGS-B, ``Σ p = 1`` через softmax.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .distributions import PARAMETERS, initial_dist, scipy_args

def source_nll(intervals, edges, probabilities, distributions):
    """Отрицательный log-likelihood; у цензуры скрытый конец суммируется."""
    result, survival_terms = 0., []
    censored = intervals.loc[~intervals["observed"], "duration"].to_numpy()
    for edge, p, dist in zip(edges, probabilities, distributions):
        rv, args, kwargs = scipy_args(dist)
        x = intervals.loc[intervals["observed"] & (intervals["dst"] == edge[1]),
                          "duration"].to_numpy()
        result -= len(x) * np.log(p) + rv.logpdf(x, *args, **kwargs).sum()
        survival_terms.append(np.log(p) + rv.logsf(censored, *args, **kwargs))
    if len(censored):
        result -= logsumexp(np.array(survival_terms), axis=0).sum()
    return float(result)


def fit_source(intervals, families, use_censoring=True):
    """Совместная MLE вероятностей выбора и параметров времени выходов одного состояния."""
    completed = intervals.loc[intervals["observed"]]
    source = intervals["src"].iloc[0]
    destinations = sorted(completed["dst"].unique())
    edges = [(source, dst) for dst in destinations]
    counts = completed.groupby("dst").size()
    diagnostic = dict(src=source, success=False, status="insufficient_data",
                      n_completed=len(completed), n_censored=int((~intervals.observed).sum()))
    if not edges or any(counts[edge[1]] < (1 if families[edge] == "exponential" else 3)
                        for edge in edges):
        return [], diagnostic
    data = intervals if use_censoring else completed
    initial = [initial_dist(families[edge], completed.loc[completed.dst == edge[1],
                                                       "duration"].to_numpy()) for edge in edges]
    k = len(edges)
    p0 = counts.reindex(destinations).to_numpy() / len(completed)
    x0 = list(np.log(p0[:-1] / p0[-1]))
    bounds = [(-15., 15.)] * (k - 1)
    for kind, par in initial:
        if kind == "uniform":
            width = par["high"] - par["low"]
            # Положительный старт survival даже если цензура позже максимума завершений.
            cens = data.loc[~data.observed, "duration"]
            extra = max(width * .1, cens.max() - par["high"] + .1) if len(cens) else width * .1
            x0.append(np.log(extra / width))
            bounds.append((-20., 12.))
        else:
            x0.extend(np.log([par[name] for name in PARAMETERS[kind]]))
            bounds.extend([(-5., 5.) if name in ("shape", "sigma") else (-15., 15.)
                           for name in PARAMETERS[kind]])

    def decode(vector):
        probabilities = softmax(np.r_[vector[:k - 1], 0.])
        distributions, offset = [], k - 1
        for kind, par in initial:
            if kind == "uniform":
                high = par["high"] + (par["high"] - par["low"]) * np.exp(vector[offset])
                distributions.append((kind, dict(low=par["low"], high=high)))
                offset += 1
            else:
                names = PARAMETERS[kind]
                values = np.exp(vector[offset:offset + len(names)])
                distributions.append((kind, dict(zip(names, values))))
                offset += len(names)
        return probabilities, distributions

    def objective(vector):
        p, distributions = decode(vector)
        value = source_nll(data, edges, p, distributions) / len(data)
        return value if np.isfinite(value) else 1e100

    attempts = []
    for shift in (0., .15):
        start = np.clip(np.array(x0) + shift, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        attempts.append(minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                                 options=dict(maxiter=500, ftol=1e-10, gtol=1e-6)))
    valid = [r for r in attempts if r.success and np.isfinite(r.fun) and r.fun < 1e90]
    best = min(valid or attempts, key=lambda r: r.fun)
    boundary = any(abs(v - lo) < 1e-4 or abs(v - hi) < 1e-4
                   for v, (lo, hi) in zip(best.x, bounds))
    diagnostic.update(success=bool(valid), status="ok" if valid else "optimizer_failed",
                      nll=float(best.fun * len(data)), message=str(best.message),
                      at_boundary=boundary)
    if not valid:
        return [], diagnostic
    probabilities, distributions = decode(best.x)
    rows = [dict(src=edge[0], dst=edge[1], name=f"{edge[0]}→{edge[1]}",
                 p=float(p), dist=dist, n_completed=int(counts[edge[1]]))
            for edge, p, dist in zip(edges, probabilities, distributions)]
    return rows, diagnostic


def estimate_network(intervals, families, use_censoring=True):
    """Оценки всех состояний: ``(estimates, diagnostics)``.

    ``estimates`` — список словарей ``src, dst, name, p, dist, n_completed``;
    его можно передать в :func:`state_petri_net`. ``use_censoring=False`` —
    наивная оценка только по завершённым интервалам.
    """
    rows, diagnostics = [], []
    for _, group in intervals.groupby("src", sort=True):
        estimated, diagnostic = fit_source(group, families, use_censoring)
        rows.extend(estimated)
        diagnostics.append(diagnostic)
    return rows, pd.DataFrame(diagnostics)
