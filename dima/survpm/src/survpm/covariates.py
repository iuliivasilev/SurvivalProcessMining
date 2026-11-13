"""Старая модель choice_then_time с признаками: softmax выбора и AFT времени с правой цензурой.

Кодирование признаков — :class:`survpm.encoding.FeatureEncoder`."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .distributions import PARAMETERS, scipy_args
from .encoding import FeatureEncoder  # noqa: F401  (прежний путь импорта)

SCALE_PARAMETER = {"exponential": "mean", "lognormal": "median", "gamma": "scale", "weibull": "scale"}


def probabilities_and_scales(edges, X):
    logits = np.log([e["p"] for e in edges]) + X @ np.array([e["choice_coef"] for e in edges]).T
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    scales = np.exp(np.clip(X @ np.array([e["time_coef"] for e in edges]).T, -30, 30))
    return logp, scales


def conditional_nll(intervals, edges, X):
    logp, scales = probabilities_and_scales(edges, X)
    t = intervals.duration.to_numpy()
    observed = intervals.observed.to_numpy()
    terms = []
    completed = np.zeros(len(intervals))
    for j, edge in enumerate(edges):
        rv, args, kwargs = scipy_args(edge["dist"])
        shifted = t / scales[:, j]
        terms.append(logp[:, j] + rv.logsf(shifted, *args, **kwargs))
        mask = observed & (intervals.dst.to_numpy() == edge["dst"])
        completed[mask] = (logp[mask, j] + rv.logpdf(shifted[mask], *args, **kwargs)
                           - np.log(scales[mask, j]))
    return -float(completed[observed].sum() + logsumexp(np.array(terms)[:, ~observed], axis=0).sum())


def fit_source(intervals, families, X, use_censoring, l2):
    from .estimation import fit_source as fit_baseline

    if any(families[(s, d)] not in SCALE_PARAMETER for s, d in
           intervals.loc[intervals.observed, ["src", "dst"]].itertuples(index=False, name=None)):
        raise ValueError("Признаки поддерживают exponential, lognormal, gamma, weibull; uniform не поддерживается")
    edges, diagnostic = fit_baseline(intervals, families, use_censoring)
    if not edges:
        return edges, diagnostic
    mask = np.ones(len(intervals), dtype=bool) if use_censoring else intervals.observed.to_numpy()
    data, design = intervals.loc[mask], X[mask]
    k, q = len(edges), X.shape[1]
    x0 = list(np.log([e["p"] / edges[-1]["p"] for e in edges[:-1]]))
    bounds = [(-15., 15.)] * (k - 1)
    for edge in edges:
        kind, par = edge["dist"]
        x0.extend(np.log([par[name] for name in PARAMETERS[kind]]))
        bounds.extend([(-5., 5.) if name in ("shape", "sigma") else (-15., 15.) for name in PARAMETERS[kind]])
    base_size = len(x0)
    n_coefficients = (2 * k - 1) * q
    x0.extend([0.] * n_coefficients)
    bounds.extend([(-8., 8.)] * n_coefficients)

    def decode(vector):
        p = softmax(np.r_[vector[:k - 1], 0.])
        offset, result = k - 1, []
        choice = np.vstack([vector[base_size:base_size + (k - 1) * q].reshape(k - 1, q), np.zeros((1, q))])
        timing = vector[base_size + (k - 1) * q:].reshape(k, q)
        for j, edge in enumerate(edges):
            kind = edge["dist"][0]
            names = PARAMETERS[kind]
            dist = (kind, dict(zip(names, np.exp(vector[offset:offset + len(names)]))))
            offset += len(names)
            result.append(dict(edge, p=float(p[j]), dist=dist, choice_coef=choice[j], time_coef=timing[j]))
        return result

    def objective(vector):
        value = (conditional_nll(data, decode(vector), design) + l2 * np.square(vector[base_size:]).sum() / 2) / len(data)
        return value if np.isfinite(value) else 1e100

    fitted = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", options=dict(maxiter=1000, ftol=1e-10))
    success = bool(fitted.success and np.isfinite(fitted.fun) and fitted.fun < 1e90)
    diagnostic.update(success=success, status="ok" if success else "optimizer_failed", message=str(fitted.message),
                      at_boundary=any(abs(v - lo) < 1e-4 or abs(v - hi) < 1e-4 for v, (lo, hi) in zip(fitted.x, bounds)))
    result = decode(fitted.x) if success else []
    diagnostic["nll"] = conditional_nll(data, result, design) if success else np.nan
    return result, diagnostic


def condition_edges(edges, x):
    result = []
    for src in sorted({e["src"] for e in edges}):
        out = [e for e in edges if e["src"] == src]
        logp, scales = probabilities_and_scales(out, x)
        for j, edge in enumerate(out):
            kind, par = edge["dist"]
            par = dict(par)
            par[SCALE_PARAMETER[kind]] *= scales[0, j]
            result.append(dict(edge, p=float(np.exp(logp[0, j])), dist=(kind, par)))
    return result
