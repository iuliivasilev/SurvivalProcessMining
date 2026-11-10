"""Признаки на входе: train-only кодирование, softmax и AFT с правой цензурой."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .distributions import PARAMETERS, scipy_args

SCALE_PARAMETER = {"exponential": "mean", "lognormal": "median", "gamma": "scale", "weibull": "scale"}


class FeatureEncoder:
    """Медиана + индикатор пропуска, one-hot, стандартизация по обучению.

    Неизвестная категория кодируется как базовая; постоянные колонки удаляются.

    ``clip=True`` (по умолчанию) обрезает стандартизованные значения до диапазона, который
    встречался при обучении: значение вне диапазона train (выброс анализа) кодируется как
    ближайшее крайнее, и линейная по z модель не экстраполирует ``exp(β·z)`` за пределы данных.
    ``log_features`` — числовые признаки, к которым до стандартизации применяется ``log1p``
    (скошенные положительные величины: анализы, суммы, длительности); отрицательные значения
    в них — ошибка.
    """

    def __init__(self, clip=True, log_features=()):
        self.clip = clip
        self.log_features = list(log_features)

    def _numbers(self, column, values):
        numbers = pd.to_numeric(values, errors="coerce")
        if column in getattr(self, "log_features", ()):
            if (numbers < 0).any():
                raise ValueError(f"log_features: отрицательные значения признака {column}")
            numbers = np.log1p(numbers)
        return numbers

    def fit(self, X):
        unknown = set(self.log_features) - set(X.columns)
        if unknown:
            raise ValueError(f"log_features нет среди признаков: {sorted(unknown)}")
        self.columns_ = list(X.columns)
        self.numeric_, self.categories_, self.medians_ = [], {}, {}
        for column in self.columns_:
            values = X[column]
            numbers = self._numbers(column, values)
            if numbers.notna().sum() == values.notna().sum():
                self.numeric_.append(column)
                self.medians_[column] = numbers.median() if numbers.notna().any() else 0.
            elif column in self.log_features:
                raise ValueError(f"log_features: признак {column} не числовой")
            else:
                self.categories_[column] = sorted(values.astype(object).fillna("<missing>").astype(str).unique())
        frame = self._frame(X)
        self.mean_ = frame.mean()
        std = frame.std(ddof=0)
        self.names_ = list(std.index[std > 1e-12])
        self.std_ = std[self.names_]
        z = (frame[self.names_] - self.mean_[self.names_]) / self.std_
        self.low_, self.high_ = z.min().to_numpy(), z.max().to_numpy()
        return self

    def _frame(self, X):
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Нет признаков X: {sorted(missing)}")
        out = {}
        for column in self.numeric_:
            values = self._numbers(column, X[column])
            if np.isinf(values).any():
                raise ValueError(f"Бесконечные значения признака {column}")
            out[column] = values.fillna(self.medians_[column])
            out[f"{column}__missing"] = values.isna().astype(float)
        for column, categories in self.categories_.items():
            values = X[column].astype(object).fillna("<missing>").astype(str)
            for category in categories[1:]:
                out[f"{column}={category}"] = (values == category).astype(float)
        return pd.DataFrame(out, index=X.index, dtype=float)

    def transform(self, X):
        frame = self._frame(X)
        z = ((frame[self.names_] - self.mean_[self.names_]) / self.std_).to_numpy()
        if getattr(self, "clip", False) and hasattr(self, "low_"):
            z = np.clip(z, self.low_, self.high_)
        return z


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
