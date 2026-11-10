"""Заменяемая единая модель независимых конкурирующих времён."""
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .covariates import FeatureEncoder


class TransitionModel(Protocol):
    def fit(self, intervals, network, X=None): ...
    def transition_parameters(self, state, X, history): ...
    def next_transition(self, state, X, history, rng, elapsed=0): ...
    def log_likelihood(self, intervals, X=None): ...


class TransitionPicker(Protocol):
    def pick(self, transition_times, rng): ...


class MinTimePicker:
    """Select the first minimum from already sampled competing times."""
    def pick(self, transition_times, rng):
        if not transition_times:
            return None, np.inf
        dst = min(transition_times, key=transition_times.get)
        return dst, float(transition_times[dst])


class ExponentialTransitionModel:
    """Cause-specific log-rate regression; fit/update возвращают новую модель.

    При отсутствии признаков lambda_sd = n_sd / sum(duration_s). Все конкурирующие
    события дают время экспозиции каждому выходу. rates можно задать вручную.
    """
    def __init__(self, rates=None, l2=1., picker=None):
        if l2 < 0:
            raise ValueError('l2 должен быть неотрицательным')
        self.rates = dict(rates or {})
        if any(not np.isfinite(v) or v < 0 for v in self.rates.values()):
            raise ValueError('Интенсивности должны быть конечными и неотрицательными')
        self.l2 = l2
        self.encoder = None
        self.coefficients = {}
        self.picker = picker if picker is not None else MinTimePicker()

    def fit(self, intervals, network, X=None):
        result = ExponentialTransitionModel(l2=self.l2, picker=self.picker)
        result.network = network
        result.training = intervals.copy()
        result.training_X = X.copy() if X is not None else None
        durations = intervals.duration.to_numpy(float)
        if len(intervals) == 0 or not np.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError('Нужны положительные конечные длительности')
        for row in intervals.loc[intervals.observed].itertuples():
            if row.dst not in network.outgoing(row.src):
                raise ValueError('Наблюдаемый переход отсутствует в сети')
        if X is not None:
            if not X.index.equals(intervals.index):
                raise ValueError('Индекс X должен совпадать с индексом intervals')
            result.encoder = FeatureEncoder().fit(X)
            design = result.encoder.transform(X)
        else:
            design = np.empty((len(intervals), 0))
        diagnostics = []
        for src, dst in network.edges.itertuples(index=False, name=None):
            mask = intervals.src.to_numpy() == src
            t, z = durations[mask], design[mask]
            y = (intervals.loc[mask, 'observed'] & (intervals.loc[mask, 'dst'] == dst)).to_numpy(float)
            rate = y.sum()/t.sum() if len(t) else 0.
            coef = np.zeros(design.shape[1])
            if len(coef) and rate > 0:
                A = np.column_stack([np.ones(len(z)), z])
                def objective(b):
                    eta = A @ b
                    exposure = t * np.exp(eta)
                    value = (exposure-y*eta).sum() + self.l2*np.square(b[1:]).sum()
                    grad = A.T @ (exposure-y)
                    grad[1:] += 2*self.l2*b[1:]
                    return value, grad
                fit = minimize(objective, np.r_[np.log(rate), coef], jac=True, method='L-BFGS-B',
                               bounds=[(-30., 30.)]*(len(coef)+1))
                if not fit.success:
                    raise ValueError(f'Не удалось оценить переход {src}→{dst}: {fit.message}')
                rate, coef = float(np.exp(fit.x[0])), fit.x[1:]
            result.rates[src, dst] = float(rate)
            result.coefficients[src, dst] = coef
            diagnostics.append(dict(src=src, dst=dst, success=True, status='ok' if rate else 'no_events',
                                    n_completed=int(y.sum())))
        result.diagnostics = pd.DataFrame(diagnostics)
        return result

    def update(self, intervals, X=None):
        if not hasattr(self, 'training'):
            raise ValueError('Перед update вызовите fit')
        if (self.training_X is None) != (X is None):
            raise ValueError('Передайте тот же набор признаков, что при fit')
        if X is not None:
            if not X.index.equals(intervals.index):
                raise ValueError('Индекс X должен совпадать с индексом intervals')
            if list(X.columns) != list(self.training_X.columns):
                raise ValueError('Колонки X должны совпадать с признаками при fit')
        data = pd.concat([self.training, intervals], ignore_index=True)
        frame = pd.concat([self.training_X, X], ignore_index=True) if X is not None else None
        return self.fit(data, self.network, frame)

    def transition_parameters(self, state, X=None, history=None):
        """Scalar X gives destination rates; a DataFrame gives one dict per row."""
        batch = isinstance(X, pd.DataFrame)
        frame = X if batch else pd.DataFrame([{} if X is None else X])
        design = self.encoder.transform(frame) if self.encoder is not None else np.empty((len(frame), 0))
        result = [{dst: rate*float(np.exp(np.clip(
                       self.coefficients.get((src, dst), np.zeros(len(row))) @ row, -60, 60)))
                   for (src, dst), rate in self.rates.items() if src == state}
                  for row in design]
        return result if batch else result[0]

    def next_transition(self, state, X, history, rng, elapsed=0):
        """Sample remaining time conditional on survival through elapsed days.

        Exponential causes are memoryless, so elapsed does not shift the draw.
        """
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('elapsed должен быть конечным и неотрицательным')
        rates = self.transition_parameters(state, X, history)
        positive = [(dst, rate) for dst, rate in rates.items() if rate > 0]
        if not positive:
            return None, np.inf
        times = rng.exponential(1 / np.array([rate for _, rate in positive]))
        return self.picker.pick(dict(zip([dst for dst, _ in positive], times)), rng)

    def log_likelihood(self, intervals, X=None):
        if self.encoder is not None and (X is None or not X.index.equals(intervals.index)):
            raise ValueError('Передайте X с индексом интервалов')
        value = 0.
        for i, row in enumerate(intervals.itertuples()):
            rates = self.transition_parameters(row.src, X.iloc[i].to_dict() if X is not None else {})
            if not rates:
                raise ValueError(f'Неизвестное исходное состояние: {row.src}')
            value -= sum(rates.values())*row.duration
            if row.observed:
                rate = rates.get(row.dst, 0.)
                if rate <= 0:
                    return -np.inf
                value += np.log(rate)
        return float(value)

    def condition(self, X):
        result = ExponentialTransitionModel(l2=self.l2, picker=self.picker)
        if hasattr(self, 'diagnostics'):
            result.diagnostics = self.diagnostics.copy()
        for src in {src for src, _ in self.rates}:
            result.rates.update({(src, dst): rate for dst, rate in self.transition_parameters(src, X).items()})
        return result

    def winner_edges(self):
        """Наблюдаемое ядро для существующего grid и адаптеров (не времена причин)."""
        edges = []
        for src in sorted({src for src, _ in self.rates}):
            rates = {dst: rate for (source, dst), rate in self.rates.items() if source == src}
            total = sum(rates.values())
            counts = (self.diagnostics.set_index(['src','dst']).n_completed.to_dict()
                      if hasattr(self, 'diagnostics') else {})
            for dst, rate in rates.items():
                if rate:
                    edges.append(dict(src=src, dst=dst, name=f'{src}→{dst}', p=rate/total,
                                      dist=('exponential', {'mean': 1/total}), n_completed=counts.get((src,dst), 0)))
        return edges


_TRANSITION_MODELS = {'exponential': ExponentialTransitionModel}


def register_transition_model(name, factory):
    """Register a model factory without silently replacing an existing name."""
    if name in _TRANSITION_MODELS:
        raise ValueError(f'Transition model already registered: {name}')
    _TRANSITION_MODELS[name] = factory


def create_transition_model(name, **kwargs):
    try:
        factory = _TRANSITION_MODELS[name]
    except KeyError:
        raise ValueError(f'Unknown transition model: {name}') from None
    return factory(**kwargs)
