"""Исполнение траекторий и прогноз с ограничением наблюдений по as_of."""
from copy import deepcopy

import numpy as np
import pandas as pd

from .eventlog import EventLog
from .generation import ModelGenerator, timestamp
from .models.features import DefaultHistoryFeatures as HistoryFeatures

DAY = pd.Timedelta(days=1)


def observed_history(log, case_id, as_of):
    cutoff = timestamp(as_of)
    events = log.events.loc[(log.events.case_id == case_id) & (log.events.timestamp <= cutoff)].copy()
    measures = log.measurements.loc[(log.measurements.case_id == case_id) & (log.measurements.timestamp <= cutoff)].copy()
    return EventLog(events, log.cases.reindex([case_id]).copy(), measures, pd.Series({case_id: cutoff}))


class ObjectStream:
    def __init__(self, initial, updater=None, seed=None):
        self.seed = seed
        self._objects = deepcopy(initial.to_dict('records') if isinstance(initial, pd.DataFrame) else list(initial))
        if len({r['case_id'] for r in self._objects}) != len(self._objects):
            raise ValueError('case_id объектов должны быть уникальны')
        for row in self._objects:
            row['t'] = timestamp(row['t'])
            row['X'] = dict(row.get('X', {}))
            row['history'] = list(row.get('history', []))
        self.updater = updater or (lambda X, state, t, rng: dict(X))

    @classmethod
    def start(cls, n, state, t, X=None, prefix='case-', updater=None, seed=None):
        """``n`` новых объектов в состоянии ``state`` в момент ``t`` с признаками ``X``."""
        return cls([dict(case_id=f'{prefix}{i}', state=state, t=t, X=dict(X or {})) for i in range(n)],
                   updater, seed)

    @classmethod
    def from_log(cls, log, state=None, X=None, prefix='', updater=None, seed=None):
        """Объекты журнала стартуют заново: момент — первое событие объекта, состояние —
        ``state`` или первое событие. ``X`` — общий словарь признаков или функция ``case_id → dict``."""
        first = log.events.sort_values('timestamp', kind='stable').groupby('case_id', sort=False).head(1)
        rows = []
        for row in first.itertuples():
            features = X(row.case_id) if callable(X) else dict(X or {})
            rows.append(dict(case_id=f'{prefix}{row.case_id}', state=state or row.activity,
                             t=row.timestamp, X=features))
        return cls(rows, updater, seed)

    def records(self):
        """Независимая копия объектов потока (движок исполняет её, сам поток не меняется)."""
        return deepcopy(self._objects)

    @property
    def objects(self):
        """Independent table of the stream's current object contexts."""
        return pd.DataFrame(deepcopy(self._objects))

    def current(self, case_id):
        row = next(r for r in self._objects if r['case_id'] == case_id)
        return row['state'], dict(row['X']), row['t']

    def history(self, case_id):
        return deepcopy(next(r for r in self._objects if r['case_id'] == case_id)['history'])


class Simulator:
    """Генерация журнала по модели: обёртка над :class:`~survpm.generation.ModelGenerator`.

    Seed задаётся у движка или у потока, но не у обоих. У каждого объекта свой поток
    случайных чисел по ``case_id``. ``max_steps=None`` — без предела переходов (тогда
    достижимый цикл требует конечного ``horizon``); по умолчанию предел 1000.
    """
    def __init__(self, model, seed=None, executor=None):
        if executor is not None:
            raise NotImplementedError('Custom executor is not supported')
        self.model, self.seed, self.executor = model, seed, executor

    def run(self, stream, horizon=None, max_steps=1000):
        """Журнал :class:`EventLog` из объектов потока; поток не меняется и годится для повтора."""
        return ModelGenerator(self.model, stream, self.seed, horizon, max_steps).generate()


class Router:
    def __init__(self, model, history_features=None, seed=None):
        self.model = model
        self.history_features = history_features or model.history_features
        self.seed = seed

    def run(self, log, case_id, as_of, future_features=None, horizon=None):
        ctx = self.history_features.transform(observed_history(log, case_id, as_of), case_id, as_of)
        ctx['initialized'] = True
        # Сценарий получает только X и сгенерированное будущее, не полный реальный журнал.
        return Simulator(self.model, self.seed).run(ObjectStream([ctx], future_features), horizon=horizon)


class Forecaster:
    def __init__(self, model, history_features=None, method='simulation', n_paths=1000, seed=None):
        if method != 'simulation':
            raise NotImplementedError(f'Forecaster method={method!r} is not implemented; use simulation')
        if n_paths < 1:
            raise ValueError('n_paths должен быть положительным')
        self.model, self.history_features = model, history_features or model.history_features
        self.n_paths, self.seed = n_paths, seed

    def _paths(self, history, as_of, times, case_id, future_features):
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError('Задайте неотрицательные конечные горизонты')
        ids = history.events.case_id.unique()
        if case_id is None and len(ids) != 1:
            raise ValueError('Для нескольких объектов укажите case_id')
        selected_id = ids[0] if case_id is None else case_id
        ctx = self.history_features.transform(observed_history(history, selected_id, as_of), selected_id, as_of)
        initial = [dict(ctx, case_id=i, initialized=True) for i in range(self.n_paths)]
        log = Simulator(self.model, self.seed).run(ObjectStream(initial, future_features),
                                                   horizon=float(times.max()), max_steps=None)
        last = log.events.groupby('case_id', sort=False).tail(1).set_index('case_id').reindex(range(self.n_paths))
        ends = log.observation.set_index('case_id').reindex(range(self.n_paths)).end
        duration = ((ends-timestamp(as_of))/DAY).to_numpy()
        terminal = last.activity.isin(self.model.network.terminals).to_numpy()
        return times, duration, terminal, last.activity.to_numpy()

    def survival(self, history, as_of, times, future_features=None, case_id=None):
        times, duration, terminal, _ = self._paths(history, as_of, times, case_id, future_features)
        cdf = [np.mean(terminal & (duration <= t)) for t in times]
        return pd.DataFrame(dict(horizon=times, survival=1-np.array(cdf), cdf=cdf))

    def terminal_probs(self, history, as_of, times, future_features=None, case_id=None):
        times, duration, terminal, states = self._paths(history, as_of, times, case_id, future_features)
        return pd.DataFrame([dict(horizon=t, outcome=k, probability=np.mean(terminal & (states == k) & (duration <= t)))
                             for t in times for k in self.model.network.terminals])

    def hazard(self, history, as_of, times, future_features=None, case_id=None):
        """Cause-specific bin rates: completions / paths at risk / bin width.

        Each horizon is the right bin edge; bins start at zero. With no paths
        remaining at risk the hazard is undefined (NaN).
        """
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(np.r_[0., times]) <= 0).any():
            raise ValueError('hazard times must be strictly increasing and positive')
        times, duration, terminal, states = self._paths(history, as_of, times, case_id, future_features)
        rows = []
        for left, right in zip(np.r_[0., times[:-1]], times):
            risk = np.count_nonzero(~terminal | (duration > left))
            for outcome in self.model.network.terminals:
                count = np.count_nonzero(terminal & (states == outcome) & (duration > left) & (duration <= right))
                rows.append(dict(horizon=right, outcome=outcome,
                                 hazard=count/risk/(right-left) if risk else np.nan))
        return pd.DataFrame(rows, columns=['horizon', 'outcome', 'hazard'])

    def expected_time(self, history, as_of, horizon, future_features=None, case_id=None):
        """Restricted mean time to any terminal, and unfinished mass at horizon."""
        _, duration, terminal, _ = self._paths(history, as_of, [horizon], case_id, future_features)
        completed = terminal & (duration <= horizon)
        return dict(restricted_mean_time=float(np.mean(np.where(completed, duration, horizon))),
                    probability_unfinished=float(np.mean(~completed)))


class Evaluator:
    """Likelihood по тому же закону, что используется моделью для генерации."""
    def __init__(self, model):
        self.model = model

    def run(self, log, metrics=('nll',)):
        if set(metrics) != {'nll'}:
            raise NotImplementedError('Evaluator ProcessModel пока поддерживает nll; остальные метрики доступны через CensoredStateNet.evaluate')
        from .log import prepare, intervals
        if isinstance(log, EventLog):
            rules = dict(log.rules or {}, starts=self.model.network.starts, terminals=self.model.network.terminals)
            log = log.prepare(rules)
            events, iv = log.events, log.intervals()
        else:
            events, cases = prepare(log, self.model.network.starts, self.model.network.terminals)
            iv = intervals(events, cases)
        frame = None
        columns = getattr(self.model, 'feature_columns', [])
        if columns:
            if not isinstance(log, EventLog):
                raise ValueError('Для признаков передайте EventLog')
            from .process import entry_features
            frame = entry_features(log, events, iv, self.model.history_features, columns)
        nll = -self.model.transitions.log_likelihood(iv, frame)
        return dict(nll_per_interval=nll/len(iv) if len(iv) else np.nan,
                    nll_per_trace=nll/iv.case_id.nunique() if len(iv) else np.nan)
