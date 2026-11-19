"""Исполнение траекторий и прогноз с ограничением наблюдений по as_of."""
from copy import deepcopy

import numpy as np
import pandas as pd

from .eventlog import EventLog
from .generation import ModelGenerator, timestamp
from .forecasting import Forecaster, observed_history  # noqa: F401  (Forecaster: прежнее место импорта)
from .models.features import DefaultHistoryFeatures as HistoryFeatures

DAY = pd.Timedelta(days=1)


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
