"""Композиция структуры и заменяемой модели переходов."""
from copy import copy, deepcopy
from inspect import signature

import numpy as np

import pandas as pd

from .eventlog import EventLog
from .log import prepare, intervals
from .models.transitions import ExponentialTransitionModel
from .models.features import DefaultHistoryFeatures


def entry_features(log, events, iv, builder, columns):
    """Признаки X на входе в src для каждого интервала (индекс как у ``iv``).

    Если у builder есть векторный ``transform_intervals`` (DefaultHistoryFeatures), признаки
    считаются одним проходом; иначе — ``transform`` по префиксу истории каждого интервала.
    """
    columns = list(columns)
    vectorized = getattr(builder, 'transform_intervals', None)
    if vectorized is not None:
        X = vectorized(log, events, iv)
        if X is not None:
            unknown = [c for c in columns if c not in X.columns]
            if unknown:
                raise KeyError(f'Нет признаков в журнале: {unknown}')
            return X[columns]
    # Один timestamp может соответствовать нескольким упорядоченным входам.
    groups = {case: group for case, group in events.groupby('case_id', sort=False)}
    measurements = {case: group for case, group in log.measurements.groupby('case_id', sort=False)}
    empty = log.measurements.iloc[:0]
    positions, rows = {}, []
    for row in iv.itertuples():
        pos = positions.get(row.case_id, 0)
        prefix = groups[row.case_id].iloc[:pos+1]
        positions[row.case_id] = pos+1
        case_measures = measurements.get(row.case_id, empty)
        measures = case_measures.loc[case_measures.timestamp <= row.enter]
        view = EventLog(prefix, log.cases.reindex([row.case_id]), measures,
                        pd.Series({row.case_id: row.enter}))
        rows.append(builder.transform(view, row.case_id, row.enter)['X'])
    return pd.DataFrame(rows, index=iv.index)[columns]


class ProcessModel:
    """Сеть + модель переходов + признаки истории.

    ``network=None`` — сеть строится в :meth:`fit` как DFG по журналу (``starts``/``terminals``
    из ``log.rules``). После ``fit`` модели доступны сеть Петри (:meth:`to_pm4py`), все метрики
    библиотеки (:meth:`evaluate`), рисунки (:meth:`plot`) и сохранение (:meth:`save`/:meth:`load`).
    """
    def __init__(self, network=None, transitions=None, history_features=None, picker=None):
        self.network = network
        self.transitions = transitions if transitions is not None else ExponentialTransitionModel()
        if picker is not None:
            self.transitions = copy(self.transitions)
            self.transitions.picker = picker
        self.picker = picker if picker is not None else getattr(self.transitions, 'picker', None)
        self.history_features = history_features if history_features is not None else DefaultHistoryFeatures()

    def fit(self, log, features=None, use_censoring=True):
        if self.network is None:
            if not isinstance(log, EventLog):
                raise ValueError('Без сети передайте EventLog после prepare(rules): сеть строится по нему')
            from .discovery import discover
            self = copy(self)
            self.network = discover(log)
        if isinstance(log, EventLog):
            rules = dict(log.rules or {}, starts=self.network.starts, terminals=self.network.terminals)
            train_log = log.prepare(rules)
            events, iv = train_log.events, train_log.intervals()
        else:
            events, cases = prepare(log, self.network.starts, self.network.terminals)
            iv = intervals(events, cases)
            train_log = EventLog(events, pd.DataFrame(index=cases.case_id),
                                 pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value']),
                                 observation=cases[['case_id', 'end', 'event', 'reason']])
        builder = self.history_features.fit(train_log)
        X = None
        if features:
            if not isinstance(log, EventLog):
                raise ValueError('Для признаков передайте EventLog')
            X = entry_features(train_log, events, iv, builder, features)
        if not use_censoring:
            iv = iv.loc[iv.observed]
            X = X.loc[iv.index] if X is not None else None
        result = copy(self)
        result.train_log = deepcopy(train_log)
        result.training_intervals = iv.copy()
        result.feature_columns = list(features or [])
        result.history_features = builder
        result.transitions = self.transitions.fit(iv, self.network, X)
        return result

    def transition_parameters(self, state, X=None, history=None):
        return self.transitions.transition_parameters(state, X, history)

    def next_transition(self, state, X, history, rng, elapsed=0):
        """Return destination and residual days after survival through elapsed.

        Legacy models remain usable at entry; conditional sampling must explicitly
        accept elapsed. Exceptions inside the model propagate unchanged.
        """
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('elapsed должен быть конечным и неотрицательным')
        method = self.transitions.next_transition
        if elapsed == 0:
            return method(state, X, history, rng)
        try:
            signature(method).bind(state, X, history, rng, elapsed=elapsed)
        except TypeError:
            raise NotImplementedError('Transition model does not support elapsed') from None
        return method(state, X, history, rng, elapsed=elapsed)

    def edges_table(self):
        """Counts and baseline rates with the observed minimum-time kernel.

        p is the destination probability among exits. mean_days describes the
        observed minimum waiting time, shared by all exits from one state.
        """
        training = getattr(self, 'training_intervals', None)
        rows = []
        for src in self.network.edges.src.drop_duplicates():
            if hasattr(self.transitions, 'rates'):
                rates = {dst: self.transitions.rates.get((src, dst), 0.)
                         for dst in self.network.outgoing(src)}
            else:
                rates = self.transition_parameters(src)
            total = sum(rates.values())
            group = training.loc[training.src == src] if training is not None else None
            for dst in self.network.outgoing(src):
                rate = rates.get(dst, 0.)
                completed = int((group.observed & (group.dst == dst)).sum()) if group is not None else 0
                rows.append(dict(src=src, dst=dst, n_completed=completed,
                                 n_visits=len(group) if group is not None else 0,
                                 rate=rate, p=rate/total if total else 0.,
                                 mean_days=1/total if total else np.inf))
        return pd.DataFrame(rows, columns=['src', 'dst', 'n_completed', 'n_visits', 'rate', 'p', 'mean_days'])

    # --- мост к сети Петри и метрикам уровней 1–4 -------------------------------------------
    def legacy(self, train=None):
        """Эквивалентный :class:`~survpm.model.CensoredStateNet` с теми же интенсивностями.

        Нужен метрикам уровней 1–3 и экспорту в PM4Py; модель не переобучается: берутся
        ``self.transitions``. ``train`` — обучающий журнал, если модель не обучалась через
        :meth:`fit` (например, истина с заданными rates).
        """
        from .model import CensoredStateNet
        if getattr(self, 'feature_columns', None):
            raise NotImplementedError('Сеть Петри и метрики уровней 1–3 пока только для модели без признаков; '
                                      'для неё доступна NLL: evaluate(model, log, "nll")')
        if not hasattr(self.transitions, 'winner_edges'):
            raise NotImplementedError('Нужна модель переходов с winner_edges (ExponentialTransitionModel)')
        train = train if train is not None else getattr(self, 'train_log', None)
        if train is None:
            raise ValueError('Модель не обучена: передайте train= (журнал, на котором считать обучающие таблицы)')
        if len(self.network.starts) != 1 or len(self.network.terminals) != 1:
            raise NotImplementedError('Сеть Петри и метрики уровней 1–3 — для одного начала и одного терминала')
        rules = dict(train.rules or {})
        result = CensoredStateNet(self.network.starts[0], self.network.terminals[0],
                                  repeats=rules.get('repeats', 'simultaneous'),
                                  min_duration=rules.get('min_duration', 1 / 1440)).fit(train)
        result.network_, result.transitions_, result.process_model_ = self.network, self.transitions, self
        result.edges_ = self.transitions.winner_edges()
        result.families_ = {(e['src'], e['dst']): 'exponential' for e in result.edges_}
        if hasattr(self.transitions, 'diagnostics'):
            result.diagnostics_ = self.transitions.diagnostics
        return result

    def to_pm4py(self, censoring=True, train=None):
        """``(net, initial_marking, final_marking)`` PM4Py: сеть состояний с местом ``CENS``."""
        return self.legacy(train).to_pm4py(censoring)

    def to_sdfa(self, censoring=False, digits=3, train=None):
        """Стохастический автомат для Ebi."""
        return self.legacy(train).to_sdfa(censoring, digits)

    def evaluate(self, log, metrics='nll', train=None):
        """Метрики на журнале ``log``; см. :func:`survpm.metrics.evaluate`."""
        from .metrics.api import evaluate
        return evaluate(self, log, metrics, train=train)

    def plot(self, kind='network', visualizer=None, **kwargs):
        """Рисунок модели: ``kind="network"`` (сеть с rate и p) или ``"petri"`` (сеть Петри с CENS)."""
        if visualizer is None:
            from .viz import GraphvizVisualizer
            visualizer = GraphvizVisualizer()
        if kind == 'network':
            return visualizer.plot_network(self, **kwargs)
        if kind == 'petri':
            return visualizer.plot_petri(self, **kwargs)
        raise ValueError(f'kind: network или petri, получено {kind!r}')

    def save(self, path):
        """Сохранить модель целиком (pickle). Загружать только из доверенных файлов."""
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """Загрузить модель, сохранённую :meth:`save` (pickle: только доверенные файлы)."""
        import pickle
        with open(path, 'rb') as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError(f'В файле не {cls.__name__}')
        return model
