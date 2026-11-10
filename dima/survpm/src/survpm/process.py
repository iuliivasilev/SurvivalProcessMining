"""Композиция структуры и заменяемой модели переходов."""
from copy import copy, deepcopy
from inspect import signature

import numpy as np

import pandas as pd

from .features import EventLog
from .log import prepare, intervals
from .transitions import ExponentialTransitionModel
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
    def __init__(self, network, transitions=None, history_features=None, picker=None):
        self.network = network
        self.transitions = transitions if transitions is not None else ExponentialTransitionModel()
        if picker is not None:
            self.transitions = copy(self.transitions)
            self.transitions.picker = picker
        self.picker = picker if picker is not None else getattr(self.transitions, 'picker', None)
        self.history_features = history_features if history_features is not None else DefaultHistoryFeatures()

    def fit(self, log, features=None, use_censoring=True):
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
