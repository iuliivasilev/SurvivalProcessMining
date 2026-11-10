"""Feature contracts and the default history builder, independent of execution."""
from copy import deepcopy
from typing import Protocol

import pandas as pd

from ..features import values_at

DAY = pd.Timedelta(days=1)


class FeatureUpdater(Protocol):
    def __call__(self, X, state, t, rng): ...


class HistoryFeatures(Protocol):
    def fit(self, train_log): ...
    def transform(self, log, case_id, as_of): ...
    def at_entry(self, X, state, t, history): ...


class LatentModel(Protocol):
    """Extension contract only; execution and latent inference are not implemented."""
    def init(self, case_id, rng): ...
    def update(self, Z, state, t, rng): ...
    def infer(self, history): ...


def timestamp(t):
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


class DefaultHistoryFeatures:
    def fit(self, train_log):
        return deepcopy(self)

    def at_entry(self, X, state, t, history):
        result = dict(X)
        result['history_visits'] = sum(r['activity'] == state for r in history)
        result['history_elapsed_days'] = float((t-history[0]['timestamp'])/DAY)
        return result

    def transform(self, log, case_id, as_of):
        cutoff = timestamp(as_of)
        events = log.events.loc[(log.events.case_id == case_id) & (log.events.timestamp <= cutoff)]
        events = events.sort_values('timestamp', kind='stable')
        if events.empty:
            raise ValueError('Нет состояния объекта на момент as_of')
        state, entered = events.iloc[-1][['activity', 'timestamp']]
        # Признаки фиксируются на входе, поздние замеры не меняют уже идущее ожидание.
        X = values_at(pd.DataFrame({'case_id': [case_id], 'time': [entered]}), log).iloc[0].to_dict()
        X = self.at_entry(X, state, entered, events.to_dict('records'))
        return dict(case_id=case_id, state=state, X=X, t=cutoff, entered_at=entered,
                    elapsed=float((cutoff-entered)/DAY), history=events.to_dict('records'))

