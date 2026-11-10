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

    def transform_intervals(self, log, events, intervals):
        """Признаки на входе для всех интервалов сразу; те же значения, что ``transform``.

        Интервал k трассы соответствует k-му событию подготовленного журнала ``events``.
        Возвращает None, если подкласс переопределил ``transform`` или ``at_entry``: тогда
        :func:`survpm.process.entry_features` считает признаки по одному интервалу.
        """
        cls = type(self)
        if (cls.transform is not DefaultHistoryFeatures.transform
                or cls.at_entry is not DefaultHistoryFeatures.at_entry):
            return None
        ev = events[['case_id', 'activity', 'timestamp']].reset_index(drop=True)
        ev['_pos'] = ev.groupby('case_id', sort=False).cumcount()
        ev['history_visits'] = ev.groupby(['case_id', 'activity'], sort=False).cumcount() + 1
        first = ev.groupby('case_id', sort=False).timestamp.transform('min')
        ev['history_elapsed_days'] = ((ev.timestamp - first) / DAY).astype(float)
        pos = intervals.groupby('case_id', sort=False).cumcount().to_numpy()
        keys = pd.DataFrame({'case_id': intervals.case_id.to_numpy(), '_pos': pos})
        rows = keys.merge(ev, on=['case_id', '_pos'], how='left', validate='many_to_one')
        if rows.timestamp.isna().any():
            raise ValueError('Интервалы не согласованы с событиями журнала')
        rows.index = intervals.index
        points = rows[['case_id']].assign(time=rows.timestamp)
        X = values_at(points, log)
        return X.assign(history_visits=rows.history_visits.to_numpy(),
                        history_elapsed_days=rows.history_elapsed_days.to_numpy())

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

