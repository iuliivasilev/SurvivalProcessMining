"""Признаки объектов к моменту «сейчас»: базовый класс :class:`StreamData`.

``StreamData().at(log)`` превращает журнал в таблицу X — одна строка на объект::

    case_id  state      elapsed  time                 Age  CRP  history_visits ...
    c42      Admission  1.0      2025-03-10 00:00     70   120  1

* ``state`` — последнее состояние объекта не позже его ``end`` (момент «сейчас»,
  ``log.observation.end``; его ставит :meth:`EventLog.cut`);
* ``elapsed`` — сколько дней объект уже находится в этом состоянии;
* ``time`` — момент «сейчас»;
* остальные колонки — признаки.

Та же таблица — вход ``model.predict``, ``Selector`` и ``Forecaster``; генератор вызывает
``at`` на каждом шаге. Своя динамика признаков — подкласс с методом :meth:`StreamData.at`.
"""
from abc import ABC, abstractmethod

import pandas as pd

from .models.features import DefaultHistoryFeatures, timestamp

DAY = pd.Timedelta(days=1)
RESERVED = ("case_id", "state", "elapsed", "time")


def observed_history(log, case_id, t):
    """Журнал одного объекта, обрезанный на ``t``: признаки не видят будущего."""
    from .eventlog import EventLog
    cutoff = timestamp(t)
    events = log.events.loc[(log.events.case_id == case_id) & (log.events.timestamp <= cutoff)].copy()
    measures = log.measurements.loc[(log.measurements.case_id == case_id) & (log.measurements.timestamp <= cutoff)].copy()
    return EventLog(events, log.cases.reindex([case_id]).copy(), measures, pd.Series({case_id: cutoff}))


def frame(X):
    """X как таблица: словарь — одна строка; ``elapsed`` по умолчанию 0."""
    X = pd.DataFrame([X]) if isinstance(X, dict) else X.copy()
    if "state" not in X:
        raise ValueError("В X нужна колонка state")
    if "case_id" not in X:
        X.insert(0, "case_id", range(len(X)))
    if "elapsed" not in X:
        X["elapsed"] = 0.
    return X.reset_index(drop=True)


def features_of(row):
    """Признаки строки X без служебных колонок (словарь)."""
    return {k: v for k, v in dict(row).items() if k not in RESERVED}


class StreamData(ABC):
    """Признаки к моменту «сейчас». Обязательный метод — :meth:`at`.

    ``derived`` — колонки, которые считаются по истории и не пишутся генератором в журнал
    как замеры (например, число посещений состояния).
    """
    derived = ()

    @abstractmethod
    def at(self, log, **kw):
        """Таблица X: ``case_id, state, elapsed, time`` и признаки, одна строка на объект."""


class DefaultStreamData(StreamData):
    """Признаки по умолчанию.

    Последнее состояние, ``elapsed``, статические признаки, последние замеры не позже входа в
    текущее состояние (признаки фиксируются на входе, поздние замеры не меняют идущее ожидание)
    и признаки истории ``history_visits`` (сколько раз объект был в текущем состоянии, считая
    текущий вход) и ``history_elapsed_days`` (дни от первого события до входа).

    ``history_features`` — свой построитель признаков истории (протокол
    :class:`~survpm.models.HistoryFeatures`); по умолчанию — обученный построитель модели или
    :class:`~survpm.DefaultHistoryFeatures`.
    """
    derived = ("history_visits", "history_elapsed_days")

    def __init__(self, history_features=None):
        self.history_features = history_features

    def at(self, log, **kw):
        current = self.current(log)
        builder = self.history_features
        if builder is None or type(builder) is DefaultHistoryFeatures:
            from .features import values_at
            X = values_at(current[["case_id"]].assign(time=current.entered), log)
            X["history_visits"] = current.visits.to_numpy()
            X["history_elapsed_days"] = current.since_start.to_numpy()
        else:
            rows = [builder.transform(observed_history(log, c, t), c, t)["X"]
                    for c, t in zip(current.case_id, current.time)]
            X = pd.DataFrame(rows, index=current.index)
        X = pd.concat([current[list(RESERVED)], X.drop(columns=[c for c in RESERVED if c in X])], axis=1)
        return X.reset_index(drop=True)

    @staticmethod
    def current(log):
        """``case_id, state, elapsed, time, entered, visits, since_start`` по объектам журнала."""
        now = log.now
        events = log.events.sort_values("timestamp", kind="stable")
        events = events.loc[events.timestamp <= events.case_id.map(now)]
        if events.empty:
            return pd.DataFrame(columns=["case_id", *RESERVED[1:], "entered", "visits", "since_start"])
        last = events.groupby("case_id", sort=False).tail(1).set_index("case_id")
        first = events.groupby("case_id", sort=False).timestamp.min()
        visits = (events.merge(last.activity.rename("current"), left_on="case_id", right_index=True)
                  .query("activity == current").groupby("case_id").size())
        ids = [c for c in log.case_ids if c in last.index]
        last = last.loc[ids]
        time = now.reindex(ids)
        return pd.DataFrame(dict(case_id=ids, state=last.activity.to_numpy(),
                                 elapsed=((time.to_numpy() - last.timestamp.to_numpy()) / DAY).astype(float),
                                 time=time.to_numpy(), entered=last.timestamp.to_numpy(),
                                 visits=visits.reindex(ids).to_numpy(),
                                 since_start=((last.timestamp - first.reindex(ids)) / DAY).to_numpy(float)))
