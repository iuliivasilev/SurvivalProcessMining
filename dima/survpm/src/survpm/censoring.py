"""Цензура поверх готового журнала: ``log.censor(StudyEnd(days=90) + RandomLoss(mean_days=60))``.

Цензура не зависит от процесса, поэтому накладывается после генерации: для каждого объекта
правило даёт момент конца наблюдения, журнал обрезается на нём (:meth:`EventLog.cut`), причина
записывается в ``observation.reason``. Своё правило — подкласс :class:`Censoring` с методом
:meth:`Censoring.times`.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def _starts(log):
    return log.events.groupby("case_id").timestamp.min().reindex(log.case_ids)


class Censoring(ABC):
    """Правило цензуры. Обязательный метод — :meth:`times`."""
    reason = "censored"

    @abstractmethod
    def times(self, log, rng, **kw):
        """Момент конца наблюдения каждого объекта: Series ``case_id → дата``."""

    def apply(self, log, seed=None, **kw):
        """Журнал, обрезанный на :meth:`times`, с причиной правила."""
        table = self.reasons(log, np.random.default_rng(seed), **kw)
        result = log.cut(table["time"])
        obs = result.observation.set_index("case_id")
        cut = obs.reason == "cut"
        obs.loc[cut, "reason"] = table["reason"].reindex(obs.index)[cut]
        result.observation = obs.reset_index()[["case_id", "end", "event", "reason"]]
        return result

    def reasons(self, log, rng, **kw):
        """Таблица ``time, reason`` по ``case_id`` (для суммы правил — самое раннее)."""
        return pd.DataFrame(dict(time=self.times(log, rng, **kw), reason=self.reason))

    def __add__(self, other):
        return AnyOf(self, other)


class AnyOf(Censoring):
    """Наблюдение кончается по самому раннему из правил (``a + b``)."""
    def __init__(self, *rules):
        self.rules = [r for rule in rules for r in (rule.rules if isinstance(rule, AnyOf) else [rule])]

    def times(self, log, rng, **kw):
        return self.reasons(log, rng, **kw)["time"]

    def reasons(self, log, rng, **kw):
        tables = [rule.reasons(log, rng, **kw) for rule in self.rules]
        times = pd.concat([t["time"] for t in tables], axis=1)
        first = times.to_numpy().argmin(axis=1)
        reasons = np.column_stack([t["reason"].to_numpy() for t in tables])
        return pd.DataFrame(dict(time=times.min(axis=1),
                                 reason=reasons[np.arange(len(first)), first]), index=times.index)


class StudyEnd(Censoring):
    """Конец исследования: ``days`` дней от первого события объекта."""
    reason = "study_end"

    def __init__(self, days):
        self.days = float(days)

    def times(self, log, rng, **kw):
        return _starts(log) + self.days*DAY


class RandomLoss(Censoring):
    """Случайная потеря наблюдения: через Exp(``mean_days``) дней от первого события."""
    reason = "lost"

    def __init__(self, mean_days):
        self.mean_days = float(mean_days)

    def times(self, log, rng, **kw):
        start = _starts(log)
        return start + pd.to_timedelta(rng.exponential(self.mean_days, len(start)), unit="D")
