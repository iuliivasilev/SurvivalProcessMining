"""Журнал событий :class:`EventLog`: события, статические признаки, замеры и конец наблюдения.

Данные в покое: подготовка трасс (:meth:`EventLog.prepare`), интервалы состояний,
выборка и разбиение по целым объектам. Признаки на моменты времени — в :mod:`survpm.features`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .log import COLUMNS, prepare as prepare_events, intervals as event_intervals

MEASUREMENT_COLUMNS = ["case_id", "timestamp", "feature", "value"]


def _as_list(value):
    return None if value is None else ([value] if isinstance(value, str) else list(value))


def process_bounds(log=None, starts=None, terminals=None):
    """``(starts, terminals)``: явно заданные или из ``log.rules`` после :meth:`EventLog.prepare`.

    Принимает и ``start``/``terminal``, и ``starts``/``terminals`` в правилах.
    """
    rules = (getattr(log, "rules", None) or {})
    starts = _as_list(starts if starts is not None else rules.get("starts", rules.get("start")))
    terminals = _as_list(terminals if terminals is not None else rules.get("terminals", rules.get("terminal")))
    if not starts or not terminals:
        raise ValueError("Задайте starts/terminals или передайте журнал после log.prepare(rules)")
    return starts, terminals


@dataclass
class EventLog:
    """Журнал с признаками: ``events``, ``cases`` и ``measurements`` (см. описание модуля).

    ``last_seen`` — последний момент, когда трасса наблюдалась (Series по ``case_id``), с учётом
    прозрачных замеров; объединяется с максимумом времени из ``events`` и ``measurements``.
    Для цензурированных трасс совпадает с ``observation.end``. Его
    :meth:`CensoredStateNet.fit` берёт как ``observation_end``: трасса, у которой после
    последнего шага были анализы, цензурируется не раньше последнего анализа.
    """
    events: pd.DataFrame
    cases: pd.DataFrame
    measurements: pd.DataFrame
    last_seen: pd.Series = None
    observation: pd.DataFrame = None
    rules: dict = None

    def __post_init__(self):
        times = pd.concat([self.events[["case_id", "timestamp"]], self.measurements[["case_id", "timestamp"]]])
        seen = times.groupby("case_id")["timestamp"].max()
        self.last_seen = (seen if self.last_seen is None
                          else pd.concat([self.last_seen, seen]).groupby(level=0).max())
        if self.observation is None:
            self.observation = self.last_seen.rename("end").rename_axis("case_id").reset_index()
            self.observation = self.observation.assign(event=0, reason="lost")
        else:
            self.observation = self.observation[["case_id", "end", "event", "reason"]].copy()
            ends = self.observation.set_index("case_id")["end"]
            self.last_seen = pd.concat([self.last_seen, ends]).groupby(level=0).max()
            censored = self.observation["event"] == 0
            self.observation.loc[censored, "end"] = self.observation.loc[censored, "case_id"].map(self.last_seen)

    @classmethod
    def from_frame(cls, data, mapping, **parse_options):
        """Parse a wide frame; mapping maps canonical names to source columns."""
        from .features import from_frame
        return from_frame(data, **{name: mapping[name] for name in COLUMNS}, **parse_options)

    def prepare(self, rules):
        """Return prepared traces, retaining observation ends and censoring reasons.

        Completed observation metadata requires an observed configured terminal event.
        """
        rules = dict(rules)
        start = rules.get("starts", rules.get("start"))
        terminal = rules.get("terminals", rules.get("terminal"))
        if start is None or terminal is None:
            raise ValueError("prepare requires start/starts and terminal/terminals")
        observation = self.observation.set_index("case_id")
        events, outcomes = prepare_events(self.events, start, terminal,
                                          observation_end=observation["end"],
                                          reason=observation["reason"],
                                          repeats=rules.get("repeats", "simultaneous"))
        for _, row in outcomes.iterrows():
            if (not row["event"] and row["case_id"] in observation.index
                    and observation.loc[row["case_id"], "event"]):
                raise ValueError(f"Case {row['case_id']!r}: completed observation requires a terminal event")
        ids = outcomes["case_id"]
        obs = outcomes[["case_id", "end", "event", "reason"]]
        return EventLog(events, self.cases.loc[self.cases.index.isin(ids)].copy(),
                        self.measurements.loc[self.measurements.case_id.isin(ids)].copy(),
                        self.last_seen.loc[self.last_seen.index.isin(ids)].copy(), obs, rules)

    def intervals(self):
        """Intervals with the prepared rules and explicit observation end."""
        if self.rules is None:
            raise ValueError("Call prepare(rules) before intervals()")
        if self.events.empty:
            return pd.DataFrame(columns=["case_id", "src", "dst", "duration", "observed", "reason", "enter"])
        outcomes = self.observation.copy()
        outcomes["entry"] = outcomes.case_id.map(self.events.groupby("case_id").timestamp.min())
        return event_intervals(self.events, outcomes, min_duration=self.rules.get("min_duration", 1 / 1440))

    def features_at(self, case_id, t, cutoff=None):
        """Last known features at min(t, cutoff); future observations are excluded."""
        time = pd.to_datetime(t, utc=True)
        if cutoff is not None:
            time = min(time, pd.to_datetime(cutoff, utc=True))
        points = pd.DataFrame({"case_id": [case_id], "time": [time]})
        from .features import values_at
        return values_at(points, self).iloc[0].to_dict()

    @property
    def case_ids(self):
        """Идентификаторы объектов в порядке первого появления в ``events``."""
        return self.events.case_id.drop_duplicates().tolist()

    def select(self, case_ids):
        """Журнал только из объектов ``case_ids``: все таблицы, конец наблюдения и правила."""
        ids = set(case_ids)
        return EventLog(self.events.loc[self.events.case_id.isin(ids)].copy(),
                        self.cases.loc[self.cases.index.isin(ids)].copy(),
                        self.measurements.loc[self.measurements.case_id.isin(ids)].copy(),
                        self.last_seen.loc[self.last_seen.index.isin(ids)].copy(),
                        self.observation.loc[self.observation.case_id.isin(ids)].copy(),
                        None if self.rules is None else dict(self.rules))

    def split(self, method="by_case", **options):
        """Разбиение по целым объектам на ``(train, test)``; таблицы и правила сохраняются.

        ``by_case``: доля ``ratio`` (по умолчанию .8) в train. Без ``seed`` — первые объекты
        по порядку журнала; с ``seed`` — случайно, с одинаковой долей цензурированных
        объектов в train и test. ``temporal`` требует ``cutoff``: train — объекты, наблюдение
        которых закончилось до cutoff, test — начавшиеся после; пересекающие cutoff
        исключаются, наблюдения не обрезаются.
        """
        ids = self.case_ids
        if method == "by_case":
            ratio = options.get("ratio", .8)
            if not 0 <= ratio <= 1:
                raise ValueError("ratio must be between 0 and 1")
            seed = options.get("seed")
            if seed is None:
                boundary = int(len(ids) * ratio)
                groups = (ids[:boundary], ids[boundary:])
            else:
                rng = np.random.default_rng(seed)
                train = []
                for _, group in self._strata(ids):
                    group = rng.permutation(group)
                    train.extend(group[:int(round(len(group) * ratio))])
                chosen = set(train)
                groups = ([i for i in ids if i in chosen], [i for i in ids if i not in chosen])
        elif method == "temporal":
            if "cutoff" not in options:
                raise ValueError("temporal split requires explicit cutoff")
            cutoff = pd.Timestamp(options["cutoff"])
            entry = self.events.groupby("case_id").timestamp.min()
            groups = (self.observation.loc[self.observation.end <= cutoff, "case_id"],
                      entry.index[entry > cutoff])
        else:
            raise ValueError(f"Unknown split method: {method}")
        return tuple(self.select(group) for group in groups)

    def folds(self, k=5, seed=2026):
        """``k`` пар ``(train, test)`` по целым объектам; доля цензурированных объектов в фолдах одинакова."""
        if k < 2:
            raise ValueError("k должно быть не меньше 2")
        rng = np.random.default_rng(seed)
        fold_of = {}
        offset = 0
        for _, group in self._strata(self.case_ids):
            for position, case in enumerate(rng.permutation(group)):
                fold_of[case] = (position + offset) % k
            offset += len(group)
        ids = self.case_ids
        return [(self.select([i for i in ids if fold_of[i] != f]), self.select([i for i in ids if fold_of[i] == f]))
                for f in range(k)]

    def _strata(self, ids):
        event = self.observation.set_index("case_id")["event"].reindex(ids).fillna(0).astype(int)
        return [(value, np.array([i for i in ids if event[i] == value], dtype=object))
                for value in sorted(event.unique())]

    def to_frame(self):
        """Independent copies of the four canonical tables."""
        return {name: getattr(self, name).copy() for name in
                ("events", "cases", "measurements", "observation")}

    @property
    def static(self):
        """Имена статических признаков."""
        return list(self.cases.columns)

    @property
    def dynamic(self):
        """Имена признаков, меняющихся во времени."""
        return sorted(self.measurements["feature"].unique())

    def add_measurements(self, table):
        """Добавить замеры (``case_id, timestamp, feature, value``), например посчитанные отдельно."""
        measurements = pd.concat([self.measurements, table[MEASUREMENT_COLUMNS]], ignore_index=True)
        times = pd.concat([self.last_seen, measurements.groupby("case_id").timestamp.max()])
        last_seen = times.groupby(level=0).max()
        measurements = measurements.sort_values(["case_id", "timestamp"], kind="stable").reset_index(drop=True)
        return EventLog(self.events, self.cases, measurements, last_seen,
                        self.observation, None if self.rules is None else dict(self.rules))
