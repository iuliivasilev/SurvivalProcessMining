"""Журнал событий → трассы и интервалы между состояниями.

Журнал — таблица с колонками ``case_id, activity, timestamp`` (одна строка — одно событие).
Время во всех таблицах — в днях.
"""
import numpy as np
import pandas as pd

COLUMNS = ["case_id", "activity", "timestamp"]
DAY = pd.Timedelta(1, unit="D")


def read_csv(path, case_id, activity, timestamp, sep=","):
    """CSV → журнал ``case_id, activity, timestamp``.

    Читает всё как строки и без подстановки пропусков: иначе pandas превращает,
    например, номер трассы ``NA`` в NaN, и трасса теряется.
    """
    raw = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = {case_id, activity, timestamp} - set(raw.columns)
    if missing:
        raise KeyError(f"Нет колонок {sorted(missing)}; в файле есть: {list(raw.columns)}")
    raw = raw[[case_id, activity, timestamp]].set_axis(COLUMNS, axis=1)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    return raw


def prepare(raw, start, terminal, observation_end=None, reason="lost"):
    """Журнал входов в состояния и таблица трасс.

    Трасса начинается с первого ``start`` (трассы без него отбрасываются) и обрезается после
    первого ``terminal``; подряд идущие повторы состояния схлопываются, время входа — первое
    событие серии. ``cases``: ``case_id, entry, end, event, reason, last_state, duration``.
    Конец незавершённой трассы — ``observation_end[case_id]`` (Series), если задан,
    иначе её последняя запись. ``reason`` — причина цензуры: строка или Series по ``case_id``.
    """
    raw = raw.sort_values(["case_id", "timestamp"], kind="stable")
    parts, case_rows = [], []
    for case_id, group in raw.groupby("case_id", sort=False):
        activities = group["activity"].to_numpy()
        starts = np.flatnonzero(activities == start)
        if not len(starts):
            continue
        group, activities = group.iloc[starts[0]:], activities[starts[0]:]
        ends = np.flatnonzero(activities == terminal)
        if len(ends):
            group, activities = group.iloc[:ends[0] + 1], activities[:ends[0] + 1]
        event = len(ends) > 0
        end = group["timestamp"].iloc[-1]
        if not event and observation_end is not None and case_id in observation_end.index:
            end = max(end, observation_end[case_id])
        parts.append(group.iloc[np.flatnonzero(np.r_[True, activities[1:] != activities[:-1]])])
        case_reason = reason.get(case_id, "lost") if isinstance(reason, pd.Series) else reason
        case_rows.append(dict(case_id=case_id, entry=group["timestamp"].iloc[0], end=end, event=int(event),
                              reason="" if event else case_reason, last_state=activities[-1]))
    if not parts:
        raise ValueError(f"Ни одна трасса не содержит start={start!r}")
    log = pd.concat(parts)[COLUMNS].reset_index(drop=True)
    cases = pd.DataFrame(case_rows)
    cases["duration"] = (cases["end"] - cases["entry"]) / DAY
    return log, cases


def intervals(log, cases, min_duration=1 / 1440):
    """Интервалы ``case_id, src, dst, duration, observed, reason`` по журналу из :func:`prepare`.

    Соседние состояния дают завершённый интервал. Незавершённая трасса добавляет
    цензурированное ожидание в последнем состоянии (``dst=None``, ``observed=False``)
    от входа в него до ``cases.end``. Длительности меньше ``min_duration`` (одинаковые
    timestamp) заменяются на ``min_duration``: плотности lognormal, gamma, weibull в нуле
    не определены.
    """
    case = log["case_id"]
    following = log.groupby(case, sort=False)
    nxt = following["activity"].shift(-1)
    gap = (following["timestamp"].shift(-1) - log["timestamp"]) / DAY
    completed = pd.DataFrame(dict(case_id=case, src=log["activity"], dst=nxt, duration=gap,
                                  observed=True, reason=""))[nxt.notna()]
    last = log.loc[following.cumcount(ascending=False) == 0].set_index("case_id")
    open_cases = cases.loc[cases["event"] == 0].set_index("case_id")
    censored = pd.DataFrame(dict(case_id=open_cases.index, src=last.loc[open_cases.index, "activity"].to_numpy(),
                                 dst=None,
                                 duration=((open_cases["end"] - last.loc[open_cases.index, "timestamp"]) / DAY).to_numpy(),
                                 observed=False, reason=open_cases["reason"].to_numpy()))
    result = pd.concat([completed, censored], ignore_index=True)
    result["duration"] = result["duration"].astype(float).clip(lower=min_duration)
    order = {c: i for i, c in enumerate(cases["case_id"])}
    return result.sort_values("case_id", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def censoring_summary(intervals):
    """Наблюдаемые вероятности за одно ожидание в состоянии, по причинам цензуры.

    Колонки: ``src, n_visits, n_completed, p_completed, q_censored`` и пары
    ``n_<reason>``, ``q_<reason>`` для каждой встретившейся причины.
    """
    reasons = sorted(intervals.loc[~intervals["observed"], "reason"].unique())
    rows = []
    for src, group in intervals.groupby("src", sort=True):
        n_visits = len(group)
        counts = group.loc[~group["observed"], "reason"].value_counts()
        row = dict(src=src, n_visits=n_visits, n_completed=int(group["observed"].sum()))
        row["p_completed"] = row["n_completed"] / n_visits
        row["q_censored"] = 1 - row["p_completed"]
        for reason in reasons:
            row[f"n_{reason}"] = int(counts.get(reason, 0))
            row[f"q_{reason}"] = row[f"n_{reason}"] / n_visits
        rows.append(row)
    return pd.DataFrame(rows)
