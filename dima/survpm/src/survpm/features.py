"""Признаки трасс: статические и меняющиеся во времени.

Журнал с признаками — три таблицы (:class:`EventLog`):

- ``events``: ``case_id, activity, timestamp`` — шаги процесса, по ним строится сеть;
- ``cases``: индекс ``case_id``, колонки — статические признаки (известны с начала трассы);
- ``measurements``: ``case_id, timestamp, feature, value`` — значения, которые меняются
  во времени (анализы, показания датчиков, сумма заказа…), одна строка — один замер.

Замер не обязан быть шагом процесса. Активности из ``measurement_activities`` в
:func:`read_log` «прозрачны»: их строки уходят из ``events`` и только обновляют признаки.

К интервалам признаки присоединяются на момент входа в состояние (:func:`at_entry`) или
по кускам между замерами (:func:`split`). Берётся последнее известное значение (LOCF):
замер в будущем никогда не попадает в признаки интервала.

Пример::

    log = survpm.read_log(path, "case_id", "activity", "timestamp",
                          measurement_activities=["Examination"])
    model = survpm.CensoredStateNet("Registration", "Discharge").fit(log.events)
    X = survpm.features.at_entry(model.intervals_, log)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .log import COLUMNS, DAY

MEASUREMENT_COLUMNS = ["case_id", "timestamp", "feature", "value"]
_TRUE, _FALSE = {"true", "yes", "1", "t", "y"}, {"false", "no", "0", "f", "n"}


@dataclass
class EventLog:
    """Журнал с признаками: ``events``, ``cases`` и ``measurements`` (см. описание модуля).

    ``last_seen`` — последний момент, когда трасса наблюдалась (Series по ``case_id``), с учётом
    прозрачных замеров; если не задан, считается по ``events`` и ``measurements``. Его
    :meth:`CensoredStateNet.fit` берёт как ``observation_end``: трасса, у которой после
    последнего шага были анализы, цензурируется не раньше последнего анализа.
    """
    events: pd.DataFrame
    cases: pd.DataFrame
    measurements: pd.DataFrame
    last_seen: pd.Series = None

    def __post_init__(self):
        if self.last_seen is None:
            times = pd.concat([self.events[["case_id", "timestamp"]], self.measurements[["case_id", "timestamp"]]])
            self.last_seen = times.groupby("case_id")["timestamp"].max()

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
        return EventLog(self.events, self.cases, _sort_measurements(measurements))  # last_seen пересчитается


def coerce(values):
    """Строки → числа, да/нет → 1/0, иначе строки как категории; пустые строки — пропуски."""
    values = pd.Series(values).replace("", np.nan)
    present = values.dropna()
    if present.empty:
        return values.astype(float)
    lower = present.astype(str).str.strip().str.lower()
    if lower.isin(_TRUE | _FALSE).all() and lower.isin(_TRUE - {"1"} | _FALSE - {"0"}).any():
        return values.map(lambda v: np.nan if pd.isna(v) else float(str(v).strip().lower() in _TRUE))
    numbers = pd.to_numeric(present, errors="coerce")
    if numbers.notna().all():
        return pd.to_numeric(values, errors="coerce").astype(float)
    return values.astype(object)


def _sort_measurements(table):
    return table.sort_values(["case_id", "timestamp"], kind="stable").reset_index(drop=True)


def classify_columns(raw, case_id="case_id", exclude=()):
    """``(static, dynamic)``: колонка статическая, если в каждой трассе у неё не больше одного
    значения; колонки с одним значением на весь журнал (константы) пропускаются."""
    static, dynamic = [], []
    for column in raw.columns:
        if column in exclude:
            continue
        values = raw[column].replace("", np.nan)
        if values.nunique(dropna=True) <= 1:
            continue
        per_case = values.groupby(raw[case_id]).nunique(dropna=True)
        (static if per_case.max() <= 1 else dynamic).append(column)
    return static, dynamic


def from_frame(raw, case_id, activity, timestamp, static=None, dynamic=None, measurement_activities=(),
               exclude=()):
    """Широкая таблица событий (одна строка — событие, признаки — колонки) → :class:`EventLog`.

    ``static``, ``dynamic`` — списки колонок; ``None`` у обоих — разобрать автоматически
    (:func:`classify_columns`). Статическое значение — первое непустое в трассе.
    Динамическое — каждое непустое значение в строке события становится замером в момент
    события. ``measurement_activities`` — активности-замеры: их строки дают только замеры
    и не попадают в ``events``. ``exclude`` — колонки, которые не нужны.
    """
    raw = raw.rename(columns={case_id: "case_id", activity: "activity", timestamp: "timestamp"})
    raw["case_id"] = raw["case_id"].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(raw["timestamp"]):
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values(["case_id", "timestamp"], kind="stable").reset_index(drop=True)
    if static is None and dynamic is None:
        static, dynamic = classify_columns(raw.drop(columns=["activity", "timestamp"]), "case_id",
                                           exclude=set(exclude) | {"case_id"})
    static, dynamic = list(static or []), list(dynamic or [])
    missing = set(static + dynamic) - set(raw.columns)
    if missing:
        raise KeyError(f"Нет колонок {sorted(missing)}; в журнале есть: {list(raw.columns)}")

    cases = pd.DataFrame(index=pd.Index(raw["case_id"].unique(), name="case_id"))
    for column in static:
        values = raw[column].replace("", np.nan)
        cases[column] = coerce(values.groupby(raw["case_id"], sort=False).first().reindex(cases.index))

    parts = []
    for column in dynamic:
        values = raw[column].replace("", np.nan)
        rows = raw.loc[values.notna(), ["case_id", "timestamp"]]
        parts.append(rows.assign(feature=column, value=coerce(values[values.notna()]).to_numpy()))
    measurements = (pd.concat(parts, ignore_index=True) if parts
                    else pd.DataFrame(columns=MEASUREMENT_COLUMNS))
    events = raw.loc[~raw["activity"].isin(list(measurement_activities)), COLUMNS].reset_index(drop=True)
    last_seen = raw.groupby("case_id")["timestamp"].max()
    return EventLog(events, cases, _sort_measurements(measurements[MEASUREMENT_COLUMNS]), last_seen)


def read_log(path, case_id, activity, timestamp, static=None, dynamic=None, measurement_activities=(),
             exclude=(), sep=","):
    """CSV → :class:`EventLog`; аргументы как у :func:`from_frame`.

    Читает всё как строки без подстановки пропусков (как :func:`survpm.read_csv`), типы
    признаков определяет :func:`coerce`.
    """
    raw = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return from_frame(raw, case_id, activity, timestamp, static, dynamic, measurement_activities, exclude)


def _asof(points, measurements, feature):
    """Для точек ``case_id, time`` — последнее значение ``feature`` не позже ``time``, его время
    и число замеров к этому моменту; порядок строк ``points`` сохраняется."""
    right = measurements.loc[measurements["feature"] == feature, ["case_id", "timestamp", "value"]]
    right = right.assign(count=right.groupby("case_id").cumcount() + 1).sort_values("timestamp", kind="stable")
    left = points.reset_index(names="_row").sort_values("time", kind="stable")
    merged = pd.merge_asof(left, right, left_on="time", right_on="timestamp", by="case_id",
                           direction="backward", allow_exact_matches=True)
    return merged.set_index("_row").sort_index()


def _window_counts(points, measurements, feature, window):
    """Число замеров ``feature`` в окне ``(time − window, time]``."""
    right = measurements.loc[measurements["feature"] == feature]
    times = {c: pd.DatetimeIndex(g["timestamp"]).asi8 for c, g in right.groupby("case_id")}
    out = np.zeros(len(points), dtype=int)
    for i, (case, time) in enumerate(zip(points["case_id"], points["time"])):
        t = times.get(case)
        if t is not None:
            out[i] = (np.searchsorted(t, time.value, "right") - np.searchsorted(t, (time - window).value, "right"))
    return out


def values_at(points, log, features=None, window=None):
    """Признаки на моменты ``points`` (колонки ``case_id, time``); индекс ``points`` сохраняется.

    Для статического признака ``f`` — колонка ``f``. Для динамического: ``f`` — последнее значение
    не позже момента (пропуск, если замеров ещё не было), ``f_age`` — сколько дней назад оно
    измерено, ``f_n`` — число замеров к этому моменту; с ``window`` (например ``"1D"``) ещё
    ``f_n_<window>`` — число замеров за окно.
    """
    features = None if features is None else set(features)
    keep = lambda f: features is None or f in features
    points = points[["case_id", "time"]]
    out = pd.DataFrame(index=points.index)
    static = [f for f in log.static if keep(f)]
    if static:
        cases = log.cases.reindex(points["case_id"])[static]
        cases.index = points.index
        out = out.join(cases)
    window = pd.Timedelta(window) if window is not None else None
    for f in log.dynamic:
        if not keep(f):
            continue
        merged = _asof(points, log.measurements, f)
        values = merged["value"]
        numbers = pd.to_numeric(values, errors="coerce")
        out[f] = (numbers if numbers.notna().sum() == values.notna().sum() else values).to_numpy()
        out[f"{f}_age"] = ((merged["time"] - merged["timestamp"]) / DAY).to_numpy()
        out[f"{f}_n"] = merged["count"].fillna(0).astype(int).to_numpy()
        if window is not None:
            out[f"{f}_n_{window_label(window)}"] = _window_counts(points, log.measurements, f, window)
    return out


def window_label(window):
    """``Timedelta`` → короткая подпись для имени колонки: ``1D``, ``12h``, ``30min``."""
    seconds = int(pd.Timedelta(window).total_seconds())
    for unit, size in (("D", 86400), ("h", 3600), ("min", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def at_entry(intervals, log, features=None, window=None):
    """Признаки на момент входа в состояние для каждой строки ``intervals`` (колонка ``enter``).

    Возвращает таблицу с тем же индексом, что у ``intervals``; колонки — как в :func:`values_at`.
    Склеить: ``intervals.join(at_entry(intervals, log))``.
    """
    return values_at(intervals.rename(columns={"enter": "time"}), log, features, window)


def split(intervals, log, features=None, window=None):
    """Интервалы, разрезанные по моментам замеров (формат counting process).

    Каждое пребывание ``[0, duration]`` режется в моменты замеров внутри него; кусок
    ``(t0, t1]`` (дни от входа в ``src``) несёт признаки на свой начальный момент. Только
    последний кусок наблюдаемого интервала имеет ``observed=True`` и ``dst``; остальные — как
    цензура в ``t1`` с отсечением слева в ``t0``. Колонки: ``interval`` (индекс строки
    ``intervals``), ``case_id, src, dst, t0, t1, observed, reason, start`` + признаки.
    """
    rows = []
    cuts = log.measurements.groupby("case_id")["timestamp"].unique()
    for index, row in intervals.iterrows():
        enter = row["enter"]
        times = cuts.get(row["case_id"])
        offsets = np.array([]) if times is None else ((pd.DatetimeIndex(times) - enter) / DAY).to_numpy()
        offsets = np.unique(offsets[(offsets > 0) & (offsets < row["duration"])])
        bounds = [0., *offsets, float(row["duration"])]
        for k, (t0, t1) in enumerate(zip(bounds[:-1], bounds[1:])):
            last = k == len(bounds) - 2
            rows.append(dict(interval=index, case_id=row["case_id"], src=row["src"],
                             dst=row["dst"] if last else None, t0=t0, t1=t1,
                             observed=bool(row["observed"]) and last,
                             reason=row["reason"] if last else "", start=enter + pd.Timedelta(t0, unit="D")))
    pieces = pd.DataFrame(rows)
    return pieces.join(values_at(pieces.rename(columns={"start": "time"}), log, features, window))
