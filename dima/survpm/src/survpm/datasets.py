"""Наборы данных: синтетическая сеть A–D с известной истиной и журнал Sepsis.

Синтетика перенесена из ``petri_net_time_estimation.ipynb``, раздел 1.

Истинная модель: ``AD_TRANSITIONS`` — выбор выхода по ``p``, время из ``dist`` (в днях).
Цензура: потеря наблюдения (экспонента со средним ``LOSS_MEAN_DAYS``) соревнуется по времени
с выбранным переходом, плюс конец исследования через ``study_days`` после старта.

Пример::

    raw, censoring = datasets.generate_ad(1000, seed=2026)
    raw = datasets.load_sepsis("Med_Eremina/Sepsis Cases - Event Log_grouped.csv")
"""
import numpy as np
import pandas as pd

from .log import read_csv

AD_START = "A"
AD_TERMINAL = "D"
AD_TRANSITIONS = [
    dict(name="A→B", src="A", dst="B", p=0.7, dist=("exponential", {"mean": 2.0})),
    dict(name="A→C", src="A", dst="C", p=0.3, dist=("lognormal", {"median": 5.0, "sigma": 0.5})),
    dict(name="B→A", src="B", dst="A", p=0.2, dist=("uniform", {"low": 1.0, "high": 3.0})),
    dict(name="B→C", src="B", dst="C", p=0.5, dist=("gamma", {"shape": 2.0, "scale": 1.5})),
    dict(name="B→D", src="B", dst="D", p=0.3, dist=("weibull", {"shape": 1.5, "scale": 10.0})),
    dict(name="C→B", src="C", dst="B", p=0.4, dist=("exponential", {"mean": 3.0})),
    dict(name="C→D", src="C", dst="D", p=0.6, dist=("weibull", {"shape": 0.8, "scale": 6.0})),
]
LOSS_MEAN_DAYS = 60


def sample_time(dist, rng):
    kind, par = dist
    if kind == "exponential":
        return rng.exponential(par["mean"])
    if kind == "lognormal":
        return rng.lognormal(np.log(par["median"]), par["sigma"])
    if kind == "gamma":
        return rng.gamma(par["shape"], par["scale"])
    if kind == "weibull":
        return par["scale"] * rng.weibull(par["shape"])
    if kind == "uniform":
        return rng.uniform(par["low"], par["high"])
    raise ValueError(kind)


def simulate_case(rng, horizon=np.inf, transitions=AD_TRANSITIONS, start=AD_START, terminal=AD_TERMINAL,
                  loss_mean_days=LOSS_MEAN_DAYS, max_steps=10_000):
    """``(events, duration, outcome)``: ``events = [(состояние, время от старта), ...]``,
    ``outcome`` — ``"event"``, ``"lost"`` или ``"study_end"``.

    Выход выбирается по ``p``; потеря наблюдения (экспонента со средним ``loss_mean_days``,
    ``None`` — потерь нет) соревнуется по времени с выбранным переходом."""
    state, now, events = start, 0., [(start, 0.)]
    for _ in range(max_steps):
        if state == terminal:
            return events, now, "event"
        candidates = [t for t in transitions if t["src"] == state]
        transition = candidates[rng.choice(len(candidates), p=[t["p"] for t in candidates])]
        dt = sample_time(transition["dist"], rng)
        loss = rng.exponential(loss_mean_days) if loss_mean_days else np.inf
        if min(dt, loss) > horizon - now:
            return events, horizon, "study_end"
        if loss < dt:
            return events, now + loss, "lost"
        now += dt
        state = transition["dst"]
        events.append((state, now))
    raise RuntimeError("превышено max_steps")


def generate_ad(n_cases, seed, transitions=AD_TRANSITIONS, start=AD_START, terminal=AD_TERMINAL,
                loss_mean_days=LOSS_MEAN_DAYS, study_start=pd.Timestamp("2025-01-01", tz="UTC"),
                enroll_days=60, study_days=90):
    """Журнал ``case_id, activity, timestamp`` и таблица цензуры ``censoring`` (индекс ``case_id``,
    колонки ``end`` — момент цензуры, ``reason`` — ``lost`` или ``study_end``) для незавершённых трасс.
    Объект входит в исследование равномерно в первые ``enroll_days`` дней, наблюдение кончается
    через ``study_days`` дней после старта исследования. В журнал цензура не пишется::

        raw, censoring = generate_ad(1000, seed=2026)
        log, cases = survpm.prepare(raw, "A", "D", censoring["end"], censoring["reason"])
    """
    rng = np.random.default_rng(seed)
    rows, censoring = [], {}
    for i in range(n_cases):
        case_id = f"case_{i + 1:04d}"
        entry = rng.uniform(0, enroll_days)
        events, duration, outcome = simulate_case(rng, study_days - entry, transitions, start, terminal, loss_mean_days)
        rows.extend(dict(case_id=case_id, activity=state,
                         timestamp=study_start + pd.to_timedelta(entry + t, unit="D").round("s"))
                    for state, t in events)
        if outcome != "event":
            censoring[case_id] = dict(end=study_start + pd.to_timedelta(entry + duration, unit="D").round("s"),
                                      reason=outcome)
    return pd.DataFrame(rows), pd.DataFrame.from_dict(censoring, orient="index", columns=["end", "reason"])


SEPSIS_START, SEPSIS_TERMINAL = "Registration", "Discharge"


def load_sepsis(path, state_column="event_group"):
    """Журнал Sepsis (``Med_Eremina/Sepsis Cases - Event Log_grouped.csv``); состояния — группы событий."""
    return read_csv(path, "case_id", state_column, "time:timestamp")
