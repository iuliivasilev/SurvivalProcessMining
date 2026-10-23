"""Синтетический журнал сети A–D с цензурой (перенесено из petri_net_time_estimation.ipynb, раздел 1).

Истинная модель: ``TRANSITIONS`` — выбор выхода по ``p``, время из ``dist`` (в днях).
Цензура: потеря наблюдения (экспонента со средним ``LOSS_MEAN_DAYS``) соревнуется по времени
с выбранным переходом, плюс конец исследования через ``study_days`` после старта.

Пример::

    raw, observation_end = synthetic_ad.generate_log(1000, seed=2026)
"""
import numpy as np
import pandas as pd

START = "A"
TERMINAL = "D"
PLACES = ["A", "B", "C", "D"]
TRANSITIONS = [
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


def simulate_case(rng, horizon=np.inf, max_steps=10_000):
    """``(events, duration, outcome)``: ``events = [(состояние, время от старта), ...]``,
    ``outcome`` — ``"event"``, ``"lost"`` или ``"study_end"``."""
    state, now, events = START, 0., [(START, 0.)]
    for _ in range(max_steps):
        if state == TERMINAL:
            return events, now, "event"
        candidates = [t for t in TRANSITIONS if t["src"] == state]
        transition = candidates[rng.choice(len(candidates), p=[t["p"] for t in candidates])]
        dt = sample_time(transition["dist"], rng)
        loss = rng.exponential(LOSS_MEAN_DAYS) if LOSS_MEAN_DAYS else np.inf
        if min(dt, loss) > horizon - now:
            return events, horizon, "study_end"
        if loss < dt:
            return events, now + loss, "lost"
        now += dt
        state = transition["dst"]
        events.append((state, now))
    raise RuntimeError("превышено max_steps")


def generate_log(n_cases, seed, study_start=pd.Timestamp("2025-01-01", tz="UTC"), enroll_days=60, study_days=90):
    """Журнал ``case_id, activity, timestamp`` и ``observation_end`` — момент цензуры каждой
    незавершённой трассы (Series по ``case_id``); в сам журнал цензура не пишется."""
    rng = np.random.default_rng(seed)
    rows, observation_end = [], {}
    for i in range(n_cases):
        case_id = f"case_{i + 1:04d}"
        entry = rng.uniform(0, enroll_days)
        events, duration, outcome = simulate_case(rng, horizon=study_days - entry)
        rows.extend(dict(case_id=case_id, activity=state,
                         timestamp=study_start + pd.to_timedelta(entry + t, unit="D").round("s"))
                    for state, t in events)
        if outcome != "event":
            observation_end[case_id] = study_start + pd.to_timedelta(entry + duration, unit="D").round("s")
    return pd.DataFrame(rows), pd.Series(observation_end, name="observation_end")
