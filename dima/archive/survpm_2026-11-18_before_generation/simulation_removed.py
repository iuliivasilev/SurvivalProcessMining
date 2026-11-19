"""Симуляция восстановленной сети: выбор выхода по ``p``, время из распределения ребра."""
import numpy as np
import pandas as pd

from .distributions import scipy_args


def simulate(edges, start, terminal, n, seed=None, stop=None, horizon=None, max_steps=1000):
    """Трассы модели ``[(states, times), ...]``, время в днях от входа в ``start``.

    ``edges`` — список словарей ``src, dst, p, dist``. ``stop`` — вероятность цензуры за одно
    ожидание в состоянии ``{state: q}`` (трасса обрывается без события); ``horizon`` — момент
    цензуры каждой трассы (массив длины ``n``). Трасса из состояния без выходов обрывается.
    """
    rng = np.random.default_rng(seed)
    outgoing = {}
    for edge in edges:
        rv, args, kwargs = scipy_args(edge["dist"])
        outgoing.setdefault(edge["src"], []).append((edge["dst"], edge["p"], rv(*args, **kwargs)))
    choice = {src: ([o[0] for o in out], np.array([o[1] for o in out]) / sum(o[1] for o in out),
                    [o[2] for o in out]) for src, out in outgoing.items()}
    traces = []
    for i in range(n):
        state, now, states, times = start, 0., [start], [0.]
        for _ in range(max_steps):
            if state == terminal or state not in choice:
                break
            if stop and rng.random() < stop.get(state, 0.):
                break
            destinations, p, frozen = choice[state]
            j = rng.choice(len(p), p=p)
            now += float(frozen[j].rvs(random_state=rng))
            if horizon is not None and now > horizon[i]:
                break
            state = destinations[j]
            states.append(state)
            times.append(now)
        traces.append((states, times))
    return traces


def completion_times(traces, terminal):
    """Время до ``terminal`` по трассам симуляции; ``inf`` — не дошла."""
    return np.array([times[-1] if states[-1] == terminal else np.inf for states, times in traces])


def traces_to_log(traces, entries, prefix="sim"):
    """Трассы симуляции → журнал ``case_id, activity, timestamp``; ``entries`` — моменты входа."""
    rows = [dict(case_id=f"{prefix}_{i}", activity=state, timestamp=entry + pd.to_timedelta(t, unit="D"))
            for i, ((states, times), entry) in enumerate(zip(traces, entries))
            for state, t in zip(states, times)]
    return pd.DataFrame(rows, columns=["case_id", "activity", "timestamp"])
