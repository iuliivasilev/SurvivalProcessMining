"""Схемы с правым цензурированием в дополнение к PM4Py.

* Сеть состояний с одним токеном и отдельным поглощающим местом ``CENS``,
  снимок маркировки после цензурированной трассы (схема 4.1).
* Граф состояний с наблюдаемыми вероятностями выходов и цензуры (схема 4.2).
* Оценка распределений времён переходов: выбор семейства по AIC и совместная MLE
  вероятностей выбора и параметров с учётом цензуры.

Основной вход — таблица интервалов с колонками
``src, dst, duration, observed, reason``: завершённый интервал имеет ``observed=True``
и известный ``dst``; цензурированное последнее ожидание — ``observed=False``,
``dst=None`` и непустую причину ``reason``. Её можно получить из журнала PM4Py
функцией :func:`intervals_from_log`.

Пример::

    import pm4py
    import censored_petri as cp

    df = pm4py.format_dataframe(raw, case_id="case_id", activity_key="activity",
                                timestamp_key="timestamp")
    intervals = cp.intervals_from_log(df, terminal="Discharge", min_duration=1 / 1440)
    families, aic_table = cp.select_families(intervals)
    estimates, diagnostics = cp.estimate_network(intervals, families)
    net, im, fm = cp.state_petri_net(intervals, start="Registration", terminal="Discharge",
                                     estimates=estimates)
    marking = cp.replay_censored(net, im, ["Registration", "Examination"], reason="lost")
    cp.draw_state_net(net, marking, fm)                                         # 4.1
    cp.state_graph(intervals, start="Registration", terminal="Discharge")       # 4.2
"""
import math

import graphviz
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.visualization.petri_net import visualizer as pn_visualizer

CENSORED = "CENS"
REASON_LABELS = {"lost": "потеря наблюдения", "study_end": "конец исследования"}
FONT = "DejaVu Sans"


def intervals_from_log(log, terminal, case_id_key="case:concept:name", activity_key="concept:name",
                       timestamp_key="time:timestamp", reason="lost", time_unit="D", min_duration=0.):
    """Интервалы между состояниями из журнала PM4Py.

    Трасса обрезается после первого ``terminal``; соседние повторы активности
    схлопываются (время входа в состояние — первое событие серии). Если трасса
    не дошла до ``terminal``, добавляется цензурированное ожидание в последнем
    состоянии длительностью до последнего события трассы с причиной ``reason``.
    Длительности — в единицах ``pd.Timedelta(1, time_unit)``; меньшие ``min_duration``
    (например, одинаковые timestamp) заменяются на ``min_duration`` — для оценки
    распределений, плотность которых в нуле не определена.
    """
    unit = pd.Timedelta(1, unit=time_unit)
    rows = []
    log = log.sort_values([case_id_key, timestamp_key], kind="stable")
    for case_id, group in log.groupby(case_id_key, sort=False):
        activities = group[activity_key].to_numpy()
        times = pd.to_datetime(group[timestamp_key]).to_numpy()
        ends = np.flatnonzero(activities == terminal)
        if len(ends):
            activities, times = activities[:ends[0] + 1], times[:ends[0] + 1]
        keep = np.r_[True, activities[1:] != activities[:-1]]
        states, entries = activities[keep], times[keep]
        for src, dst, t0, t1 in zip(states, states[1:], entries, entries[1:]):
            rows.append(dict(case_id=case_id, src=src, dst=dst, duration=(t1 - t0) / unit,
                             observed=True, reason=""))
        if states[-1] != terminal:
            rows.append(dict(case_id=case_id, src=states[-1], dst=None,
                             duration=(times[-1] - entries[-1]) / unit, observed=False, reason=reason))
    intervals = pd.DataFrame(rows, columns=["case_id", "src", "dst", "duration", "observed", "reason"])
    intervals["duration"] = intervals["duration"].clip(lower=min_duration)
    return intervals


def censoring_summary(intervals):
    """Наблюдаемые вероятности за одно ожидание в состоянии, по причинам цензуры.

    Колонки: ``n_visits, n_completed, p_completed, q_censored`` и пары
    ``n_<reason>``, ``q_<reason>`` для каждой встретившейся причины.
    """
    reasons = sorted(intervals.loc[~intervals.observed, "reason"].unique())
    rows = []
    for src, group in intervals.groupby("src", sort=True):
        n_visits = len(group)
        counts = group.loc[~group.observed, "reason"].value_counts()
        row = dict(src=src, n_visits=n_visits, n_completed=int(group.observed.sum()))
        row["p_completed"] = row["n_completed"] / n_visits
        row["q_censored"] = 1 - row["p_completed"]
        for reason in reasons:
            row[f"n_{reason}"] = int(counts.get(reason, 0))
            row[f"q_{reason}"] = row[f"n_{reason}"] / n_visits
        rows.append(row)
    return pd.DataFrame(rows)


# --- Оценка распределений (перенесено из petri_net_time_estimation.ipynb, раздел 3) ---

PARAMETERS = {
    "exponential": ("mean",), "lognormal": ("median", "sigma"),
    "gamma": ("shape", "scale"), "weibull": ("shape", "scale"),
    "uniform": ("low", "high"),
}
CANDIDATE_FAMILIES = ("exponential", "lognormal", "gamma", "weibull")


def scipy_args(dist):
    """``(семейство, параметры)`` → ``(rv, args, kwargs)`` для ``scipy.stats``."""
    kind, par = dist
    if kind == "exponential":
        return stats.expon, (), dict(scale=par["mean"])
    if kind == "lognormal":
        return stats.lognorm, (par["sigma"],), dict(scale=par["median"])
    if kind == "gamma":
        return stats.gamma, (par["shape"],), dict(scale=par["scale"])
    if kind == "weibull":
        return stats.weibull_min, (par["shape"],), dict(scale=par["scale"])
    if kind == "uniform":
        return stats.uniform, (), dict(loc=par["low"], scale=par["high"] - par["low"])
    raise ValueError(kind)


def mean_time(dist):
    """Математическое ожидание времени перехода."""
    kind, par = dist
    if kind == "exponential":
        return par["mean"]
    if kind == "lognormal":
        return par["median"] * math.exp(par["sigma"] ** 2 / 2)
    if kind == "gamma":
        return par["shape"] * par["scale"]
    if kind == "weibull":
        return par["scale"] * math.gamma(1 + 1 / par["shape"])
    if kind == "uniform":
        return (par["low"] + par["high"]) / 2
    raise ValueError(kind)


def initial_dist(kind, x):
    """MLE семейства по завершённым длительностям ``x`` без учёта цензуры."""
    if kind == "exponential":
        par = dict(mean=x.mean())
    elif kind == "lognormal":
        par = dict(median=np.exp(np.log(x).mean()), sigma=max(np.log(x).std(), .05))
    elif kind == "gamma":
        shape, _, scale = stats.gamma.fit(x, floc=0)
        par = dict(shape=shape, scale=scale)
    elif kind == "weibull":
        shape, _, scale = stats.weibull_min.fit(x, floc=0)
        par = dict(shape=shape, scale=scale)
    elif kind == "uniform":
        par = dict(low=x.min(), high=x.max())
    else:
        raise ValueError(kind)
    return kind, par


def select_families(intervals, candidates=CANDIDATE_FAMILIES):
    """Семейство каждого ребра по минимальному AIC на завершённых интервалах.

    Ребро с менее чем тремя завершениями получает exponential.
    Возвращает ``(families, aic_table)``: словарь ``(src, dst) → семейство`` и таблицу AIC.
    """
    rows = []
    for (src, dst), group in intervals.loc[intervals.observed].groupby(["src", "dst"]):
        x = group.duration.to_numpy()
        if len(x) < 3:
            rows.append(dict(src=src, dst=dst, n_completed=len(x), family="exponential", aic=np.nan))
            continue
        for kind in candidates:
            rv, args, kwargs = scipy_args(initial_dist(kind, x))
            aic = 2 * len(PARAMETERS[kind]) - 2 * rv.logpdf(x, *args, **kwargs).sum()
            rows.append(dict(src=src, dst=dst, n_completed=len(x), family=kind, aic=aic))
    table = pd.DataFrame(rows)
    table["delta_aic"] = table.aic - table.groupby(["src", "dst"]).aic.transform("min")
    best = table.sort_values("aic", na_position="last").groupby(["src", "dst"]).head(1)
    return {(row.src, row.dst): row.family for row in best.itertuples()}, table


def source_nll(intervals, edges, probabilities, distributions):
    """Отрицательный log-likelihood; у цензуры скрытый конец суммируется."""
    result, survival_terms = 0., []
    censored = intervals.loc[~intervals["observed"], "duration"].to_numpy()
    for edge, p, dist in zip(edges, probabilities, distributions):
        rv, args, kwargs = scipy_args(dist)
        x = intervals.loc[intervals["observed"] & (intervals["dst"] == edge[1]),
                          "duration"].to_numpy()
        result -= len(x) * np.log(p) + rv.logpdf(x, *args, **kwargs).sum()
        survival_terms.append(np.log(p) + rv.logsf(censored, *args, **kwargs))
    if len(censored):
        result -= logsumexp(np.array(survival_terms), axis=0).sum()
    return float(result)


def fit_source(intervals, families, use_censoring=True):
    """Совместная MLE вероятностей выбора и параметров времени выходов одного состояния."""
    completed = intervals.loc[intervals["observed"]]
    source = intervals["src"].iloc[0]
    destinations = sorted(completed["dst"].unique())
    edges = [(source, dst) for dst in destinations]
    counts = completed.groupby("dst").size()
    diagnostic = dict(src=source, success=False, status="insufficient_data",
                      n_completed=len(completed), n_censored=int((~intervals.observed).sum()))
    if not edges or any(counts[edge[1]] < (1 if families[edge] == "exponential" else 3)
                        for edge in edges):
        return [], diagnostic
    data = intervals if use_censoring else completed
    initial = [initial_dist(families[edge], completed.loc[completed.dst == edge[1],
                                                       "duration"].to_numpy()) for edge in edges]
    k = len(edges)
    p0 = counts.reindex(destinations).to_numpy() / len(completed)
    x0 = list(np.log(p0[:-1] / p0[-1]))
    bounds = [(-15., 15.)] * (k - 1)
    for kind, par in initial:
        if kind == "uniform":
            width = par["high"] - par["low"]
            # Положительный старт survival даже если цензура позже максимума завершений.
            cens = data.loc[~data.observed, "duration"]
            extra = max(width * .1, cens.max() - par["high"] + .1) if len(cens) else width * .1
            x0.append(np.log(extra / width))
            bounds.append((-20., 12.))
        else:
            x0.extend(np.log([par[name] for name in PARAMETERS[kind]]))
            bounds.extend([(-5., 5.) if name in ("shape", "sigma") else (-15., 15.)
                           for name in PARAMETERS[kind]])

    def decode(vector):
        probabilities = softmax(np.r_[vector[:k - 1], 0.])
        distributions, offset = [], k - 1
        for kind, par in initial:
            if kind == "uniform":
                high = par["high"] + (par["high"] - par["low"]) * np.exp(vector[offset])
                distributions.append((kind, dict(low=par["low"], high=high)))
                offset += 1
            else:
                names = PARAMETERS[kind]
                values = np.exp(vector[offset:offset + len(names)])
                distributions.append((kind, dict(zip(names, values))))
                offset += len(names)
        return probabilities, distributions

    def objective(vector):
        p, distributions = decode(vector)
        value = source_nll(data, edges, p, distributions) / len(data)
        return value if np.isfinite(value) else 1e100

    attempts = []
    for shift in (0., .15):
        start = np.clip(np.array(x0) + shift, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        attempts.append(minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                                 options=dict(maxiter=500, ftol=1e-10, gtol=1e-6)))
    valid = [r for r in attempts if r.success and np.isfinite(r.fun) and r.fun < 1e90]
    best = min(valid or attempts, key=lambda r: r.fun)
    boundary = any(abs(v - lo) < 1e-4 or abs(v - hi) < 1e-4
                   for v, (lo, hi) in zip(best.x, bounds))
    diagnostic.update(success=bool(valid), status="ok" if valid else "optimizer_failed",
                      nll=float(best.fun * len(data)), message=str(best.message),
                      at_boundary=boundary)
    if not valid:
        return [], diagnostic
    probabilities, distributions = decode(best.x)
    rows = [dict(src=edge[0], dst=edge[1], name=f"{edge[0]}→{edge[1]}",
                 p=float(p), dist=dist, n_completed=int(counts[edge[1]]))
            for edge, p, dist in zip(edges, probabilities, distributions)]
    return rows, diagnostic


def estimate_network(intervals, families, use_censoring=True):
    """Оценки всех состояний: ``(estimates, diagnostics)``.

    ``estimates`` — список словарей ``src, dst, name, p, dist, n_completed``;
    его можно передать в :func:`state_petri_net`. ``use_censoring=False`` —
    наивная оценка только по завершённым интервалам.
    """
    rows, diagnostics = [], []
    for _, group in intervals.groupby("src", sort=True):
        estimated, diagnostic = fit_source(group, families, use_censoring)
        rows.extend(estimated)
        diagnostics.append(diagnostic)
    return rows, pd.DataFrame(diagnostics)


def dist_str(dist):
    """``("weibull", {"shape": 1.4, "scale": 3.7})`` → ``"weibull(shape=1.4, scale=3.7)"``."""
    kind, par = dist
    return kind + "(" + ", ".join(f"{k}={v:g}" for k, v in par.items()) + ")"


def state_petri_net(intervals, start, terminal, estimates=None):
    """Сеть состояний с одним токеном и местом ``CENS``.

    Места — наблюдаемые состояния; переход ``src→dst`` — каждая наблюдаемая пара;
    тихий переход ``src→CENS:<reason>`` — каждая встретившаяся причина цензуры.
    ``estimates`` — необязательный список словарей ``src, dst, p, dist``
    (``p`` — вероятность выбора, ``dist`` — пара ``(семейство, параметры)``).
    В ``properties`` переходов пишутся ``choice_probability``,
    ``observed_probability``, ``time_distribution`` и ``censor_reason``.
    Возвращает ``(net, initial_marking, final_marking)``; начальная фишка — в ``source``.
    """
    net = PetriNet("estimated_state_net")
    completed = intervals.loc[intervals.observed]
    states = set(intervals.src) | set(completed.dst) | {start, terminal}
    if (~intervals.observed).any():
        states.add(CENSORED)
    places = {s: PetriNet.Place(s) for s in sorted(states)}
    source = PetriNet.Place("source")
    net.places.update([source, *places.values()])
    start_transition = PetriNet.Transition("start", start)
    net.transitions.add(start_transition)
    petri_utils.add_arc_from_to(source, start_transition, net)
    petri_utils.add_arc_from_to(start_transition, places[start], net)

    summary = censoring_summary(intervals).set_index("src")
    lookup = {(t["src"], t["dst"]): t for t in estimates or []}
    for (src, dst), count in completed.groupby(["src", "dst"]).size().items():
        tr = PetriNet.Transition(f"{src}→{dst}", dst)
        fitted = lookup.get((src, dst))
        tr.properties.update(choice_probability=fitted["p"] if fitted else None,
                             observed_probability=float(count / summary.loc[src, "n_visits"]),
                             time_distribution=fitted["dist"] if fitted else None)
        net.transitions.add(tr)
        petri_utils.add_arc_from_to(places[src], tr, net)
        petri_utils.add_arc_from_to(tr, places[dst], net)

    reasons = [c[2:] for c in summary.columns if c.startswith("q_") and c != "q_censored"]
    for src, row in summary.iterrows():
        places[src].properties["censoring_probability"] = float(row.q_censored)
        for reason in reasons:
            if row[f"q_{reason}"] == 0:
                continue
            tr = PetriNet.Transition(f"{src}→{CENSORED}:{reason}", None)
            tr.properties.update(censor_reason=reason, choice_probability=None,
                                 observed_probability=float(row[f"q_{reason}"]), time_distribution=None)
            net.transitions.add(tr)
            petri_utils.add_arc_from_to(places[src], tr, net)
            petri_utils.add_arc_from_to(tr, places[CENSORED], net)
    return net, Marking({source: 1}), Marking({places[terminal]: 1})


def replay_censored(net, initial_marking, states, reason):
    """Маркировка после трассы ``states`` (начинается со стартового состояния) и её цензуры.

    Выполняет ``start``, переходы ``states[i]→states[i+1]`` и ``states[-1]→CENS:reason``.
    Ожидаемый результат — одна фишка в ``CENS``.
    """
    transitions = {tr.name: tr for tr in net.transitions}
    names = ["start", *(f"{a}→{b}" for a, b in zip(states, states[1:])), f"{states[-1]}→{CENSORED}:{reason}"]
    marking = initial_marking
    for name in names:
        if name not in transitions:
            raise KeyError(f"В сети нет перехода {name}")
        if not semantics.is_enabled(transitions[name], net, marking):
            raise ValueError(f"Переход {name} не разрешён в маркировке {marking}")
        marking = semantics.execute(transitions[name], net, marking)
    return marking


def state_net_decorations(net, reason_labels=None):
    """Подписи для ``pm4py`` визуализатора: π выбора, p наблюд., распределение; цензура серым."""
    reason_labels = {**REASON_LABELS, **(reason_labels or {})}
    decorations = {p: dict(label="Цензура\nCENS" if p.name == CENSORED else p.name,
                           color="#dce6f1" if p.name == CENSORED else "white")
                   for p in net.places}
    for tr in net.transitions:
        reason = tr.properties.get("censor_reason")
        if reason:
            label = f"τ: {reason_labels.get(reason, reason)}\nq наблюд.={tr.properties['observed_probability']:.3f}"
            decorations[tr] = dict(label=label, color="#eeeeee")
        elif tr.name != "start":
            choice = tr.properties.get("choice_probability")
            label = f"{tr.name}\nπ выбор={choice:.3f}" if choice is not None else f"{tr.name}\nπ не оценена"
            label += f"\np наблюд.={tr.properties['observed_probability']:.3f}"
            if tr.properties.get("time_distribution"):
                label += f"\n{dist_str(tr.properties['time_distribution'])}"
            decorations[tr] = dict(label=label, color="white")
    return decorations


def draw_state_net(net, marking, final_marking, decorations=None, reason_labels=None):
    """Рисунок сети состояний (``graphviz.Digraph``); фишка в ``CENS`` рисуется точкой.

    С начальной маркировкой — схема сети; с результатом :func:`replay_censored` —
    снимок 4.1. Сохранить: ``picture.pipe(format="png")``.
    """
    decorations = decorations or state_net_decorations(net, reason_labels)
    picture = pn_visualizer.apply(net, marking, final_marking,
                                  parameters={"format": "svg", "decorations": decorations})
    for place in net.places:
        if place.name == CENSORED:
            token = "●\n" if marking.get(place, 0) else ""
            picture.node(str(id(place)), label=f"{token}Цензура\nCENS", xlabel="", shape="circle",
                         fixedsize="false", width="1.1", height="1.1", fontname=FONT, fontsize="12")
    return picture


def state_graph(intervals, start, terminal, title=None, reason_labels=None):
    """Схема 4.2: граф состояний с отдельным поглощающим состоянием «Цензура (CENS)».

    Подписи — наблюдаемые вероятности за одно ожидание: ``p наблюд.`` на обычных
    выходах, ``q`` и разбивка по причинам — на пунктирной стрелке в цензуру.
    Сумма выходов из каждого состояния равна единице.
    """
    reason_labels = {**REASON_LABELS, **(reason_labels or {})}
    graph = graphviz.Digraph(graph_attr={"rankdir": "LR", "labelloc": "t", "fontsize": "14"},
                             node_attr={"fontname": FONT, "fontsize": "12"},
                             edge_attr={"fontname": FONT, "fontsize": "10"})
    completed = intervals.loc[intervals.observed]
    for state in sorted(set(intervals.src) | set(completed.dst)):
        graph.node(state, label=state, shape="doublecircle" if state == terminal else "circle",
                   style="filled", fillcolor="#cde2fb" if state == start else "white")
    summary = censoring_summary(intervals).set_index("src")
    for (src, dst), count in completed.groupby(["src", "dst"]).size().items():
        graph.edge(src, dst, label=f"p наблюд.={count / summary.loc[src, 'n_visits']:.1%}")
    reasons = [c[2:] for c in summary.columns if c.startswith("q_") and c != "q_censored"]
    if reasons:
        graph.node(CENSORED, label="Цензура\nCENS", shape="doublecircle", style="filled", fillcolor="#dce6f1")
    for src, row in summary.iterrows():
        if row.q_censored > 0:
            label = f"q={row.q_censored:.1%}"
            if len(reasons) > 1:
                label += "\n" + "; ".join(f"{reason_labels.get(r, r)}={row[f'q_{r}']:.1%}" for r in reasons)
            graph.edge(src, CENSORED, label=label, style="dashed", color="#607d8b", fontcolor="#455a64")
    if title:
        graph.attr(label=title)
    return graph
