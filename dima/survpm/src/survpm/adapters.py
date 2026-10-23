"""Адаптеры к другим библиотекам: PM4Py (сеть Петри, журнал, схема 4.1) и Ebi (автомат SDFA).

PM4Py импортируется только внутри функций: ядро пакета от него не зависит
(PM4Py распространяется под AGPL v3).
"""
import json

import numpy as np

from .distributions import dist_str
from .log import censoring_summary

CENSORED = "CENS"
REASON_LABELS = {"lost": "потеря наблюдения", "study_end": "конец исследования"}
FONT = "DejaVu Sans"
PM4PY_COLUMNS = {"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"}


def to_pm4py_log(log):
    """Журнал ``case_id, activity, timestamp`` → DataFrame с колонками PM4Py."""
    return log.rename(columns=PM4PY_COLUMNS)


def _decimals(values, total, digits):
    """Вероятности с ``digits`` знаками после запятой, сумма ровно ``total``: Ebi считает точно,
    и уже при знаменателе 10⁴ entropic relevance на Sepsis не вычисляется («cannot compute ER»)."""
    unit = 10 ** digits
    counts = [max(1, round(v * unit)) for v in values]
    counts[int(np.argmax(counts))] += round(total * unit) - sum(counts)
    return [f"{c}/{unit}" for c in counts]


def to_sdfa(edges, start, stop=None, digits=3):
    """Стохастический детерминированный автомат для Ebi (JSON-строка).

    Состояние 0 — до ``start``; метка перехода — состояние, в которое он ведёт.
    ``stop`` — вероятность закончить трассу в состоянии ``{state: q}`` (цензура);
    вероятности округляются до ``digits`` знаков.
    """
    states = sorted({start} | {e["src"] for e in edges} | {e["dst"] for e in edges})
    index = {state: i + 1 for i, state in enumerate(states)}
    transitions = [{"from": 0, "to": index[start], "label": start, "prob": "1"}]
    for src in states:
        out = [e for e in edges if e["src"] == src]
        if not out:
            continue
        keep = 1 - (stop or {}).get(src, 0.)
        probabilities = _decimals([e["p"] * keep for e in out], round(keep, digits), digits)
        transitions += [{"from": index[src], "to": index[e["dst"]], "label": e["dst"], "prob": p}
                        for e, p in zip(out, probabilities)]
    return json.dumps({"initialState": 0, "transitions": transitions})


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
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

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
    from pm4py.objects.petri_net import semantics

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
    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    decorations = decorations or state_net_decorations(net, reason_labels)
    picture = pn_visualizer.apply(net, marking, final_marking,
                                  parameters={"format": "svg", "decorations": decorations})
    for place in net.places:
        if place.name == CENSORED:
            token = "●\n" if marking.get(place, 0) else ""
            picture.node(str(id(place)), label=f"{token}Цензура\nCENS", xlabel="", shape="circle",
                         fixedsize="false", width="1.1", height="1.1", fontname=FONT, fontsize="12")
    return picture
