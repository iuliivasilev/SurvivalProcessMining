"""Схема 4.2: граф состояний с отдельным поглощающим состоянием цензуры (graphviz)."""
from .adapters import CENSORED, FONT, REASON_LABELS
from .log import censoring_summary


def state_graph(intervals, start, terminal, title=None, reason_labels=None):
    """Схема 4.2: граф состояний с отдельным поглощающим состоянием «Цензура (CENS)».

    Подписи — наблюдаемые вероятности за одно ожидание: ``p наблюд.`` на обычных
    выходах, ``q`` и разбивка по причинам — на пунктирной стрелке в цензуру.
    Сумма выходов из каждого состояния равна единице.
    """
    import graphviz

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
