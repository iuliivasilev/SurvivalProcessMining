"""Рисунки: один интерфейс :class:`Visualizer`, два бэкенда.

* :class:`GraphvizVisualizer` — DFG журнала, сеть модели с rate/p, сеть Петри с ``CENS``,
  граф цензуры (нужны пакет и программа graphviz; сеть Петри — ещё PM4Py);
* :class:`MatplotlibVisualizer` — сеть по слоям от начального состояния и кривые переходов.

:func:`state_graph` — схема 4.2 (граф состояний с поглощающим состоянием цензуры).
"""
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


from typing import Protocol


class Visualizer(Protocol):
    def plot_network(self, network, **kwargs): ...
    def plot_edge_curves(self, model, edge, times=None, **kwargs): ...
    def compare_km(self, model, log, edge): ...
    def network(self, network, **kwargs): ...
    def edge_curves(self, model, src, dst, times, **kwargs): ...


class MatplotlibVisualizer:
    """Optional plots. Import matplotlib only when actually drawing."""
    def network(self, network, ax=None):
        """Сеть по слоям: столбец — кратчайшее число шагов от начального состояния."""
        import numpy as np
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        positions = _layered_positions(network)
        for src, dst in network.edges[['src', 'dst']].itertuples(index=False, name=None):
            a, b = positions[src], positions[dst]
            if src == dst:
                ax.text(a[0], a[1]+.16, '↺', ha='center')
            else:
                ax.annotate('', xy=b, xytext=a,
                            arrowprops=dict(arrowstyle='-|>', mutation_scale=14, shrinkA=24, shrinkB=24,
                                            connectionstyle='arc3,rad=.15', color='#455a64'))
        for state, (x, y) in positions.items():
            ax.text(x, y, str(state), ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white'))
        xs, ys = zip(*positions.values())
        ax.set(xlim=(min(xs)-.6, max(xs)+.6), ylim=(min(ys)-.8, max(ys)+.8))
        ax.axis('off')
        return ax

    def edge_curves(self, model, src, dst, times, X=None, ax=None):
        """Plot P(next destination=dst, waiting time<=t) under minimum-time races."""
        import numpy as np
        from .models.transitions import ExponentialTransitionModel
        if not isinstance(model.transitions, ExponentialTransitionModel):
            raise NotImplementedError('Edge curves currently support exponential transitions only')
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) < 0).any():
            raise ValueError('times must be finite, nonnegative and increasing')
        if dst not in model.network.outgoing(src):
            raise ValueError('Edge is absent from the network')
        rates = model.transition_parameters(src, X)
        total = sum(rates.values())
        values = rates.get(dst, 0.)/total*(-np.expm1(-total*times)) if total else np.zeros_like(times)
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(times, values, label=f'{src}→{dst}')
        ax.set(xlabel='Days', ylabel='Joint transition CDF', ylim=(0, 1))
        return ax

    def plot_network(self, network, **kwargs):
        return self.network(getattr(network, 'network', network), **kwargs)

    def plot_edge_curves(self, model, edge, times=None, **kwargs):
        """Plot an edge tuple on a supplied grid or 101 points over five mean waits.

        With no positive exit rate, the default grid spans zero to five days.
        """
        src, dst = edge
        if times is None:
            import numpy as np
            from .models.transitions import ExponentialTransitionModel
            if not isinstance(model.transitions, ExponentialTransitionModel):
                raise NotImplementedError('Edge curves currently support exponential transitions only')
            total = sum(model.transition_parameters(src, kwargs.get('X')).values())
            times = np.linspace(0., 5/total if total else 5., 101)
        return self.edge_curves(model, src, dst, times, **kwargs)

    def compare_km(self, model, log, edge):
        raise NotImplementedError('KM comparison visualization is not implemented')

    def latent(self, model, **kwargs):
        raise NotImplementedError('Latent visualization is not implemented')


def _layered_positions(network):
    """Координаты состояний: x — слой (BFS от starts), y — порядок внутри слоя; терминалы — справа."""
    depth, frontier = {s: 0 for s in network.starts}, list(network.starts)
    while frontier:
        nxt = []
        for state in frontier:
            for dst in network.outgoing(state):
                if dst not in depth:
                    depth[dst] = depth[state] + 1
                    nxt.append(dst)
        frontier = nxt
    last = max(depth.values(), default=0) + 1
    for state in network.states:
        depth.setdefault(state, last)
    for state in network.terminals:
        depth[state] = max(depth.values()) if depth[state] < max(depth.values()) else depth[state]
    layers = {}
    for state in network.states:
        layers.setdefault(depth[state], []).append(state)
    return {state: (layer * 1.5, (len(states) - 1) / 2 - i)
            for layer, states in layers.items() for i, state in enumerate(sorted(states))}


class GraphvizVisualizer:
    """Рисунки graphviz (``graphviz.Digraph``; в Jupyter отображается сразу, сохранить —
    ``picture.render("file", format="png")``). Импорт graphviz/PM4Py — только при рисовании."""

    def __init__(self, rankdir="LR"):
        self.rankdir = rankdir

    def _graph(self):
        import graphviz
        return graphviz.Digraph(graph_attr={"rankdir": self.rankdir, "fontname": FONT},
                                node_attr={"fontname": FONT, "fontsize": "12"},
                                edge_attr={"fontname": FONT, "fontsize": "10"})

    def _nodes(self, graph, network, visits=None):
        for state in network.states:
            label = state if visits is None else f"{state}\n{visits.get(state, 0)}"
            graph.node(state, label=label, shape="doublecircle" if state in network.terminals else "circle",
                       style="filled", fillcolor="#cde2fb" if state in network.starts else "white")

    def plot_dfg(self, log, network=None):
        """DFG журнала: рёбра с числом наблюдённых переходов, узлы — с числом посещений.
        ``log`` — EventLog после ``prepare(rules)``; ``network`` по умолчанию — DFG этого журнала."""
        from .discovery import discover
        network = network or discover(log)
        iv = log.intervals()
        graph = self._graph()
        self._nodes(graph, network, iv.src.value_counts().add(
            iv.loc[iv.observed].dst.value_counts(), fill_value=0).astype(int).to_dict())
        counts = iv.loc[iv.observed].groupby(["src", "dst"]).size()
        for src, dst in network.edges[["src", "dst"]].itertuples(index=False, name=None):
            graph.edge(src, dst, label=str(int(counts.get((src, dst), 0))))
        return graph

    def plot_network(self, model, log=None, **kwargs):
        """Сеть :class:`Network` или :class:`ProcessModel`. У модели на рёбрах ``p`` (вероятность
        выхода) и ``rate`` (1/день), в узлах — среднее ожидание; с ``log`` — ещё число переходов."""
        network = getattr(model, "network", model)
        graph = self._graph()
        table = model.edges_table().set_index(["src", "dst"]) if hasattr(model, "edges_table") else None
        counts = None
        if log is not None:
            iv = log.intervals()
            counts = iv.loc[iv.observed].groupby(["src", "dst"]).size()
        for state in network.states:
            label = state
            if table is not None and state in table.index.get_level_values(0):
                label += f"\n{table.loc[state].mean_days.iloc[0]:.2g} дн."
            graph.node(state, label=label, shape="doublecircle" if state in network.terminals else "circle",
                       style="filled", fillcolor="#cde2fb" if state in network.starts else "white")
        for src, dst in network.edges[["src", "dst"]].itertuples(index=False, name=None):
            parts = []
            if table is not None and (src, dst) in table.index:
                row = table.loc[(src, dst)]
                parts.append(f"p={row.p:.2f}, rate={row.rate:.3g}")
            if counts is not None:
                parts.append(f"n={int(counts.get((src, dst), 0))}")
            graph.edge(src, dst, label="\n".join(parts))
        return graph

    def plot_petri(self, model, censoring=True, train=None):
        """Сеть Петри модели (сеть состояний с местом ``CENS``), через PM4Py."""
        from .adapters import draw_state_net
        net, initial, final = model.to_pm4py(censoring=censoring, train=train) \
            if hasattr(model, "legacy") else model.to_pm4py(censoring=censoring)
        return draw_state_net(net, initial, final)

    def plot_censoring(self, log):
        """Схема 4.2: наблюдаемые доли выходов и цензуры по журналу после ``prepare(rules)``."""
        from .eventlog import process_bounds
        starts, terminals = process_bounds(log)
        return state_graph(log.intervals(), starts[0], terminals[0])

    def plot_edge_curves(self, model, edge, times=None, **kwargs):
        return MatplotlibVisualizer().plot_edge_curves(model, edge, times, **kwargs)

    def compare_km(self, model, log, edge):
        raise NotImplementedError('KM comparison visualization is not implemented')

    def network(self, network, **kwargs):
        return self.plot_network(network, **kwargs)
