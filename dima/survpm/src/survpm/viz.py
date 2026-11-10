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
        """Draw directed topology on a circle, without invented edge probabilities."""
        import numpy as np
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        states = network.states
        positions = {s: (np.cos(t), np.sin(t)) for s, t in
                     zip(states, np.linspace(0, 2*np.pi, len(states), endpoint=False))}
        for src, dst in network.edges[['src', 'dst']].itertuples(index=False, name=None):
            a, b = positions[src], positions[dst]
            if src == dst:
                ax.text(a[0], a[1]+.16, '↺', ha='center')
            else:
                ax.annotate('', xy=b, xytext=a,
                            arrowprops=dict(arrowstyle='->', shrinkA=18, shrinkB=18))
        for state, (x, y) in positions.items():
            ax.text(x, y, str(state), ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white'))
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect='equal')
        ax.axis('off')
        return ax

    def edge_curves(self, model, src, dst, times, X=None, ax=None):
        """Plot P(next destination=dst, waiting time<=t) under minimum-time races."""
        import numpy as np
        from .transitions import ExponentialTransitionModel
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
            from .transitions import ExponentialTransitionModel
            if not isinstance(model.transitions, ExponentialTransitionModel):
                raise NotImplementedError('Edge curves currently support exponential transitions only')
            total = sum(model.transition_parameters(src, kwargs.get('X')).values())
            times = np.linspace(0., 5/total if total else 5., 101)
        return self.edge_curves(model, src, dst, times, **kwargs)

    def compare_km(self, model, log, edge):
        raise NotImplementedError('KM comparison visualization is not implemented')

    def latent(self, model, **kwargs):
        raise NotImplementedError('Latent visualization is not implemented')
