"""Структура процесса, независимая от оценивания и исполнения."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Network:
    edges: pd.DataFrame
    starts: list
    terminals: list

    @classmethod
    def from_edges(cls, edges, starts, terminals):
        table = edges[['src', 'dst']].copy() if isinstance(edges, pd.DataFrame) else pd.DataFrame(edges, columns=['src', 'dst'])
        table = table.drop_duplicates().reset_index(drop=True)
        vertices = set(table.src) | set(table.dst) | set(starts) | set(terminals)
        if not starts or not terminals or not set(starts) <= vertices or table.isna().any().any():
            raise ValueError('Задайте допустимые начальные и конечные состояния')
        if table.src.isin(terminals).any():
            raise ValueError('Конечные состояния не имеют исходящих переходов')
        return cls(table, list(starts), list(terminals))

    @classmethod
    def discover(cls, log, starts=None, terminals=None):
        """DFG по журналу; без ``starts``/``terminals`` они берутся из ``log.rules``."""
        from .eventlog import EventLog, process_bounds
        from .log import prepare, intervals
        if isinstance(log, EventLog):
            starts, terminals = process_bounds(log, starts, terminals)
            rules = dict(log.rules or {}, starts=starts, terminals=terminals)
            iv = log.prepare(rules).intervals()
        else:
            events, cases = prepare(log, starts, terminals)
            iv = intervals(events, cases)
        return cls.from_edges(iv.loc[iv.observed, ['src', 'dst']], starts, terminals)

    @property
    def states(self):
        return sorted(set(self.edges.src) | set(self.edges.dst) | set(self.starts) | set(self.terminals))

    def outgoing(self, state):
        return self.edges.loc[self.edges.src == state, 'dst'].tolist()

    def validate(self):
        reachable, frontier = set(), list(self.starts)
        while frontier:
            s = frontier.pop()
            if s not in reachable:
                reachable.add(s)
                frontier.extend(self.outgoing(s))
        can_finish, frontier = set(), list(self.terminals)
        while frontier:
            s = frontier.pop()
            if s not in can_finish:
                can_finish.add(s)
                frontier.extend(self.edges.loc[self.edges.dst == s, 'src'])
        rows = []
        for s in self.states:
            if s not in reachable:
                rows.append(dict(state=s, issue='unreachable'))
            if s not in can_finish:
                rows.append(dict(state=s, issue='no_terminal_path'))
            if s not in self.terminals and not self.outgoing(s):
                rows.append(dict(state=s, issue='dead_end'))
        return pd.DataFrame(rows, columns=['state', 'issue'])

    def plot(self, visualizer=None, **kwargs):
        """Delegate drawing to an optional visualizer."""
        if visualizer is None:
            from .viz import MatplotlibVisualizer
            visualizer = MatplotlibVisualizer()
        renderer = getattr(visualizer, 'plot_network', None)
        if renderer is None:
            renderer = visualizer.network
        return renderer(self, **kwargs)
