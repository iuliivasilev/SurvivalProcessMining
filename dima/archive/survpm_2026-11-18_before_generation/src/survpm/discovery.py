"""Discovery protocol and a dependency-free directly-follows backend."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .eventlog import EventLog, process_bounds
from .network import Network


@runtime_checkable
class PetriBackend(Protocol):
    def discover(self, log: EventLog) -> Network: ...


def discover(log: EventLog, backend: PetriBackend = None) -> Network:
    """Сеть по журналу; ``backend`` по умолчанию — :class:`DirectlyFollowsBackend` по ``log.rules``."""
    return (backend or DirectlyFollowsBackend()).discover(log)


@dataclass
class DirectlyFollowsBackend:
    """DFG без внешних зависимостей. ``starts``/``terminals`` по умолчанию берутся из ``log.rules``."""
    starts: list = None
    terminals: list = None

    def discover(self, log: EventLog) -> Network:
        starts, terminals = process_bounds(log, self.starts, self.terminals)
        rules = dict(log.rules or {}, starts=starts, terminals=terminals)
        prepared = log.prepare(rules)
        events = prepared.events
        following = events.groupby('case_id', sort=False).activity.shift(-1)
        edges = events[['activity']].rename(columns={'activity': 'src'}).assign(dst=following).dropna()
        return Network.from_edges(edges, starts, terminals)
