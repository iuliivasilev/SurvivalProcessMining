"""Discovery protocol and a dependency-free directly-follows backend."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .features import EventLog
from .network import Network


@runtime_checkable
class PetriBackend(Protocol):
    def discover(self, log: EventLog) -> Network: ...


def discover(log: EventLog, backend: PetriBackend) -> Network:
    return backend.discover(log)


@dataclass
class DirectlyFollowsBackend:
    starts: list
    terminals: list

    def discover(self, log: EventLog) -> Network:
        rules = dict(log.rules or {}, starts=self.starts, terminals=self.terminals)
        prepared = log.prepare(rules)
        events = prepared.events
        following = events.groupby('case_id', sort=False).activity.shift(-1)
        edges = events[['activity']].rename(columns={'activity': 'src'}).assign(dst=following).dropna()
        return Network.from_edges(edges, self.starts, self.terminals)
