"""Exact answers for the teacher's finite, single-token Petri net and firing policy."""
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Case:
    case_id: str
    labels: tuple[str, ...]
    times: tuple[float, ...]  # seconds


def at_anchor(labels, anchor):
    """Prefix before the first observed anchor, and its immediately following event."""
    if anchor not in labels:
        return None
    index = labels.index(anchor)
    if index + 1 == len(labels):
        return None
    return labels[:index], labels[index + 1]


@dataclass(frozen=True)
class _Route:
    labels: tuple[str, ...]
    times: tuple[float, ...]
    mass: float


class TeacherNet:
    def __init__(self, p=0.3):
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        net = json.loads(Path(__file__).with_name("teacher-net.json").read_text())
        # The original experiment specifies firing probabilities outside the net.
        weights = {"P": 0.5, "B": 0.5, "F_P": p, "OK_P": 1 - p}
        outgoing = defaultdict(list)
        for tr in net["transitions"]:
            (before, consumed), = tr["pre"].items()
            (after, produced), = tr["post"].items()
            earliest, latest = tr["interval_ms"]
            if consumed != 1 or produced != 1 or earliest != latest:
                raise ValueError("This fixture requires one token and fixed firing delays")
            outgoing[before].append((after, tr["label"], earliest, weights.get(tr["name"], 1.0)))

        def walk(place, elapsed_ms, labels, times, mass):
            if place not in outgoing:
                yield _Route(labels, times, mass)
                return
            transitions = outgoing[place]
            if not math.isclose(sum(tr[3] for tr in transitions), 1.0):
                raise ValueError(f"Invalid firing policy at {place}")
            for after, label, delay, probability in transitions:
                if probability > 0:
                    stamp = elapsed_ms + delay
                    yield from walk(after, stamp, labels + (label,), times + (stamp / 1000,),
                                    mass * probability)

        start, = net["initial"]
        self._routes = tuple(walk(start, 0, (), (), 1.0))

    def sample(self, n, seed, partition):
        """Return only observations, with N/2 cases per branch as in the original policy."""
        if n < 2 or n % 2:
            raise ValueError("The number of cases must be positive and even")
        rng = random.Random(f"{seed}/{partition}")
        branches = ["P", "B"] * (n // 2)
        rng.shuffle(branches)
        cases = []
        for i, branch in enumerate(branches):
            routes = [r for r in self._routes if branch in r.labels]
            route = rng.choices(routes, weights=[r.mass for r in routes])[0]
            cases.append(Case(f"{partition}-{i}", route.labels, route.times))
        return cases

    def _matching(self, anchor, event):
        for route in self._routes:
            observation = at_anchor(route.labels, anchor)
            if observation is not None:
                prefix, following = observation
                if event in prefix:
                    yield route, prefix, following

    def risk(self, anchor, event):
        matched = list(self._matching(anchor, event))
        total = sum(route.mass for route, _, _ in matched)
        if total == 0:
            return None
        return sum(r.mass for r, _, following in matched if following == "F") / total

    def compare(self, anchor, left, right):
        a, b = self.risk(anchor, left), self.risk(anchor, right)
        if a is None or b is None:
            return "unknown"
        if math.isclose(a, b, rel_tol=1e-12):
            return "tie"
        return "left" if a > b else "right"

    def counterexample(self, anchor, event):
        for route, _, following in self._matching(anchor, event):
            if following != "F":
                return route.labels
        return None

    def context_support(self, anchor, event, context):
        """Existence of an earlier context, and of its absence, before the specified event."""
        observed = {context in prefix[:prefix.index(event)]
                    for _, prefix, _ in self._matching(anchor, event)}
        return True in observed, False in observed
