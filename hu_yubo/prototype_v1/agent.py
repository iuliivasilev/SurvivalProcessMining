"""Pair selection and a shared linear ranking model. Only the answer callback sees the net."""
from dataclasses import dataclass
from itertools import combinations
import math
import random


@dataclass(frozen=True)
class Candidate:
    name: str
    baseline: float
    contrast: float
    ranked: bool  # False for an observed comparison reference.


@dataclass(frozen=True)
class Weights:
    baseline: float = 1.0
    contrast: float = 0.0

    def score(self, candidate):
        return self.baseline * candidate.baseline + self.contrast * candidate.contrast

    def prefer(self, winner, loser):
        # One pairwise logistic-loss step with learning rate 1.
        margin = self.score(winner) - self.score(loser)
        step = math.exp(-max(margin, 0)) / (1 + math.exp(-abs(margin)))
        return Weights(self.baseline + step * (winner.baseline - loser.baseline),
                       self.contrast + step * (winner.contrast - loser.contrast))


@dataclass(frozen=True)
class Question:
    left: str
    right: str
    answer: str
    weights: Weights


@dataclass(frozen=True)
class Fit:
    weights: Weights
    history: tuple[Question, ...]


def ranking(candidates, weights):
    return sorted((c for c in candidates if c.ranked), key=lambda c: (-weights.score(c), c.name))


def learn(candidates, budget, compare, *, policy="active", seed=7):
    """Choose each unordered pair at most once. Only strict preferences update the model."""
    if budget < 0 or policy not in ("active", "random"):
        raise ValueError("Expected a nonnegative budget and active or random policy")
    ordered = sorted(candidates, key=lambda c: c.name)
    if len({c.name for c in ordered}) != len(ordered):
        raise ValueError("Candidate names must be unique at this anchor")
    remaining = [(a, b) for a, b in combinations(ordered, 2) if a.ranked or b.ranked]
    rng = random.Random(seed)
    weights = Weights()
    history = []
    for _ in range(min(budget, len(remaining))):
        if policy == "random":
            pair = rng.choice(remaining)
        else:
            pair = min(remaining, key=lambda pair: abs(weights.score(pair[0]) - weights.score(pair[1])))
        remaining.remove(pair)
        left, right = pair
        answer = compare(left.name, right.name)
        if answer == "left":
            weights = weights.prefer(left, right)
        elif answer == "right":
            weights = weights.prefer(right, left)
        elif answer not in ("tie", "unknown"):
            raise ValueError(f"Unknown answer: {answer!r}")
        history.append(Question(left.name, right.name, answer, weights))
    return Fit(weights, tuple(history))
