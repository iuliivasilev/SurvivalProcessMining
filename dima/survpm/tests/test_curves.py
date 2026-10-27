import numpy as np
import pytest
from scipy import stats

import survpm
from survpm import datasets
from survpm.curves import completion_cdf, residual_cdf

EXP = ("exponential", {"mean": 1.})


def test_chain_is_convolution():
    # A → B → D, обе экспоненты со средним 1: время до D — Gamma(2, 1).
    edges = [dict(src="A", dst="B", p=1., dist=EXP), dict(src="B", dst="D", p=1., dist=EXP)]
    times, states, F = completion_cdf(edges, "D", 15, .01)
    assert np.abs(F[states.index("A")] - stats.gamma.cdf(times, 2)).max() < 1e-4
    assert np.abs(F[states.index("B")] - stats.expon.cdf(times)).max() < 1e-4


def test_self_loop_counts_every_return():
    # Из A с вероятностью ½ в D и ½ обратно в A: геометрическая сумма экспонент — экспонента со средним 2.
    edges = [dict(src="A", dst="D", p=.5, dist=EXP), dict(src="A", dst="A", p=.5, dist=EXP)]
    times, states, F = completion_cdf(edges, "D", 30, .01)
    assert np.abs(F[states.index("A")] - stats.expon.cdf(times, scale=2)).max() < 1e-4


def test_residual_of_exponential_is_memoryless():
    edges = [dict(src="A", dst="B", p=1., dist=EXP), dict(src="B", dst="D", p=1., dist=EXP)]
    grid = completion_cdf(edges, "D", 15, .01)
    assert np.allclose(residual_cdf(edges, "A", 3., *grid), grid[2][grid[1].index("A")])


def test_grid_matches_simulation():
    model = survpm.CensoredStateNet("A", "D")
    model.edges_ = [dict(t) for t in datasets.AD_TRANSITIONS]
    t = np.linspace(0, 100, 101)
    simulated = model.survival(t, n=50_000, seed=1)
    assert np.abs(model.survival(t, method="grid", step=.05) - simulated).max() < .015
