"""Кривые времени до ``terminal`` без симуляции: свёртка со сдвигом на сетке.

``F_s(t)`` — вероятность дойти до ``terminal`` не позже ``t`` после входа в ``s``:
``F_s(t) = Σ_d p_sd ∫₀ᵗ f_sd(u) F_d(t − u) du``, ``F_terminal ≡ 1``, у тупиков ``F ≡ 0``.
Функция места ``d`` сдвинута на момент прихода в него ``u``. Время перехода округляется
до ближайшего узла сетки ``k·h``, значение ``F[k]`` относится к моменту ``(k + ½)·h``; масса ``[0, h/2)`` — мгновенный переход, поэтому на
каждом шаге решается линейная система ``(I − P₀) F[k] = b[k]``, где ``b[k]`` зависит только от
``F[<k]``. Так циклы не мешают счёту вперёд по ``k``.
"""
import numpy as np
import pandas as pd

from .distributions import scipy_args


def _cdf(dist, x):
    rv, args, kwargs = scipy_args(dist)
    return rv.cdf(x, *args, **kwargs)


def _masses(dist, step, n_steps, elapsed=0.):
    """Масса времени перехода в узлах ``0..n_steps`` (округление до ближайшего узла);
    ``elapsed > 0`` — условно на то, что за ``elapsed`` переход не произошёл (без деления на ``S``)."""
    edges = elapsed + np.r_[0., (np.arange(n_steps + 1) + .5) * step]
    return np.diff(_cdf(dist, edges))


def completion_cdf(edges, terminal, horizon, step):
    """``(times, states, F)``: сетка ``times``, список мест и матрица ``F[место, k]``.

    ``edges`` — список словарей ``src, dst, p, dist``, как ``CensoredStateNet.edges_``.
    """
    n_steps = int(np.ceil(horizon / step))
    states = sorted({e["src"] for e in edges} | {e["dst"] for e in edges} | {terminal})
    index = {s: i for i, s in enumerate(states)}
    src = np.array([index[e["src"]] for e in edges], dtype=int)
    dst = np.array([index[e["dst"]] for e in edges], dtype=int)
    p = np.array([e["p"] for e in edges], dtype=float)
    p = p / np.bincount(src, weights=p, minlength=len(states))[src]
    mass = np.array([_masses(e["dist"], step, n_steps) for e in edges]).reshape(len(edges), n_steps + 1)

    # Переходы из terminal не рассматриваются: процесс там останавливается.
    active = src != index[terminal]
    src, dst, p, mass = src[active], dst[active], p[active], mass[active]
    instant = np.zeros((len(states), len(states)))
    np.add.at(instant, (src, dst), p * mass[:, 0])
    free = np.array([s != index[terminal] for s in range(len(states))])
    solve = np.linalg.inv(np.eye(free.sum()) - instant[np.ix_(free, free)])
    to_terminal = instant[free, index[terminal]]

    F = np.zeros((len(states), n_steps + 1))
    F[index[terminal]] = 1.
    weighted = p[:, None] * mass[:, 1:]
    for k in range(n_steps + 1):
        lagged = (weighted[:, :k] * F[dst, k - 1::-1]).sum(1) if k else np.zeros(len(src))
        b = np.bincount(src, weights=lagged, minlength=len(states))[free] + to_terminal
        F[free, k] = solve @ b
    # Округлённое время ≤ k означает истинное ≤ (k + ½)·h, поэтому узлы сдвинуты на полшага.
    return (np.arange(n_steps + 1) + .5) * step, states, np.clip(F, 0., 1.)


def residual_cdf(edges, state, elapsed, times, states, F):
    """``F(t | s, a)`` — дойти до ``terminal`` за ``t`` после текущего момента, если токен уже
    ``elapsed`` дней в ``state``. Выбор выхода пересчитан: ``p_sd(a) ∝ p_sd S_sd(a)``,
    ожидание — остаток ``f_sd(a + u) / S_sd(a)``. ``times, states, F`` — из :func:`completion_cdf`."""
    index = {s: i for i, s in enumerate(states)}
    out = [e for e in edges if e["src"] == state]
    if not out:
        return F[index[state]].copy()
    step, n_steps = times[1] - times[0], len(times) - 1
    alive = sum(e["p"] * (1 - _cdf(e["dist"], elapsed)) for e in out)
    result = np.zeros(n_steps + 1)
    for e in out:
        result += e["p"] * np.convolve(_masses(e["dist"], step, n_steps, elapsed), F[index[e["dst"]]])[:n_steps + 1]
    return np.clip(result / alive, 0., 1.) if alive > 0 else result


def curves_table(times, states, F):
    """Длинная таблица ``state, time_days, cdf, survival``."""
    return pd.DataFrame(dict(state=np.repeat(states, len(times)), time_days=np.tile(times, len(states)),
                             cdf=F.ravel(), survival=1 - F.ravel()))
