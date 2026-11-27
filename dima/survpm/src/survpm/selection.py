"""Правило выбора перехода: базовый класс :class:`Selector`.

Правило задаётся весами выходов — :meth:`Selector.weights` (единственный обязательный метод).
По этим весам генератор разыгрывает один переход (:meth:`Selector.select`, готов в базе), а
прогноз суммирует все продолжения. Время берётся из закона модели для выбранного выхода::

    class NoSurgery(survpm.Selector):
        def weights(self, X, steps, **kw):
            table = survpm.SampleSelector().weights(X, steps)
            table.loc[table.dst == "Surgery", "weight"] = 0     # запретить переход
            return table                                        # база нормирует веса

Готовые правила: :class:`SampleSelector` (по закону, по умолчанию), :class:`MostLikelySelector`,
:class:`BestSelector`, :class:`WorstSelector`.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

WEIGHT_COLUMNS = ["case_id", "dst", "weight"]
MOVE_COLUMNS = ["case_id", "dst", "dt"]


def law_weights(X, steps):
    """Веса по закону модели: p_d каждого шага (таблица ``case_id, dst, weight``)."""
    rows = [(cid, d, float(p)) for cid, step in zip(X.case_id, steps) for d, p in zip(step.dst, step.p)]
    if any(np.isnan(w) for _, _, w in rows):
        raise NotImplementedError("Модель не даёт вероятностей p_d (только розыгрыш шага): "
                                  "используйте ModelGenerator или SimulationForecaster")
    return pd.DataFrame(rows, columns=WEIGHT_COLUMNS)


class Selector(ABC):
    """Правило выбора. ``time="sample"`` — время разыгрывается из закона, ``"mean"`` — среднее."""

    def __init__(self, time="sample"):
        if time not in ("sample", "mean"):
            raise ValueError('time: "sample" или "mean"')
        self.time = time

    @abstractmethod
    def weights(self, X, steps, **kw):
        """Веса выходов: таблица ``case_id, dst, weight`` (по строкам X; нормировать не обязательно)."""

    def weight_matrix(self, X, steps, **kw):
        """Веса в порядке ``step.dst`` для каждой строки X (нормированные; нули, если выбирать нечего)."""
        table = pd.DataFrame(self.weights(X, steps, **kw))
        if list(table.columns[:3]) != WEIGHT_COLUMNS and not set(WEIGHT_COLUMNS) <= set(table.columns):
            raise ValueError(f"weights должен вернуть колонки {WEIGHT_COLUMNS}")
        lookup = {(c, d): w for c, d, w in table[WEIGHT_COLUMNS].itertuples(index=False)}
        result = []
        for cid, step in zip(X.case_id, steps):
            w = np.array([lookup.get((cid, d), 0.) for d in step.dst], float)
            if (w < 0).any() or not np.isfinite(w).all():
                raise ValueError("Веса должны быть конечными и неотрицательными")
            result.append(w / w.sum() if w.sum() > 0 else w)
        return result

    def select(self, X, steps, rng, **kw):
        """Один переход на строку: таблица ``case_id, dst, dt``; ``dst=None`` — выхода нет.

        ``rng`` — поток случайных чисел каждой строки (список той же длины, что X).
        """
        W = self.weight_matrix(X, steps, **kw)
        rows = []
        for cid, step, w, r in zip(X.case_id, steps, W, rng):
            if step.empty or w.sum() <= 0:
                rows.append((cid, None, np.inf))
                continue
            j = int(np.argmax(w)) if w.max() >= 1 - 1e-12 else int(r.choice(len(w), p=w))
            dt = step.mean_time(j) if self.time == "mean" else step.sample_time(j, r)
            rows.append((cid, step.dst[j], float(dt)))
        return pd.DataFrame(rows, columns=MOVE_COLUMNS)


class SampleSelector(Selector):
    """Розыгрыш по закону модели: веса — p_d. С ``time="sample"`` шаг разыгрывается
    :meth:`~survpm.steps.StepDistribution.sample` целиком (прежние числа при том же seed)."""

    def weights(self, X, steps, **kw):
        return law_weights(X, steps)

    def select(self, X, steps, rng, **kw):
        if self.time != "sample":
            return super().select(X, steps, rng, **kw)
        rows = []
        for cid, step, r in zip(X.case_id, steps, rng):
            dst, dt = (None, np.inf) if step.empty else step.sample(r)
            rows.append((cid, dst, float(dt)))
        return pd.DataFrame(rows, columns=MOVE_COLUMNS)


def _one_hot(X, steps, score, best, target=None):
    """Вес 1 у выхода с наибольшим (``best``) или наименьшим значением ``score``.

    При равных значениях: для лучшего — сам ``target``, затем переход в другое состояние, затем
    больший p; для худшего — не ``target``, затем переход в другое состояние, затем больший p.
    Петля s → s при равенстве не выбирается: иначе объект мог бы остаться в s навсегда.
    """
    rows = []
    for (_, row), step in zip(X.iterrows(), steps):
        allowed = [j for j, p in enumerate(step.p) if p > 0]
        if not allowed:
            continue
        values = np.round(np.asarray(score(row, step, allowed), float), 12)

        def key(i):
            j, d = allowed[i], step.dst[allowed[i]]
            hit = (d == target) if best else (d != target)
            return (values[i] if best else -values[i], hit, d != row.state, step.p[j])
        chosen = allowed[max(range(len(allowed)), key=key)]
        rows.extend((row.case_id, d, float(j == chosen)) for j, d in enumerate(step.dst))
    return pd.DataFrame(rows, columns=WEIGHT_COLUMNS)


class MostLikelySelector(Selector):
    """Выход с наибольшим p_d; время по умолчанию — среднее."""

    def __init__(self, time="mean"):
        super().__init__(time)

    def weights(self, X, steps, **kw):
        return _one_hot(X, steps, lambda row, step, allowed: step.p[allowed], True)


def absorption(model, row, target):
    """P(когда-нибудь дойти до ``target``) из каждого состояния при признаках строки ``row``.

    Считается по вероятностям выходов p_d (время не важно): h = P h, h[target] = 1,
    h[другой терминал] = 0. Признаки после «сейчас» считаются неизменными.
    """
    from .streamdata import features_of
    states = list(model.network.states)
    terminals = set(model.network.terminals)
    inner = [s for s in states if s not in terminals]
    X = pd.DataFrame([dict(features_of(row), case_id=s, state=s, elapsed=0.) for s in inner])
    steps = model.predict(X) if inner else []
    index = {s: i for i, s in enumerate(inner)}
    A, b = np.eye(len(inner)), np.zeros(len(inner))
    for s, step in zip(inner, steps):
        for d, p in zip(step.dst, step.p):
            if d in index:
                A[index[s], index[d]] -= p
            elif d == target:
                b[index[s]] += p
    h = np.linalg.lstsq(A, b, rcond=None)[0] if inner else b
    result = {s: float(h[index[s]]) for s in inner}
    result.update({k: float(k == target) for k in terminals})
    return result


class _TargetSelector(Selector):
    best = True

    def __init__(self, target, model=None, time="sample"):
        super().__init__(time)
        self.target, self.model, self._cache = target, model, {}

    def weights(self, X, steps, model=None, **kw):
        model = model or self.model
        if model is None:
            raise ValueError(f"{type(self).__name__}: нужна модель (model=...)")
        from .streamdata import features_of

        def score(row, step, allowed):
            key = (id(model), tuple(sorted((k, str(v)) for k, v in features_of(row).items())))
            if key not in self._cache:
                self._cache[key] = absorption(model, row, self.target)
            h = self._cache[key]
            return np.array([h.get(step.dst[j], 0.) for j in allowed])
        return _one_hot(X, steps, score, self.best, self.target)


class BestSelector(_TargetSelector):
    """Выход с наибольшей вероятностью когда-нибудь дойти до ``target`` (дальше — по закону).

    Если терминал один, эта вероятность обычно равна 1 у всех выходов; тогда при равенстве
    выбирается сам ``target``, затем переход в другое состояние (см. :func:`_one_hot`)."""
    best = True


class WorstSelector(_TargetSelector):
    """Выход с наименьшей вероятностью исхода ``target`` (дальше — по закону)."""
    best = False
