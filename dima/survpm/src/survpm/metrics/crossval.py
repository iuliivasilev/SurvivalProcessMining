"""Кросс-валидация по трассам: k фолдов и набор метрик, который можно менять.

Метрика — функция ``metric(fold)`` от :class:`Fold`; она возвращает строку (``dict``) или список
строк. Строка с колонкой ``model`` относится к модели (``naive``, ``censor_aware`` или своей).
Набор метрик меняется через :meth:`CrossValidator.set_metrics`::

    def median_days(model, fold):
        return float(np.median(model.completion_times(2000, fold.seed)))

    cv = crossval.CrossValidator("A", "D").set_metrics(median=crossval.per_model(median_days), level2=None)
    results = cv.run(raw, censoring["end"], censoring["reason"])
    crossval.summarize(results["median"])
"""
from dataclasses import dataclass
from functools import wraps
from numbers import Number

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..log import intervals as make_intervals, prepare
from ..model import CensoredStateNet
from .process import level1
from .stochastic import level2
from .temporal import level3


@dataclass
class Fold:
    """Всё, что известно метрике об одном фолде: обученные модели и обучающая/тестовая части."""
    index: int
    seed: int
    n_sim: int
    models: dict
    train_log: pd.DataFrame
    test_log: pd.DataFrame
    train_cases: pd.DataFrame
    test_cases: pd.DataFrame
    train_intervals: pd.DataFrame
    test_intervals: pd.DataFrame

    @property
    def complete_ids(self):
        """Тестовые трассы, дошедшие до терминала."""
        return self.test_cases.loc[self.test_cases["event"] == 1, "case_id"]


def process_metrics(fold):
    """Уровень 1: сеть состояний ``censor_aware`` против Inductive Miner."""
    return level1(fold.models["censor_aware"], fold.train_log, fold.test_log, fold.complete_ids)


def stochastic_metrics(fold):
    """Уровень 2: entropic relevance, uEMSC, EMD языков."""
    return level2(fold.models, fold.test_log, fold.n_sim, fold.seed)


def temporal_metrics(fold):
    """Уровень 3: temporal profile, расстояния журналов, KS по рёбрам, выживание."""
    return level3(fold.models, fold.train_log, fold.test_log, fold.train_cases, fold.test_cases,
                  fold.test_intervals, fold.n_sim, fold.seed)


def likelihood_metrics(fold):
    """Уровень 4: −log L тестовых интервалов и доля трасс с невиданными переходами."""
    return [dict(model=name, **model.nll(fold.test_intervals)) for name, model in fold.models.items()]


DEFAULT_METRICS = {"level1": process_metrics, "level2": stochastic_metrics,
                   "level3": temporal_metrics, "level4": likelihood_metrics}


def per_model(metric):
    """Метрика ``metric(model, fold)`` для одной модели → метрика по всем моделям фолда.

    ``metric`` возвращает число (колонка называется по имени функции) или ``dict`` колонок.
    """
    @wraps(metric)
    def wrapped(fold):
        rows = []
        for name, model in fold.models.items():
            value = metric(model, fold)
            rows.append(dict(model=name, **({metric.__name__: value} if isinstance(value, Number) else value)))
        return rows
    return wrapped


def trace_folds(cases, k=5, seed=2026):
    """``k`` пар ``(train_ids, test_ids)``; доля цензурированных трасс в фолдах одинакова."""
    ids = cases["case_id"].to_numpy()
    folds = StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, cases["event"])
    return [(ids[train], ids[test]) for train, test in folds]


class CrossValidator:
    """Наивная MLE (``naive``) и MLE с цензурой (``censor_aware``) на ``k`` фолдах.

    По умолчанию считаются метрики ``level1 … level4`` (:data:`DEFAULT_METRICS`).
    """

    def __init__(self, start, terminal, k=5, seed=2026, min_duration=1 / 1440, n_sim=5000):
        self.start, self.terminal, self.k, self.seed = start, terminal, k, seed
        self.min_duration, self.n_sim = min_duration, n_sim
        self.metrics = dict(DEFAULT_METRICS)

    def set_metrics(self, **metrics):
        """Добавляет метрику под новым именем, заменяет под старым, убирает при значении ``None``.

        Возвращает ``self``, поэтому вызовы можно сцеплять.
        """
        for name, metric in metrics.items():
            if metric is None:
                self.metrics.pop(name, None)
            elif callable(metric):
                self.metrics[name] = metric
            else:
                raise TypeError(f"Метрика {name!r} должна быть функцией fold → строки, получено {type(metric).__name__}")
        return self

    def fit_models(self, train_intervals):
        """Модели фолда; переопределите, чтобы сравнивать другие модели."""
        naive = CensoredStateNet(self.start, self.terminal, use_censoring=False).fit(intervals=train_intervals)
        return {"naive": naive,
                "censor_aware": CensoredStateNet(self.start, self.terminal,
                                                 families=naive.families_).fit(intervals=train_intervals)}

    def run(self, raw, observation_end=None, reason="lost", progress=print):
        """Словарь ``имя метрики → таблица`` с колонкой ``fold``."""
        log, cases = prepare(raw, self.start, self.terminal, observation_end, reason)
        intervals = make_intervals(log, cases, self.min_duration)

        def part(table, ids):
            return table.loc[table["case_id"].isin(ids)]

        tables = {name: [] for name in self.metrics}
        for index, (train_ids, test_ids) in enumerate(trace_folds(cases, self.k, self.seed)):
            train_intervals = part(intervals, train_ids)
            fold = Fold(index, self.seed + 1000 * index, self.n_sim, self.fit_models(train_intervals),
                        part(log, train_ids), part(log, test_ids), part(cases, train_ids), part(cases, test_ids),
                        train_intervals, part(intervals, test_ids))
            for name, metric in self.metrics.items():
                rows = metric(fold)
                tables[name].extend(dict(fold=index, **row) for row in ([rows] if isinstance(rows, dict) else rows))
            if progress:
                progress(f"фолд {index + 1}/{self.k} готов")
        return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def cross_validate(raw, start, terminal, observation_end=None, reason="lost", k=5, seed=2026,
                   min_duration=1 / 1440, n_sim=5000, progress=print, metrics=None):
    """Короткая форма :class:`CrossValidator`; ``metrics`` передаётся в :meth:`~CrossValidator.set_metrics`."""
    validator = CrossValidator(start, terminal, k, seed, min_duration, n_sim).set_metrics(**(metrics or {}))
    return validator.run(raw, observation_end, reason, progress)


def summarize(table):
    """Среднее и стандартное отклонение по фолдам (по моделям, если есть колонка ``model``)."""
    table = table.drop(columns="fold")
    if "model" not in table.columns:
        return table.agg(["mean", "std"])
    return table.groupby("model", sort=False).agg(["mean", "std"])


def paired_differences(table, metric, baseline="naive", other="censor_aware"):
    """Разница ``other − baseline`` по метрике внутри каждого фолда."""
    wide = table.loc[table["model"].isin([baseline, other])].pivot_table(index="fold", columns="model", values=metric)
    wide["difference"] = wide[other] - wide[baseline]
    return wide
