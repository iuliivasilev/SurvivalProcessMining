"""Фолд кросс-валидации и обёртка ``per_model``: общее для всех метрик."""
from dataclasses import dataclass, field
from functools import wraps
from numbers import Number

import pandas as pd


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
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def complete_ids(self):
        """Тестовые трассы, дошедшие до терминала."""
        return self.test_cases.loc[self.test_cases["event"] == 1, "case_id"]

    def cached(self, key, compute):
        """``compute()`` один раз на фолд: метрики делят симуляции, сети и журналы PM4Py."""
        if key not in self.cache:
            self.cache[key] = compute()
        return self.cache[key]


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
