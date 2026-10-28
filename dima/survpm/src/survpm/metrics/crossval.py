"""Кросс-валидация по трассам: k фолдов и метрики, которые выбираются явно.

По умолчанию не считается ничего. Метрики добавляются через :meth:`CrossValidator.add_metrics`
по одной (``"km_sup"``), уровнем (``"level3"``) или все сразу (``"all"``); там же добавляются свои::

    def median_days(model, fold):
        return float(np.median(model.completion_times(2000, fold.seed)))

    cv = crossval.CrossValidator("A", "D").add_metrics("level1", "uemsc", "ibs", median=crossval.per_model(median_days))
    results = cv.run(raw, censoring["end"], censoring["reason"])
    crossval.summarize(results["level3"])     # таблица уровня: только выбранные метрики (здесь ibs)

Своя метрика — функция ``metric(fold)`` от :class:`Fold`; она возвращает строку (``dict``) или
список строк и попадает в отдельную таблицу под своим именем. Строка с колонкой ``model``
относится к модели (``naive``, ``censor_aware`` или своей).
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..log import intervals as make_intervals, prepare
from ..model import CensoredStateNet
from . import likelihood, process, stochastic, temporal
from .fold import Fold, per_model  # noqa: F401  (часть API crossval)

LEVELS = {"level1": process.METRICS, "level2": stochastic.METRICS,
          "level3": temporal.METRICS, "level4": likelihood.METRICS}
METRICS = {name: metric for metrics in LEVELS.values() for name, metric in metrics.items()}
"""Все встроенные метрики по имени, в порядке уровней."""
LEVEL_OF = {name: level for level, metrics in LEVELS.items() for name in metrics}


def _expand(name):
    """``"all"``, уровень или имя метрики → имена встроенных метрик."""
    if name == "all":
        return list(METRICS)
    if name in LEVELS:
        return list(LEVELS[name])
    if name in METRICS:
        return [name]
    raise ValueError(f"Неизвестная метрика {name!r}; есть 'all', {', '.join(LEVELS)}, {', '.join(METRICS)}")


def trace_folds(cases, k=5, seed=2026):
    """``k`` пар ``(train_ids, test_ids)``; доля цензурированных трасс в фолдах одинакова."""
    ids = cases["case_id"].to_numpy()
    folds = StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, cases["event"])
    return [(ids[train], ids[test]) for train, test in folds]


class CrossValidator:
    """Наивная MLE (``naive``) и MLE с цензурой (``censor_aware``) на ``k`` фолдах.

    Метрик по умолчанию нет: их добавляет :meth:`add_metrics`.
    """

    def __init__(self, start, terminal, k=5, seed=2026, min_duration=1 / 1440, n_sim=5000, repeats="simultaneous"):
        self.start, self.terminal, self.k, self.seed = start, terminal, k, seed
        self.min_duration, self.n_sim, self.repeats = min_duration, n_sim, repeats
        self.metrics = {}

    def add_metrics(self, *names, **custom):
        """Встроенные метрики по имени, уровню (``"level1"`` … ``"level4"``) или все (``"all"``);
        свои — ``имя=функция``. Возвращает ``self``, поэтому вызовы можно сцеплять."""
        for name in names:
            self.metrics.update((metric, METRICS[metric]) for metric in _expand(name))
        for name, metric in custom.items():
            if name == "all" or name in LEVELS or name in METRICS:
                raise ValueError(f"Имя {name!r} занято встроенной метрикой или уровнем")
            if not callable(metric):
                raise TypeError(f"Метрика {name!r} должна быть функцией fold → строки, получено {type(metric).__name__}")
            self.metrics[name] = metric
        return self

    def remove_metrics(self, *names):
        """Убирает метрики: по имени (и своих), уровнем или все (``"all"``)."""
        for name in names:
            for metric in [name] if name in self.metrics and name not in METRICS else _expand(name):
                self.metrics.pop(metric, None)
        return self

    def fit_models(self, train_intervals):
        """Модели фолда; переопределите, чтобы сравнивать другие модели."""
        naive = CensoredStateNet(self.start, self.terminal, use_censoring=False).fit(intervals=train_intervals)
        return {"naive": naive,
                "censor_aware": CensoredStateNet(self.start, self.terminal,
                                                 families=naive.families_).fit(intervals=train_intervals)}

    def run(self, raw, observation_end=None, reason="lost", progress=print):
        """Словарь ``таблица → DataFrame`` с колонкой ``fold``: по таблице на уровень
        (только выбранные метрики) и на каждую свою метрику."""
        if not self.metrics:
            raise ValueError("Метрики не выбраны: add_metrics('all'), add_metrics('level3'), add_metrics('km_sup') …")
        log, cases = prepare(raw, self.start, self.terminal, observation_end, reason, self.repeats)
        intervals = make_intervals(log, cases, self.min_duration)
        builtin = [name for name in METRICS if name in self.metrics]
        own = [name for name in self.metrics if name not in METRICS]

        def part(table, ids):
            return table.loc[table["case_id"].isin(ids)]

        tables = {LEVEL_OF.get(name, name): [] for name in builtin + own}
        for index, (train_ids, test_ids) in enumerate(trace_folds(cases, self.k, self.seed)):
            train_intervals = part(intervals, train_ids)
            fold = Fold(index, self.seed + 1000 * index, self.n_sim, self.fit_models(train_intervals),
                        part(log, train_ids), part(log, test_ids), part(cases, train_ids), part(cases, test_ids),
                        train_intervals, part(intervals, test_ids))
            merged = {}  # встроенные метрики уровня сливаются в одну строку на модель
            for name in builtin:
                for row in self.metrics[name](fold):
                    merged.setdefault(LEVEL_OF[name], {}).setdefault(row["model"], dict(fold=index)).update(row)
            for level, rows in merged.items():
                tables[level].extend(rows.values())
            for name in own:
                rows = self.metrics[name](fold)
                tables[name].extend(dict(fold=index, **row) for row in ([rows] if isinstance(rows, dict) else rows))
            if progress:
                progress(f"фолд {index + 1}/{self.k} готов")
        return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def cross_validate(raw, start, terminal, observation_end=None, reason="lost", k=5, seed=2026,
                   min_duration=1 / 1440, n_sim=5000, progress=print, metrics="all", repeats="simultaneous"):
    """Короткая форма :class:`CrossValidator`; ``metrics`` — имя, список имён или ``{имя: функция}``
    для :meth:`~CrossValidator.add_metrics`."""
    validator = CrossValidator(start, terminal, k, seed, min_duration, n_sim, repeats)
    if isinstance(metrics, dict):
        validator.add_metrics(**metrics)
    else:
        validator.add_metrics(*([metrics] if isinstance(metrics, str) else metrics))
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
