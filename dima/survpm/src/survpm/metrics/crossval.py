"""Кросс-валидация по трассам: k фолдов, все четыре уровня метрик."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..log import intervals as make_intervals, prepare
from ..model import CensoredStateNet
from .process import level1
from .stochastic import level2
from .temporal import level3


def trace_folds(cases, k=5, seed=2026):
    """``k`` пар ``(train_ids, test_ids)``; доля цензурированных трасс в фолдах одинакова."""
    ids = cases["case_id"].to_numpy()
    folds = StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, cases["event"])
    return [(ids[train], ids[test]) for train, test in folds]


def cross_validate(raw, start, terminal, observation_end=None, reason="lost", k=5, seed=2026,
                   min_duration=1 / 1440, n_sim=5000, progress=print):
    """Наивная MLE (``naive``) и MLE с цензурой (``censor_aware``) на ``k`` фолдах.

    Возвращает словарь таблиц ``level1 … level4`` с колонкой ``fold``.
    """
    log, cases = prepare(raw, start, terminal, observation_end, reason)
    intervals = make_intervals(log, cases, min_duration)
    tables = {f"level{i}": [] for i in range(1, 5)}
    for fold, (train_ids, test_ids) in enumerate(trace_folds(cases, k, seed)):
        train_log, test_log = log.loc[log["case_id"].isin(train_ids)], log.loc[log["case_id"].isin(test_ids)]
        train_cases, test_cases = cases.loc[cases["case_id"].isin(train_ids)], cases.loc[cases["case_id"].isin(test_ids)]
        train_intervals = intervals.loc[intervals["case_id"].isin(train_ids)]
        test_intervals = intervals.loc[intervals["case_id"].isin(test_ids)]
        complete_ids = test_cases.loc[test_cases["event"] == 1, "case_id"]
        naive = CensoredStateNet(start, terminal, use_censoring=False).fit(intervals=train_intervals)
        models = {"naive": naive,
                  "censor_aware": CensoredStateNet(start, terminal, families=naive.families_).fit(intervals=train_intervals)}
        fold_seed = seed + 1000 * fold
        parts = {
            "level1": level1(models["censor_aware"], train_log, test_log, complete_ids),
            "level2": level2(models, test_log, complete_ids, n_sim, fold_seed),
            "level3": level3(models, train_log, test_log, train_cases, test_cases, test_intervals, n_sim, fold_seed),
            "level4": [dict(model=name, **model.nll(test_intervals)) for name, model in models.items()],
        }
        for name, rows in parts.items():
            tables[name].extend(dict(fold=fold, **row) for row in rows)
        if progress:
            progress(f"фолд {fold + 1}/{k} готов")
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def summarize(table):
    """Среднее и стандартное отклонение по фолдам."""
    keys = [c for c in ("model", "variant") if c in table.columns]
    return table.drop(columns="fold").groupby(keys, sort=False).agg(["mean", "std"])


def paired_differences(table, metric, baseline="naive", other="censor_aware"):
    """Разница ``other − baseline`` по метрике внутри каждого фолда (и варианта, если есть)."""
    keys = ["fold"] + (["variant"] if "variant" in table.columns else [])
    wide = table.loc[table["model"].isin([baseline, other])].pivot_table(index=keys, columns="model", values=metric)
    wide["difference"] = wide[other] - wide[baseline]
    return wide
