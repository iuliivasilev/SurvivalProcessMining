"""Уровень 3: время.

* temporal profile (PM4Py): доля тестовых трасс с отклонениями от профиля, построенного по
  обучающему журналу (``model="train_log"``, базовая линия) или по симуляции модели;
* расстояния журналов (log-distance-measures): тест против симуляции того же размера с теми же
  моментами входа; симулированная трасса обрывается в момент цензуры парной
  тестовой трассы; ``cycle_time_emd`` и ``absolute_event_emd`` — в часах;
* KS по рёбрам на тестовых интервалах;
* выживание: Каплан–Майер теста против модели (``km_sup``, ``km_l1_days``), log-rank,
  integrated Brier score.

log-distance-measures формально требует Python < 3.13:
``pip install --ignore-requires-python log-distance-measures jellyfish pulp``.
"""
import datetime

import numpy as np
import pm4py
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from ..adapters import to_pm4py_log
from ..distributions import scipy_args
from ..simulation import completion_times, traces_to_log

try:
    from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
    from log_distance_measures.config import EventLogIDs
    from log_distance_measures.control_flow_log_distance import control_flow_log_distance
    from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
    from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
except ImportError as error:
    raise ImportError("Нужен log-distance-measures: pip install --ignore-requires-python "
                      "log-distance-measures jellyfish pulp") from error

LOG_IDS = EventLogIDs(case="case_id", activity="activity", start_time="timestamp", end_time="timestamp")


def temporal_deviation_share(log, profile):
    """Доля трасс журнала ``case_id, activity, timestamp`` с отклонением от temporal profile."""
    deviations = pm4py.conformance_temporal_profile(to_pm4py_log(log), profile)
    return float(np.mean([len(d) > 0 for d in deviations]))


def survival_metrics(test_cases, train_cases, model_times):
    """Кривая выживания модели (выборка времён до исхода) против Каплан–Майера теста."""
    km = KaplanMeierFitter().fit(test_cases["duration"], test_cases["event"])
    grid = np.linspace(0, test_cases["duration"].max(), 400)
    model_times = np.sort(model_times)
    model_survival = 1 - np.searchsorted(model_times, grid, side="right") / len(model_times)
    difference = np.abs(model_survival - km.survival_function_at_times(grid).to_numpy())
    finite = np.isfinite(model_times)
    logrank = logrank_test(test_cases["duration"], np.where(finite, model_times, grid[-1] * 10),
                           test_cases["event"], finite.astype(int))
    # IPCW-веса считаются по цензуре обучения: тестовые времена должны быть меньше последнего в обучении.
    scored = test_cases.loc[test_cases["duration"] < train_cases["duration"].max()]
    event_times = scored.loc[scored["event"] == 1, "duration"]
    times = np.unique(np.quantile(event_times, np.linspace(.1, .8, 15)))
    times = times[(times > scored["duration"].min()) & (times < scored["duration"].max())]
    estimate = 1 - np.searchsorted(model_times, times, side="right") / len(model_times)
    ibs = integrated_brier_score(Surv.from_arrays(train_cases["event"].astype(bool), train_cases["duration"]),
                                 Surv.from_arrays(scored["event"].astype(bool), scored["duration"]),
                                 np.tile(estimate, (len(scored), 1)), times) if len(times) > 1 else np.nan
    return dict(km_sup=difference.max(), km_l1_days=np.trapezoid(difference, grid),
                logrank_statistic=logrank.test_statistic, logrank_p=logrank.p_value, ibs=ibs)


def edge_ks(test_intervals, edges):
    """Средний по тестовым интервалам KS завершённых времён против распределения ребра."""
    completed = test_intervals.loc[test_intervals["observed"]]
    values, weights = [], []
    for edge in edges:
        x = completed.loc[(completed["src"] == edge["src"]) & (completed["dst"] == edge["dst"]), "duration"]
        if len(x):
            rv, args, kwargs = scipy_args(edge["dist"])
            values.append(stats.kstest(x, lambda v: rv.cdf(v, *args, **kwargs)).statistic)
            weights.append(len(x))
    return float(np.average(values, weights=weights))


def level3(models, train_log, test_log, train_cases, test_cases, test_intervals, n_sim=5000, seed=0):
    """Строки метрик для ``train_log`` (базовая линия temporal profile) и каждой модели."""
    test_cases = test_cases.set_index("case_id").loc[test_log["case_id"].unique()].reset_index()
    train_profile = pm4py.discover_temporal_profile(to_pm4py_log(train_log))
    rows = [dict(model="train_log", temporal_deviation_share=temporal_deviation_share(test_log, train_profile))]
    horizon = np.where(test_cases["event"] == 1, np.inf, test_cases["duration"])
    for name, model in models.items():
        free = model.simulate(n_sim, seed)
        profile = pm4py.discover_temporal_profile(to_pm4py_log(traces_to_log(free, [train_cases["entry"].min()] * n_sim)))
        simulated = traces_to_log(model.simulate(len(test_cases), seed + 1, horizon=horizon), test_cases["entry"])
        rows.append(dict(model=name,
                         temporal_deviation_share=temporal_deviation_share(test_log, profile),
                         cfld=control_flow_log_distance(test_log, LOG_IDS, simulated, LOG_IDS, parallel=False),
                         ngram_3=n_gram_distribution_distance(test_log, LOG_IDS, simulated, LOG_IDS, n=3, normalize=True),
                         cycle_time_emd=cycle_time_distribution_distance(test_log, LOG_IDS, simulated, LOG_IDS,
                                                                         bin_size=datetime.timedelta(hours=1)),
                         absolute_event_emd=absolute_event_distribution_distance(test_log, LOG_IDS, simulated, LOG_IDS),
                         edge_ks=edge_ks(test_intervals, model.edges_),
                         **survival_metrics(test_cases, train_cases, completion_times(free, model.terminal))))
    return rows
