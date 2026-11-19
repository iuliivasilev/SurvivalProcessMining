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
from .fold import per_model

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


def _deviation_share(log, profile):
    """Доля трасс журнала ``case_id, activity, timestamp`` с отклонением от temporal profile."""
    deviations = pm4py.conformance_temporal_profile(to_pm4py_log(log), profile)
    return float(np.mean([len(d) > 0 for d in deviations]))


def _test_cases(fold):
    """Тестовые трассы в порядке журнала (так парная симуляция совпадает с тестом по входу и цензуре)."""
    return fold.cached("level3_test_cases", lambda: fold.test_cases.set_index("case_id")
                       .loc[fold.test_log["case_id"].unique()].reset_index())


def _free(model, fold):
    """``n_sim`` трасс модели без цензуры: temporal profile и кривая выживания."""
    return fold.cached(("free", id(model)), lambda: model.simulate(fold.n_sim, fold.seed))


def _paired_log(model, fold):
    """По симулированной трассе на тестовую: тот же вход, обрыв в момент её цензуры."""
    def compute():
        cases = _test_cases(fold)
        horizon = np.where(cases["event"] == 1, np.inf, cases["duration"])
        return traces_to_log(model.simulate(len(cases), fold.seed + 1, horizon=horizon), cases["entry"])
    return fold.cached(("paired", id(model)), compute)


def _model_times(model, fold):
    return fold.cached(("times", id(model)), lambda: np.sort(completion_times(_free(model, fold), model.terminal)))


def temporal_deviation_share(fold):
    """Строка ``train_log`` (профиль по обучающему журналу, базовая линия) и строка на модель."""
    rows = [dict(model="train_log", temporal_deviation_share=_deviation_share(
        fold.test_log, pm4py.discover_temporal_profile(to_pm4py_log(fold.train_log))))]
    for name, model in fold.models.items():
        free_log = traces_to_log(_free(model, fold), [fold.train_cases["entry"].min()] * fold.n_sim)
        profile = pm4py.discover_temporal_profile(to_pm4py_log(free_log))
        rows.append(dict(model=name, temporal_deviation_share=_deviation_share(fold.test_log, profile)))
    return rows


@per_model
def cfld(model, fold):
    return control_flow_log_distance(fold.test_log, LOG_IDS, _paired_log(model, fold), LOG_IDS, parallel=False)


@per_model
def ngram_3(model, fold):
    return n_gram_distribution_distance(fold.test_log, LOG_IDS, _paired_log(model, fold), LOG_IDS, n=3, normalize=True)


@per_model
def cycle_time_emd(model, fold):
    """Часы."""
    return cycle_time_distribution_distance(fold.test_log, LOG_IDS, _paired_log(model, fold), LOG_IDS,
                                            bin_size=datetime.timedelta(hours=1))


@per_model
def absolute_event_emd(model, fold):
    """Часы."""
    return absolute_event_distribution_distance(fold.test_log, LOG_IDS, _paired_log(model, fold), LOG_IDS)


@per_model
def edge_ks(model, fold):
    """Средний по тестовым интервалам KS завершённых времён против распределения ребра."""
    completed = fold.test_intervals.loc[fold.test_intervals["observed"]]
    values, weights = [], []
    for edge in model.edges_:
        x = completed.loc[(completed["src"] == edge["src"]) & (completed["dst"] == edge["dst"]), "duration"]
        if len(x):
            rv, args, kwargs = scipy_args(edge["dist"])
            values.append(stats.kstest(x, lambda v: rv.cdf(v, *args, **kwargs)).statistic)
            weights.append(len(x))
    return float(np.average(values, weights=weights))


def _km_difference(model, fold):
    """``|S_модели − S_KM теста|`` на сетке до максимальной тестовой длительности."""
    def compute():
        cases = _test_cases(fold)
        km = KaplanMeierFitter().fit(cases["duration"], cases["event"])
        grid = np.linspace(0, cases["duration"].max(), 400)
        times = _model_times(model, fold)
        survival = 1 - np.searchsorted(times, grid, side="right") / len(times)
        return grid, np.abs(survival - km.survival_function_at_times(grid).to_numpy())
    return fold.cached(("km", id(model)), compute)


@per_model
def km_sup(model, fold):
    return _km_difference(model, fold)[1].max()


@per_model
def km_l1_days(model, fold):
    return np.trapezoid(*_km_difference(model, fold)[::-1])


@per_model
def logrank(model, fold):
    """Тест против симуляции модели; ``logrank_statistic`` и ``logrank_p``."""
    cases, times = _test_cases(fold), _model_times(model, fold)
    finite = np.isfinite(times)
    result = logrank_test(cases["duration"], np.where(finite, times, cases["duration"].max() * 10),
                          cases["event"], finite.astype(int))
    return dict(logrank_statistic=result.test_statistic, logrank_p=result.p_value)


@per_model
def ibs(model, fold):
    """Integrated Brier score с IPCW."""
    cases, train = _test_cases(fold), fold.train_cases
    # IPCW-веса считаются по цензуре обучения: тестовые времена должны быть меньше последнего в обучении.
    scored = cases.loc[cases["duration"] < train["duration"].max()]
    event_times = scored.loc[scored["event"] == 1, "duration"]
    if event_times.empty:
        return np.nan
    times = np.unique(np.quantile(event_times, np.linspace(.1, .8, 15)))
    times = times[(times > scored["duration"].min()) & (times < scored["duration"].max())]
    if len(times) < 2:
        return np.nan
    model_times = _model_times(model, fold)
    estimate = 1 - np.searchsorted(model_times, times, side="right") / len(model_times)
    return integrated_brier_score(Surv.from_arrays(train["event"].astype(bool), train["duration"]),
                                  Surv.from_arrays(scored["event"].astype(bool), scored["duration"]),
                                  np.tile(estimate, (len(scored), 1)), times)


METRICS = {f.__name__: f for f in (temporal_deviation_share, cfld, ngram_3, cycle_time_emd, absolute_event_emd,
                                   edge_ks, km_sup, km_l1_days, logrank, ibs)}
