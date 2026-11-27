"""Наборы данных: синтетическая сеть A–D с известной истиной и журнал Sepsis.

Синтетика перенесена из ``petri_net_time_estimation.ipynb``, раздел 1.

Истинная модель: ``AD_TRANSITIONS`` — выбор выхода по ``p``, время из ``dist`` (в днях).
Цензура: потеря наблюдения (экспонента со средним ``LOSS_MEAN_DAYS``) соревнуется по времени
с выбранным переходом, плюс конец исследования через ``study_days`` после старта.

Пример::

    raw, censoring = datasets.generate_ad(1000, seed=2026)
    raw = datasets.load_sepsis("Med_Eremina/Sepsis Cases - Event Log_grouped.csv")
    log = datasets.load_sepsis_log("Med_Eremina/Sepsis Cases - Event Log_grouped.csv")  # с признаками
"""
import numpy as np
import pandas as pd

from .features import read_log
from ._truth import TruthGenerator
from .log import read_csv

AD_START = "A"
AD_TERMINAL = "D"
AD_TRANSITIONS = [
    dict(name="A→B", src="A", dst="B", p=0.7, dist=("exponential", {"mean": 2.0})),
    dict(name="A→C", src="A", dst="C", p=0.3, dist=("lognormal", {"median": 5.0, "sigma": 0.5})),
    dict(name="B→A", src="B", dst="A", p=0.2, dist=("uniform", {"low": 1.0, "high": 3.0})),
    dict(name="B→C", src="B", dst="C", p=0.5, dist=("gamma", {"shape": 2.0, "scale": 1.5})),
    dict(name="B→D", src="B", dst="D", p=0.3, dist=("weibull", {"shape": 1.5, "scale": 10.0})),
    dict(name="C→B", src="C", dst="B", p=0.4, dist=("exponential", {"mean": 3.0})),
    dict(name="C→D", src="C", dst="D", p=0.6, dist=("weibull", {"shape": 0.8, "scale": 6.0})),
]
LOSS_MEAN_DAYS = 60


def generate_ad(n_cases, seed, transitions=AD_TRANSITIONS, start=AD_START, terminal=AD_TERMINAL,
                loss_mean_days=LOSS_MEAN_DAYS, study_start=pd.Timestamp("2025-01-01", tz="UTC"),
                enroll_days=60, study_days=90):
    """Журнал ``case_id, activity, timestamp`` и таблица цензуры ``censoring`` (индекс ``case_id``,
    колонки ``end`` — момент цензуры, ``reason`` — ``lost`` или ``study_end``) для незавершённых трасс.
    Объект входит в исследование равномерно в первые ``enroll_days`` дней, наблюдение кончается
    через ``study_days`` дней после старта исследования. В журнал цензура не пишется::

        raw, censoring = generate_ad(1000, seed=2026)
        log, cases = survpm.prepare(raw, "A", "D", censoring["end"], censoring["reason"])

    Генератор — внутренний :class:`survpm._truth.TruthGenerator`; :func:`generate_ad_log` даёт тот же журнал как ``EventLog``.
    """
    generator = ad_generator(seed, transitions, start, terminal, loss_mean_days, study_start, enroll_days)
    rows, censoring = [], {}
    for path in generator._paths(n_cases):
        case_id = path.case.case_id
        entry = generator.entries[case_id]
        path.case.horizon = None
        rows.extend(dict(case_id=case_id, activity=state,
                         timestamp=study_start + pd.to_timedelta(entry + t, unit="D").round("s"))
                    for state, t in zip(path.states, path.times))
        if not path.event:
            censoring[case_id] = dict(end=study_start + pd.to_timedelta(entry + path.end, unit="D").round("s"),
                                      reason="study_end" if path.reason == "horizon" else path.reason)
    return pd.DataFrame(rows), pd.DataFrame.from_dict(censoring, orient="index", columns=["end", "reason"])


class _StudyGenerator(TruthGenerator):
    """Синтетика A–D: конец наблюдения — общий конец исследования (``study_days`` от старта)."""
    def __init__(self, *args, study_days=90, **kwargs):
        super().__init__(*args, **kwargs)
        self.study_days = study_days

    def start_cases(self, n, rng):
        for case in super().start_cases(n, rng):
            case.horizon = self.study_days - self.entries[case.case_id]
            yield case


def ad_generator(seed, transitions=AD_TRANSITIONS, start=AD_START, terminal=AD_TERMINAL,
                 loss_mean_days=LOSS_MEAN_DAYS, study_start=pd.Timestamp("2025-01-01", tz="UTC"),
                 enroll_days=60, study_days=90):
    """внутренний :class:`survpm._truth.TruthGenerator` синтетики A–D (общий поток случайных чисел, как прежде)."""
    return _StudyGenerator(transitions, start, terminal, seed=seed, rng="shared", max_steps=10_000,
                           loss_mean_days=loss_mean_days, origin=study_start, enroll_days=enroll_days,
                           study_days=study_days)


def generate_ad_log(n_cases, seed, **options):
    """Синтетика A–D сразу как :class:`survpm.EventLog` (с ``observation``: ``end``, ``event``, ``reason``)."""
    return ad_generator(seed, **options).generate(n_cases)


SEPSIS_START, SEPSIS_TERMINAL = "Registration", "Discharge"


def load_sepsis(path, state_column="event_group"):
    """Журнал Sepsis (``Med_Eremina/Sepsis Cases - Event Log_grouped.csv``); состояния — группы событий."""
    return read_csv(path, "case_id", state_column, "time:timestamp")


SEPSIS_MEASUREMENTS = ["Examination"]
SEPSIS_EXCLUDE = ["concept:name", "org:group", "lifecycle:transition"]


def load_sepsis_log(path, state_column="event_group", measurement_activities=(), exclude=SEPSIS_EXCLUDE):
    """Журнал Sepsis с признаками (:class:`survpm.features.EventLog`).

    Признаки разбираются автоматически: Age, Diagnose и флаги SIRS/Diagnostic* — статические
    (записаны в Registration), CRP, Leucocytes, LacticAcid — замеры. По умолчанию трассы не
    упрощаются: Examination остаётся состоянием (как в :func:`load_sepsis`), а значения
    анализов дополнительно попадают в ``measurements``.
    ``measurement_activities=SEPSIS_MEASUREMENTS`` делает Examination «прозрачной»: анализы
    только обновляют признаки и не считаются состоянием.
    """
    return read_log(path, "case_id", state_column, "time:timestamp",
                    measurement_activities=measurement_activities, exclude=exclude)
