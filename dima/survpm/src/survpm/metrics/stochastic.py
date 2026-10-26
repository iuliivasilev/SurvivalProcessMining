"""Уровень 2: вероятности путей — entropic relevance и uEMSC (Ebi), EMD языков (POT).

Тестовые трассы, включая цензурированные, сравниваются с моделью, которая обрывается
в состоянии с наблюдаемой вероятностью цензуры ``q``.
"""
import ebi
import jellyfish
import numpy as np
import ot
import pandas as pd
import pm4py

from ..adapters import to_pm4py_log
from .fold import per_model


def _ebi_number(result):
    """Ebi отдаёт точное выражение строкой (``"... Approximately 0.93"``) или список."""
    if isinstance(result, list):
        return float(result[0])
    return float(str(result).split("Approximately")[-1])


def _entropic_relevance(event_log, model, censoring=False):
    """Entropic relevance (Ebi), бит на трассу; при «cannot compute ER» округление грубее."""
    for digits in (3, 2):
        try:
            return _ebi_number(ebi.conformance_entropic_relevance(event_log, model.to_sdfa(censoring, digits)))
        except Exception as error:
            if "cannot compute ER" not in str(error):
                raise
    return np.nan


def language(log):
    """Стохастический язык журнала ``case_id, activity, timestamp``: ``{трасса: доля}``."""
    traces = log.groupby("case_id", sort=False)["activity"].agg(tuple)
    return traces.value_counts(normalize=True).to_dict()


def simulated_language(traces):
    return pd.Series([tuple(states) for states, _ in traces]).value_counts(normalize=True).to_dict()


def language_emd(log_language, model_language):
    """EMD между стохастическими языками с нормированным Левенштейном, как в PM4Py.

    PM4Py строит матрицу по объединению вариантов и без пакета POT решает плотную задачу ЛП:
    на Sepsis это >14 ГБ памяти; здесь матрица только «варианты журнала × варианты модели»."""
    activities = sorted({a for trace in [*log_language, *model_language] for a in trace})
    code = {a: chr(0x4E00 + i) for i, a in enumerate(activities)}
    left = ["".join(code[a] for a in trace) for trace in log_language]
    right = ["".join(code[a] for a in trace) for trace in model_language]
    distance = np.array([[jellyfish.levenshtein_distance(x, y) / max(len(x), len(y)) for y in right] for x in left])
    weights = [np.array(list(lang.values()), dtype=float) for lang in (log_language, model_language)]
    return float(ot.emd2(weights[0] / weights[0].sum(), weights[1] / weights[1].sum(), distance))


def _event_log(fold):
    return fold.cached("ebi_test_log", lambda: pm4py.convert_to_event_log(to_pm4py_log(fold.test_log)))


@per_model
def entropic_relevance(model, fold):
    """Бит на трассу, меньше — лучше."""
    return _entropic_relevance(_event_log(fold), model, censoring=True)


@per_model
def uemsc(model, fold):
    """1 − TV по трассам, больше — лучше."""
    return _ebi_number(ebi.conformance_unit_earth_movers(_event_log(fold), model.to_sdfa(censoring=True)))


@per_model
def emd(model, fold):
    """EMD языков теста и симуляции, меньше — лучше."""
    traces = model.simulate(fold.n_sim, fold.seed, censoring=True)
    return language_emd(language(fold.test_log), simulated_language(traces))


METRICS = {f.__name__: f for f in (entropic_relevance, uemsc, emd)}
