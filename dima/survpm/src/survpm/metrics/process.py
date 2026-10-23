"""Уровень 1: fitness, precision, generalization, simplicity (PM4Py).

Вариант ``A`` — только полные трассы. Вариант ``Б`` — все трассы, цензурированная трасса
считается префиксом: token replay без финальной маркировки, оставшиеся фишки не штрафуются;
precision (ETC) и так строится по префиксам; alignments только в ``A``.
"""
import numpy as np
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.petri_net.obj import Marking

from ..adapters import to_pm4py_log


def _replay_sums(log, net, im, fm):
    """Суммы ``missing, consumed, remaining, produced`` token replay по журналу PM4Py."""
    replay = token_replay.apply(pm4py.convert_to_event_log(log), net, im, fm)
    return np.array([[r["missing_tokens"], r["consumed_tokens"], r["remaining_tokens"], r["produced_tokens"]]
                     for r in replay], dtype=float).sum(axis=0)


def prefix_fitness(complete, censored, net, im, fm):
    """Token-based fitness, где цензурированная трасса — префикс (журналы в колонках PM4Py)."""
    missing, consumed, remaining, produced = _replay_sums(complete, net, im, fm)
    if len(censored):
        m, c, _, p = _replay_sums(censored, net, im, Marking())
        missing, consumed, produced = missing + m, consumed + c, produced + p
    return .5 * (1 - missing / consumed) + .5 * (1 - remaining / produced)


def conformance(net, im, fm, log, complete_ids):
    """Строки метрик для вариантов A и Б; ``log`` — журнал ``case_id, activity, timestamp``."""
    log = to_pm4py_log(log)
    is_complete = log["case:concept:name"].isin(complete_ids)
    complete, censored = log.loc[is_complete], log.loc[~is_complete]
    simplicity = pm4py.simplicity_petri_net(net, im, fm)
    return [
        dict(variant="A",
             fitness_token=pm4py.fitness_token_based_replay(complete, net, im, fm)["log_fitness"],
             fitness_alignments=pm4py.fitness_alignments(complete, net, im, fm)["log_fitness"],
             precision_token=pm4py.precision_token_based_replay(complete, net, im, fm),
             precision_alignments=pm4py.precision_alignments(complete, net, im, fm),
             generalization=pm4py.generalization_tbr(complete, net, im, fm),
             simplicity=simplicity),
        dict(variant="Б",
             fitness_token=prefix_fitness(complete, censored, net, im, fm),
             precision_token=pm4py.precision_token_based_replay(log, net, im, fm),
             generalization=pm4py.generalization_tbr(log, net, im, fm),
             simplicity=simplicity),
    ]


def level1(model, train_log, test_log, complete_ids):
    """Сеть состояний модели (без места CENS) против Inductive Miner на обучающем журнале."""
    nets = {"state_net": model.to_pm4py(censoring=False),
            "inductive": pm4py.discover_petri_net_inductive(to_pm4py_log(train_log))}
    return [dict(model=name, **row) for name, (net, im, fm) in nets.items()
            for row in conformance(net, im, fm, test_log, complete_ids)]
