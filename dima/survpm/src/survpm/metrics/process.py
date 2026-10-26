"""Уровень 1: fitness, precision, generalization, simplicity (PM4Py).

Сеть состояний модели ``censor_aware`` сравнивается с Inductive Miner (строки ``state_net``, ``inductive``).

Цензурированная трасса считается префиксом: token replay без финальной маркировки,
оставшиеся фишки не штрафуются; precision (ETC) и так строится по префиксам.
"""
from functools import wraps

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


def _nets(fold):
    """Сеть состояний ``censor_aware`` (без места CENS) и Inductive Miner на обучающем журнале."""
    return fold.cached("level1_nets", lambda: {
        "state_net": fold.models["censor_aware"].to_pm4py(censoring=False),
        "inductive": pm4py.discover_petri_net_inductive(to_pm4py_log(fold.train_log))})


def _test_log(fold):
    return fold.cached("pm4py_test_log", lambda: to_pm4py_log(fold.test_log))


def per_net(metric):
    """``metric(net, im, fm, fold)`` → строки ``state_net`` и ``inductive``."""
    @wraps(metric)
    def wrapped(fold):
        return [dict(model=name, **{metric.__name__: metric(net, im, fm, fold)})
                for name, (net, im, fm) in _nets(fold).items()]
    return wrapped


@per_net
def fitness_token(net, im, fm, fold):
    """Token-based fitness; трассы, не дошедшие до терминала, — префиксы."""
    log = _test_log(fold)
    is_complete = log["case:concept:name"].isin(fold.complete_ids)
    return prefix_fitness(log.loc[is_complete], log.loc[~is_complete], net, im, fm)


@per_net
def precision_token(net, im, fm, fold):
    return pm4py.precision_token_based_replay(_test_log(fold), net, im, fm)


@per_net
def generalization(net, im, fm, fold):
    return pm4py.generalization_tbr(_test_log(fold), net, im, fm)


@per_net
def simplicity(net, im, fm, fold):
    return pm4py.simplicity_petri_net(net, im, fm)


METRICS = {f.__name__: f for f in (fitness_token, precision_token, generalization, simplicity)}
