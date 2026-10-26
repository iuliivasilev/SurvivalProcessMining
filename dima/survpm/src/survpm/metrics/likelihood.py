"""Уровень 4: −log L тестовых интервалов с цензурой (:meth:`survpm.CensoredStateNet.nll`)."""
from .fold import per_model


def _nll(model, fold):
    """Трассы с переходом, которого не было в обучении, исключены; их доля — ``unseen_share``."""
    return fold.cached(("nll", id(model)), lambda: model.nll(fold.test_intervals))


@per_model
def unseen_share(model, fold):
    return _nll(model, fold)["unseen_share"]


@per_model
def nll_per_interval(model, fold):
    return _nll(model, fold)["nll_per_interval"]


@per_model
def nll_per_trace(model, fold):
    return _nll(model, fold)["nll_per_trace"]


METRICS = {f.__name__: f for f in (unseen_share, nll_per_interval, nll_per_trace)}
