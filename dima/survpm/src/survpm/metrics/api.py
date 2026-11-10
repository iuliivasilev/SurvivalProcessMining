"""Evaluation of already fitted ProcessModels; independent of legacy fold metrics."""
from collections.abc import Mapping
from typing import Protocol

import pandas as pd


class Metric(Protocol):
    """A callable returning a scalar or a table with metric/value columns."""
    def __call__(self, model, log): ...


def _nll(model, log):
    from ..engine import Evaluator
    return pd.DataFrame(Evaluator(model).run(log).items(), columns=['metric', 'value'])


def _ibs(model, log):
    raise NotImplementedError('ProcessModel IBS requires train-only IPCW; use the legacy cross-validation API')


_METRICS = {'nll': _nll, 'ibs': _ibs}


def register_metric(name, func):
    """Register a model/log callable without replacing an existing metric."""
    if name in _METRICS:
        raise ValueError(f'Metric already registered: {name}')
    if not isinstance(name, str) or not name or not callable(func):
        raise ValueError('A metric needs a nonempty name and a callable')
    _METRICS[name] = func


def evaluate(model, log, metrics=('nll',)):
    """Return {requested_name: metric/value table}; never fit any component.

    NLL contains nll_per_interval and nll_per_trace, including censored exposure.
    Unknown observed edges receive infinite NLL, rather than being dropped.
    """
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    for name in names:
        if name not in _METRICS:
            raise ValueError(f'Unknown metric: {name}')
    result = {}
    for name in names:
        value = _METRICS[name](model, log)
        table = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame({'metric': [name], 'value': [value]})
        if not {'metric', 'value'} <= set(table.columns):
            raise ValueError('Metric tables must contain metric/value columns')
        result[name] = table[['metric', 'value']]
    return result


def compare(models, log, metrics=('nll',)):
    """Stack results for named models or an iterable labelled model_0, model_1, … ."""
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    tables = []
    models = models.items() if isinstance(models, Mapping) else ((f'model_{i}', model) for i, model in enumerate(models))
    for name, model in models:
        for table in evaluate(model, log, names).values():
            tables.append(table.assign(model=name)[['model', 'metric', 'value']])
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=['model', 'metric', 'value'])


def compare_networks(reference, estimated, state_mapping=None):
    """Compare directed topology only, accepting Network or ProcessModel.

    state_mapping maps estimated labels into reference labels; unmapped labels
    retain their names. Probabilities and fitted parameters are not compared.
    An empty predicted/reference edge set has precision/recall one respectively.
    """
    mapping = state_mapping or {}
    ref = getattr(reference, 'network', reference)
    est = getattr(estimated, 'network', estimated)
    expected = set(ref.edges[['src', 'dst']].itertuples(index=False, name=None))
    found = {(mapping.get(src, src), mapping.get(dst, dst))
             for src, dst in est.edges[['src', 'dst']].itertuples(index=False, name=None)}
    common = len(expected & found)
    precision = common/len(found) if found else 1.
    recall = common/len(expected) if expected else 1.
    def edges(values):
        return pd.DataFrame(sorted(values, key=repr), columns=['src', 'dst'])
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                missing=edges(expected-found), extra=edges(found-expected))
