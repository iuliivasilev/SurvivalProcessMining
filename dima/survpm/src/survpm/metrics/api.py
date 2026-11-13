"""Оценка обученных моделей: одна точка входа для всех метрик библиотеки, единый формат результата."""
from collections.abc import Mapping
from typing import Protocol

import pandas as pd


COLUMNS = ['level', 'model', 'metric', 'value']


class Metric(Protocol):
    """A callable ``(model, log)`` returning a scalar or a table with metric/value columns."""
    def __call__(self, model, log): ...


def _nll(model, log):
    from ..engine import Evaluator
    return pd.DataFrame(Evaluator(model).run(log).items(), columns=['metric', 'value'])


_METRICS = {'nll': _nll}


def register_metric(name, func):
    """Register a model/log callable without replacing an existing metric."""
    from .registry import LEVELS, METRICS
    if name in _METRICS or name in METRICS or name in LEVELS or name == 'all':
        raise ValueError(f'Metric already registered: {name}')
    if not isinstance(name, str) or not name or not callable(func):
        raise ValueError('A metric needs a nonempty name and a callable')
    _METRICS[name] = func


def available_metrics():
    """Все имена для :func:`evaluate`: ``nll``, свои метрики, уровни ``level1``…``level4`` и их метрики."""
    from .registry import LEVELS
    rows = [dict(level='nll', metric='nll')] + [dict(level='custom', metric=n) for n in _METRICS if n != 'nll']
    rows += [dict(level=level, metric=name) for level, names in LEVELS.items() for name in names]
    return pd.DataFrame(rows)


def long_format(tables, model=None):
    """Широкие таблицы ``{уровень: DataFrame}`` (старый evaluate, CV) → длинная таблица.

    Колонки ``level, model, metric, value`` (+ ``fold``, если он был). Строки модели без колонки
    ``model`` получают имя ``model``; модель ``"model"`` переименовывается в ``model``, если задано.
    """
    parts = []
    for level, table in tables.items():
        table = table.copy()
        if 'model' not in table.columns:
            table['model'] = 'model'
        if model is not None:
            table['model'] = table['model'].replace('model', model)
        ids = ['model'] + (['fold'] if 'fold' in table.columns else [])
        long = table.melt(id_vars=ids, var_name='metric', value_name='value')
        # Пустые клетки справочной строки train_log — «не применимо»; NaN у модели оставляем.
        long = long.loc[~((long['model'] == 'train_log') & long['value'].isna())]
        parts.append(long.assign(level=level))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    result = pd.concat(parts, ignore_index=True)
    extra = ['fold'] if 'fold' in result.columns else []
    return result[COLUMNS[:2] + extra + COLUMNS[2:]]


def evaluate(model, log, metrics='nll', train=None, label='model'):
    """Метрики уже обученной модели на журнале ``log``; модель не переобучается.

    ``metrics`` — имя, список имён или уровней: ``"nll"`` (NLL с цензурой, v2), свои метрики
    (:func:`register_metric`), уровни ``"level1"``…``"level4"``, отдельные метрики уровней
    (``"km_sup"``, ``"ibs"``, …) или ``"all"`` (``nll`` и все уровни). Список — :func:`available_metrics`.
    Уровни 1–4 считаются через :meth:`ProcessModel.legacy` и требуют обученную модель без
    признаков (или ``train=`` для необученной). Возвращает длинную таблицу
    ``level, model, metric, value``; ``label`` — имя модели в колонке ``model``.

    NLL (``nll_per_interval``, ``nll_per_trace``) включает цензурированную экспозицию;
    неизвестный наблюдаемый переход даёт бесконечный NLL, а не отбрасывается.
    В ``level4`` такие трассы исключаются, а их доля — ``unseen_share``.
    """
    from .registry import LEVELS, METRICS, _expand
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    own, legacy = [], []
    for name in names:
        if name == 'all':
            own.append('nll')
            legacy.extend(_expand('all'))
        elif name in _METRICS:
            own.append(name)
        elif name in LEVELS or name in METRICS:
            legacy.extend(_expand(name))
        else:
            raise ValueError(f'Unknown metric: {name}; see survpm.metrics.available_metrics()')
    tables = []
    for name in dict.fromkeys(own):
        value = _METRICS[name](model, log)
        table = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame({'metric': [name], 'value': [value]})
        if not {'metric', 'value'} <= set(table.columns):
            raise ValueError('Metric tables must contain metric/value columns')
        tables.append(table[['metric', 'value']].assign(level='nll' if name == 'nll' else 'custom', model=label))
    if legacy:
        bridge = model.legacy(train) if hasattr(model, 'legacy') else model
        tables.append(long_format(bridge.evaluate(log, list(dict.fromkeys(legacy))), label))
    if not tables:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(tables, ignore_index=True)[COLUMNS]


def compare(models, log, metrics='nll', train=None):
    """Одна таблица ``level, model, metric, value`` для словаря ``{имя: модель}`` или списка
    (имена ``model_0``, ``model_1``, …)."""
    models = models.items() if isinstance(models, Mapping) else ((f'model_{i}', m) for i, m in enumerate(models))
    tables = [evaluate(model, log, metrics, train=train, label=name) for name, model in models]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=COLUMNS)


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
