"""Общий реестр метрик: зависимости уровня загружаются только при его вычислении."""
from importlib import import_module


def _lazy_metric(module, name):
    def metric(fold):
        return getattr(import_module(f".{module}", __package__), name)(fold)
    metric.__name__ = name
    return metric


LEVELS = {
    "level1": {name: _lazy_metric("process", name) for name in
               ("fitness_token", "precision_token", "generalization", "simplicity")},
    "level2": {name: _lazy_metric("stochastic", name) for name in
               ("entropic_relevance", "uemsc", "emd")},
    "level3": {name: _lazy_metric("temporal", name) for name in
               ("temporal_deviation_share", "cfld", "ngram_3", "cycle_time_emd", "absolute_event_emd",
                "edge_ks", "km_sup", "km_l1_days", "logrank", "ibs")},
    "level4": {name: _lazy_metric("likelihood", name) for name in
               ("unseen_share", "nll_per_interval", "nll_per_trace")},
}
METRICS = {name: metric for metrics in LEVELS.values() for name, metric in metrics.items()}
LEVEL_OF = {name: level for level, metrics in LEVELS.items() for name in metrics}


def _expand(name):
    """``"all"``, уровень или имя метрики → имена встроенных метрик."""
    if name == "all":
        return list(METRICS)
    if name in LEVELS:
        return list(LEVELS[name])
    if name in METRICS:
        return [name]
    raise ValueError(f"Неизвестная метрика {name!r}; есть 'all', {', '.join(LEVELS)}, {', '.join(METRICS)}")
