"""Likelihood модели на журнале (:class:`Evaluator`) и прежнее имя построителя признаков истории."""
import numpy as np

from .eventlog import EventLog
from .models.features import DefaultHistoryFeatures as HistoryFeatures  # noqa: F401  (прежнее имя)


class Evaluator:
    """Likelihood по тому же закону, что используется моделью для генерации."""
    def __init__(self, model):
        self.model = model

    def run(self, log, metrics=('nll',)):
        if set(metrics) != {'nll'}:
            raise NotImplementedError('Evaluator ProcessModel пока поддерживает nll; остальные метрики доступны через CensoredStateNet.evaluate')
        from .log import prepare, intervals
        if isinstance(log, EventLog):
            rules = dict(log.rules or {}, starts=self.model.network.starts, terminals=self.model.network.terminals)
            log = log.prepare(rules)
            events, iv = log.events, log.intervals()
        else:
            events, cases = prepare(log, self.model.network.starts, self.model.network.terminals)
            iv = intervals(events, cases)
        frame = None
        columns = getattr(self.model, 'feature_columns', [])
        if columns:
            if not isinstance(log, EventLog):
                raise ValueError('Для признаков передайте EventLog')
            from .process import entry_features
            frame = entry_features(log, events, iv, self.model.history_features, columns)
        nll = -self.model.transitions.log_likelihood(iv, frame)
        return dict(nll_per_interval=nll/len(iv) if len(iv) else np.nan,
                    nll_per_trace=nll/iv.case_id.nunique() if len(iv) else np.nan)
