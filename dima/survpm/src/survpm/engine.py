"""Исполнение траекторий и прогноз с ограничением наблюдений по as_of."""
from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from .features import EventLog
from .models.features import DefaultHistoryFeatures as HistoryFeatures

DAY = pd.Timedelta(days=1)


def timestamp(t):
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def observed_history(log, case_id, as_of):
    cutoff = timestamp(as_of)
    events = log.events.loc[(log.events.case_id == case_id) & (log.events.timestamp <= cutoff)].copy()
    measures = log.measurements.loc[(log.measurements.case_id == case_id) & (log.measurements.timestamp <= cutoff)].copy()
    return EventLog(events, log.cases.reindex([case_id]).copy(), measures, pd.Series({case_id: cutoff}))


class ObjectStream:
    def __init__(self, initial, updater=None, seed=None):
        self.seed = seed
        self._objects = deepcopy(initial.to_dict('records') if isinstance(initial, pd.DataFrame) else list(initial))
        if len({r['case_id'] for r in self._objects}) != len(self._objects):
            raise ValueError('case_id объектов должны быть уникальны')
        for row in self._objects:
            row['t'] = timestamp(row['t'])
            row['X'] = dict(row.get('X', {}))
            row['history'] = list(row.get('history', []))
        self.updater = updater or (lambda X, state, t, rng: dict(X))
        self._started = False

    @property
    def objects(self):
        """Independent table of the stream's current object contexts."""
        return pd.DataFrame(deepcopy(self._objects))

    def current(self, case_id):
        row = next(r for r in self._objects if r['case_id'] == case_id)
        return row['state'], dict(row['X']), row['t']

    def history(self, case_id):
        return deepcopy(next(r for r in self._objects if r['case_id'] == case_id)['history'])


class Simulator:
    """Serial executor with stable per-case RNGs shared by updater and model.

    Set seed on either the engine or the stream; supplying both is an error.
    max_steps=None allows unlimited strictly advancing transitions. A reachable
    cycle additionally requires a finite horizon; the default cap is 1000.
    """
    def __init__(self, model, seed=None, executor=None):
        if executor is not None:
            raise NotImplementedError('Custom executor is not supported')
        self.model, self.seed, self.executor = model, seed, executor

    def run(self, stream, horizon=None, max_steps=1000):
        if stream._started:
            raise ValueError('Передайте свежий ObjectStream для нового запуска')
        if horizon is not None and (not np.isfinite(horizon) or horizon < 0):
            raise ValueError('horizon должен быть конечным и неотрицательным')
        if max_steps is not None and (not isinstance(max_steps, (int, np.integer)) or max_steps < 1):
            raise ValueError('max_steps должен быть положительным')
        if max_steps is None and horizon is None:
            visited, active = set(), set()
            def cycle(state):
                if state in active:
                    return True
                if state in visited:
                    return False
                active.add(state)
                if any(cycle(dst) for dst in self.model.network.outgoing(state)):
                    return True
                active.remove(state)
                visited.add(state)
                return False
            if any(cycle(row['state']) for row in stream._objects):
                raise ValueError('Reachable cycles require horizon or max_steps')
        # A stream seed is a fallback; two explicit seeds are ambiguous.
        if self.seed is not None and stream.seed is not None:
            raise ValueError('Specify seed on the engine or stream, not both')
        seed = self.seed if self.seed is not None else stream.seed
        entropy = np.random.SeedSequence(seed).entropy
        stream._started = True
        events, measurements, observations, last_seen = [], [], [], {}
        for row in stream._objects:
            # Built-in hash is process-randomized; digest canonical case identity.
            identity = json.dumps([type(row['case_id']).__name__, row['case_id']],
                                  default=str, ensure_ascii=False).encode('utf-8')
            digest = hashlib.sha256(identity).digest()
            words = np.frombuffer(digest, dtype='<u4').tolist()
            rng = np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=words))
            start = row['t']
            limit = start + horizon*DAY if horizon is not None else None
            reason = 'max_steps'
            for step in itertools.count():
                state, now = row['state'], row['t']
                if step or not row.get('initialized', False):
                    row['X'] = dict(stream.updater(dict(row['X']), state, now, rng))
                    row['history'].append(dict(activity=state, timestamp=now, X=dict(row['X'])))
                    row['X'] = self.model.history_features.at_entry(row['X'], state, now, row['history'])
                events.append(dict(case_id=row['case_id'], activity=state, timestamp=now))
                for feature, value in row['X'].items():
                    measurements.append(dict(case_id=row['case_id'], timestamp=now, feature=feature, value=value))
                if state in self.model.network.terminals:
                    reason = ''
                    break
                if limit is not None and now >= limit:
                    row['t'], reason = limit, 'horizon'
                    break
                if max_steps is not None and step == max_steps:
                    break
                if not self.model.network.outgoing(state):
                    reason = 'dead_end'
                    break
                elapsed = row.get('elapsed', 0) if step == 0 else 0
                dst, dt = self.model.next_transition(state, row['X'], row['history'], rng, elapsed=elapsed)
                if dst is None and dt == np.inf:
                    reason = 'dead_end'
                    break
                if dst not in self.model.network.outgoing(state) or not np.isfinite(dt) or dt < 0:
                    raise ValueError('Модель вернула недопустимый переход или длительность')
                next_time = now + dt*DAY
                if max_steps is None and next_time <= now:
                    raise ValueError('Unlimited steps require strictly advancing time')
                if limit is not None and next_time > limit:
                    row['t'], reason = limit, 'horizon'
                    break
                row['state'], row['t'] = dst, next_time
            last_seen[row['case_id']] = row['t']
            observations.append(dict(case_id=row['case_id'], end=row['t'], reason=reason,
                                     event=int(row['state'] in self.model.network.terminals)))
        log = EventLog(pd.DataFrame(events, columns=['case_id', 'activity', 'timestamp']),
                       pd.DataFrame(index=pd.Index([r['case_id'] for r in stream._objects], name='case_id')),
                       pd.DataFrame(measurements, columns=['case_id', 'timestamp', 'feature', 'value']),
                       pd.Series(last_seen), pd.DataFrame(observations, columns=['case_id', 'end', 'event', 'reason']))
        return log


class Router:
    def __init__(self, model, history_features=None, seed=None):
        self.model = model
        self.history_features = history_features or model.history_features
        self.seed = seed

    def run(self, log, case_id, as_of, future_features=None, horizon=None):
        ctx = self.history_features.transform(observed_history(log, case_id, as_of), case_id, as_of)
        ctx['initialized'] = True
        # Сценарий получает только X и сгенерированное будущее, не полный реальный журнал.
        return Simulator(self.model, self.seed).run(ObjectStream([ctx], future_features), horizon=horizon)


class Forecaster:
    def __init__(self, model, history_features=None, method='simulation', n_paths=1000, seed=None):
        if method != 'simulation':
            raise NotImplementedError(f'Forecaster method={method!r} is not implemented; use simulation')
        if n_paths < 1:
            raise ValueError('n_paths должен быть положительным')
        self.model, self.history_features = model, history_features or model.history_features
        self.n_paths, self.seed = n_paths, seed

    def _paths(self, history, as_of, times, case_id, future_features):
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError('Задайте неотрицательные конечные горизонты')
        ids = history.events.case_id.unique()
        if case_id is None and len(ids) != 1:
            raise ValueError('Для нескольких объектов укажите case_id')
        selected_id = ids[0] if case_id is None else case_id
        ctx = self.history_features.transform(observed_history(history, selected_id, as_of), selected_id, as_of)
        initial = [dict(ctx, case_id=i, initialized=True) for i in range(self.n_paths)]
        log = Simulator(self.model, self.seed).run(ObjectStream(initial, future_features),
                                                   horizon=float(times.max()), max_steps=None)
        last = log.events.groupby('case_id', sort=False).tail(1).set_index('case_id').reindex(range(self.n_paths))
        ends = log.observation.set_index('case_id').reindex(range(self.n_paths)).end
        duration = ((ends-timestamp(as_of))/DAY).to_numpy()
        terminal = last.activity.isin(self.model.network.terminals).to_numpy()
        return times, duration, terminal, last.activity.to_numpy()

    def survival(self, history, as_of, times, future_features=None, case_id=None):
        times, duration, terminal, _ = self._paths(history, as_of, times, case_id, future_features)
        cdf = [np.mean(terminal & (duration <= t)) for t in times]
        return pd.DataFrame(dict(horizon=times, survival=1-np.array(cdf), cdf=cdf))

    def terminal_probs(self, history, as_of, times, future_features=None, case_id=None):
        times, duration, terminal, states = self._paths(history, as_of, times, case_id, future_features)
        return pd.DataFrame([dict(horizon=t, outcome=k, probability=np.mean(terminal & (states == k) & (duration <= t)))
                             for t in times for k in self.model.network.terminals])

    def hazard(self, history, as_of, times, future_features=None, case_id=None):
        """Cause-specific bin rates: completions / paths at risk / bin width.

        Each horizon is the right bin edge; bins start at zero. With no paths
        remaining at risk the hazard is undefined (NaN).
        """
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(np.r_[0., times]) <= 0).any():
            raise ValueError('hazard times must be strictly increasing and positive')
        times, duration, terminal, states = self._paths(history, as_of, times, case_id, future_features)
        rows = []
        for left, right in zip(np.r_[0., times[:-1]], times):
            risk = np.count_nonzero(~terminal | (duration > left))
            for outcome in self.model.network.terminals:
                count = np.count_nonzero(terminal & (states == outcome) & (duration > left) & (duration <= right))
                rows.append(dict(horizon=right, outcome=outcome,
                                 hazard=count/risk/(right-left) if risk else np.nan))
        return pd.DataFrame(rows, columns=['horizon', 'outcome', 'hazard'])

    def expected_time(self, history, as_of, horizon, future_features=None, case_id=None):
        """Restricted mean time to any terminal, and unfinished mass at horizon."""
        _, duration, terminal, _ = self._paths(history, as_of, [horizon], case_id, future_features)
        completed = terminal & (duration <= horizon)
        return dict(restricted_mean_time=float(np.mean(np.where(completed, duration, horizon))),
                    probability_unfinished=float(np.mean(~completed)))


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
