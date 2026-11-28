"""Генерация журналов: базовый класс :class:`LogGenerator` и :class:`ModelGenerator`.

Вход и выход — :class:`~survpm.EventLog`. Генератор продолжает каждый объект входного журнала
от его ``end`` («сейчас»). Общий цикл записан в базовом классе и не меняется::

    while есть незавершённые объекты:
        X    = data.at(log)                    # StreamData: где объект и какие признаки
        move = self.step(X, rng)               # куда и через сколько (у ModelGenerator:
                                               #   selector.select(X, model.predict(X), rng))
        log  = log.append(move)                # новое событие, end += dt

Свой генератор — подкласс с одним методом :meth:`LogGenerator.step`::

    class Coin(survpm.LogGenerator):
        terminals = {"Done"}

        def step(self, X, rng, **kw):          # по строке на объект: case_id, dst, dt
            return pd.DataFrame(dict(case_id=X.case_id,
                                     dst=["Done" if r.random() < .5 else "Start" for r in rng],
                                     dt=[r.exponential(1.) for r in rng]))

    log = Coin(seed=1).generate(survpm.EventLog.start(200, "Start"), horizon=30)
    log = log.censor(survpm.StudyEnd(days=20))          # цензура — поверх готового журнала
    log.traces()                                        # трассы того же прогона
"""
from abc import ABC, abstractmethod
import hashlib
import json

import numpy as np
import pandas as pd

from .eventlog import EventLog, MEASUREMENT_COLUMNS
from .selection import SampleSelector
from .streamdata import DefaultStreamData, RESERVED

DAY = pd.Timedelta(days=1)


def case_rng(seed, case_id):
    """Поток случайных чисел объекта: зависит только от ``seed`` и ``case_id``."""
    identity = json.dumps([type(case_id).__name__, case_id], default=str, ensure_ascii=False).encode('utf-8')
    words = np.frombuffer(hashlib.sha256(identity).digest(), dtype='<u4').tolist()
    entropy = np.random.SeedSequence(seed).entropy
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=words))


class LogGenerator(ABC):
    """Базовый генератор. Обязательный метод — :meth:`step`.

    ``data`` — :class:`~survpm.StreamData` (по умолчанию :class:`~survpm.DefaultStreamData`),
    ``terminals`` — завершающие состояния, ``seed`` — у каждого объекта свой поток случайных
    чисел по ``case_id``, поэтому результат не зависит от порядка объектов.
    """
    terminals = frozenset()

    def __init__(self, data=None, seed=None, terminals=None, **kw):
        self.data = data if data is not None else DefaultStreamData()
        self.seed = seed
        if terminals is not None:
            self.terminals = frozenset([terminals] if isinstance(terminals, str) else terminals)

    # --- обязательно -------------------------------------------------------------------------
    @abstractmethod
    def step(self, X, rng, **kw):
        """Следующий переход каждой строки X: таблица ``case_id, dst, dt`` (+ ``reason``).

        ``rng`` — список потоков случайных чисел, по одному на строку X. ``dst=None`` —
        выхода нет (``dead_end``) или наблюдение обрывается через ``dt`` (``reason="lost"``).
        """

    # --- общий цикл ----------------------------------------------------------------------------
    def generate(self, log, max_steps=1000, horizon=None, **kw):
        """Продолжить каждый объект ``log`` от его ``end``; результат — :class:`~survpm.EventLog`.

        Остановка объекта (``observation.reason``): ``""`` — дошёл до терминала, ``horizon`` —
        прошло ``horizon`` дней от входного ``end``, ``max_steps`` — сделано ``max_steps`` шагов
        (``None`` — без предела), ``dead_end`` — выхода нет, ``lost`` или своя причина из ``step``.
        Признаки, которые ``data.at`` вернул и которые отличаются от последних записанных,
        пишутся в журнал замерами на момент входа в состояние.
        """
        if not isinstance(log, EventLog):
            raise TypeError('generate(log): нужен EventLog (например, EventLog.start(n, state) или log.cut(t))')
        if horizon is not None and (not np.isfinite(horizon) or horizon < 0):
            raise ValueError('horizon должен быть конечным и неотрицательным')
        if max_steps is not None and (not isinstance(max_steps, (int, np.integer)) or max_steps < 1):
            raise ValueError('max_steps должен быть положительным')
        if len(set(log.case_ids)) != len(log.case_ids):
            raise ValueError('case_id объектов должны быть уникальны')
        terminals = set(self.terminals)
        origin = log.now.to_dict()                 # «сейчас» на входе; время ведём в днях от него
        days = {c: 0. for c in origin}
        now = dict(origin)
        rngs = {c: case_rng(self.seed, c) for c in log.case_ids}
        observation = log.observation.set_index('case_id')[['end', 'event', 'reason']].to_dict('index')
        events = log.events.loc[log.events.timestamp <= log.events.case_id.map(log.now)]
        static = set(log.cases.columns)
        recorded = (log.measurements.sort_values('timestamp', kind='stable')
                    .groupby(['case_id', 'feature']).value.last().to_dict())
        new_events, new_measures = [], []
        state = DefaultStreamData.current(log).set_index('case_id').state.to_dict()
        active = [c for c in log.case_ids if c in state and state[c] not in terminals]
        step_no = 0
        while active:
            if max_steps is not None and step_no >= max_steps:
                for c in active:
                    observation[c].update(event=0, reason='max_steps')
                break
            view = self._view(log, events, new_events, new_measures, active, now)
            X = self.data.at(view, rng=[rngs[c] for c in active], **kw)
            X = X.set_index('case_id').loc[active].reset_index()
            self._record(X, now, recorded, static, new_measures, entry=step_no > 0)
            moves = pd.DataFrame(self.step(X, [rngs[c] for c in X.case_id], log=view, **kw))
            if not {'case_id', 'dst', 'dt'} <= set(moves.columns):
                raise ValueError('step должен вернуть колонки case_id, dst, dt')
            moves = moves.set_index('case_id').reindex(active)
            still = []
            for c, move in moves.iterrows():
                dst, dt = move['dst'], float(move['dt'])
                reason = move.get('reason') if 'reason' in moves else None
                dst = None if dst is None or (isinstance(dst, float) and np.isnan(dst)) else dst
                if dst is None and not np.isfinite(dt):
                    observation[c].update(end=now[c], event=0, reason=reason or 'dead_end')
                    continue
                if not np.isfinite(dt) or dt < 0:
                    raise ValueError('Генератор вернул недопустимую длительность')
                if max_steps is None and dt <= 0 and dst is not None:
                    raise ValueError('Без max_steps время должно строго расти (unlimited steps require strictly advancing time)')
                elapsed = days[c] + dt
                if horizon is not None and elapsed > horizon:
                    now[c] = origin[c] + horizon*DAY
                    observation[c].update(end=now[c], event=0, reason='horizon')
                    continue
                days[c] = elapsed
                arrival = now[c] = origin[c] + elapsed*DAY
                if dst is None:
                    observation[c].update(end=arrival, event=0, reason=reason or 'lost')
                    continue
                new_events.append((c, dst, arrival))
                if dst in terminals:
                    observation[c].update(end=arrival, event=1, reason='')
                else:
                    observation[c].update(end=arrival, event=0, reason='')
                    still.append(c)
            active = still
            step_no += 1
        added = pd.DataFrame(new_events, columns=['case_id', 'activity', 'timestamp'])
        all_events = pd.concat([events, added], ignore_index=True) if len(added) else events.reset_index(drop=True)
        all_events = all_events.sort_values(['timestamp'], kind='stable')
        all_events = all_events.set_index('case_id').loc[[c for c in log.case_ids if c in set(all_events.case_id)]].reset_index()
        measures = pd.DataFrame(new_measures, columns=MEASUREMENT_COLUMNS)
        measurements = pd.concat([log.measurements, measures], ignore_index=True) if len(measures) else log.measurements.copy()
        obs = pd.DataFrame([dict(case_id=c, **observation[c]) for c in log.case_ids],
                           columns=['case_id', 'end', 'event', 'reason'])
        obs['event'] = obs.event.astype(int)
        return EventLog(all_events[['case_id', 'activity', 'timestamp']], log.cases.copy(), measurements,
                        pd.Series(obs.set_index('case_id').end), obs,
                        None if log.rules is None else dict(log.rules))

    def _view(self, log, events, new_events, new_measures, active, now):
        ids = set(active)
        base = events.loc[events.case_id.isin(ids)]
        added = pd.DataFrame([e for e in new_events if e[0] in ids], columns=['case_id', 'activity', 'timestamp'])
        ev = pd.concat([base, added], ignore_index=True) if len(added) else base
        ms = log.measurements.loc[log.measurements.case_id.isin(ids)]
        extra = pd.DataFrame([m for m in new_measures if m[0] in ids], columns=MEASUREMENT_COLUMNS)
        ms = pd.concat([ms, extra], ignore_index=True) if len(extra) else ms
        ends = pd.Series({c: now[c] for c in active})
        obs = pd.DataFrame(dict(case_id=active, end=[now[c] for c in active], event=0, reason=''))
        view = EventLog.__new__(EventLog)        # без пересчёта last_seen: «сейчас» задаём явно
        view.events, view.cases, view.measurements = ev, log.cases.loc[log.cases.index.isin(ids)], ms
        view.last_seen, view.observation, view.rules = ends, obs, log.rules
        return view

    def _record(self, X, now, recorded, static, out, entry):
        if not entry:          # на старте признаки взяты из журнала: писать нечего
            return
        derived = set(getattr(self.data, 'derived', ())) | set(RESERVED) | static
        names = [c for c in X.columns if c not in derived
                 and not (c.endswith('_age') or c.endswith('_n')) and not c.startswith('history_')]
        for row in X[['case_id', *names]].itertuples(index=False):
            c = row[0]
            for name, value in zip(names, row[1:]):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                if recorded.get((c, name)) != value:
                    recorded[c, name] = value
                    out.append((c, now[c], name, value))


class ModelGenerator(LogGenerator):
    """Генерация по модели: ``step = selector.select(X, model.predict(X), rng)``.

    ``model`` — :class:`~survpm.ProcessModel` или любая модель с ``predict(X)`` и ``network``
    (например :class:`~survpm.ChoiceThenTimeModel`). ``selector`` — правило выбора
    (по умолчанию :class:`~survpm.SampleSelector`), ``data`` — признаки (по умолчанию
    :class:`~survpm.DefaultStreamData` с построителем истории модели).
    """
    def __init__(self, model, selector=None, data=None, seed=None, **kw):
        if data is None:
            data = DefaultStreamData(getattr(model, 'history_features', None))
        super().__init__(data, seed, model.network.terminals, **kw)
        self.model = model
        self.selector = selector if selector is not None else SampleSelector()

    def step(self, X, rng, log=None, **kw):
        return self.selector.select(X, self.model.predict(X, log=log), rng, model=self.model)
