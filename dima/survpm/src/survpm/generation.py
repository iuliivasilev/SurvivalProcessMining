"""Генерация журналов: базовый класс :class:`LogGenerator` и две готовые реализации.

Свой генератор — подкласс с двумя методами::

    class Coin(survpm.LogGenerator):
        terminals = {"Done"}

        def start_cases(self, n, rng):          # с чего начинают объекты
            for i in range(n):
                yield survpm.Case(f"c{i}", "Start", "2025-01-01")

        def next_step(self, case, rng):         # следующий переход из case.state
            return survpm.Step("Done" if rng.random() < .5 else "Start", rng.exponential(1.))

    log = Coin(seed=1, horizon=30).generate(200)    # EventLog с observation (end, event, reason)

Цикл по шагам, горизонт, ``max_steps``, случайность и сборку :class:`~survpm.EventLog` делает
база. Готовые реализации:

* :class:`ModelGenerator` — по обученной :class:`~survpm.ProcessModel` и
  :class:`~survpm.ObjectStream` (на нём работают ``Simulator``, ``Router``, ``Forecaster``);
* :class:`TruthGenerator` — по заданным рёбрам ``src, dst, p, dist`` (выбор выхода по ``p``,
  время из ``dist``) с потерей наблюдения; на нём ``datasets.generate_ad``.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .eventlog import EventLog

DAY = pd.Timedelta(days=1)


def timestamp(t):
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


@dataclass
class Case:
    """Объект во время генерации.

    ``t`` — момент старта, ``elapsed`` — дни от старта до текущего момента (ведёт база),
    ``horizon`` — свой горизонт наблюдения в днях от старта (иначе общий у генератора),
    ``sojourn`` — сколько дней объект уже провёл в начальном состоянии (для прогноза с середины),
    ``initialized=True`` — признаки на входе в начальное состояние уже посчитаны.
    """
    case_id: object
    state: str
    t: pd.Timestamp
    X: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    horizon: Optional[float] = None
    sojourn: float = 0.
    initialized: bool = False
    elapsed: float = 0.
    step: int = 0

    def __post_init__(self):
        self.t = timestamp(self.t)
        self.X = dict(self.X)
        self.history = list(self.history)

    @property
    def now(self):
        return self.t + self.elapsed*DAY


class Step(NamedTuple):
    """Следующий шаг: переход в ``dst`` через ``dt`` дней.

    ``dst=None`` — наблюдение обрывается через ``dt`` дней без события (цензура, ``reason``
    по умолчанию ``"lost"``), например потеря наблюдения раньше перехода.
    """
    dst: Optional[str]
    dt: float
    reason: Optional[str] = None


@dataclass
class _Path:
    case: Case
    states: list
    times: list
    X: list
    end: float
    reason: str
    event: int


class LogGenerator(ABC):
    """Базовый генератор журнала.

    Обязательные методы: :meth:`start_cases` и :meth:`next_step`. Необязательный хук —
    :meth:`on_enter` (признаки на входе в состояние). ``terminals`` — завершающие состояния.

    ``horizon`` — конец наблюдения в днях от старта объекта (``None`` — без ограничения),
    ``max_steps`` — предел числа переходов (``None`` — без предела). ``rng="per_case"`` даёт
    каждому объекту свой поток случайных чисел по ``case_id`` (результат не зависит от порядка
    объектов), ``rng="shared"`` — один поток на весь журнал по порядку объектов.
    """
    terminals = frozenset()

    def __init__(self, seed=None, horizon=None, max_steps=1000, rng='per_case'):
        if horizon is not None and (not np.isfinite(horizon) or horizon < 0):
            raise ValueError('horizon должен быть конечным и неотрицательным')
        if max_steps is not None and (not isinstance(max_steps, (int, np.integer)) or max_steps < 1):
            raise ValueError('max_steps должен быть положительным')
        if rng not in ('per_case', 'shared'):
            raise ValueError('rng: per_case или shared')
        self.seed, self.horizon, self.max_steps, self.rng_mode = seed, horizon, max_steps, rng

    # --- обязательно -------------------------------------------------------------------------
    @abstractmethod
    def start_cases(self, n, rng):
        """Начальные объекты :class:`Case` (список или генератор; читается по одному)."""

    @abstractmethod
    def next_step(self, case, rng):
        """:class:`Step` из ``case.state`` или ``None``, если выхода нет (``dead_end``)."""

    # --- необязательно -----------------------------------------------------------------------
    def on_enter(self, case, rng):
        """Признаки ``X`` на входе в ``case.state`` в момент ``case.now``; по умолчанию без изменений."""
        return case.X

    def check_step(self, case, step):
        """Проверка шага подкласса; по умолчанию — только конечное неотрицательное ``dt``."""

    # --- общее -------------------------------------------------------------------------------
    def _entropy(self):
        return np.random.SeedSequence(self.seed).entropy

    def case_rng(self, case_id, entropy=None):
        """Поток случайных чисел объекта: зависит только от seed и ``case_id``."""
        identity = json.dumps([type(case_id).__name__, case_id], default=str, ensure_ascii=False).encode('utf-8')
        words = np.frombuffer(hashlib.sha256(identity).digest(), dtype='<u4').tolist()
        entropy = self._entropy() if entropy is None else entropy
        return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=words))

    def _paths(self, n):
        entropy = self._entropy()
        shared = np.random.default_rng(np.random.SeedSequence(entropy))
        seen = set()
        for case in self.start_cases(n, shared):
            if not isinstance(case, Case):
                raise TypeError('start_cases должен возвращать объекты Case')
            if case.case_id in seen:
                raise ValueError('case_id объектов должны быть уникальны')
            seen.add(case.case_id)
            rng = shared if self.rng_mode == 'shared' else self.case_rng(case.case_id, entropy)
            yield self._run_case(case, rng)

    def _run_case(self, case, rng):
        horizon = case.horizon if case.horizon is not None else self.horizon
        states, times, snapshots, reason, end = [], [], [], 'max_steps', None
        while True:
            if case.step or not case.initialized:
                case.X = dict(self.on_enter(case, rng))
            states.append(case.state)
            times.append(case.elapsed)
            snapshots.append(dict(case.X))
            if case.state in self.terminals:
                reason = ''
                break
            if horizon is not None and case.elapsed > horizon:
                reason, end = 'horizon', horizon
                break
            if self.max_steps is not None and case.step == self.max_steps:
                break
            step = self.next_step(case, rng)
            if step is None:
                reason = 'dead_end'
                break
            step = Step(*step)
            if not np.isfinite(step.dt) or step.dt < 0:
                raise ValueError('Генератор вернул недопустимую длительность')
            self.check_step(case, step)
            arrival = case.elapsed + step.dt
            if horizon is not None and arrival > horizon:
                reason, end = 'horizon', horizon
                break
            if step.dst is None:
                reason, end = step.reason or 'lost', arrival
                break
            if self.max_steps is None and step.dt <= 0:
                raise ValueError('Без max_steps время должно строго расти (unlimited steps require strictly advancing time)')
            case.state, case.elapsed, case.step = step.dst, arrival, case.step + 1
        case.elapsed = case.elapsed if end is None else end
        return _Path(case, states, times, snapshots, case.elapsed, reason, int(reason == ''))

    def traces(self, n=None):
        """Быстрый вывод без журнала: ``[(состояния, дни от старта), ...]``."""
        return [(p.states, p.times) for p in self._paths(n)]

    def generate(self, n=None):
        """Журнал :class:`~survpm.EventLog`: события, признаки ``X`` на каждом входе как замеры,
        ``observation`` (``end``, ``event``, ``reason``: ``""`` — дошёл до terminal, иначе
        ``horizon``, ``max_steps``, ``dead_end``, ``lost`` или своя причина)."""
        events, measurements, observations, ids, last_seen = [], [], [], [], {}
        for path in self._paths(n):
            case = path.case
            ids.append(case.case_id)
            entries = [case.t + t*DAY for t in path.times]
            events.extend(dict(case_id=case.case_id, activity=s, timestamp=t) for s, t in zip(path.states, entries))
            for entry, snapshot in zip(entries, path.X):
                measurements.extend(dict(case_id=case.case_id, timestamp=entry, feature=k, value=v)
                                    for k, v in snapshot.items())
            end = case.t + path.end*DAY
            last_seen[case.case_id] = end
            observations.append(dict(case_id=case.case_id, end=end, reason=path.reason, event=path.event))
        return EventLog(pd.DataFrame(events, columns=['case_id', 'activity', 'timestamp']),
                        pd.DataFrame(index=pd.Index(ids, name='case_id')),
                        pd.DataFrame(measurements, columns=['case_id', 'timestamp', 'feature', 'value']),
                        pd.Series(last_seen),
                        pd.DataFrame(observations, columns=['case_id', 'end', 'event', 'reason']))


# --- реализация по обученной модели -----------------------------------------------------------

class ModelGenerator(LogGenerator):
    """Генерация по :class:`~survpm.ProcessModel` из объектов :class:`~survpm.ObjectStream`.

    На входе в состояние ``updater`` потока обновляет ``X``, вход пишется в ``history``, затем
    ``model.history_features.at_entry`` добавляет признаки истории; переход — ``model.next_transition``.
    """
    def __init__(self, model, stream, seed=None, horizon=None, max_steps=1000):
        if seed is not None and stream.seed is not None:
            raise ValueError('Specify seed on the engine or stream, not both')
        super().__init__(seed if seed is not None else stream.seed, horizon, max_steps)
        self.model, self.stream = model, stream
        self.terminals = frozenset(model.network.terminals)
        if max_steps is None and horizon is None and self._has_cycle(r['state'] for r in stream.records()):
            raise ValueError('Reachable cycles require horizon or max_steps')

    def _has_cycle(self, starts):
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
        return any(cycle(s) for s in starts)

    def start_cases(self, n, rng):
        if n is not None:
            raise ValueError('Число объектов задаёт поток; n не указывается')
        for row in self.stream.records():
            yield Case(row['case_id'], row['state'], row['t'], row['X'], row['history'],
                       sojourn=row.get('elapsed', 0), initialized=row.get('initialized', False))

    def on_enter(self, case, rng):
        X = dict(self.stream.updater(dict(case.X), case.state, case.now, rng))
        case.history.append(dict(activity=case.state, timestamp=case.now, X=dict(X)))
        return self.model.history_features.at_entry(X, case.state, case.now, case.history)

    def next_step(self, case, rng):
        if not self.model.network.outgoing(case.state):
            return None
        dst, dt = self.model.next_transition(case.state, case.X, case.history, rng,
                                             elapsed=case.sojourn if case.step == 0 else 0)
        if dst is None and dt == np.inf:
            return None
        return Step(dst, dt)

    def check_step(self, case, step):
        if step.dst not in self.model.network.outgoing(case.state):
            raise ValueError('Модель вернула недопустимый переход или длительность')


# --- реализация по заданной истине ------------------------------------------------------------

def sample_time(dist, rng):
    """Время перехода из ``(семейство, параметры)``: exponential(mean), lognormal(median, sigma),
    gamma(shape, scale), weibull(shape, scale), uniform(low, high); или ``callable(rng)``."""
    if callable(dist):
        return float(dist(rng))
    kind, par = dist
    if kind == 'exponential':
        return rng.exponential(par['mean'])
    if kind == 'lognormal':
        return rng.lognormal(np.log(par['median']), par['sigma'])
    if kind == 'gamma':
        return rng.gamma(par['shape'], par['scale'])
    if kind == 'weibull':
        return par['scale'] * rng.weibull(par['shape'])
    if kind == 'uniform':
        return rng.uniform(par['low'], par['high'])
    raise ValueError(kind)


class TruthGenerator(LogGenerator):
    """Генерация по заданным рёбрам: выход по ``p``, затем время из ``dist`` выбранного ребра.

    ``edges`` — словари ``src, dst, p, dist`` (``dist`` — см. :func:`sample_time`).
    Цензура: ``stop`` — вероятность оборвать наблюдение перед каждым ожиданием ``{state: q}``;
    ``loss_mean_days`` — потеря наблюдения ~ Exp(mean), соревнуется по времени с переходом.
    Старт — ``start`` в момент ``origin`` (+ равномерно ``enroll_days`` дней, если задано).
    """
    def __init__(self, edges, start, terminals, seed=None, horizon=None, max_steps=1000, rng='per_case',
                 stop=None, loss_mean_days=None, origin='2025-01-01', enroll_days=0., prefix='case_'):
        super().__init__(seed, horizon, max_steps, rng)
        self.start, self.terminals = start, frozenset([terminals] if isinstance(terminals, str) else terminals)
        self.stop, self.loss_mean_days = dict(stop or {}), loss_mean_days
        self.origin, self.enroll_days, self.prefix = timestamp(origin), enroll_days, prefix
        self.edges = list(edges)
        outgoing = {}
        for edge in self.edges:
            outgoing.setdefault(edge['src'], []).append(edge)
        self._outgoing = {}
        for src, out in outgoing.items():
            p = np.array([e['p'] for e in out], float)
            if abs(p.sum() - 1) > 1e-12:
                p = p / p.sum()
            self._outgoing[src] = ([e['dst'] for e in out], p, [e['dist'] for e in out])

    def start_cases(self, n, rng):
        self.entries = {}
        for i in range(n):
            entry = rng.uniform(0, self.enroll_days) if self.enroll_days else 0.
            self.entries[f'{self.prefix}{i + 1:04d}'] = entry
            yield Case(f'{self.prefix}{i + 1:04d}', self.start, self.origin + entry*DAY)

    def transition(self, state, rng):
        """``(dst, dt)`` выбранного перехода или ``None``, если выходов нет."""
        if state not in self._outgoing:
            return None
        destinations, p, dists = self._outgoing[state]
        j = rng.choice(len(p), p=p)
        return destinations[j], sample_time(dists[j], rng)

    def has_exit(self, state):
        return state in self._outgoing

    def next_step(self, case, rng):
        if not self.has_exit(case.state):
            return None
        if self.stop and rng.random() < self.stop.get(case.state, 0.):
            return Step(None, 0., 'lost')
        dst, dt = self.transition(case.state, rng)
        if self.loss_mean_days:
            loss = rng.exponential(self.loss_mean_days)
            if loss < dt:
                return Step(None, loss, 'lost')
        return Step(dst, dt)
