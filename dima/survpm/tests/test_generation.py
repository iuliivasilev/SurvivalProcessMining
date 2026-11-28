"""Генерация: формат EventLog (start, cut, censor, traces), StreamData, LogGenerator, ModelGenerator."""
import numpy as np
import pandas as pd
import pytest

import survpm
from survpm import EventLog, LogGenerator

T0 = pd.Timestamp('2025-01-01', tz='UTC')
DAY = pd.Timedelta(days=1)


class Coin(LogGenerator):
    """Start → Done с вероятностью 1/2, иначе снова Start; ожидание ~ Exp(1)."""
    terminals = {'Done'}

    def step(self, X, rng, **kw):
        return pd.DataFrame(dict(case_id=X.case_id, dst=['Done' if r.random() < .5 else 'Start' for r in rng],
                                 dt=[r.exponential(1.) for r in rng]))


def exp_model(edges=(('A', 'D'),), rates=None, starts=('A',), terminals=('D',)):
    net = survpm.Network.from_edges(list(edges), list(starts), list(terminals))
    return survpm.ProcessModel(net, survpm.ExponentialTransitionModel(rates or {e: 1. for e in edges}))


def log_of(*cases):
    rows = [(cid, state, T0 + day*DAY) for cid, path in cases for state, day in path]
    events = pd.DataFrame(rows, columns=['case_id', 'activity', 'timestamp'])
    return EventLog(events, pd.DataFrame(index=pd.Index([c for c, _ in cases], name='case_id')),
                    pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value']))


def test_user_generator_one_method_plugs_into_fit():
    log = Coin(seed=1).generate(EventLog.start(2000, 'Start', T0), horizon=30)
    assert isinstance(log, EventLog)
    assert set(log.observation.reason) <= {'', 'horizon'}
    model = survpm.ProcessModel().fit(log.prepare({'starts': ['Start'], 'terminals': ['Done']}))
    assert model.transitions.rates[('Start', 'Done')] == pytest.approx(.5, abs=.05)
    assert model.transitions.rates[('Start', 'Start')] == pytest.approx(.5, abs=.05)
    with pytest.raises(TypeError):
        type('Nothing', (LogGenerator,), {})()


def test_same_seed_same_log_and_order_independence():
    a = Coin(seed=3).generate(EventLog.start(50, 'Start', T0))
    b = Coin(seed=3).generate(EventLog.start(50, 'Start', T0))
    pd.testing.assert_frame_equal(a.events, b.events)
    start = EventLog.start(50, 'Start', T0)
    reversed_log = start.select(start.case_ids[::-1])
    reversed_log.events = reversed_log.events.iloc[::-1].reset_index(drop=True)
    c = Coin(seed=3).generate(reversed_log)
    key = ['case_id', 'timestamp']
    pd.testing.assert_frame_equal(a.events.sort_values(key).reset_index(drop=True),
                                  c.events.sort_values(key).reset_index(drop=True))


def test_dead_end_lost_invalid_and_duplicates():
    class Lossy(LogGenerator):
        terminals = {'D'}

        def step(self, X, rng, **kw):
            rows = []
            for cid, state in zip(X.case_id, X.state):
                if state == 'B':
                    rows.append((cid, None, np.inf, None))          # выхода нет
                elif len(self.seen.setdefault(cid, [])) == 2:
                    rows.append((cid, None, .5, 'lost'))            # потеря наблюдения
                else:
                    self.seen[cid].append(1)
                    rows.append((cid, 'A', 1., None))
            return pd.DataFrame(rows, columns=['case_id', 'dst', 'dt', 'reason'])
    gen = Lossy()
    gen.seen = {}
    start = pd.concat([EventLog.start(1, 'A', T0, prefix='x').events, EventLog.start(1, 'B', T0, prefix='y').events])
    log = gen.generate(EventLog(start, pd.DataFrame(index=['x0', 'y0']), pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value'])))
    obs = log.observation.set_index('case_id')
    assert obs.loc['x0', 'reason'] == 'lost' and obs.loc['x0', 'end'] == T0 + 2.5*DAY
    assert obs.loc['y0', 'reason'] == 'dead_end' and obs.loc['y0', 'event'] == 0

    class Bad(Coin):
        def step(self, X, rng, **kw):
            return pd.DataFrame(dict(case_id=X.case_id, dst='Done', dt=-1.))
    with pytest.raises(ValueError, match='длительность'):
        Bad().generate(EventLog.start(1, 'Start', T0))
    with pytest.raises(ValueError, match='max_steps'):
        Coin().generate(EventLog.start(1, 'Start', T0), max_steps=0)
    with pytest.raises(TypeError):
        Coin().generate(10)                                    # вход — журнал, а не число


def test_cut_censor_traces():
    log = log_of(('a', [('A', 0), ('B', 2), ('D', 5)]), ('b', [('A', 4), ('D', 6)]))
    log.observation['event'] = 1
    log.observation['reason'] = ''
    cut = log.cut(T0 + 3*DAY)
    assert cut.case_ids == ['a']                              # b начался позже
    assert cut.events.activity.tolist() == ['A', 'B']
    obs = cut.observation.iloc[0]
    assert obs.end == T0 + 3*DAY and obs.event == 0 and obs.reason == 'cut'
    late = log.cut(T0 + 10*DAY).observation.set_index('case_id')
    assert late.event.tolist() == [1, 1]                       # закончились раньше — без изменений
    per_case = log.cut(pd.Series({'a': T0 + DAY, 'b': T0 + 5*DAY}))
    assert per_case.observation.set_index('case_id').end.tolist() == [T0 + DAY, T0 + 5*DAY]
    censored = log.censor(survpm.StudyEnd(days=3))
    c = censored.observation.set_index('case_id')
    assert c.loc['a', 'reason'] == 'study_end' and c.loc['a', 'end'] == T0 + 3*DAY
    assert c.loc['b', 'event'] == 1                            # у b исход через 2 дня от старта
    assert log.traces() == [(['A', 'B', 'D'], [0., 2., 5.]), (['A', 'D'], [0., 2.])]
    both = log.censor(survpm.StudyEnd(days=100) + survpm.RandomLoss(mean_days=.01), seed=1)
    assert set(both.observation.reason) == {'lost'}


def test_stream_data_at_gives_one_row_per_object():
    log = log_of(('c42', [('A', 0), ('Admission', 2), ('Admission', 2.5)]))
    log = log.add_measurements(pd.DataFrame(dict(case_id='c42', timestamp=[T0 + 1*DAY, T0 + 2.7*DAY],
                                                 feature='CRP', value=[120., 50.])))
    X = survpm.DefaultStreamData().at(log.cut(T0 + 3*DAY))
    row = X.iloc[0]
    assert list(X.columns[:4]) == ['case_id', 'state', 'elapsed', 'time']
    # наблюдение кончилось на последнем замере (2.7 дня): «сейчас» не позже него
    assert (row.state, row.elapsed, row.time) == ('Admission', pytest.approx(.2), T0 + 2.7*DAY)
    assert row.CRP == 120.                                      # замер после входа не виден
    assert row.history_visits == 2 and row.history_elapsed_days == pytest.approx(2.5)


def test_choice_then_time_model_recovers_branch_probabilities():
    edges = [dict(src='A', dst='B', p=.7, dist=('exponential', {'mean': 1.})),
             dict(src='A', dst='D', p=.3, dist=('gamma', {'shape': 2., 'scale': 1.})),
             dict(src='B', dst='D', p=1., dist=lambda rng: 2.)]
    truth = survpm.ChoiceThenTimeModel(edges)
    assert truth.network.terminals == ['D']
    traces = survpm.ModelGenerator(truth, seed=1).generate(EventLog.start(4000, 'A', T0)).traces()
    share = np.mean([states[1] == 'B' for states, _ in traces])
    assert share == pytest.approx(.7, abs=.03)
    assert all(times[2] - times[1] == pytest.approx(2.) for states, times in traces if states[1] == 'B')


def test_continuation_from_cut_and_selectors():
    model = exp_model([('A', 'B'), ('A', 'E'), ('B', 'D')], {('A', 'B'): 1., ('A', 'E'): 3., ('B', 'D'): 1.},
                      terminals=('D', 'E'))
    now = log_of(('p', [('A', 0)]))                            # «сейчас» = последнее наблюдение, T0
    best = survpm.ModelGenerator(model, survpm.BestSelector('D'), seed=1).generate(now)
    assert best.events.activity.tolist() == ['A', 'B', 'D']
    worst = survpm.ModelGenerator(model, survpm.WorstSelector('D'), seed=1).generate(now)
    assert worst.events.activity.tolist() == ['A', 'E']
    likely = survpm.ModelGenerator(model, survpm.MostLikelySelector(), seed=1).generate(now)
    assert likely.events.activity.tolist() == ['A', 'E']
    assert likely.events.timestamp.iloc[-1] == T0 + .25*DAY           # среднее время 1/Λ

    class Forbid(survpm.Selector):
        def weights(self, X, steps, **kw):
            table = survpm.SampleSelector().weights(X, steps)
            table.loc[table.dst == 'E', 'weight'] = 0.
            return table
    forced = survpm.ModelGenerator(model, Forbid(), seed=2).generate(EventLog.start(50, 'A', T0))
    assert 'E' not in set(forced.events.activity)


def test_generated_dynamic_features_are_written_as_measurements():
    class Fading(survpm.DefaultStreamData):
        def at(self, log, **kw):
            X = super().at(log, **kw)
            X['severity'] = 1. / X.history_visits
            return X
    model = exp_model([('A', 'A'), ('A', 'D')], {('A', 'A'): 1., ('A', 'D'): .1})
    log = survpm.ModelGenerator(model, data=Fading(), seed=1).generate(EventLog.start(5, 'A', T0), max_steps=3)
    values = log.measurements.loc[log.measurements.feature == 'severity', 'value']
    assert len(values) and values.between(0, 1).all()


def test_generate_ad_log_matches_generate_ad():
    raw, censoring = survpm.datasets.generate_ad(200, seed=4)
    log = survpm.datasets.generate_ad_log(200, seed=4)
    assert log.events.case_id.nunique() == raw.case_id.nunique() == 200
    assert (log.observation.event == 0).sum() == len(censoring)


def test_starts_restart_objects_of_a_log():
    log = log_of(('a', [('A', 0), ('B', 2)]), ('b', [('B', 1)]))
    start = log.starts(prefix='sim-')
    assert start.case_ids == ['sim-a', 'sim-b']
    assert start.events.activity.tolist() == ['A', 'B']
    assert start.now.tolist() == [T0, T0 + DAY]
    model = exp_model([('A', 'B'), ('B', 'D')], {('A', 'B'): 1., ('B', 'D'): 1.})
    first = survpm.ModelGenerator(model, seed=2).generate(start, horizon=5.)
    again = survpm.ModelGenerator(model, seed=2).generate(start, horizon=5.)
    pd.testing.assert_frame_equal(first.events, again.events)          # вход не меняется


def test_generator_horizon_max_steps_empty_and_zero_time_cycles():
    class Fixed:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'D', 2.
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    start = EventLog.start(1, 'A', T0, prefix='one')
    short = survpm.ModelGenerator(survpm.ProcessModel(net, Fixed())).generate(start, horizon=1., max_steps=None)
    assert short.observation.iloc[0].to_dict() == dict(case_id='one0', end=T0 + DAY, event=0, reason='horizon')
    assert short.last_seen['one0'] == short.observation.end.iloc[0]
    complete = survpm.ModelGenerator(survpm.ProcessModel(net, Fixed())).generate(start, max_steps=None)
    assert complete.observation.event.tolist() == [1]

    class Loop:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'A', 0.
    loop = survpm.ProcessModel(survpm.Network.from_edges([('A', 'A')], ['A'], ['D']), Loop())
    assert survpm.ModelGenerator(loop).generate(start, max_steps=2).observation.reason.tolist() == ['max_steps']
    with pytest.raises(ValueError, match='time'):
        survpm.ModelGenerator(loop).generate(start, horizon=1., max_steps=None)
    empty = EventLog.start(0, 'A', T0)
    result = survpm.ModelGenerator(survpm.ProcessModel(net)).generate(empty)
    assert result.events.empty and result.observation.empty
    assert result.observation.columns.tolist() == ['case_id', 'end', 'event', 'reason']


def test_replaceable_model_and_history_features_in_generation():
    class HistoryRate(survpm.ExponentialTransitionModel):
        def next_transition(self, state, X, history, rng, elapsed=0):
            return ('A', 1.) if state == 'A' and X['history_visits'] == 1 else ('D', 1.)
    net = survpm.Network.from_edges([('A', 'A'), ('A', 'D')], ['A'], ['D'])
    log = survpm.ModelGenerator(survpm.ProcessModel(net, HistoryRate())).generate(EventLog.start(1, 'A', T0))
    assert log.events.activity.tolist() == ['A', 'A', 'D']
    assert (log.events.timestamp.diff().dropna() == DAY).all()
    trained = survpm.ProcessModel(net).fit(log, features=['history_visits'])
    assert 'history_visits' in trained.transitions.encoder.columns_

