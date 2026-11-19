"""Базовый класс генерации: пользовательский подкласс, хуки и готовые реализации."""
import numpy as np
import pandas as pd
import pytest

import survpm
from survpm import Case, LogGenerator, Step

T0 = pd.Timestamp('2025-01-01', tz='UTC')


class Coin(LogGenerator):
    """Start → Done с вероятностью 1/2, иначе снова Start; ожидание ~ Exp(1)."""
    terminals = {'Done'}

    def start_cases(self, n, rng):
        return [Case(f'c{i}', 'Start', T0) for i in range(n)]

    def next_step(self, case, rng):
        return Step('Done' if rng.random() < .5 else 'Start', rng.exponential(1.))


def test_user_subclass_plugs_into_fit_and_evaluate():
    log = Coin(seed=1, horizon=30).generate(2000)
    assert isinstance(log, survpm.EventLog)
    assert set(log.observation.columns) == {'case_id', 'end', 'event', 'reason'}
    assert set(log.observation.reason) <= {'', 'horizon'}
    model = survpm.ProcessModel().fit(log.prepare({'starts': ['Start'], 'terminals': ['Done']}))
    rates = model.transitions.rates
    assert rates[('Start', 'Done')] == pytest.approx(.5, abs=.05)
    assert rates[('Start', 'Start')] == pytest.approx(.5, abs=.05)


def test_missing_required_method_is_an_error():
    class OnlyStart(LogGenerator):
        def start_cases(self, n, rng):
            return []
    with pytest.raises(TypeError):
        OnlyStart()


def test_same_seed_same_log_and_order_independence():
    a, b = Coin(seed=3).generate(50), Coin(seed=3).generate(50)
    pd.testing.assert_frame_equal(a.events, b.events)

    class Reversed(Coin):
        def start_cases(self, n, rng):
            return list(reversed(super().start_cases(n, rng)))
    c = Reversed(seed=3).generate(50)
    key = ['case_id', 'timestamp']
    pd.testing.assert_frame_equal(a.events.sort_values(key).reset_index(drop=True),
                                  c.events.sort_values(key).reset_index(drop=True))


def test_censoring_step_dead_end_and_features():
    class Lossy(LogGenerator):
        terminals = {'D'}

        def start_cases(self, n, rng):
            yield Case('x', 'A', T0, X={'visits': 0})
            yield Case('y', 'B', T0)

        def on_enter(self, case, rng):
            return dict(case.X, visits=case.X.get('visits', 0) + 1)

        def next_step(self, case, rng):
            if case.state == 'B':
                return None                       # выхода нет
            if case.step == 2:
                return Step(None, .5, 'lost')     # потеря наблюдения через полдня
            return Step('A', 1.)

    log = Lossy().generate()
    obs = log.observation.set_index('case_id')
    assert obs.loc['x', 'reason'] == 'lost' and obs.loc['x', 'end'] == T0 + pd.Timedelta(days=2.5)
    assert obs.loc['y', 'reason'] == 'dead_end' and obs.loc['y', 'event'] == 0
    visits = log.measurements.loc[log.measurements.case_id == 'x', 'value'].tolist()
    assert visits == [1, 2, 3]


def test_invalid_step_and_duplicate_ids():
    class Bad(Coin):
        def next_step(self, case, rng):
            return Step('Done', -1.)
    with pytest.raises(ValueError, match='длительность'):
        Bad().generate(1)

    class Twice(Coin):
        def start_cases(self, n, rng):
            return [Case('a', 'Start', T0), Case('a', 'Start', T0)]
    with pytest.raises(ValueError, match='уникальны'):
        Twice().generate()


def test_truth_generator_recovers_branch_probabilities():
    edges = [dict(src='A', dst='B', p=.7, dist=('exponential', {'mean': 1.})),
             dict(src='A', dst='D', p=.3, dist=('gamma', {'shape': 2., 'scale': 1.})),
             dict(src='B', dst='D', p=1., dist=lambda rng: 2.)]
    traces = survpm.TruthGenerator(edges, 'A', 'D', seed=1).traces(4000)
    share = np.mean([states[1] == 'B' for states, _ in traces])
    assert share == pytest.approx(.7, abs=.03)
    assert all(times[2] - times[1] == pytest.approx(2.) for states, times in traces if states[1] == 'B')


def test_generate_ad_log_matches_generate_ad():
    raw, censoring = survpm.datasets.generate_ad(200, seed=4)
    log = survpm.datasets.generate_ad_log(200, seed=4)
    assert log.events.case_id.nunique() == raw.case_id.nunique() == 200
    assert (log.observation.event == 0).sum() == len(censoring)
    reasons = log.observation.loc[log.observation.event == 0, 'reason'].replace({'horizon': 'study_end'})
    assert sorted(reasons) == sorted(censoring.reason)


def test_simulator_is_model_generator():
    model = survpm.ProcessModel(survpm.Network.from_edges([('A', 'D')], ['A'], ['D']),
                                survpm.ExponentialTransitionModel({('A', 'D'): 1.}))
    stream = survpm.ObjectStream.start(20, 'A', T0)
    a = survpm.Simulator(model, seed=2).run(stream, horizon=5)
    b = survpm.ModelGenerator(model, stream, seed=2, horizon=5).generate()
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.observation, b.observation)
