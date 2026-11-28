"""Forecaster: пользовательский подкласс, проверки ответа, Markov/Grid/Simulation, Selector."""
import numpy as np
import pandas as pd
import pytest

import survpm
from survpm import Forecaster, GridForecaster, MarkovForecaster, SimulationForecaster

T0 = pd.Timestamp('2025-01-01', tz='UTC')
DAY = pd.Timedelta(days=1)
NO_MEASUREMENTS = pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value'])


def log_of(*cases):
    rows = [(cid, state, T0 + day*DAY) for cid, path in cases for state, day in path]
    events = pd.DataFrame(rows, columns=['case_id', 'activity', 'timestamp'])
    return survpm.EventLog(events, pd.DataFrame(index=pd.Index([c for c, _ in cases], name='case_id')), NO_MEASUREMENTS)


class AlwaysExp(Forecaster):
    """Один терминал D, оставшееся время ~ Exp(0.5)."""
    def cumulative_incidence(self, X, times, **kw):
        return pd.DataFrame(dict(horizon=times, outcome='D', cdf=1 - np.exp(-.5*np.asarray(times))))


def test_user_subclass_gets_all_derived_quantities():
    f = AlwaysExp(terminals=['D'])
    log = log_of(('c', [('A', 0)]))
    surv = f.survival(log, [0., 1., 2.])
    assert surv.columns.tolist() == ['case_id', 'horizon', 'survival', 'cdf']
    assert surv.survival.to_numpy() == pytest.approx(np.exp(-.5*np.array([0., 1., 2.])))
    assert f.hazard(log, [.01, .02]).hazard.to_numpy() == pytest.approx([.5, .5], rel=.01)
    estimate = f.expected_time(log, 2.).iloc[0]
    assert estimate.restricted_mean_time == pytest.approx((1 - np.exp(-1))/.5, abs=1e-6)
    assert estimate.probability_unfinished == pytest.approx(np.exp(-1))
    X = pd.DataFrame(dict(case_id=['x', 'y'], state='A'))      # можно подать готовую таблицу X
    assert f.survival(X, [1.]).case_id.tolist() == ['x', 'y']


def test_abstract_method_is_required_and_answer_is_checked():
    with pytest.raises(TypeError):
        type('Nothing', (Forecaster,), {})(terminals=['D'])

    class Decreasing(Forecaster):
        def cumulative_incidence(self, X, times, **kw):
            return pd.DataFrame(dict(horizon=times, outcome='D', cdf=np.linspace(.9, .1, len(times))))
    with pytest.raises(ValueError, match='убывает'):
        Decreasing(terminals=['D']).survival(log_of(('c', [('A', 0)])), [1., 2.])

    class TooMuch(Forecaster):
        def cumulative_incidence(self, X, times, **kw):
            return pd.DataFrame([dict(horizon=t, outcome=k, cdf=.7) for t in times for k in self.terminals])
    with pytest.raises(ValueError, match='больше 1'):
        TooMuch(terminals=['D', 'E']).terminal_probs(log_of(('c', [('A', 0)])), [1.])


def test_forecast_sees_only_cut_history():
    seen = []

    class Spy(AlwaysExp):
        def cumulative_incidence(self, X, times, **kw):
            seen.append((X.state.iloc[0], X.elapsed.iloc[0]))
            return super().cumulative_incidence(X, times)
    log = log_of(('c', [('A', 0), ('B', 2), ('D', 5)]))
    Spy(terminals=['D']).survival(log.cut(T0 + 3*DAY), [1.])
    assert seen == [('B', 1.)]


def competing():
    net = survpm.Network.from_edges([('A', 'B'), ('B', 'A'), ('B', 'D'), ('A', 'E')], ['A'], ['D', 'E'])
    rates = {('A', 'B'): 1., ('B', 'A'): .3, ('B', 'D'): .5, ('A', 'E'): .2}
    return survpm.ProcessModel(net, survpm.ExponentialTransitionModel(rates))


def ongoing(as_of=1.5):
    log = log_of(('c', [('A', 0), ('B', 1)]), ('d', [('A', 0)]))
    log.observation['end'] = T0 + as_of*DAY             # наблюдаем до as_of
    return log


def test_markov_matches_closed_form_for_single_exponential():
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    log = log_of(('c', [('A', 0)]))
    t = np.array([.5, 1., 4.])
    assert MarkovForecaster(model).survival(log, t).survival.to_numpy() == pytest.approx(np.exp(-.5*t))
    assert MarkovForecaster(model).expected_time(log, 2.).restricted_mean_time.iloc[0] == pytest.approx((1 - np.exp(-1))/.5)


def test_grid_and_simulation_agree_with_exact_markov():
    model, log, times = competing(), ongoing(), [.5, 2., 8.]
    exact = MarkovForecaster(model).terminal_probs(log, times)
    grid = GridForecaster(model, n_grid=800).terminal_probs(log, times)
    assert np.abs(grid.probability - exact.probability).max() < 1e-4
    mc = SimulationForecaster(model, n_paths=4000, seed=3).terminal_probs(log, times)
    assert (np.abs(mc.probability - exact.probability) <= 4*mc.se + 1e-12).all()
    e_exact = MarkovForecaster(model).expected_time(log, 4.)
    e_mc = SimulationForecaster(model, n_paths=4000, seed=3).expected_time(log, 4.)
    assert e_mc.restricted_mean_time.to_numpy() == pytest.approx(e_exact.restricted_mean_time.to_numpy(), abs=.06)
    base = Forecaster.restricted_mean(MarkovForecaster(model), MarkovForecaster(model).features(log, 'c')[0], 4.)
    assert base[0] == pytest.approx(e_exact.restricted_mean_time.iloc[0], abs=1e-5)


def test_grid_handles_non_exponential_time_and_elapsed():
    gamma = ('gamma', {'shape': 3., 'scale': 1.})
    truth = survpm.ChoiceThenTimeModel([dict(src='A', dst='D', p=1., dist=gamma)])
    log = log_of(('c', [('A', 0)]))
    log.observation['end'] = T0 + 2*DAY                  # уже ждёт 2 дня
    t = np.array([.5, 1., 3.])
    from scipy import stats
    exact = (stats.gamma(3).cdf(2 + t) - stats.gamma(3).cdf(2)) / stats.gamma(3).sf(2)
    assert GridForecaster(truth).terminal_probs(log, t).probability.to_numpy() == pytest.approx(exact, abs=1e-5)
    with pytest.raises(ValueError, match='экспоненциальный'):
        MarkovForecaster(truth).survival(log, t)


def test_selector_changes_the_forecast():
    model = competing()
    log = log_of(('c', [('A', 0)]))
    worst = MarkovForecaster(model, selector=survpm.WorstSelector('D')).terminal_probs(log, [50.])
    best = MarkovForecaster(model, selector=survpm.BestSelector('D')).terminal_probs(log, [50.])
    law = MarkovForecaster(model).terminal_probs(log, [50.])
    p = lambda table, k: table.loc[table.outcome == k, 'probability'].iloc[0]
    assert p(worst, 'D') == pytest.approx(0, abs=1e-9) and p(worst, 'E') == pytest.approx(1)
    assert p(best, 'D') == pytest.approx(1, abs=1e-6)
    assert p(worst, 'D') < p(law, 'D') < p(best, 'D')
    sim = SimulationForecaster(model, n_paths=200, seed=1, selector=survpm.BestSelector('D')).terminal_probs(log, [50.])
    assert p(sim, 'D') == 1.


def test_many_cases_give_long_table():
    table = MarkovForecaster(competing()).survival(ongoing(), [1., 2.])
    assert table.columns.tolist() == ['case_id', 'horizon', 'survival', 'cdf']
    assert table.case_id.tolist() == ['c', 'c', 'd', 'd']
    assert len(MarkovForecaster(competing()).expected_time(ongoing(), 2., case_id=['c'])) == 1


def test_self_loop_does_not_leak_probability():
    net = survpm.Network.from_edges([('A', 'A'), ('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'A'): 2., ('A', 'D'): .5}))
    log = log_of(('c', [('A', 0)]))
    t = np.array([1., 10., 40.])
    for f in (MarkovForecaster(model), GridForecaster(model)):
        table = f.survival(log, t)
        tol = 1e-4 if isinstance(f, GridForecaster) else 1e-9     # сетка: O(h²)
        assert table.survival.to_numpy() == pytest.approx(np.exp(-.5*t), abs=tol)


def test_custom_residual_model_and_elapsed_in_simulation():
    class Residual:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'D', max(3 - elapsed, 0)
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, Residual())
    log = log_of(('one', [('A', 0)]))
    log.observation['end'] = T0 + 2*DAY                   # наблюдаем до дня 2: ждёт уже 2 дня
    path = survpm.ModelGenerator(model).generate(log)
    assert path.events.timestamp.iloc[-1] == T0 + 3*DAY
    assert SimulationForecaster(model, n_paths=3).survival(log, [.5, 1.]).survival.tolist() == [1., 0.]
    with pytest.raises(NotImplementedError):
        MarkovForecaster(model).survival(log, [1.])


def test_features_are_cut_and_elapsed_conditions_the_forecast():
    net = survpm.Network.from_edges([('A', 'B'), ('A', 'D')], ['A'], ['B', 'D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'B'): 1., ('A', 'D'): 3.}))
    log = survpm.EventLog(pd.DataFrame([('one', 'A', T0)], columns=['case_id', 'activity', 'timestamp']),
                          pd.DataFrame(index=pd.Index(['one'], name='case_id')),
                          pd.DataFrame([('one', T0, 'x', 1.), ('one', T0 + 20*DAY, 'x', 99.)],
                                       columns=['case_id', 'timestamp', 'feature', 'value']))
    now = log.cut(T0 + 5*DAY)
    X = survpm.DefaultStreamData().at(now)
    assert X.x.iloc[0] == 1. and X.elapsed.iloc[0] == 5.
    table = SimulationForecaster(model, n_paths=10000, seed=2).survival(now, [0., 1.])
    assert table.survival.iloc[0] == 1.
    assert table.survival.iloc[1] == pytest.approx(np.exp(-4), abs=.01)
    outcomes = MarkovForecaster(model).terminal_probs(log.cut(T0), [10.])
    assert outcomes.set_index('outcome').loc['D', 'probability'] == pytest.approx(.75, abs=1e-9)


def test_forecast_unchanged_when_future_measurements_change():
    X = pd.DataFrame({'x': [-1., -1., 1., 1.]})
    iv = pd.DataFrame(dict(case_id=['a', 'b', 'c', 'd'], src='A', dst='D', duration=[4., 4., 1., 1.],
                           observed=True, reason=''))
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel(l2=0).fit(iv, net, X))
    events = pd.DataFrame([('one', 'A', T0)], columns=['case_id', 'activity', 'timestamp'])
    cases = pd.DataFrame(index=pd.Index(['one'], name='case_id'))
    measurements = pd.DataFrame([('one', T0, 'x', -1.), ('one', T0 + 2*DAY, 'x', 99.)],
                                columns=['case_id', 'timestamp', 'feature', 'value'])
    before = survpm.EventLog(events, cases, measurements)
    changed = measurements.copy()
    changed.loc[1, 'value'] = -999.
    after = survpm.EventLog(events, cases, changed)
    f = SimulationForecaster(model, n_paths=200, seed=4)
    pd.testing.assert_frame_equal(f.survival(before.cut(T0), [1., 5.]), f.survival(after.cut(T0), [1., 5.]))


def test_custom_history_builder_sees_only_the_past():
    class PastOnly(survpm.DefaultHistoryFeatures):
        def transform(self, log, case_id, as_of):
            assert (log.events.timestamp <= as_of).all()
            assert (log.measurements.timestamp <= as_of).all()
            return super().transform(log, case_id, as_of)
    events = pd.DataFrame([('one', 'A', T0), ('one', 'D', T0 + 10*DAY)], columns=['case_id', 'activity', 'timestamp'])
    measurements = pd.DataFrame([('one', T0 + 5*DAY, 'x', 100.)], columns=['case_id', 'timestamp', 'feature', 'value'])
    log = survpm.EventLog(events, pd.DataFrame(index=['one']), measurements)
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'D'): 1}), PastOnly())
    survpm.ModelGenerator(model).generate(log.cut(T0), horizon=1.)
    SimulationForecaster(model, n_paths=5).survival(log.cut(T0), [1.])


def test_hazard_rmst_and_long_paths_are_not_truncated():
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    log = log_of(('one', [('A', 0)]))
    f = SimulationForecaster(model, n_paths=4000, seed=10)
    hazard = f.hazard(log, [.1, .2, .3, .4])
    assert list(hazard.columns) == ['case_id', 'horizon', 'outcome', 'hazard']
    assert hazard.hazard.mean() == pytest.approx(.5, abs=.07)
    estimate = f.expected_time(log, 2.).iloc[0]
    assert estimate.restricted_mean_time == pytest.approx((1-np.exp(-1))/.5, abs=.04)
    for times in ([0., 1.], [1., 1.], [2., 1.]):
        with pytest.raises(ValueError):
            f.hazard(log, times)
    loops = survpm.Network.from_edges([('A', 'A'), ('A', 'D')], ['A'], ['D'])
    busy = survpm.ProcessModel(loops, survpm.ExponentialTransitionModel({('A', 'A'): 700., ('A', 'D'): 1e-9}))
    out = survpm.ModelGenerator(busy, seed=1).generate(log, horizon=2., max_steps=None)
    assert len(out.events) > 1000 and out.observation.reason.tolist() == ['horizon']

