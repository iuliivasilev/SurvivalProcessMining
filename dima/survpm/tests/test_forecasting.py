"""Базовый Forecaster: пользовательский подкласс, проверки ответа, точное решение против Монте-Карло."""
import numpy as np
import pandas as pd
import pytest

import survpm
from survpm import Forecaster, MarkovForecaster, SimulationForecaster

T0 = pd.Timestamp('2025-01-01', tz='UTC')
NO_MEASUREMENTS = pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value'])


def log_of(*cases):
    rows = [(cid, state, T0 + pd.Timedelta(days=day)) for cid, path in cases for state, day in path]
    events = pd.DataFrame(rows, columns=['case_id', 'activity', 'timestamp'])
    return survpm.EventLog(events, pd.DataFrame(index=pd.Index([c for c, _ in cases], name='case_id')), NO_MEASUREMENTS)


class AlwaysExp(Forecaster):
    """Один терминал D, оставшееся время ~ Exp(0.5)."""
    def cumulative_incidence(self, case, times):
        return pd.DataFrame(dict(horizon=times, outcome='D', cdf=1 - np.exp(-.5*np.asarray(times))))


def test_user_subclass_gets_all_derived_quantities():
    f = AlwaysExp(terminals=['D'])
    log = log_of(('c', [('A', 0)]))
    surv = f.survival(log, T0, [0., 1., 2.])
    assert surv.survival.to_numpy() == pytest.approx(np.exp(-.5*np.array([0., 1., 2.])))
    hazard = f.hazard(log, T0, [.01, .02])
    assert hazard.hazard.to_numpy() == pytest.approx([.5, .5], rel=.01)
    estimate = f.expected_time(log, T0, horizon=2.)
    assert estimate['restricted_mean_time'] == pytest.approx((1 - np.exp(-1))/.5, abs=1e-6)
    assert estimate['probability_unfinished'] == pytest.approx(np.exp(-1))


def test_abstract_method_is_required_and_answer_is_checked():
    class Nothing(Forecaster):
        pass
    with pytest.raises(TypeError):
        Nothing(terminals=['D'])

    class Decreasing(Forecaster):
        def cumulative_incidence(self, case, times):
            return pd.DataFrame(dict(horizon=times, outcome='D', cdf=np.linspace(.9, .1, len(times))))
    with pytest.raises(ValueError, match='убывает'):
        Decreasing(terminals=['D']).survival(log_of(('c', [('A', 0)])), T0, [1., 2.])

    class TooMuch(Forecaster):
        def cumulative_incidence(self, case, times):
            return pd.DataFrame([dict(horizon=t, outcome=k, cdf=.7) for t in times for k in self.terminals])
    with pytest.raises(ValueError, match='больше 1'):
        TooMuch(terminals=['D', 'E']).terminal_probs(log_of(('c', [('A', 0)])), T0, [1.])


def test_context_sees_only_history_until_as_of():
    seen = []

    class Spy(AlwaysExp):
        def cumulative_incidence(self, case, times):
            seen.append((case.state, case.sojourn))
            return super().cumulative_incidence(case, times)
    log = log_of(('c', [('A', 0), ('B', 2), ('D', 5)]))
    Spy(terminals=['D']).survival(log, T0 + pd.Timedelta(days=3), [1.])
    assert seen == [('B', 1.)]


def test_list_of_cases_gives_long_table():
    log = log_of(('a', [('A', 0)]), ('b', [('A', 0)]))
    table = AlwaysExp(terminals=['D']).survival(log, T0, [1., 2.], case_id=['a', 'b'])
    assert table.columns.tolist() == ['case_id', 'horizon', 'survival', 'cdf']
    assert table.case_id.tolist() == ['a', 'a', 'b', 'b']
    many = AlwaysExp(terminals=['D']).expected_time(log, T0, 2., case_id=['a', 'b'])
    assert len(many) == 2


def competing():
    net = survpm.Network.from_edges([('A', 'B'), ('B', 'A'), ('B', 'D'), ('A', 'E')], ['A'], ['D', 'E'])
    rates = {('A', 'B'): 1., ('B', 'A'): .3, ('B', 'D'): .5, ('A', 'E'): .2}
    return survpm.ProcessModel(net, survpm.ExponentialTransitionModel(rates))


def test_markov_matches_closed_form_for_single_exponential():
    net = survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    f = MarkovForecaster(model)
    log = log_of(('c', [('A', 0)]))
    t = np.array([.5, 1., 4.])
    assert f.survival(log, T0, t).survival.to_numpy() == pytest.approx(np.exp(-.5*t))
    estimate = f.expected_time(log, T0, 2.)
    assert estimate['restricted_mean_time'] == pytest.approx((1 - np.exp(-1))/.5)


def test_simulation_agrees_with_exact_markov_within_confidence_interval():
    model = competing()
    log = log_of(('c', [('A', 0), ('B', 1)]))
    as_of = T0 + pd.Timedelta(days=1.5)
    exact = MarkovForecaster(model).terminal_probs(log, as_of, [.5, 2., 8.])
    mc = SimulationForecaster(model, n_paths=4000, seed=3).terminal_probs(log, as_of, [.5, 2., 8.])
    assert (np.abs(mc.probability - exact.probability) <= 4*mc.se + 1e-12).all()
    e_exact = MarkovForecaster(model).expected_time(log, as_of, 4.)
    e_mc = SimulationForecaster(model, n_paths=4000, seed=3).expected_time(log, as_of, 4.)
    assert e_mc['restricted_mean_time'] == pytest.approx(e_exact['restricted_mean_time'], abs=.06)
    # базовая сетка RMST согласована с точной формулой
    grid = Forecaster.restricted_mean(MarkovForecaster(model), MarkovForecaster(model).context(log, 'c', as_of), 4.)
    assert grid[0] == pytest.approx(e_exact['restricted_mean_time'], abs=1e-5)


def test_forecaster_with_model_keeps_old_constructor():
    model = competing()
    assert type(Forecaster(model, n_paths=10, seed=1)) is SimulationForecaster
    assert type(Forecaster(model, method='markov')) is MarkovForecaster
    with pytest.raises(NotImplementedError, match='exact'):
        Forecaster(model, method='exact')
