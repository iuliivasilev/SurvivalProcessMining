import numpy as np
import pandas as pd
import pytest

import survpm


def intervals():
    return pd.DataFrame(dict(case_id=['a', 'b', 'c'], src='A', dst=['B', 'D', None],
                             duration=[1., 2., 3.], observed=[True, True, False], reason=''))


def graph():
    return survpm.Network.from_edges([('A', 'B'), ('A', 'D')], ['A'], ['B', 'D'])


def test_fit_rates_counts_all_exposure_and_likelihood():
    model = survpm.ExponentialTransitionModel().fit(intervals(), graph())
    params = model.transition_parameters('A', {}, [])
    assert params == pytest.approx({'B': 1/6, 'D': 1/6})
    assert model.log_likelihood(intervals()) == pytest.approx(-2*np.log(6)-2)
    assert model.update(intervals()).transition_parameters('A', {}, []) == pytest.approx(params)


def test_minimum_time_winner_and_waiting_distribution():
    model = survpm.ExponentialTransitionModel({('A', 'B'): 1., ('A', 'D'): 3.})
    rng = np.random.default_rng(42)
    samples = [model.next_transition('A', {}, [], rng) for _ in range(10000)]
    assert np.mean([s == 'D' for s, _ in samples]) == pytest.approx(.75, abs=.02)
    assert np.mean([t for _, t in samples]) == pytest.approx(.25, abs=.01)


def test_censored_state_net_default_uses_competing_likelihood_and_grid():
    model = survpm.CensoredStateNet('A', 'D').fit(intervals=intervals())
    assert model.transition_mode == 'min_time'
    assert model.nll(intervals())['nll_per_interval'] == pytest.approx((2*np.log(6)+2)/3)
    # A->B is a dead end; A->D is half the exits, total exit rate 1/3.
    assert model.survival([3.], method='grid')[0] == pytest.approx(.5+.5*np.exp(-1), abs=.002)
    assert model.residual_survival('A', 20., [3.])[0] == pytest.approx(.5+.5*np.exp(-1), abs=.002)


def test_features_recover_rate_difference():
    rng = np.random.default_rng(7)
    x = np.tile([-1., 1.], 400)
    rate = np.exp(.7*x)
    duration = rng.exponential(1/rate)
    iv = pd.DataFrame(dict(case_id=np.arange(len(x)), src='A', dst='D', duration=duration,
                           observed=True, reason=''))
    m = survpm.CensoredStateNet('A', 'D', feature_columns=['x'], l2=0).fit(intervals=iv, X=pd.DataFrame({'x': x}))
    low = m.for_features({'x': -1}).completion_times(10000, 1).mean()
    high = m.for_features({'x': 1}).completion_times(10000, 1).mean()
    assert low/high == pytest.approx(np.exp(1.4), rel=.15)
    assert m.coefficients_table()['rate'].iloc[0] == pytest.approx(.7, abs=.12)


def test_stream_updates_at_entry_records_x_and_horizon():
    net = survpm.Network.from_edges([('A', 'B'), ('B', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'B'): 2., ('B', 'D'): 2.}))
    stream = survpm.ObjectStream(pd.DataFrame([dict(case_id='one', state='A', t=pd.Timestamp('2025-01-01', tz='UTC'), X={'x': 0})]),
                                updater=lambda X, state, t, rng: {'x': X['x']+1})
    log = survpm.Simulator(model, seed=1).run(stream, horizon=10.)
    assert log.events.activity.tolist() == ['A', 'B', 'D']
    assert log.measurements.loc[log.measurements.feature == 'x', 'value'].tolist() == [1, 2, 3]
    assert stream.current('one')[0] == 'D'
    with pytest.raises(ValueError, match='свеж|fresh'):
        survpm.Simulator(model).run(stream)
    short = survpm.ObjectStream(pd.DataFrame([dict(case_id='two', state='A', t=pd.Timestamp('2025-01-01', tz='UTC'), X={})]))
    censored = survpm.Simulator(model, seed=1).run(short, horizon=0.)
    assert censored.events.activity.tolist() == ['A']
    assert censored.last_seen['two'] == pd.Timestamp('2025-01-01', tz='UTC')


def test_forecaster_limits_real_features_to_cutoff_and_conditions_elapsed():
    net = graph()
    model = survpm.ProcessModel(net, survpm.ExponentialTransitionModel({('A', 'B'): 1., ('A', 'D'): 3.}))
    origin = pd.Timestamp('2025-01-01', tz='UTC')
    log = survpm.EventLog(pd.DataFrame([('one', 'A', origin)], columns=['case_id','activity','timestamp']),
                         pd.DataFrame(index=pd.Index(['one'], name='case_id')),
                         pd.DataFrame([('one', origin, 'x', 1.), ('one', origin+pd.Timedelta(days=20), 'x', 99.)],
                                      columns=['case_id','timestamp','feature','value']))
    history = survpm.HistoryFeatures().transform(log, 'one', origin+pd.Timedelta(days=5))
    assert history['X']['x'] == 1.
    assert history['elapsed'] == 5.
    forecast = survpm.Forecaster(model, n_paths=10000, seed=2)
    table = forecast.survival(log, origin+pd.Timedelta(days=5), [0., 1.], case_id='one')
    assert table.survival.iloc[0] == 1.
    assert table.survival.iloc[1] == pytest.approx(np.exp(-4), abs=.01)
    outcomes = forecast.terminal_probs(log, origin, [10.], case_id='one')
    assert outcomes.set_index('outcome').loc['D','probability'] == pytest.approx(.75, abs=.02)


def test_network_validation_and_invalid_rates():
    with pytest.raises(ValueError):
        survpm.ExponentialTransitionModel({('A','D'): -1})
    with pytest.raises(ValueError):
        survpm.Network.from_edges([('A','D')], [], ['D'])
    net = survpm.Network.from_edges([('A','B')], ['A'], ['D'])
    assert 'B' in net.validate().state.tolist()


def test_process_fit_roundtrip_multiple_terminals_and_evaluator():
    origin = pd.Timestamp('2025-01-01', tz='UTC')
    events = pd.DataFrame([('one','A',origin), ('one','B',origin+pd.Timedelta(days=1)),
                           ('two','A',origin), ('two','D',origin+pd.Timedelta(days=2))],
                          columns=['case_id','activity','timestamp'])
    model = survpm.ProcessModel(graph()).fit(events)
    assert model.transition_parameters('A') == pytest.approx({'B': 1/3, 'D': 1/3})
    assert survpm.Evaluator(model).run(events)['nll_per_interval'] == pytest.approx(np.log(3)+1)


def test_zero_event_edge_and_invalid_feature_alignment():
    iv = intervals()
    net = survpm.Network.from_edges([('A','B'), ('A','D'), ('A','E')], ['A'], ['B','D','E'])
    model = survpm.ExponentialTransitionModel().fit(iv, net)
    assert model.transition_parameters('A')['E'] == 0.
    rng = np.random.default_rng(1)
    assert all(model.next_transition('A', {}, [], rng)[0] != 'E' for _ in range(100))
    with pytest.raises(ValueError, match='Индекс'):
        model.fit(iv, net, pd.DataFrame({'x':[1,2,3]}, index=[4,5,6]))


def test_min_time_rejects_nonexponential_instead_of_silent_switch():
    with pytest.raises(NotImplementedError, match='exponential'):
        survpm.CensoredStateNet('A','D', candidates=('weibull',)).fit(intervals=intervals())


def test_forecast_unchanged_when_future_real_measurements_are_changed():
    origin = pd.Timestamp('2025-01-01', tz='UTC')
    X = pd.DataFrame({'x': [-1., -1., 1., 1.]})
    iv = pd.DataFrame(dict(case_id=['a','b','c','d'], src='A', dst='D', duration=[4.,4.,1.,1.],
                           observed=True, reason=''))
    net = survpm.Network.from_edges([('A','D')], ['A'], ['D'])
    transition = survpm.ExponentialTransitionModel(l2=0).fit(iv, net, X)
    model = survpm.ProcessModel(net, transition)
    events = pd.DataFrame([('one','A',origin)], columns=['case_id','activity','timestamp'])
    cases = pd.DataFrame(index=pd.Index(['one'],name='case_id'))
    measurements = pd.DataFrame([('one',origin,'x',-1.), ('one',origin+pd.Timedelta(days=2),'x',99.)],
                                columns=['case_id','timestamp','feature','value'])
    before = survpm.EventLog(events,cases,measurements)
    measurements = measurements.copy()
    measurements.loc[1,'value'] = -999.
    after = survpm.EventLog(events,cases,measurements)
    forecast = survpm.Forecaster(model,n_paths=200,seed=4)
    pd.testing.assert_frame_equal(forecast.survival(before,origin,[1.,5.]), forecast.survival(after,origin,[1.,5.]))


def test_transition_model_rejects_censored_unknown_source():
    model = survpm.ExponentialTransitionModel({('A','D'):1.})
    iv = pd.DataFrame(dict(src=['Z'],dst=[None],duration=[2.],observed=[False]))
    with pytest.raises(ValueError,match='состояни'):
        model.log_likelihood(iv)


def test_simulator_accepts_replaceable_model_without_rate_dictionary():
    class FixedTransition:
        def next_transition(self,state,X,history,rng):
            return 'D', 2.
    net = survpm.Network.from_edges([('A','D')], ['A'], ['D'])
    model = survpm.ProcessModel(net, FixedTransition())
    stream = survpm.ObjectStream([dict(case_id='one',state='A',t='2025-01-01',X={})])
    log = survpm.Simulator(model).run(stream)
    assert log.events.activity.tolist() == ['A','D']
    assert log.events.timestamp.iloc[1]-log.events.timestamp.iloc[0] == pd.Timedelta(days=2)


def test_history_features_are_used_in_fit_and_updated_after_each_entry():
    class HistoryRate(survpm.ExponentialTransitionModel):
        def next_transition(self,state,X,history,rng):
            if state == 'A' and X['history_visits'] == 1:
                return 'A', 1.
            return 'D', 1.
    net = survpm.Network.from_edges([('A','A'),('A','D')], ['A'], ['D'])
    m = survpm.ProcessModel(net, HistoryRate())
    log = survpm.Simulator(m).run(survpm.ObjectStream([dict(case_id='one',state='A',t='2025-01-01',X={})]))
    assert log.events.activity.tolist() == ['A','A','D']
    assert log.measurements.loc[log.measurements.feature == 'history_visits','value'].tolist() == [1,2,1]
    trained = survpm.ProcessModel(net).fit(log, features=['history_visits'])
    assert 'history_visits' in trained.transitions.encoder.columns_


def test_forecast_does_not_add_fake_visit_at_as_of():
    class CheckHistory(survpm.ExponentialTransitionModel):
        def next_transition(self,state,X,history,rng,elapsed=0):
            assert len(history) == 1
            assert X['history_visits'] == 1
            return 'D', 1.
    origin = pd.Timestamp('2025-01-01',tz='UTC')
    log = survpm.EventLog(pd.DataFrame([('one','A',origin)],columns=['case_id','activity','timestamp']),
                         pd.DataFrame(index=['one']),pd.DataFrame(columns=['case_id','timestamp','feature','value']))
    net = survpm.Network.from_edges([('A','D')], ['A'], ['D'])
    result = survpm.Router(survpm.ProcessModel(net,CheckHistory())).run(log,'one',origin+pd.Timedelta(days=2))
    assert result.events.activity.tolist() == ['A','D']


def test_entry_counts_are_preserved_in_diagnostics_and_profile():
    m = survpm.CensoredStateNet('A','D').fit(intervals=intervals())
    assert m.edges_table().n_completed.tolist() == [1,1]


def test_replacement_history_builder_receives_only_past_data():
    class PastOnly(survpm.HistoryFeatures):
        def transform(self,log,case_id,as_of):
            assert (log.events.timestamp <= as_of).all()
            assert (log.measurements.timestamp <= as_of).all()
            return super().transform(log,case_id,as_of)
    origin = pd.Timestamp('2025-01-01',tz='UTC')
    events = pd.DataFrame([('one','A',origin),('one','D',origin+pd.Timedelta(days=10))],
                          columns=['case_id','activity','timestamp'])
    measurements = pd.DataFrame([('one',origin+pd.Timedelta(days=5),'x',100.)],
                                columns=['case_id','timestamp','feature','value'])
    log = survpm.EventLog(events,pd.DataFrame(index=['one']),measurements)
    net = survpm.Network.from_edges([('A','D')],['A'],['D'])
    model = survpm.ProcessModel(net,survpm.ExponentialTransitionModel({('A','D'):1}),PastOnly())
    survpm.Router(model).run(log,'one',origin,horizon=1.)
    survpm.Forecaster(model,n_paths=5).survival(log,origin,[1.])


def test_training_features_respect_event_order_at_identical_timestamps():
    origin = pd.Timestamp('2025-01-01',tz='UTC')
    events = pd.DataFrame([('one','A',origin),('one','B',origin),('one','A',origin),
                           ('one','D',origin+pd.Timedelta(days=1))], columns=['case_id','activity','timestamp'])
    log = survpm.EventLog(events,pd.DataFrame(index=['one']),
                         pd.DataFrame(columns=['case_id','timestamp','feature','value']))
    net = survpm.Network.from_edges([('A','B'),('B','A'),('A','D')],['A'],['D'])
    model = survpm.ProcessModel(net).fit(log,features=['history_visits'])
    assert model.transitions.training_X.history_visits.tolist() == [1,1,2]
