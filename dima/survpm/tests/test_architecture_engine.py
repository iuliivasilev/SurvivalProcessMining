"""Execution uses public model contracts and explicit censoring metadata."""
import numpy as np
import pandas as pd
import pytest

import survpm

ORIGIN = pd.Timestamp('2025-01-01', tz='UTC')


def graph():
    return survpm.Network.from_edges([('A', 'D')], ['A'], ['D'])


def history():
    return survpm.EventLog(pd.DataFrame([('one', 'A', ORIGIN)], columns=['case_id', 'activity', 'timestamp']),
                          pd.DataFrame(index=['one']),
                          pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value']))


def initial(ids=('one',)):
    return [dict(case_id=i, state='A', t=ORIGIN, X={}) for i in ids]


def test_custom_residual_model_is_used_in_router_and_forecaster():
    class Residual:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'D', max(3-elapsed, 0)
    model = survpm.ProcessModel(graph(), Residual())
    cutoff = ORIGIN+pd.Timedelta(days=2)
    routed = survpm.Router(model).run(history(), 'one', cutoff)
    assert routed.events.timestamp.iloc[-1] == ORIGIN+pd.Timedelta(days=3)
    result = survpm.Forecaster(model, n_paths=3).survival(history(), cutoff, [.5, 1.])
    assert result.survival.tolist() == [1., 0.]


@pytest.mark.parametrize('timezone', [None, 'Asia/Shanghai'])
def test_from_frame_datetime_sources_fit_and_forecast_in_utc(timezone):
    times = pd.to_datetime(['2025-01-01', '2025-01-03'])
    if timezone is not None:
        times = times.tz_localize('UTC').tz_convert(timezone)
    frame = pd.DataFrame(dict(case_id='one', activity=['A', 'D'], timestamp=times, x=[1., 2.]))
    source = survpm.EventLog.from_frame(frame, {c: c for c in ('case_id', 'activity', 'timestamp')},
                                       static=[], dynamic=['x'])
    model = survpm.ProcessModel(graph()).fit(source)
    result = survpm.Forecaster(model, n_paths=3, seed=1).survival(source, ORIGIN, [1., 2.])
    assert result.survival.between(0., 1.).all()
    assert str(source.events.timestamp.dt.tz) == 'UTC'
    assert str(source.measurements.timestamp.dt.tz) == 'UTC'
    assert source.events.timestamp.tolist() == [ORIGIN, ORIGIN+pd.Timedelta(days=2)]
    assert source.measurements.timestamp.tolist() == [ORIGIN, ORIGIN+pd.Timedelta(days=2)]
    assert source.observation.end.tolist() == [ORIGIN+pd.Timedelta(days=2)]


def test_stream_dataframe_and_case_seed_do_not_depend_on_input_order():
    model = survpm.ProcessModel(graph(), survpm.ExponentialTransitionModel({('A', 'D'): 1.}))
    def run(ids):
        stream = survpm.ObjectStream(initial(ids), seed=9)
        assert isinstance(stream.objects, pd.DataFrame)
        result = survpm.Simulator(model).run(stream)
        assert (stream.objects.state == 'D').all()
        return result.events.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
    pd.testing.assert_frame_equal(run(['a', 'b']), run(['b', 'a']))
    with pytest.raises(ValueError, match='seed'):
        survpm.Simulator(model, seed=1).run(survpm.ObjectStream(initial(), seed=2))


def test_executor_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match='executor'):
        survpm.Simulator(survpm.ProcessModel(graph()), executor=object())


def test_simulator_observation_horizon_max_steps_and_unbounded_steps():
    class Fixed:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'D', 2.
    model = survpm.ProcessModel(graph(), Fixed())
    short = survpm.Simulator(model).run(survpm.ObjectStream(initial()), horizon=1., max_steps=None)
    assert short.observation.iloc[0].to_dict() == dict(case_id='one', end=ORIGIN+pd.Timedelta(days=1), event=0, reason='horizon')
    assert short.last_seen['one'] == short.observation.end.iloc[0]
    complete = survpm.Simulator(model).run(survpm.ObjectStream(initial()), max_steps=None)
    assert complete.observation.event.tolist() == [1]
    loop = survpm.ProcessModel(survpm.Network.from_edges([('A', 'A')], ['A'], ['D']), Fixed())
    loop.transitions.next_transition = lambda *args, **kwargs: ('A', 1.)
    stopped = survpm.Simulator(loop).run(survpm.ObjectStream(initial()), max_steps=2)
    assert stopped.observation.reason.tolist() == ['max_steps']


def test_features_update_only_at_entries_and_cutoff_reaches_custom_builder():
    class PastOnly(survpm.HistoryFeatures):
        def transform(self, log, case_id, as_of):
            assert log.observation.end.max() <= as_of
            assert log.events.timestamp.max() <= as_of
            assert log.measurements.empty or log.measurements.timestamp.max() <= as_of
            return super().transform(log, case_id, as_of)
    class Fixed:
        def next_transition(self, state, X, history, rng, elapsed=0):
            assert X['x'] == 1.
            return 'D', 2.
    source = history()
    source.cases['x'] = 1.
    source.observation['end'] = ORIGIN+pd.Timedelta(days=20)
    calls = []
    def updater(X, state, t, rng):
        calls.append((state, t))
        return X
    model = survpm.ProcessModel(graph(), Fixed(), PastOnly())
    survpm.Router(model).run(source, 'one', ORIGIN+pd.Timedelta(days=1), future_features=updater)
    assert calls == [('D', ORIGIN+pd.Timedelta(days=3))]


def test_hazard_bins_and_rmst_for_exponential_completion():
    model = survpm.ProcessModel(graph(), survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    forecast = survpm.Forecaster(model, n_paths=4000, seed=10)
    hazard = forecast.hazard(history(), ORIGIN, [.1, .2, .3, .4])
    assert list(hazard.columns) == ['horizon', 'outcome', 'hazard']
    assert hazard.outcome.tolist() == ['D']*4
    assert hazard.hazard.mean() == pytest.approx(.5, abs=.07)
    estimate = forecast.expected_time(history(), ORIGIN, horizon=2.)
    assert estimate['restricted_mean_time'] == pytest.approx((1-np.exp(-1))/.5, abs=.04)
    assert estimate['probability_unfinished'] == pytest.approx(np.exp(-1), abs=.025)
    with pytest.raises(TypeError):
        forecast.expected_time(history(), ORIGIN)
    for times in ([0., 1.], [1., 1.], [2., 1.]):
        with pytest.raises(ValueError):
            forecast.hazard(history(), ORIGIN, times)
    with pytest.raises(NotImplementedError, match='exact'):
        survpm.Forecaster(model, method='exact')


def test_unlimited_steps_reject_nonterminating_and_zero_time_cycles():
    class Loop:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'A', 0.
    model = survpm.ProcessModel(survpm.Network.from_edges([('A', 'A')], ['A'], ['D']), Loop())
    with pytest.raises(ValueError, match='horizon|max_steps'):
        survpm.Simulator(model).run(survpm.ObjectStream(initial()), max_steps=None)
    with pytest.raises(ValueError, match='time|duration'):
        survpm.Simulator(model).run(survpm.ObjectStream(initial()), horizon=1., max_steps=None)


def test_seed_is_reproducible_for_updater_and_model_together():
    model = survpm.ProcessModel(graph(), survpm.ExponentialTransitionModel({('A', 'D'): 1.}))
    def run(ids):
        stream = survpm.ObjectStream(initial(ids), updater=lambda X, state, t, rng: {'x': rng.normal()})
        return survpm.Simulator(model, seed=16).run(stream)
    a, b = run(['a', 'b']), run(['b', 'a'])
    for name in ('events', 'measurements', 'observation'):
        left, right = getattr(a, name), getattr(b, name)
        columns = ['case_id', 'timestamp'] if name != 'observation' else ['case_id']
        pd.testing.assert_frame_equal(left.sort_values(columns).reset_index(drop=True),
                                      right.sort_values(columns).reset_index(drop=True))


def test_empty_stream_returns_empty_canonical_tables():
    result = survpm.Simulator(survpm.ProcessModel(graph())).run(survpm.ObjectStream([]))
    assert result.events.empty and result.measurements.empty and result.observation.empty
    assert result.observation.columns.tolist() == ['case_id', 'end', 'event', 'reason']


def test_latent_model_is_a_protocol_hook_only():
    from survpm.models.features import LatentModel
    assert all(callable(getattr(LatentModel, name)) for name in ('init', 'update', 'infer'))
    with pytest.raises(TypeError):
        LatentModel()


def test_forecasts_do_not_silently_truncate_paths_after_one_thousand_transitions():
    class ManySteps:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return ('D' if len(history) >= 1002 else 'A'), .001
    model = survpm.ProcessModel(survpm.Network.from_edges([('A', 'A'), ('A', 'D')], ['A'], ['D']), ManySteps())
    forecast = survpm.Forecaster(model, n_paths=1, seed=1)
    source = history()
    assert forecast.survival(source, ORIGIN, [2.]).survival.tolist() == [0.]
    assert forecast.terminal_probs(source, ORIGIN, [2.]).probability.tolist() == [1.]
    result = forecast.expected_time(source, ORIGIN, horizon=2.)
    assert result['restricted_mean_time'] == pytest.approx(1.002)
    assert result['probability_unfinished'] == 0.
    assert forecast.hazard(source, ORIGIN, [2.]).hazard.tolist() == [.5]
