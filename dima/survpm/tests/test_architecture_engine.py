"""Генерация и прогноз через публичные контракты; время в UTC; порядок объектов не важен."""
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


@pytest.mark.parametrize('timezone', [None, 'Asia/Shanghai'])
def test_from_frame_datetime_sources_fit_and_forecast_in_utc(timezone):
    times = pd.to_datetime(['2025-01-01', '2025-01-03'])
    if timezone is not None:
        times = times.tz_localize('UTC').tz_convert(timezone)
    frame = pd.DataFrame(dict(case_id='one', activity=['A', 'D'], timestamp=times, x=[1., 2.]))
    source = survpm.EventLog.from_frame(frame, {c: c for c in ('case_id', 'activity', 'timestamp')},
                                       static=[], dynamic=['x'])
    model = survpm.ProcessModel(graph()).fit(source)
    result = survpm.SimulationForecaster(model, n_paths=3, seed=1).survival(source.cut(ORIGIN), [1., 2.])
    assert result.survival.between(0., 1.).all()
    assert str(source.events.timestamp.dt.tz) == 'UTC'
    assert str(source.measurements.timestamp.dt.tz) == 'UTC'
    assert source.events.timestamp.tolist() == [ORIGIN, ORIGIN+pd.Timedelta(days=2)]
    assert source.measurements.timestamp.tolist() == [ORIGIN, ORIGIN+pd.Timedelta(days=2)]
    assert source.observation.end.tolist() == [ORIGIN+pd.Timedelta(days=2)]


def test_generation_does_not_depend_on_input_order():
    model = survpm.ProcessModel(graph(), survpm.ExponentialTransitionModel({('A', 'D'): 1.}))
    a = survpm.EventLog.start(2, 'A', ORIGIN, prefix='c')
    b = a.select(['c1', 'c0'])
    first = survpm.ModelGenerator(model, seed=9).generate(a)
    second = survpm.ModelGenerator(model, seed=9).generate(b)
    key = ['case_id', 'timestamp']
    pd.testing.assert_frame_equal(first.events.sort_values(key).reset_index(drop=True),
                                  second.events.sort_values(key).reset_index(drop=True))
    assert (a.observation.reason == 'start').all()          # вход не меняется генератором


def test_empty_log_returns_empty_canonical_tables():
    empty = survpm.EventLog.start(0, 'A', ORIGIN)
    result = survpm.ModelGenerator(survpm.ProcessModel(graph())).generate(empty)
    assert result.events.empty and result.measurements.empty and result.observation.empty
    assert result.observation.columns.tolist()[:4] == ['case_id', 'end', 'event', 'reason']


def test_latent_model_is_a_protocol_hook_only():
    from survpm.models.features import LatentModel
    assert all(callable(getattr(LatentModel, name)) for name in ('init', 'update', 'infer'))
    with pytest.raises(TypeError):
        LatentModel()
