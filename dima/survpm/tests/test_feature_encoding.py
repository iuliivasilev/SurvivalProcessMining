"""Обрезка признаков по диапазону обучения, log1p и векторная сборка признаков на входе."""
import numpy as np
import pandas as pd
import pytest

from survpm import DefaultHistoryFeatures, EventLog, Network, ProcessModel
from survpm.covariates import FeatureEncoder
from survpm.features import MEASUREMENT_COLUMNS
from survpm.models.features import HistoryFeatures
from survpm.process import entry_features
from survpm.transitions import ExponentialTransitionModel


def test_encoder_clips_to_training_range():
    train = pd.DataFrame({'x': [1., 2., 3., 4.]})
    encoder = FeatureEncoder().fit(train)
    z_train = encoder.transform(train)
    z = encoder.transform(pd.DataFrame({'x': [100., -50., 2.5]}))
    assert z[0, 0] == pytest.approx(z_train[:, 0].max())
    assert z[1, 0] == pytest.approx(z_train[:, 0].min())
    assert z[2, 0] == pytest.approx(0.)
    raw = FeatureEncoder(clip=False).fit(train).transform(pd.DataFrame({'x': [100.]}))
    assert raw[0, 0] > z_train[:, 0].max()


def test_encoder_log_features():
    train = pd.DataFrame({'x': [0., 1., 10., 100.], 'y': [1., 2., 3., 4.]})
    encoder = FeatureEncoder(log_features=['x']).fit(train)
    expected = np.log1p(train.x)
    assert encoder.transform(train)[:, 0] == pytest.approx(((expected-expected.mean())/expected.std(ddof=0)).to_numpy())
    assert encoder.medians_['x'] == pytest.approx(expected.median())
    with pytest.raises(ValueError):
        FeatureEncoder(log_features=['x']).fit(pd.DataFrame({'x': [-1., 2.]}))
    with pytest.raises(ValueError):
        FeatureEncoder(log_features=['z']).fit(train)


def test_transition_model_keeps_encoder_options():
    model = ExponentialTransitionModel(clip=False, log_features=['x'])
    iv = pd.DataFrame(dict(case_id=['1', '2', '3'], src='A', dst='B', duration=[1., 2., 3.],
                           observed=True, reason=''))
    fitted = model.fit(iv, Network.from_edges([('A', 'B')], ['A'], ['B']), pd.DataFrame({'x': [1., 5., 9.]}, index=iv.index))
    assert fitted.encoder.clip is False and fitted.encoder.log_features == ['x']
    assert fitted.condition({'x': 2.}).clip is False


def ts(day):
    return pd.Timestamp('2025-01-01', tz='UTC') + pd.Timedelta(days=day)


def sample_log():
    events = pd.DataFrame([('1', 'A', ts(0)), ('1', 'B', ts(1)), ('1', 'B', ts(2)), ('1', 'D', ts(4)),
                           ('2', 'A', ts(0)), ('2', 'B', ts(3)),
                           ('3', 'A', ts(1)), ('3', 'D', ts(2))],
                          columns=['case_id', 'activity', 'timestamp'])
    cases = pd.DataFrame({'age': [50, 70, 60]}, index=pd.Index(['1', '2', '3'], name='case_id'))
    measurements = pd.DataFrame([('1', ts(0), 'crp', 10.), ('1', ts(1.5), 'crp', 30.),
                                 ('2', ts(2), 'crp', 5.), ('3', ts(3), 'crp', 7.)],
                                columns=MEASUREMENT_COLUMNS)
    return EventLog(events, cases, measurements).prepare({'start': 'A', 'terminal': 'D'})


class LoopFeatures:
    """Та же логика, что DefaultHistoryFeatures, но без векторного пути."""
    def __init__(self):
        self.inner = DefaultHistoryFeatures()

    def fit(self, train_log):
        return self

    def transform(self, log, case_id, as_of):
        return self.inner.transform(log, case_id, as_of)


def test_vectorized_entry_features_match_loop():
    log = sample_log()
    iv = log.intervals()
    columns = ['age', 'crp', 'crp_age', 'history_visits', 'history_elapsed_days']
    fast = entry_features(log, log.events, iv, DefaultHistoryFeatures(), columns)
    slow = entry_features(log, log.events, iv, LoopFeatures(), columns)
    pd.testing.assert_frame_equal(fast, slow, check_dtype=False)
    assert fast.history_visits.tolist() == [1, 1, 2, 1, 1, 1]
    with pytest.raises(KeyError):
        entry_features(log, log.events, iv, DefaultHistoryFeatures(), ['missing'])


def test_subclass_override_uses_loop():
    class Custom(DefaultHistoryFeatures):
        def at_entry(self, X, state, t, history):
            return dict(super().at_entry(X, state, t, history), custom=1.)
    log = sample_log()
    assert Custom().transform_intervals(log, log.events, log.intervals()) is None
    X = entry_features(log, log.events, log.intervals(), Custom(), ['custom'])
    assert X.custom.tolist() == [1.]*len(X)


def test_process_model_fit_with_features_is_consistent():
    log = sample_log()
    network = Network.from_edges([('A', 'B'), ('B', 'B'), ('B', 'D'), ('A', 'D')], ['A'], ['D'])
    model = ProcessModel(network).fit(log, features=['age', 'crp'])
    assert list(model.transitions.encoder.columns_) == ['age', 'crp']
    assert np.isfinite(model.transitions.log_likelihood(
        log.intervals(), entry_features(log, log.events, log.intervals(), model.history_features, ['age', 'crp'])))
