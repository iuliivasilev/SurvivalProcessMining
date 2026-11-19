import numpy as np
import pandas as pd
import pytest

from survpm import EventLog, ExponentialTransitionModel, Network, ProcessModel


def graph():
    return Network.from_edges([('A', 'B'), ('A', 'D'), ('A', 'E')], ['A'], ['B', 'D', 'E'])


def iv():
    return pd.DataFrame(dict(case_id=['a', 'b', 'c'], src='A', dst=['B', 'D', None],
                             duration=[1., 2., 3.], observed=[True, True, False], reason=''))


def event_log():
    origin = pd.Timestamp('2025-01-01', tz='UTC')
    events = pd.DataFrame([('a', 'A', origin), ('a', 'B', origin+pd.Timedelta(days=1)),
                           ('b', 'A', origin), ('b', 'D', origin+pd.Timedelta(days=2)),
                           ('c', 'A', origin)], columns=['case_id', 'activity', 'timestamp'])
    observation = pd.DataFrame(dict(case_id=['a', 'b', 'c'],
                                    end=[origin+pd.Timedelta(days=i) for i in [1, 2, 3]],
                                    event=[True, True, False], reason=['', '', 'lost']))
    return EventLog(events, pd.DataFrame(index=['a', 'b', 'c']),
                    pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value']),
                    observation=observation)


def test_registry_creates_replacement_and_rejects_duplicate_unknown():
    from survpm.models.transitions import register_transition_model, create_transition_model
    class Alternative:
        def __init__(self, value):
            self.value = value
    register_transition_model('architecture_alternative', Alternative)
    assert create_transition_model('architecture_alternative', value=7).value == 7
    assert isinstance(create_transition_model('exponential'), ExponentialTransitionModel)
    with pytest.raises(ValueError, match='already|duplicate'):
        register_transition_model('architecture_alternative', Alternative)
    with pytest.raises(ValueError, match='Unknown|unknown'):
        create_transition_model('absent_model')


@pytest.mark.parametrize('new_X', [pd.DataFrame({'x': [1., 2., 3.]}, index=[3, 4, 5]),
                                   pd.DataFrame({'other': [1., 2., 3.]}),
                                   pd.DataFrame({'x': [1., 2., 3.], 'extra': 0.})])
def test_update_validates_features_before_concat(new_X):
    fitted = ExponentialTransitionModel().fit(iv(), graph(), pd.DataFrame({'x': [0., 1., 2.]}))
    with pytest.raises(ValueError):
        fitted.update(iv(), new_X)
    assert len(fitted.training) == 3


def test_batch_parameters_preserve_scalar_and_empty_batch():
    fitted = ExponentialTransitionModel(l2=0).fit(iv(), graph(), pd.DataFrame({'x': [-1., 0., 1.]}))
    frame = pd.DataFrame({'x': [-1., 1.]}, index=[8, 9])
    result = fitted.transition_parameters('A', frame)
    assert len(result) == 2
    assert result[0] == fitted.transition_parameters('A', {'x': -1.})
    assert result[1] == fitted.transition_parameters('A', {'x': 1.})
    assert fitted.transition_parameters('A', frame.iloc[:0]) == []
    with pytest.raises(ValueError):
        fitted.transition_parameters('A', pd.DataFrame({'wrong': [1.]}))


def test_fit_and_update_leave_input_model_and_frames_unchanged():
    original = ExponentialTransitionModel({('A', 'D'): 9.})
    data = iv()
    fitted = original.fit(data, graph())
    updated = fitted.update(data)
    data.loc[0, 'duration'] = 99.
    assert original.rates == {('A', 'D'): 9.}
    assert not hasattr(original, 'training')
    assert len(fitted.training) == 3
    assert fitted.training.duration.tolist() == [1., 2., 3.]
    assert len(updated.training) == 6


def test_picker_selects_from_sampled_times_and_survives_fit_condition():
    from survpm.models.transitions import MinTimePicker
    assert MinTimePicker().pick({'B': 5., 'D': 2.}, np.random.default_rng(2)) == ('D', 2.)
    class LastPicker:
        def pick(self, times, rng):
            dst = list(times)[-1]
            return dst, times[dst]
    model = ProcessModel(graph(), picker=LastPicker()).fit(event_log())
    assert model.next_transition('A', {}, [], np.random.default_rng(1))[0] == 'D'
    conditioned = model.transitions.condition({})
    assert conditioned.next_transition('A', {}, [], np.random.default_rng(1))[0] == 'D'


def test_process_fit_keeps_training_log_and_observed_edge_kernel():
    original = ProcessModel(graph())
    log = event_log()
    fitted = original.fit(log)
    assert not hasattr(original, 'train_log')
    assert fitted.train_log is not log
    log.events.loc[0, 'activity'] = 'changed'
    assert fitted.train_log.events.activity.iloc[0] == 'A'
    edges = fitted.edges_table().set_index('dst')
    assert edges.loc['B', 'rate'] == pytest.approx(1/6)
    assert edges.loc['D', 'n_completed'] == 1
    assert edges.loc['E', 'n_completed'] == 0
    assert edges.loc['B', 'p'] == pytest.approx(.5)
    assert edges.loc['B', 'mean_days'] == pytest.approx(3.)
    assert edges.loc['E', 'p'] == 0.


def test_elapsed_is_residual_memoryless_and_rejects_negative():
    model = ExponentialTransitionModel({('A', 'D'): 2.})
    assert model.next_transition('A', {}, [], np.random.default_rng(5), elapsed=20.) == model.next_transition('A', {}, [], np.random.default_rng(5))
    with pytest.raises(ValueError):
        model.next_transition('A', {}, [], np.random.default_rng(5), elapsed=-1.)


def test_process_rejects_unsupported_elapsed_but_supports_explicit_contract():
    class Legacy:
        def next_transition(self, state, X, history, rng):
            return 'D', 1.
    model = ProcessModel(graph(), Legacy())
    assert model.next_transition('A', {}, [], np.random.default_rng(5)) == ('D', 1.)
    with pytest.raises(NotImplementedError, match='elapsed'):
        model.next_transition('A', {}, [], np.random.default_rng(5), elapsed=1.)
    class Conditional:
        def next_transition(self, state, X, history, rng, elapsed=0):
            return 'D', 3.-elapsed
    model = ProcessModel(graph(), Conditional())
    assert model.next_transition('A', {}, [], np.random.default_rng(5), elapsed=1.) == ('D', 2.)


def test_default_history_features_fix_measurements_at_entry():
    from survpm.models.features import DefaultHistoryFeatures
    log = event_log()
    origin = log.events.timestamp.min()
    log.measurements = pd.DataFrame([('c', origin, 'x', 1.),
                                     ('c', origin+pd.Timedelta(days=1), 'x', 99.)],
                                    columns=['case_id', 'timestamp', 'feature', 'value'])
    builder = DefaultHistoryFeatures()
    assert builder.fit(log) is not builder
    context = builder.transform(log, 'c', origin+pd.Timedelta(days=2))
    assert context['elapsed'] == 2.
    assert context['X']['x'] == 1.
    assert context['X']['history_visits'] == 1
