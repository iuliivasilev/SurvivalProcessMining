"""Data/discovery contracts for the v2 architecture."""
import sys

import pandas as pd
import pytest

from survpm.features import EventLog, MEASUREMENT_COLUMNS
from survpm.network import Network


def ts(day):
    return pd.Timestamp('2025-01-01', tz='UTC') + pd.Timedelta(day, unit='D')


def sample():
    events = pd.DataFrame([
        ('1', 'A', ts(0)), ('1', 'B', ts(1)), ('1', 'D', ts(3)),
        ('2', 'A', ts(0)), ('2', 'B', ts(2)),
        ('3', 'A', ts(8)), ('3', 'E', ts(9)),
    ], columns=['case_id', 'activity', 'timestamp'])
    cases = pd.DataFrame({'age': [30, 40, 50]}, index=pd.Index(['1', '2', '3'], name='case_id'))
    measurements = pd.DataFrame([('2', ts(1), 'x', 10), ('2', ts(4), 'x', 20)], columns=MEASUREMENT_COLUMNS)
    observation = pd.DataFrame([('1', ts(3), 1, ''), ('2', ts(6), 0, 'study_end'),
                                ('3', ts(9), 1, '')], columns=['case_id', 'end', 'event', 'reason'])
    return EventLog(events, cases, measurements, observation=observation)


def test_prepare_keeps_observation_after_last_measurement_and_multiple_terminals():
    log = sample()
    prepared = log.prepare({'starts': ['A'], 'terminals': ['D', 'E'], 'min_duration': .01})
    assert prepared is not log
    assert prepared.observation.set_index('case_id').loc['2', 'end'] == ts(6)
    iv = prepared.intervals()
    censored = iv.loc[~iv.observed].iloc[0]
    assert (censored.case_id, censored.duration, censored.reason) == ('2', 4, 'study_end')
    assert set(iv.loc[iv.observed, 'dst']) == {'B', 'D', 'E'}
    assert prepared.rules['min_duration'] == .01
    pd.testing.assert_frame_equal(log.events, sample().events)


def test_positional_last_seen_and_measurement_addition_preserve_metadata():
    log = sample()
    legacy = EventLog(log.events, log.cases, log.measurements, pd.Series({'2': ts(7)}))
    assert legacy.last_seen['2'] == ts(7)
    prepared = log.prepare({'start': 'A', 'terminal': ['D', 'E']})
    updated = prepared.add_measurements(pd.DataFrame([('2', ts(5), 'x', 30)], columns=MEASUREMENT_COLUMNS))
    pd.testing.assert_frame_equal(updated.observation, prepared.observation)
    assert updated.rules == prepared.rules
    assert updated.intervals().loc[lambda d: ~d.observed, 'duration'].tolist() == [4]
    assert updated.last_seen['2'] == ts(6)


def test_from_frame_mapping_and_to_frame_are_independent_tables():
    frame = sample().events.rename(columns={'case_id': 'id', 'activity': 'state', 'timestamp': 'time'})
    log = EventLog.from_frame(frame, {'case_id': 'id', 'activity': 'state', 'timestamp': 'time'},
                              static=[], dynamic=[])
    tables = log.to_frame()
    assert set(tables) == {'events', 'cases', 'measurements', 'observation'}
    assert set(tables['observation'].columns) == {'case_id', 'end', 'event', 'reason'}
    tables['events'].loc[0, 'activity'] = 'changed'
    assert log.events.iloc[0].activity == 'A'


def test_features_at_cutoff_excludes_future_measurements():
    log = sample()
    assert log.features_at('2', ts(5))['x'] == 20
    assert log.features_at('2', ts(5), cutoff=ts(2))['x'] == 10
    assert log.features_at('2', ts(5), cutoff=ts(2))['age'] == 40
    assert log.features_at('2', ts(0))['x_n'] == 0


def test_case_split_preserves_all_tables_and_observation():
    log = sample().prepare({'start': 'A', 'terminal': ['D', 'E']})
    train, test = log.split('by_case', ratio=2 / 3)
    assert set(train.events.case_id).isdisjoint(test.events.case_id)
    assert set(train.events.case_id) | set(test.events.case_id) == {'1', '2', '3'}
    for part in (train, test):
        ids = set(part.events.case_id)
        assert set(part.cases.index) == ids
        assert set(part.measurements.case_id) <= ids
        assert set(part.observation.case_id) == ids
        assert set(part.last_seen.index) == ids
        assert part.rules == log.rules


def test_temporal_split_uses_explicit_cutoff_and_excludes_crossing_cases():
    log = sample().prepare({'start': 'A', 'terminal': ['D', 'E']})
    train, test = log.split('temporal', cutoff=ts(4))
    assert set(train.events.case_id) == {'1'}
    assert set(test.events.case_id) == {'3'}
    assert train.observation.end.tolist() == [ts(3)]
    with pytest.raises(ValueError, match='cutoff'):
        log.split('temporal')


def test_prepare_repeat_rules_and_min_duration():
    log = sample()
    duplicate = log.events.iloc[[0]].assign(timestamp=ts(.5))
    log.events = pd.concat([log.events, duplicate], ignore_index=True)
    kept = log.prepare({'start': 'A', 'terminal': ['D', 'E'], 'repeats': 'keep'})
    collapsed = log.prepare({'start': 'A', 'terminal': ['D', 'E'], 'repeats': 'collapse'})
    assert len(kept.events) == len(collapsed.events) + 1
    assert (kept.intervals().duration > 0).all()


def test_discovery_protocol_roundtrip():
    from survpm.discovery import DirectlyFollowsBackend, PetriBackend, discover
    backend = DirectlyFollowsBackend(starts=['A'], terminals=['D', 'E'])
    assert isinstance(backend, PetriBackend)
    network = discover(sample(), backend)
    assert set(map(tuple, network.edges.to_numpy())) == {('A', 'B'), ('B', 'D'), ('A', 'E')}
    assert network.starts == ['A'] and network.terminals == ['D', 'E']


@pytest.mark.parametrize('backend', ['directly_follows', 'network', 'pm4py'])
def test_discovery_preserves_same_time_repeat_for_fit_roundtrip(backend, monkeypatch):
    from survpm import DirectlyFollowsBackend, ProcessModel
    events = pd.DataFrame([('1', 'A', ts(0)), ('1', 'A', ts(0)), ('1', 'D', ts(1))],
                          columns=['case_id', 'activity', 'timestamp'])
    log = EventLog.from_frame(events, {c: c for c in events.columns}, static=[], dynamic=[])
    prepared = log.prepare(dict(starts=['A'], terminals=['D'], repeats='keep', min_duration=5.))
    if backend == 'directly_follows':
        network = DirectlyFollowsBackend(['A'], ['D']).discover(prepared)
    elif backend == 'network':
        network = Network.discover(prepared, ['A'], ['D'])
    else:
        from types import SimpleNamespace
        from survpm import Pm4pyBackend
        def discover_dfg(frame):
            assert frame['concept:name'].tolist() == ['A', 'A', 'D']
            return {('A', 'A'): 1, ('A', 'D'): 1}, {'A': 1}, {'D': 1}
        monkeypatch.setitem(sys.modules, 'pm4py', SimpleNamespace(discover_dfg=discover_dfg))
        network = Pm4pyBackend(starts=['A'], terminals=['D']).discover(prepared)
    assert set(map(tuple, network.edges.to_numpy())) == {('A', 'A'), ('A', 'D')}
    fitted = ProcessModel(network).fit(prepared)
    assert fitted.training_intervals.duration.tolist() == [5., 5.]
    assert fitted.transitions.rates == {('A', 'A'): .1, ('A', 'D'): .1}


@pytest.mark.parametrize('t, cutoff', [('2025-01-06', ts(2)), (ts(5), '2025-01-03')])
def test_features_at_normalizes_mixed_naive_and_aware_cutoffs(t, cutoff):
    assert sample().features_at('2', t, cutoff=cutoff)['x'] == 10


def test_network_reports_cycles_without_terminal_path_and_isolated_start():
    network = Network.from_edges([('A', 'B'), ('B', 'B'), ('X', 'D')], ['A', 'I'], ['D'])
    issues = set(map(tuple, network.validate().to_numpy()))
    assert {('A', 'no_terminal_path'), ('B', 'no_terminal_path'), ('I', 'dead_end'),
            ('I', 'no_terminal_path'), ('X', 'unreachable')} <= issues
    assert ('D', 'no_terminal_path') not in issues


def test_pm4py_backend_rejects_general_petri_without_optional_import():
    from survpm.adapters import Pm4pyBackend
    loaded = 'pm4py' in sys.modules  # другие тесты могли уже импортировать pm4py
    for method in ('inductive', 'heuristics', 'heu'):
        with pytest.raises(NotImplementedError, match='state|Petri'):
            Pm4pyBackend(method=method).discover(sample())
    assert loaded or 'pm4py' not in sys.modules


def test_pm4py_dfg_optional_dependency_error():
    from importlib.util import find_spec
    from survpm.adapters import Pm4pyBackend
    if find_spec('pm4py') is not None:
        pytest.skip('optional pm4py is installed')
    with pytest.raises(ImportError, match='pm4py'):
        Pm4pyBackend().discover(sample())


def test_prepare_rejects_completed_metadata_without_terminal_event():
    events = pd.DataFrame([('1', 'A', ts(0)), ('1', 'B', ts(1))],
                          columns=['case_id', 'activity', 'timestamp'])
    observation = pd.DataFrame([('1', ts(5), 1, '')],
                               columns=['case_id', 'end', 'event', 'reason'])
    log = EventLog(events, pd.DataFrame(index=pd.Index(['1'], name='case_id')),
                   pd.DataFrame(columns=MEASUREMENT_COLUMNS), observation=observation)
    with pytest.raises(ValueError, match='completed.*terminal'):
        log.prepare({'start': 'A', 'terminal': 'D'})


def test_initial_inferred_observation_includes_measurements_despite_older_last_seen():
    events = pd.DataFrame([('1', 'A', ts(0)), ('1', 'B', ts(1))],
                          columns=['case_id', 'activity', 'timestamp'])
    measurements = pd.DataFrame([('1', ts(7), 'x', 10)], columns=MEASUREMENT_COLUMNS)
    log = EventLog(events, pd.DataFrame(index=pd.Index(['1'], name='case_id')), measurements,
                   pd.Series({'1': ts(1)}))
    assert log.observation.end.tolist() == [ts(7)]
    assert log.last_seen['1'] == ts(7)
    assert log.prepare({'start': 'A', 'terminal': 'D'}).intervals().duration.tolist() == [1, 6]
    train, test = log.split('temporal', cutoff=ts(3))
    assert train.events.empty and test.events.empty


def test_added_measurements_extend_censored_observation_but_keep_completed_end():
    original = sample().prepare({'start': 'A', 'terminal': ['D', 'E']})
    measurements = pd.DataFrame([('1', ts(7), 'x', 30), ('2', ts(7), 'x', 40)],
                                columns=MEASUREMENT_COLUMNS)
    updated = original.add_measurements(measurements)
    observation = updated.observation.set_index('case_id')
    assert observation.loc['2', 'end'] == updated.last_seen['2'] == ts(7)
    assert observation.loc['2', 'reason'] == 'study_end'
    assert observation.loc['1', 'end'] == ts(3)
    assert observation.loc['1', 'event'] == 1
    assert updated.intervals().loc[lambda d: ~d.observed, 'duration'].tolist() == [5]
    assert original.observation.set_index('case_id').loc['2', 'end'] == ts(6)


def test_added_follow_up_corrects_inferred_end_intervals_and_temporal_split():
    events = pd.DataFrame([('1', 'A', ts(0)), ('1', 'B', ts(1))],
                          columns=['case_id', 'activity', 'timestamp'])
    log = EventLog(events, pd.DataFrame(index=pd.Index(['1'], name='case_id')),
                   pd.DataFrame(columns=MEASUREMENT_COLUMNS))
    updated = log.add_measurements(pd.DataFrame([('1', ts(7), 'x', 10)], columns=MEASUREMENT_COLUMNS))
    assert updated.observation.end.tolist() == [ts(7)]
    assert updated.last_seen['1'] == ts(7)
    assert updated.prepare({'start': 'A', 'terminal': 'D'}).intervals().duration.tolist() == [1, 6]
    train, test = updated.split('temporal', cutoff=ts(3))
    assert train.events.empty and test.events.empty
