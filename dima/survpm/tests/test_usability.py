"""Удобство API: правила из журнала, выборки, потоки, единые метрики, сохранение, рисунки."""
import numpy as np
import pandas as pd
import pytest

import survpm
from survpm.features import MEASUREMENT_COLUMNS

RULES = {'starts': ['A'], 'terminals': ['D']}
ORIGIN = pd.Timestamp('2025-01-01', tz='UTC')


def generated(n=60, seed=0):
    network = survpm.Network.from_edges([('A', 'B'), ('A', 'D'), ('B', 'B'), ('B', 'D')], ['A'], ['D'])
    truth = survpm.ProcessModel(network, survpm.ExponentialTransitionModel(
        {('A', 'B'): 1., ('A', 'D'): .5, ('B', 'B'): .3, ('B', 'D'): .6}))
    log = survpm.Simulator(truth, seed=seed).run(survpm.ObjectStream.start(n, 'A', ORIGIN), horizon=3.)
    return truth, log.prepare(RULES)


def test_bounds_come_from_rules():
    _, log = generated()
    assert survpm.process_bounds(log) == (['A'], ['D'])
    with pytest.raises(ValueError, match='prepare'):
        survpm.process_bounds(survpm.EventLog(log.events, log.cases, log.measurements))
    network = survpm.discover(log)
    assert network.starts == ['A'] and network.terminals == ['D']
    assert survpm.Network.discover(log).edges.equals(network.edges)
    model = survpm.ProcessModel().fit(log)
    assert set(model.network.edges.itertuples(index=False, name=None)) == set(network.edges.itertuples(index=False, name=None))
    legacy = survpm.CensoredStateNet().fit(log)
    assert (legacy.start, legacy.terminal) == ('A', 'D')


def test_select_split_and_folds_keep_whole_cases():
    _, log = generated(100)
    part = log.select(log.case_ids[:3])
    assert part.case_ids == log.case_ids[:3] and part.rules == log.rules
    train, test = log.split(seed=4, ratio=.7)
    assert set(train.case_ids).isdisjoint(test.case_ids)
    assert len(train.case_ids) + len(test.case_ids) == 100
    assert train.case_ids == log.split(seed=4, ratio=.7)[0].case_ids
    assert abs(train.observation.event.mean() - test.observation.event.mean()) < .05
    folds = log.folds(4, seed=1)
    tests = [set(t.case_ids) for _, t in folds]
    assert set().union(*tests) == set(log.case_ids) and sum(map(len, tests)) == 100
    assert max(map(len, tests)) - min(map(len, tests)) <= 1


def test_streams_are_reusable_and_built_from_logs():
    truth, log = generated(20)
    stream = survpm.ObjectStream.from_log(log, prefix='sim-')
    first = survpm.Simulator(truth, seed=2).run(stream, horizon=5.)
    again = survpm.Simulator(truth, seed=2).run(stream, horizon=5.)
    pd.testing.assert_frame_equal(first.events, again.events)
    assert sorted(first.case_ids) == sorted(f'sim-{c}' for c in log.case_ids)
    starts = first.events.groupby('case_id').timestamp.min()
    assert (starts.values == log.events.groupby('case_id').timestamp.min().reindex(
        [c[4:] for c in starts.index]).values).all()


def test_evaluate_one_long_table_with_legacy_levels():
    truth, train = generated(200, seed=1)
    _, test = generated(80, seed=2)
    model = survpm.ProcessModel().fit(train)
    table = survpm.evaluate(model, test, ['nll', 'level4'])
    assert list(table.columns) == ['level', 'model', 'metric', 'value']
    assert set(table.level) == {'nll', 'level4'}
    nll = table.set_index(['level', 'metric']).value
    assert nll[('nll', 'nll_per_interval')] == pytest.approx(nll[('level4', 'nll_per_interval')])
    with pytest.raises(ValueError, match='train'):
        survpm.evaluate(truth, test, 'level4')
    assert np.isfinite(survpm.evaluate(truth, test, 'level4', train=train).value).all()
    both = survpm.compare({'fit': model, 'truth': truth}, test, 'nll')
    assert both.model.unique().tolist() == ['fit', 'truth']
    assert 'km_sup' in set(survpm.available_metrics().metric)


def test_legacy_bridge_uses_same_rates(tmp_path):
    _, train = generated(150, seed=3)
    model = survpm.ProcessModel().fit(train)
    bridge = model.legacy()
    assert bridge.transitions_ is model.transitions
    assert {(e['src'], e['dst']): e['p'] for e in bridge.edges_} == pytest.approx(
        model.edges_table().set_index(['src', 'dst']).p.to_dict())
    path = tmp_path / 'model.pkl'
    model.save(path)
    loaded = survpm.ProcessModel.load(path)
    assert loaded.transitions.rates == model.transitions.rates


def test_featured_model_has_clear_limits():
    _, log = generated(80, seed=5)
    rows = [(c, ORIGIN, 'x', float(i % 4)) for i, c in enumerate(log.case_ids)]
    log = log.add_measurements(pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS))
    model = survpm.ProcessModel().fit(log, features=['x'])
    assert np.isfinite(survpm.evaluate(model, log).value).all()
    with pytest.raises(NotImplementedError, match='признак'):
        survpm.evaluate(model, log, 'level4')


def test_graphviz_pictures():
    pytest.importorskip('graphviz')
    _, log = generated(50, seed=6)
    model = survpm.ProcessModel().fit(log)
    picture = survpm.GraphvizVisualizer()
    assert 'p=' in model.plot().source
    assert '->' in picture.plot_dfg(log).source
    assert 'CENS' in picture.plot_censoring(log).source
    pytest.importorskip('pm4py')
    assert 'CENS' in model.plot('petri').source


def test_cross_validator_reads_rules():
    pytest.importorskip('sklearn')
    from survpm.metrics.crossval import CrossValidator
    _, log = generated(60, seed=7)
    result = CrossValidator(k=3).add_metrics('level4').run(log, progress=None)
    long = survpm.long_format(result)
    assert {'level', 'model', 'fold', 'metric', 'value'} <= set(long.columns)
    assert set(long.model) == {'naive', 'censor_aware'}
