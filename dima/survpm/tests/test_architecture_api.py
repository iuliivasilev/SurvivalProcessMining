"""Public evaluation and extension hooks exercise the exponential baseline."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import survpm


def sample():
    origin = pd.Timestamp('2025-01-01', tz='UTC')
    events = pd.DataFrame([('a', 'A', origin), ('a', 'D', origin + pd.Timedelta(days=2)),
                           ('b', 'A', origin)], columns=['case_id', 'activity', 'timestamp'])
    log = survpm.EventLog(events, pd.DataFrame(index=['a', 'b']),
                         pd.DataFrame(columns=['case_id', 'timestamp', 'feature', 'value']),
                         pd.Series({'a': origin + pd.Timedelta(days=2),
                                    'b': origin + pd.Timedelta(days=3)}))
    return survpm.Network.from_edges([('A', 'D')], ['A'], ['D']), log


def test_nll_censored_exposure_and_no_refit():
    network, log = sample()
    model = survpm.ProcessModel(network, survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    result = survpm.evaluate(model, log).set_index('metric').value
    assert result['nll_per_interval'] == pytest.approx((2.5 + np.log(2))/2)
    table = survpm.compare({'manual': model}, log, ['nll'])
    assert list(table.columns) == ['level', 'model', 'metric', 'value']
    assert table.model.tolist() == ['manual', 'manual']
    assert table.value.tolist() == pytest.approx(result.tolist())
    assert model.transitions.rates == {('A', 'D'): .5}
    assert not hasattr(model, 'train_log')


@pytest.mark.parametrize('repeats, expected_nll, count', [('keep', 20., 2), ('collapse', 10., 1)])
def test_evaluate_preserves_prepared_repeat_and_duration_rules(repeats, expected_nll, count):
    origin = pd.Timestamp('2025-01-01', tz='UTC')
    events = pd.DataFrame([('a', 'A', origin), ('a', 'A', origin),
                           ('a', 'D', origin + pd.Timedelta(days=1))],
                          columns=['case_id', 'activity', 'timestamp'])
    log = survpm.EventLog.from_frame(events, {c: c for c in events.columns}, static=[], dynamic=[])
    prepared = log.prepare(dict(start='A', terminal='D', repeats=repeats, min_duration=5.))
    network = survpm.Network.from_edges([('A', 'A'), ('A', 'D')], ['A'], ['D'])
    model = survpm.ProcessModel(network, survpm.ExponentialTransitionModel({('A', 'A'): 1., ('A', 'D'): 1.}))
    direct_nll = -model.transitions.log_likelihood(prepared.intervals())
    assert direct_nll == pytest.approx(expected_nll)
    result = survpm.evaluate(model, prepared).set_index('metric').value
    assert result['nll_per_trace'] == pytest.approx(direct_nll)
    assert result['nll_per_interval'] == pytest.approx(expected_nll/count)
    assert model.transitions.rates == {('A', 'A'): 1., ('A', 'D'): 1.}
    assert not hasattr(model, 'train_log')


def test_metric_registry_and_errors():
    network, log = sample()
    model = survpm.ProcessModel(network)
    survpm.register_metric('api_case_count', lambda model, log: log.events.case_id.nunique())
    custom = survpm.evaluate(model, log, ['api_case_count'])
    assert custom.value.tolist() == [2] and custom.level.tolist() == ['custom']
    with pytest.raises(ValueError, match='registered'):
        survpm.register_metric('api_case_count', lambda model, log: 0)
    with pytest.raises(ValueError, match='registered'):
        survpm.register_metric('km_sup', lambda model, log: 0)
    with pytest.raises(ValueError, match='Unknown'):
        survpm.evaluate(model, log, ['absent_metric'])
    with pytest.raises(ValueError, match='train'):
        survpm.evaluate(model, log, ['level4'])  # уровни 1–4 требуют обученную модель или train=


def test_network_comparison_mapping_and_empty():
    reference = survpm.Network.from_edges([('A', 'B'), ('B', 'D')], ['A'], ['D'])
    estimated = survpm.Network.from_edges([('x', 'y'), ('x', 'z')], ['x'], ['z'])
    result = survpm.compare_networks(reference, estimated, {'x': 'A', 'y': 'B', 'z': 'D'})
    assert (result['precision'], result['recall'], result['f1']) == (.5, .5, .5)
    assert result['missing'].to_records(index=False).tolist() == [('B', 'D')]
    assert result['extra'].to_records(index=False).tolist() == [('A', 'D')]
    empty = survpm.Network.from_edges([], ['A'], ['D'])
    result = survpm.compare_networks(empty, empty)
    assert (result['precision'], result['recall'], result['f1']) == (1., 1., 1.)


def test_roundtrip_and_public_imports():
    from survpm.log import EventLog
    from survpm import (discover, DirectlyFollowsBackend, PetriBackend, Pm4pyBackend,
                        DefaultHistoryFeatures, FeatureUpdater, LatentModel, Metric,
                        register_transition_model, create_transition_model)
    from survpm.models import LatentModel as ModelLatent
    assert ModelLatent is LatentModel
    network, train = sample()
    model = survpm.ProcessModel(network).fit(train)
    assert isinstance(train, EventLog)
    stream = survpm.ObjectStream([dict(case_id='new', state='A', t='2025-01-01', X={})])
    generated = survpm.Simulator(model, seed=7).run(stream, horizon=10)
    found = discover(generated, DirectlyFollowsBackend(['A'], ['D']))
    assert isinstance(found, survpm.Network)
    estimate = survpm.Forecaster(model, n_paths=5, seed=7).survival(
        generated, pd.Timestamp('2025-01-01', tz='UTC'), [1., 2.], case_id='new')
    assert estimate.survival.between(0, 1).all()
    assert np.isfinite(survpm.evaluate(model, train).value).all()


def test_public_import_does_not_load_optional_dependencies():
    subprocess.run([sys.executable, '-c', '''
import sys
import survpm
from survpm.viz import Visualizer, MatplotlibVisualizer
assert not any(x in sys.modules for x in ('pm4py', 'graphviz', 'matplotlib.pyplot'))
'''], check=True)


def test_plot_delegate_and_unsupported():
    from survpm.viz import MatplotlibVisualizer
    network, _ = sample()
    class Renderer:
        def network(self, value, **kwargs):
            return value.states
    assert network.plot(Renderer()) == ['A', 'D']
    with pytest.raises(NotImplementedError):
        MatplotlibVisualizer().latent(None)


def test_exponential_edge_curves_use_minimum_kernel():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from survpm.viz import MatplotlibVisualizer
    network = survpm.Network.from_edges([('A', 'D'), ('A', 'E')], ['A'], ['D', 'E'])
    model = survpm.ProcessModel(network, survpm.ExponentialTransitionModel({('A', 'D'): .5, ('A', 'E'): 1.5}))
    renderer = MatplotlibVisualizer()
    ax = renderer.plot_edge_curves(model, ('A', 'D'), [0., 1.])
    assert ax.lines[0].get_ydata().tolist() == pytest.approx([0., .25*(1-np.exp(-2))])
    assert renderer.plot_network(network).get_xlabel() == ''
    import matplotlib.pyplot as plt
    plt.close('all')


def test_compare_accepts_model_iterable_with_stable_labels():
    network, log = sample()
    model = survpm.ProcessModel(network, survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    table = survpm.compare(iter([model, model]), log, ['nll'])
    assert table.model.tolist() == ['model_0', 'model_0', 'model_1', 'model_1']
    assert table.value.tolist() == pytest.approx([(2.5 + np.log(2))/2]*4)
    assert survpm.compare([], log).columns.tolist() == ['level', 'model', 'metric', 'value']


def test_required_visualizer_interface_and_network_plot_delegate():
    from survpm.viz import Visualizer, MatplotlibVisualizer
    for method in ('plot_network', 'plot_edge_curves', 'compare_km'):
        assert callable(getattr(Visualizer, method, None))
        assert callable(getattr(MatplotlibVisualizer, method, None))
    network, _ = sample()
    class Renderer:
        def plot_network(self, value, **kwargs):
            return value.states
    assert network.plot(Renderer()) == ['A', 'D']
    with pytest.raises(NotImplementedError, match='KM'):
        MatplotlibVisualizer().compare_km(None, None, ('A', 'D'))


def test_architectural_km_signature_fails_explicitly():
    from survpm.viz import MatplotlibVisualizer
    network, log = sample()
    with pytest.raises(NotImplementedError, match='KM'):
        MatplotlibVisualizer().compare_km(survpm.ProcessModel(network), log, ('A', 'D'))


def test_architectural_edge_plot_signature_and_default_grid():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from survpm.viz import MatplotlibVisualizer
    network, _ = sample()
    model = survpm.ProcessModel(network, survpm.ExponentialTransitionModel({('A', 'D'): .5}))
    renderer = MatplotlibVisualizer()
    ax = renderer.plot_edge_curves(model, ('A', 'D'))
    assert ax.lines[0].get_xdata()[[0, -1]].tolist() == [0., 10.]
    assert ax.lines[0].get_ydata()[[0, -1]].tolist() == pytest.approx([0., 1-np.exp(-5)])
    explicit = renderer.plot_edge_curves(model, ('A', 'D'), [0., 1.])
    assert explicit.lines[0].get_ydata().tolist() == pytest.approx([0., 1-np.exp(-.5)])
    import matplotlib.pyplot as plt
    plt.close('all')


def test_architectural_network_plot_accepts_process_model():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from survpm.viz import MatplotlibVisualizer
    network, _ = sample()
    ax = MatplotlibVisualizer().plot_network(survpm.ProcessModel(network))
    assert {label.get_text() for label in ax.texts} >= {'A', 'D'}
    import matplotlib.pyplot as plt
    plt.close('all')
