"""Known-answer tests for the portable, label-free precursor ranking core."""
from dataclasses import replace
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def miner():
    assert importlib.util.find_spec('precursor_mining') is not None, 'Portable miner is not implemented'
    import precursor_mining
    return precursor_mining


def config(**changes):
    return miner().Config(
        target=('F',), time_column='t', time_unit='s',
        scope_columns=('case',), event_columns=('event',),
        lookback_seconds=10, anchor_max_gap_seconds=3,
        edge_max_gap_seconds=3, half_life_seconds=2,
        **changes,
    )


def rows_for(case, events):
    return [dict(case=case, t=str(t), event=event) for t, event in events]


def example():
    # F-A gap is 2s; X-F gap is 4s, so only A can be the anchor.
    return (
        rows_for('p1', [(0, 'X'), (2, 'A'), (4, 'F')])
        + rows_for('p2', [(0, 'X'), (2, 'A'), (4, 'F')])
        + rows_for('n1', [(0, 'Z'), (2, 'A'), (5, 'END')])
        + rows_for('n2', [(0, 'Z'), (2, 'A'), (5, 'END')])
    )


def test_known_answer_conditional_lift_and_actual_chain():
    result = miner().analyze(example(), config())
    assert result['summary']['status'] == 'ok'
    assert result['summary']['anchor'] == ['A']
    assert result['summary']['target_cycles'] == 2
    assert result['summary']['matched_target_cycles'] == 2
    edge = next(e for e in result['edges'] if e['source'] == ['X'] and e['target'] == ['A'])
    # Two dangerous supports, zero controls, Laplace smoothing=1: lift=3.
    # Two-second lag and half-life=2 give time weight=.5; .5*(1-1/3)=1/3.
    assert edge['dangerous_support'] == 2
    assert edge['ordinary_support'] == 0
    assert edge['conditional_lift'] == pytest.approx(3)
    assert edge['weight'] == pytest.approx(1 / 3)
    assert result['nodes'][0]['node'] == ['X']
    chain = next(p for p in result['paths'] if p['path'] == [['X'], ['A'], ['F']])
    assert chain['dangerous_support'] == 2
    assert chain['ordinary_support'] == 0


def test_node_names_do_not_imply_background_or_special_retention():
    original = miner().analyze(example(), config())
    names = {'X': 'Microsoft-Windows-Dhcp', 'A': '430', 'F': '512', 'Z': 'StorDiag', 'END': 'done'}
    renamed = [dict(row, event=names[row['event']]) for row in example()]
    result = miner().analyze(renamed, replace(config(), target=('512',)))
    for old in original['edges']:
        new = next(e for e in result['edges'] if e['source'] == [names[old['source'][0]]]
                   and e['target'] == [names[old['target'][0]]])
        assert new['weight'] == pytest.approx(old['weight'])
    assert result['nodes'][0]['node'] == ['Microsoft-Windows-Dhcp']


def test_cycles_are_isolated_even_with_identical_clocks():
    rows = rows_for('p', [(2, 'A'), (4, 'F')])
    rows += rows_for('n', [(0, 'X'), (2, 'A'), (5, 'END')])
    result = miner().analyze(rows, config())
    assert all(e['source'] != ['X'] for e in result['edges'])


def test_target_repeats_and_future_events_do_not_change_mined_evidence():
    original = miner().analyze(example(), config())
    extra = rows_for('p1', [(4, 'SIMULTANEOUS'), (4.5, 'F'), (5, 'F'), (6, 'FUTURE')])
    modified = miner().analyze(example() + extra, config())
    assert modified['summary']['target_cycles'] == 2
    assert modified['summary']['target_records'] == 4
    for field in ('edges', 'nodes', 'paths', 'windows'):
        assert modified[field] == original[field]


def test_ignored_columns_cannot_change_ranking():
    poisoned = [dict(row, resource_free='0', scenario='failure', truth='yes', record_id='secret')
                for row in example()]
    assert miner().analyze(poisoned, config()) == miner().analyze(example(), config())


def test_same_timestamp_does_not_create_temporal_edge():
    rows = example() + rows_for('p1', [(2, 'SAME')]) + rows_for('p2', [(2, 'SAME')])
    result = miner().analyze(rows, config())
    assert not any(e['source'] == ['SAME'] and e['target'] == ['A'] for e in result['edges'])


def test_full_chain_is_not_invented_from_edges_in_different_cycles():
    cfg = replace(config(), anchor_max_gap_seconds=1, edge_max_gap_seconds=2)
    rows = rows_for('p1', [(0, 'X'), (1, 'Y'), (4, 'A'), (5, 'F')])
    rows += rows_for('p2', [(3, 'Y'), (4, 'A'), (5, 'F')])
    rows += rows_for('n1', [(0, 'Z'), (4, 'A'), (6, 'END')])
    rows += rows_for('n2', [(0, 'Z'), (4, 'A'), (6, 'END')])
    result = miner().analyze(rows, cfg)
    assert any(e['source'] == ['X'] and e['target'] == ['Y'] for e in result['edges'])
    assert any(e['source'] == ['Y'] and e['target'] == ['A'] for e in result['edges'])
    assert [['X'], ['Y'], ['A'], ['F']] not in [p['path'] for p in result['paths']]


def test_chain_must_reach_the_selected_anchor_occurrence():
    cfg = replace(config(), anchor_max_gap_seconds=1, edge_max_gap_seconds=2)
    rows = rows_for('p', [(0, 'X'), (1, 'A'), (2.5, 'A'), (3, 'F')])
    rows += rows_for('n', [(0, 'Z'), (1, 'A'), (2.5, 'A'), (4, 'END')])
    result = miner().analyze(rows, cfg)
    # X -> early A -> F exists, but X cannot reach the selected A@2.5 in <=2s.
    assert result['windows'][0]['anchor'] == 2.5
    assert [['X'], ['A'], ['F']] not in [p['path'] for p in result['paths']]


def test_depth_can_exceed_four_when_the_data_supports_a_longer_chain():
    cfg = replace(config(), anchor_max_gap_seconds=1, edge_max_gap_seconds=1, max_depth=6)
    rows = rows_for('p', [(0, 'U'), (1, 'V'), (2, 'W'), (3, 'X'), (4, 'Y'), (5, 'A'), (6, 'F')])
    rows += rows_for('n', [(0, 'Z'), (5, 'A'), (7, 'END')])
    result = miner().analyze(rows, cfg)
    assert any(p['depth'] == 6 and p['dangerous_support'] == 1 for p in result['paths'])


def test_controls_per_target_is_not_silently_capped_at_three():
    rows = rows_for('p', [(0, 'X'), (2, 'A'), (4, 'F')])
    for i in range(4):
        rows += rows_for(f'n{i}', [(0, 'Z'), (2, 'A'), (5, 'END')])
    result = miner().analyze(rows, config(controls_per_target=4))
    assert result['summary']['control_cycles'] == 4
    assert [w['weight'] for w in result['windows'] if w['kind'] == 'ordinary'] == [0.25] * 4


@pytest.mark.parametrize('changes', [
    {'scope_columns': ()}, {'event_columns': ()}, {'target': ('F', 'extra')},
    {'half_life_seconds': 0}, {'smoothing': -1}, {'prediction_gap_seconds': -1},
    {'pagerank_alpha': 1}, {'beam_width': 0}, {'top_k': True},
    {'scope_columns': ('case', 'case')}, {'time_unit': 'minutes'},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_timezone_less_datetime_is_not_silently_interpreted_as_local_time():
    rows = [dict(r, t='2026-01-01T00:00:00') for r in example()]
    with pytest.raises(ValueError, match='time'):
        miner().analyze(rows, replace(config(), time_unit='iso'))


def test_inconsistent_matching_stratum_within_cycle_is_rejected():
    rows = [dict(r, host='h') for r in example()]
    rows[0]['host'] = 'different'
    with pytest.raises(ValueError, match='match_columns'):
        miner().analyze(rows, config(match_columns=('host',)))


def test_control_cycles_are_not_reused_as_independent_samples():
    rows = example() + rows_for('p3', [(0, 'X'), (2, 'A'), (4, 'F')])
    result = miner().analyze(rows, config())
    controls = [w for w in result['windows'] if w['kind'] == 'ordinary']
    assert len(controls) == len({tuple(w['scope']) for w in controls}) == 2
    assert result['summary']['matched_target_cycles'] == 2


def test_multiple_controls_have_total_weight_one_per_dangerous_cycle():
    rows = [r for r in example() if r['case'] != 'p2']
    result = miner().analyze(rows, config(controls_per_target=2))
    weights = [w['weight'] for w in result['windows'] if w['kind'] == 'ordinary']
    assert weights == [0.5, 0.5]


def test_no_controls_reports_insufficient_evidence():
    result = miner().analyze([r for r in example() if r['case'].startswith('p')], config())
    assert result['summary']['status'] == 'no_matched_controls'
    assert not result['paths']


def test_no_target_is_reported_without_fabricating_a_target():
    result = miner().analyze(example(), replace(config(), target=('OTHER',)))
    assert result['summary']['status'] == 'no_target'
    assert not result['nodes']


def test_short_control_followup_is_not_treated_as_observed_nonfailure():
    rows = [r for r in example() if r['event'] != 'END']
    result = miner().analyze(rows, config())
    assert result['summary']['status'] == 'no_matched_controls'


def test_matching_respects_explicit_observed_strata():
    rows = [dict(r, host='positive' if r['case'].startswith('p') else 'negative') for r in example()]
    result = miner().analyze(rows, config(match_columns=('host',)))
    assert result['summary']['status'] == 'no_matched_controls'


def test_time_units_and_column_mapping_are_configurable():
    rows = [dict(run=r['case'], ms=str(float(r['t']) * 1000), provider='P', code=r['event'])
            for r in example()]
    cfg = replace(config(), scope_columns=('run',), time_column='ms', time_unit='ms',
                  event_columns=('provider', 'code'), target=('P', 'F'))
    result = miner().analyze(rows, cfg)
    assert result['summary']['anchor'] == ['P', 'A']
    assert result['paths'][0]['dangerous_support'] == 2


def test_iso_time_matches_numeric_time_statistics():
    rows = [dict(r, t=f"2026-01-01T00:00:0{r['t']}+00:00") for r in example()]
    iso = miner().analyze(rows, replace(config(), time_unit='iso'))
    numeric = miner().analyze(example(), config())
    assert iso['edges'] == numeric['edges']
    assert iso['paths'] == numeric['paths']


@pytest.mark.parametrize('value', ['nan', 'inf', '', 'not-a-time'])
def test_invalid_time_is_rejected_instead_of_silently_dropped(value):
    rows = example()
    rows[0]['t'] = value
    with pytest.raises(ValueError, match='time|empty t'):
        miner().analyze(rows, config())


def test_anchor_matching_honors_lookback_for_every_target_cycle():
    cfg = replace(config(), lookback_seconds=2)
    rows = rows_for('p1', [(2, 'A'), (4, 'F')])
    rows += rows_for('p2', [(1, 'A'), (4, 'F')])
    rows += rows_for('n1', [(2, 'A'), (5, 'END')])
    rows += rows_for('n2', [(2, 'A'), (5, 'END')])
    result = miner().analyze(rows, cfg)
    assert result['summary']['matched_target_cycles'] == 1


def test_missing_scope_rejected_instead_of_combining_cycles():
    rows = [{k: v for k, v in r.items() if k != 'case'} for r in example()]
    with pytest.raises(ValueError, match='case'):
        miner().analyze(rows, config())


def test_prediction_gap_excludes_late_features():
    result = miner().analyze(example(), config(prediction_gap_seconds=2))
    # A occurs exactly at the feature cutoff, so it must not be used.
    assert result['summary']['status'] == 'no_anchor'


def test_event_identity_includes_all_configured_columns():
    rows = [dict(r, provider='p') for r in example()]
    rows += [dict(r, provider='other') for r in rows_for('other-cycle', [(0, 'A'), (4, 'F')])]
    result = miner().analyze(rows, replace(config(), event_columns=('provider', 'event'), target=('p', 'F')))
    assert result['summary']['target_cycles'] == 2


def test_depth_limit_is_not_fixed_to_four():
    result = miner().analyze(example(), config(max_depth=1))
    assert result['summary']['status'] == 'ok'
    assert not result['paths']  # The connector alone is never a discovered chain.


def test_cli_runs_from_another_directory_and_refuses_overwrite(tmp_path):
    data = tmp_path / 'logs.csv'
    with data.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 't', 'event'])
        writer.writeheader()
        writer.writerows(example())
    from dataclasses import asdict
    settings = dict(asdict(config()), input='logs.csv')
    config_path = tmp_path / 'settings.json'
    config_path.write_text(json.dumps(settings), encoding='utf-8')
    output = tmp_path / 'result'
    command = [sys.executable, str(ROOT / 'precursor_mining.py'), '--config', str(config_path), '--output', str(output)]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'ok'
    assert summary['evaluation'] == 'exploratory_only_no_held_out_test'
    assert summary['input_sha256']
    assert summary['implementation_sha256']
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_duplicate_csv_headers_are_rejected_before_they_can_erase_targets(tmp_path):
    from dataclasses import asdict
    data = tmp_path / 'duplicate.csv'
    data.write_text('case,t,event,event\np,0,A,A\np,2,F,END\n', encoding='utf-8')
    settings = dict(asdict(config()), input=data.name)
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings), encoding='utf-8')
    output = tmp_path / 'must-not-exist'
    completed = subprocess.run(
        [sys.executable, str(ROOT / 'precursor_mining.py'), '--config', str(path), '--output', str(output)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert 'duplicate' in completed.stderr
    assert not output.exists()
