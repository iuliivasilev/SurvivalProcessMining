"""Portable conditioned-anchor baseline; ranks associations, not causes.

Input units are COMPLETE, explicitly identified work cycles. Only the first
configured target in each cycle is an outcome; later records cannot be features.
No notebook, generator, hidden truth, provider dictionary, or expert labels are
loaded. See README.md for the algorithm and its limits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
from typing import Iterable, Mapping

import networkx as nx


Node = tuple[str, ...]
VERSION = '1.0.0'


@dataclass(frozen=True)
class Config:
    target: Node
    time_column: str
    time_unit: str
    scope_columns: tuple[str, ...]
    event_columns: tuple[str, ...]
    match_columns: tuple[str, ...] = ()
    lookback_seconds: float = 60.0
    anchor_max_gap_seconds: float = 15.0
    edge_max_gap_seconds: float = 15.0
    half_life_seconds: float = 5.0
    prediction_gap_seconds: float = 0.0
    controls_per_target: int = 1
    smoothing: float = 1.0
    pagerank_alpha: float = 0.85
    max_depth: int = 4
    beam_width: int = 500
    top_k: int = 10

    def __post_init__(self):
        for name in ('target', 'scope_columns', 'event_columns', 'match_columns'):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f'{name} must be a list')
            if any(v is None or isinstance(v, (list, dict, bool)) or not str(v).strip() for v in value):
                raise ValueError(f'{name} contains an invalid value')
            object.__setattr__(self, name, tuple(str(v).strip() for v in value))
        if not self.scope_columns or not self.event_columns:
            raise ValueError('scope_columns and event_columns must not be empty')
        if len(self.target) != len(self.event_columns):
            raise ValueError('target must match event_columns in length and order')
        for name in ('scope_columns', 'event_columns', 'match_columns'):
            if len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f'{name} contains duplicate columns')
        if not isinstance(self.time_column, str) or not self.time_column.strip():
            raise ValueError('time_column must be a nonempty column name')
        if self.time_unit not in ('s', 'ms', 'iso'):
            raise ValueError('time_unit must be s, ms, or iso')
        for name in ('lookback_seconds', 'anchor_max_gap_seconds', 'edge_max_gap_seconds',
                     'half_life_seconds', 'smoothing', 'prediction_gap_seconds', 'pagerank_alpha'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite and numeric')
            if value < 0 or (value == 0 and name != 'prediction_gap_seconds'):
                raise ValueError(f'{name} is outside its valid range')
        if not 0 < self.pagerank_alpha < 1:
            raise ValueError('pagerank_alpha must be between zero and one')
        for name in ('controls_per_target', 'max_depth', 'beam_width', 'top_k'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class Event:
    time: float
    node: Node


@dataclass(frozen=True)
class Cycle:
    scope: tuple[str, ...]
    stratum: tuple[str, ...]
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Window:
    group: int
    kind: str
    scope: tuple[str, ...]
    anchor: float
    start: float
    cutoff: float
    endpoint: float
    weight: float
    events: tuple[Event, ...]


def _value(row: Mapping, name: str, line: int) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f'row {line}: missing or empty {name}')
    return str(value).strip()


def _time(value: str, cfg: Config, line: int) -> float:
    try:
        if cfg.time_unit == 'iso':
            stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError('ISO timestamps must specify a timezone')
            number = stamp.timestamp()
        else:
            number = float(value) / (1000 if cfg.time_unit == 'ms' else 1)
        if not math.isfinite(number):
            raise ValueError('nonfinite time')
        return number
    except (ValueError, OverflowError) as error:
        raise ValueError(f'row {line}: invalid time in {cfg.time_column}: {value!r}') from error


def _cycles(rows: Iterable[Mapping], cfg: Config) -> tuple[list[Cycle], int]:
    groups: dict[tuple, list[Event]] = defaultdict(list)
    strata = {}
    count = 0
    for count, row in enumerate(rows, start=1):
        scope = tuple(_value(row, c, count) for c in cfg.scope_columns)
        stratum = tuple(_value(row, c, count) for c in cfg.match_columns)
        if scope in strata and strata[scope] != stratum:
            raise ValueError(f'match_columns change inside cycle {scope}')
        strata[scope] = stratum
        node = tuple(_value(row, c, count) for c in cfg.event_columns)
        groups[scope].append(Event(_time(_value(row, cfg.time_column, count), cfg, count), node))
    cycles = [Cycle(key, strata[key], tuple(sorted(events, key=lambda e: (e.time, e.node))))
              for key, events in sorted(groups.items())]
    return cycles, count


def _window(cycle: Cycle, anchor: float, endpoint: float, group: int,
            kind: str, weight: float, cfg: Config) -> Window:
    start = max(cycle.events[0].time, anchor - cfg.lookback_seconds)
    cutoff = endpoint - cfg.prediction_gap_seconds
    events = tuple(e for e in cycle.events
                   if start <= e.time < cutoff and e.node != cfg.target)
    return Window(group, kind, cycle.scope, anchor, start, cutoff, endpoint, weight, events)


def _match_anchor(anchor: Node, positives: list[tuple[Cycle, float]],
                  controls: list[Cycle], cfg: Config) -> tuple[list[Window], list[Window], dict]:
    """Greedily match whole control cycles once; never count overlapping controls twice."""
    dangerous, ordinary, lags = [], [], []
    unused = {c.scope: c for c in controls}
    eligible_targets = 0
    for group, (cycle, target_time) in enumerate(positives):
        times = [e.time for e in cycle.events if e.node == anchor
                 and max(target_time - cfg.anchor_max_gap_seconds,
                         target_time - cfg.lookback_seconds) <= e.time
                 < target_time - cfg.prediction_gap_seconds]
        if not times:
            continue
        eligible_targets += 1
        anchor_time = max(times)
        lag = target_time - anchor_time
        danger = _window(cycle, anchor_time, target_time, group, 'dangerous', 1.0, cfg)
        candidates = []
        for control in unused.values():
            if control.stratum != cycle.stratum:
                continue
            for time in sorted({e.time for e in control.events if e.node == anchor}):
                # The input cycle must demonstrably extend through the matched outcome horizon.
                if time + lag > control.events[-1].time:
                    continue
                normal = _window(control, time, time + lag, group, 'ordinary', 1.0, cfg)
                key = (abs(len(normal.events) - len(danger.events)),
                       abs((time - control.events[0].time) - (anchor_time - cycle.events[0].time)),
                       control.scope, time)
                candidates.append((key, normal))
        selected = []
        selected_scopes = set()
        for _, normal in sorted(candidates, key=lambda p: p[0]):
            if normal.scope in selected_scopes:
                continue
            selected.append(normal)
            selected_scopes.add(normal.scope)
            if len(selected) == cfg.controls_per_target:
                break
        if not selected:
            continue
        dangerous.append(danger)
        lags.append(lag)
        for normal in selected:
            ordinary.append(Window(**{**normal.__dict__, 'weight': 1 / len(selected)}))
            unused.pop(normal.scope)
    summary = dict(anchor=list(anchor), eligible_target_cycles=eligible_targets,
                   matched_target_cycles=len(dangerous), control_cycles=len(ordinary),
                   median_anchor_lead_seconds=median(lags) if lags else None)
    return dangerous, ordinary, summary


def _edge_scores(window: Window, cfg: Config) -> dict[tuple[Node, Node], tuple[float, float]]:
    # K is a virtual endpoint in ordinary windows, NOT an observed failure there.
    events = window.events + (Event(window.endpoint, cfg.target),)
    best = {}
    for index, earlier in enumerate(events):
        for later in events[index + 1:]:
            delta = later.time - earlier.time
            if delta > cfg.edge_max_gap_seconds:
                break
            if delta <= 0:
                continue
            edge = (earlier.node, later.node)
            score = 2 ** (-delta / cfg.half_life_seconds)
            if edge not in best or score > best[edge][0]:
                best[edge] = score, delta
    return best


def _edge_statistics(dangerous: list[Window], ordinary: list[Window], cfg: Config) -> list[dict]:
    positive = defaultdict(list)
    negative = defaultdict(float)
    for window in dangerous:
        for edge, values in _edge_scores(window, cfg).items():
            positive[edge].append(values)
    for window in ordinary:
        for edge in _edge_scores(window, cfg):
            negative[edge] += window.weight
    n = len(dangerous)
    result = []
    for (source, target), evidence in sorted(positive.items()):
        sd, so = len(evidence), negative[(source, target)]
        # Both classes have equal total window weight n, including missing-edge windows.
        pd = (sd + cfg.smoothing) / (n + 2 * cfg.smoothing)
        po = (so + cfg.smoothing) / (n + 2 * cfg.smoothing)
        lift = pd / po
        temporal = sum(score for score, _ in evidence) / n
        result.append(dict(source=list(source), target=list(target), dangerous_support=sd,
                           ordinary_support=so, dangerous_coverage=sd / n, ordinary_coverage=so / n,
                           conditional_lift=lift, mean_time_score=mean(s for s, _ in evidence),
                           median_delta_seconds=median(d for _, d in evidence),
                           weight=temporal * max(1 - 1 / lift, 0)))
    return result


def _occurs(window: Window, path: tuple[Node, ...], cfg: Config) -> tuple[bool, float | None]:
    """Require one full ordered occurrence, not just separate supporting edges."""
    events = window.events + (Event(window.endpoint, cfg.target),)
    reachable = {e.time: e.time for e in events if e.node == path[0]}
    for position, node in enumerate(path[1:], start=1):
        following = {}
        for event in events:
            if event.node != node:
                continue
            if position == len(path) - 2 and event.time != window.anchor:
                continue  # The final A must be this window's selected anchor occurrence.
            starts = [start for time, start in reachable.items()
                      if 0 < event.time - time <= cfg.edge_max_gap_seconds]
            if starts:
                following[event.time] = max(starts)
        reachable = following
        if not reachable:
            return False, None
    return bool(reachable), window.endpoint - max(reachable.values()) if reachable else None


def _rank(edges: list[dict], anchor: Node, dangerous: list[Window],
          ordinary: list[Window], cfg: Config) -> tuple[list[dict], list[dict]]:
    graph = nx.DiGraph()
    for edge in edges:
        source, target = tuple(edge['source']), tuple(edge['target'])
        if edge['weight'] > 0 and target != cfg.target:
            graph.add_edge(source, target, weight=edge['weight'], connector=False)
    graph.add_edge(anchor, cfg.target, weight=1.0, connector=True)
    graph = graph.subgraph(nx.ancestors(graph, cfg.target) | {cfg.target}).copy()
    scores = nx.pagerank(graph.reverse(copy=False), personalization={cfg.target: 1.0},
                         alpha=cfg.pagerank_alpha, weight='weight', tol=1e-12, max_iter=1000)
    scale = max(v for node, v in scores.items() if node != cfg.target)
    normalized = {node: value / scale for node, value in scores.items()}
    nodes = [dict(node=list(node), pagerank=scores[node], normalized_pagerank=normalized[node])
             for node in sorted(scores, key=lambda n: (-scores[n], n))
             if node not in (cfg.target, anchor)]
    # The connector A -> K is structural and never earns a score by itself.
    def path_score(path):
        weights, relevance = [], []
        for source, target in zip(path, path[1:]):
            if not graph[source][target]['connector']:
                weights.append(graph[source][target]['weight'])
                relevance.append(normalized[source])
        return (math.exp(mean(math.log(w) for w in weights)) * mean(relevance)) if weights else 0.0

    beam = [(cfg.target,)]
    paths = []
    for depth in range(1, cfg.max_depth + 1):
        expanded = []
        for path in beam:
            for predecessor in sorted(graph.predecessors(path[0])):
                if predecessor not in path:
                    new_path = (predecessor,) + path
                    expanded.append((path_score(new_path), new_path))
        expanded.sort(key=lambda p: (-p[0], p[1]))
        beam = [path for _, path in expanded[:cfg.beam_width]]
        verified = []
        for score, path in expanded[:cfg.beam_width]:
            if score <= 0:
                continue
            positive = [_occurs(w, path, cfg) for w in dangerous]
            leads = [lead for found, lead in positive if found]
            if not leads:
                continue
            ordinary_support = sum(w.weight for w in ordinary if _occurs(w, path, cfg)[0])
            verified.append(dict(path=[list(n) for n in path], depth=depth, score=score,
                                 dangerous_support=len(leads), ordinary_support=ordinary_support,
                                 dangerous_coverage=len(leads) / len(dangerous),
                                 ordinary_coverage=ordinary_support / len(dangerous),
                                 conditional_lift=(len(leads) + cfg.smoothing) / (ordinary_support + cfg.smoothing),
                                 median_start_lead_seconds=median(leads)))
        paths.extend(verified[:cfg.top_k])
        if not beam:
            break
    return nodes[:cfg.top_k], paths


def analyze(rows: Iterable[Mapping], cfg: Config) -> dict:
    """Mine one supplied historical partition. Does not create or inspect a test split."""
    cycles, row_count = _cycles(rows, cfg)
    positives, controls = [], []
    target_records = 0
    for cycle in cycles:
        times = [e.time for e in cycle.events if e.node == cfg.target]
        target_records += len(times)
        if times:
            positives.append((cycle, min(times)))
        else:
            controls.append(cycle)
    summary = dict(version=VERSION, evaluation='exploratory_only_no_held_out_test',
                   rows=row_count, cycles=len(cycles), target=list(cfg.target),
                   target_cycles=len(positives), non_target_cycles=len(controls),
                   target_records=target_records, target_policy='first_per_scope',
                   matched_target_cycles=0, control_cycles=0, anchor=None,
                   status='no_target', config=asdict(cfg))
    result = dict(summary=summary, edges=[], nodes=[], paths=[], windows=[], anchors=[])
    if not positives:
        return result
    anchor_strength = defaultdict(float)
    for cycle, target_time in positives:
        latest = {}
        for event in cycle.events:
            if (event.node != cfg.target
                    and max(cycle.events[0].time, target_time - cfg.lookback_seconds,
                            target_time - cfg.anchor_max_gap_seconds) <= event.time
                    < target_time - cfg.prediction_gap_seconds):
                latest[event.node] = event.time
        for node, time in latest.items():
            anchor_strength[node] += 2 ** (-(target_time - time) / cfg.half_life_seconds)
    if not anchor_strength:
        summary['status'] = 'no_anchor'
        return result
    best, best_quality = None, None
    for anchor in sorted(anchor_strength):
        dangerous, ordinary, info = _match_anchor(anchor, positives, controls, cfg)
        result['anchors'].append(info)
        quality = (len(dangerous), len(ordinary), anchor_strength[anchor],
                   -(info['median_anchor_lead_seconds'] or 0))
        if dangerous and (best_quality is None or quality > best_quality):
            best = anchor, dangerous, ordinary, info
            best_quality = quality
    if best is None:
        summary['status'] = 'no_matched_controls'
        return result
    anchor, dangerous, ordinary, info = best
    summary.update(info)
    edges = _edge_statistics(dangerous, ordinary, cfg)
    nodes, paths = _rank(edges, anchor, dangerous, ordinary, cfg)
    summary.update(status='ok' if nodes else 'no_enriched_upstream_edges',
                   matching_coverage=len(dangerous) / len(positives),
                   candidate_nodes=len(nodes), candidate_paths=len(paths))
    result.update(edges=edges, nodes=nodes, paths=paths,
                  windows=[{k: v for k, v in asdict(w).items() if k != 'events'}
                           for w in dangerous + ordinary])
    return result


CSV_FIELDS = {
    'nodes': ('node', 'pagerank', 'normalized_pagerank'),
    'edges': ('source', 'target', 'dangerous_support', 'ordinary_support', 'dangerous_coverage',
              'ordinary_coverage', 'conditional_lift', 'mean_time_score', 'median_delta_seconds', 'weight'),
    'paths': ('path', 'depth', 'score', 'dangerous_support', 'ordinary_support', 'dangerous_coverage',
              'ordinary_coverage', 'conditional_lift', 'median_start_lead_seconds'),
    'windows': ('group', 'kind', 'scope', 'anchor', 'start', 'cutoff', 'endpoint', 'weight'),
    'anchors': ('anchor', 'eligible_target_cycles', 'matched_target_cycles', 'control_cycles',
                'median_anchor_lead_seconds'),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--input', type=Path, help='Override the CSV in config (relative to current directory)')
    parser.add_argument('--output', required=True, type=Path, help='A NEW output directory; never overwritten')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f'output already exists: {args.output}')
        raw = json.loads(args.config.read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            raise ValueError('config must be a JSON object')
        input_name = raw.pop('input', None)
        unknown = sorted(set(raw) - {f.name for f in fields(Config)})
        if unknown:
            raise ValueError(f'unknown config fields: {unknown}')
        cfg = Config(**raw)
        if args.input is None and (not isinstance(input_name, str) or not input_name.strip()):
            raise ValueError('input CSV is required in config or --input')
        source = (args.input if args.input is not None else args.config.parent / input_name).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            duplicates = sorted({h for h in headers if headers.count(h) > 1})
            if duplicates:
                raise ValueError(f'duplicate CSV columns: {duplicates}')
            required = {cfg.time_column, *cfg.scope_columns, *cfg.event_columns, *cfg.match_columns}
            if not required.issubset(headers):
                raise ValueError(f'missing CSV columns: {sorted(required - set(headers))}')
            result = analyze(reader, cfg)
        result['summary'].update(input=str(source), input_sha256=digest,
                                 implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 python_version=platform.python_version(),
                                 networkx_version=nx.__version__,
                                 created_utc=datetime.now(timezone.utc).isoformat())
        args.output.mkdir(parents=True, exist_ok=False)
        for name, fieldnames in CSV_FIELDS.items():
            with (args.output / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in result[name]:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (tuple, list)) else v
                                     for k, v in record.items()})
        (args.output / 'summary.json').write_text(
            json.dumps(result['summary'], ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({k: result['summary'][k] for k in ('status', 'target_cycles', 'matched_target_cycles', 'anchor')},
                         ensure_ascii=False))
        print(f'Exploratory output: {args.output.resolve()}')
        return 0
    except (ValueError, TypeError, OSError, nx.PowerIterationFailedConvergence) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
