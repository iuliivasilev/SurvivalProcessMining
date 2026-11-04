"""Parameterized state-machine Petri net that generates one XuetangX behavior profile.

Places are actions. A transition is an event: it consumes the token in the current
action, waits, and puts the token in the next action. The log records that next
action. Probability and waiting-time parameters are separate. One run uses a single
distribution family for every edge.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROB_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    prob: float
    rate: float | None = None
    shape: float | None = None
    scale: float | None = None

    @property
    def transition_id(self) -> str:
        return f"t_{self.src}__{self.dst}"


@dataclass(frozen=True)
class PetriNet:
    profile: str
    family: str
    time_unit: str
    parameter_status: str
    note: str
    absorbing: frozenset[str]
    edges: tuple[Edge, ...]

    @property
    def places(self) -> tuple[str, ...]:
        names = {"start"}
        for edge in self.edges:
            names.add(edge.src)
            names.add(edge.dst)
        ordered = ["start"]
        ordered.extend(sorted(names - {"start"} - set(self.absorbing)))
        ordered.extend(sorted(self.absorbing))
        return tuple(ordered)

    def outgoing(self) -> dict[str, tuple[Edge, ...]]:
        grouped: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            grouped[edge.src].append(edge)
        return {src: tuple(items) for src, items in grouped.items()}


def load_net(path: Path) -> PetriNet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    family = raw["family"]
    if family not in {"exponential", "weibull"}:
        raise ValueError("family must be 'exponential' or 'weibull'")
    absorbing = frozenset(raw["absorbing"])
    if "start" in absorbing:
        raise ValueError("start cannot be absorbing")
    edges: list[Edge] = []
    for src, outgoing in raw["transitions"].items():
        for dst, spec in outgoing.items():
            edges.append(
                Edge(
                    src=src,
                    dst=dst,
                    prob=float(spec["prob"]),
                    rate=_optional_float(spec, "rate"),
                    shape=_optional_float(spec, "shape"),
                    scale=_optional_float(spec, "scale"),
                )
            )
    net = PetriNet(
        profile=raw["profile"],
        family=family,
        time_unit=raw.get("time_unit", "hour"),
        parameter_status=raw.get("parameter_status", "specified_not_estimated"),
        note=raw.get("note", ""),
        absorbing=absorbing,
        edges=tuple(edges),
    )
    validate(net)
    return net


def _optional_float(spec: dict, key: str) -> float | None:
    if key not in spec or spec[key] is None:
        return None
    return float(spec[key])


def validate(net: PetriNet) -> None:
    if not net.edges:
        raise ValueError("the net has no transitions")
    outgoing = net.outgoing()
    if "start" not in outgoing:
        raise ValueError("start has no outgoing transitions")
    for src, edges in outgoing.items():
        if src in net.absorbing:
            raise ValueError(f"absorbing place {src} has outgoing transitions")
        total = sum(edge.prob for edge in edges)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities from {src} sum to {total}, not 1")
        if src != "start" and not any(edge.dst in net.absorbing for edge in edges):
            raise ValueError(f"{src} cannot reach an absorbing place in one step")
        for edge in edges:
            if edge.prob <= 0:
                raise ValueError(f"non-positive probability on {edge.src} -> {edge.dst}")
            _validate_delay(net.family, edge)
    transient_sources = set(outgoing) - {"start"}
    for edge in net.edges:
        if edge.dst == "start":
            raise ValueError("a transition cannot return to start")
        if edge.dst not in net.absorbing and edge.dst not in transient_sources:
            raise ValueError(f"{edge.dst} is reached but has no outgoing transitions")


def _validate_delay(family: str, edge: Edge) -> None:
    if family == "exponential":
        if edge.rate is None or edge.rate <= 0:
            raise ValueError(f"{edge.src} -> {edge.dst} needs a positive exponential rate")
        return
    if edge.shape is None or edge.shape <= 0 or edge.scale is None or edge.scale <= 0:
        raise ValueError(f"{edge.src} -> {edge.dst} needs positive Weibull shape and scale")


def sample_delay(rng: random.Random, family: str, edge: Edge) -> float:
    if family == "exponential":
        assert edge.rate is not None
        return rng.expovariate(edge.rate)
    assert edge.shape is not None and edge.scale is not None
    return rng.weibullvariate(edge.scale, edge.shape)


def choose_edge(rng: random.Random, edges: tuple[Edge, ...]) -> Edge:
    draw = rng.random()
    cumulative = 0.0
    for edge in edges:
        cumulative += edge.prob
        if draw <= cumulative + PROB_TOLERANCE:
            return edge
    return edges[-1]


def generate_trace(
    rng: random.Random,
    net: PetriNet,
    *,
    horizon: float | None,
    max_events: int,
) -> tuple[list[tuple[float, str]], str, bool]:
    """Return events, observed absorbing action, and whether the trace was censored.

    A censored trace stops at the horizon without writing the event that would
    have fallen past it. The absorbing action is then unknown.
    """
    outgoing = net.outgoing()
    place = "start"
    clock = 0.0
    events: list[tuple[float, str]] = []
    for _ in range(max_events):
        edge = choose_edge(rng, outgoing[place])
        delay = sample_delay(rng, net.family, edge)
        next_clock = clock + delay
        if horizon is not None and next_clock > horizon:
            return events, "", True
        clock = next_clock
        events.append((clock, edge.dst))
        place = edge.dst
        if place in net.absorbing:
            return events, place, False
    raise RuntimeError(f"trace exceeded {max_events} events; increase absorption probabilities")


def generate(
    net: PetriNet,
    n_objects: int,
    seed: int,
    *,
    horizon: float | None = None,
    max_events: int = 10_000,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    log_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    for index in range(n_objects):
        object_id = f"e{index:06d}"
        events, absorbing_action, censored = generate_trace(rng, net, horizon=horizon, max_events=max_events)
        for clock, action in events:
            log_rows.append({"objectId": object_id, "time": clock, "action": action})
        if censored and horizon is not None:
            end_time = horizon
        elif events:
            end_time = events[-1][0]
        else:
            end_time = 0.0
        object_rows.append(
            {
                "objectId": object_id,
                "profile": net.profile,
                "censored": int(censored),
                "absorbing_action": absorbing_action,
                "n_events": len(events),
                "end_time": end_time,
            }
        )
    return log_rows, object_rows


def write_outputs(
    out_dir: Path,
    net: PetriNet,
    log_rows: list[dict[str, object]],
    object_rows: list[dict[str, object]],
    *,
    seed: int,
    n_objects: int,
    horizon: float | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "event_log.csv", ["objectId", "time", "action"], log_rows)
    _write_csv(
        out_dir / "objects.csv",
        ["objectId", "profile", "censored", "absorbing_action", "n_events", "end_time"],
        object_rows,
    )
    (out_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "profile": net.profile,
                "family": net.family,
                "time_unit": net.time_unit,
                "parameter_status": net.parameter_status,
                "note": net.note,
                "seed": seed,
                "n_objects": n_objects,
                "horizon": horizon,
                "absorbing": sorted(net.absorbing),
                "edges": [
                    {
                        "src": edge.src,
                        "dst": edge.dst,
                        "prob": edge.prob,
                        "rate": edge.rate,
                        "shape": edge.shape,
                        "scale": edge.scale,
                    }
                    for edge in net.edges
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_pnml(out_dir / "generator_net.pnml", net)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pnml(path: Path, net: PetriNet) -> None:
    pnml = ET.Element("pnml")
    net_el = ET.SubElement(
        pnml,
        "net",
        id=f"xuetangx_{net.profile}",
        type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel",
    )
    page = ET.SubElement(net_el, "page", id="page1")
    for place in net.places:
        place_el = ET.SubElement(page, "place", id=place)
        ET.SubElement(ET.SubElement(place_el, "name"), "text").text = place
        if place == "start":
            marking = ET.SubElement(place_el, "initialMarking")
            ET.SubElement(marking, "text").text = "1"
    for index, edge in enumerate(net.edges):
        transition = ET.SubElement(page, "transition", id=edge.transition_id)
        ET.SubElement(ET.SubElement(transition, "name"), "text").text = edge.dst
        tool = ET.SubElement(transition, "toolspecific", tool="xuetangx-generator", version="1")
        ET.SubElement(tool, "probability").text = repr(edge.prob)
        ET.SubElement(tool, "family").text = net.family
        if edge.rate is not None:
            ET.SubElement(tool, "rate").text = repr(edge.rate)
        if edge.shape is not None:
            ET.SubElement(tool, "shape").text = repr(edge.shape)
        if edge.scale is not None:
            ET.SubElement(tool, "scale").text = repr(edge.scale)
        ET.SubElement(page, "arc", id=f"a{index}_in", source=edge.src, target=edge.transition_id)
        ET.SubElement(page, "arc", id=f"a{index}_out", source=edge.transition_id, target=edge.dst)
    ET.ElementTree(pnml).write(path, encoding="utf-8", xml_declaration=True)


def empirical_check(net: PetriNet, n_objects: int, seed: int) -> float:
    log_rows, object_rows = generate(net, n_objects, seed, horizon=None)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    previous: dict[str, str] = {}
    for row in log_rows:
        object_id = str(row["objectId"])
        src = previous.get(object_id, "start")
        dst = str(row["action"])
        counts[(src, dst)] += 1
        previous[object_id] = dst
    totals: dict[str, int] = defaultdict(int)
    for (src, _dst), count in counts.items():
        totals[src] += count
    max_error = 0.0
    print(f"profile={net.profile} family={net.family} objects={n_objects} events={len(log_rows)}")
    for edge in net.edges:
        total = totals[edge.src]
        empirical = counts[(edge.src, edge.dst)] / total if total else 0.0
        max_error = max(max_error, abs(empirical - edge.prob))
    censored = sum(int(row["censored"]) for row in object_rows)
    unfinished = sum(1 for row in object_rows if not row["absorbing_action"])
    print(f"censored={censored} unfinished={unfinished} max_abs_error={max_error:.4f}")
    if censored or unfinished:
        raise RuntimeError("uncensored generation produced an unfinished trace")
    return max_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an event log from the parameterized Petri net.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("converter.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=None, help="Censor traces still running after this many time units.")
    parser.add_argument("--check", action="store_true", help="Compare empirical transition frequencies with the configured probabilities.")
    parser.add_argument("--check-n", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    net = load_net(args.config)
    if args.check:
        max_error = empirical_check(net, args.check_n, args.seed)
        if max_error > 0.04:
            raise SystemExit(f"empirical transition error {max_error:.4f} exceeds 0.04")
    log_rows, object_rows = generate(net, args.n, args.seed, horizon=args.horizon)
    write_outputs(args.out, net, log_rows, object_rows, seed=args.seed, n_objects=args.n, horizon=args.horizon)
    print(f"wrote {len(log_rows)} events for {args.n} objects to {args.out}")


if __name__ == "__main__":
    main()
