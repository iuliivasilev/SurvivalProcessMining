"""Task 2: one shared Petri net for a mixture of all client profiles.

Two views of "average over all clients":

1. Mixture log: each student is first drawn from a profile (equal weight by
   default), then walks that profile's net. The DFG estimated on this mixed
   log is the shared net you would discover if you did not segment students.

2. Parameter average: for every edge that appears in any profile, take the
   mean probability and mean rate across profiles (missing edges count as
   probability 0), then renormalize probabilities from each source. This is
   the closed-form mean of the four parameter tables.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_log import Edge, PetriNet, generate, generate_trace, load_net, write_outputs, write_pnml  # noqa: E402

PROFILES = ("browser", "explorer", "abandoner", "converter")
N_OBJECTS = 4000
SEED = 42
MIXTURE_WEIGHTS = {name: 0.25 for name in PROFILES}


def average_parameter_net(nets: dict[str, PetriNet]) -> PetriNet:
    absorbing = frozenset.union(*(net.absorbing for net in nets.values()))
    sources = sorted({edge.src for net in nets.values() for edge in net.edges})
    destinations = sorted({edge.dst for net in nets.values() for edge in net.edges})
    edges: list[Edge] = []
    for src in sources:
        raw: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for net in nets.values():
            outgoing = {edge.dst: edge for edge in net.edges if edge.src == src}
            for dst in destinations:
                if dst == src and dst == "start":
                    continue
                edge = outgoing.get(dst)
                if edge is None:
                    raw[dst].append((0.0, 0.0))
                else:
                    raw[dst].append((edge.prob, edge.rate or 0.0))
        mean_prob = {dst: sum(item[0] for item in values) / len(values) for dst, values in raw.items()}
        total = sum(mean_prob.values())
        if total <= 0:
            continue
        for dst, values in raw.items():
            prob = mean_prob[dst] / total
            if prob <= 0:
                continue
            rates = [rate for prob_i, rate in values if prob_i > 0 and rate > 0]
            rate = sum(rates) / len(rates) if rates else 0.4
            edges.append(Edge(src=src, dst=dst, prob=prob, rate=rate))
    return PetriNet(
        profile="mixed_parameter_average",
        family="exponential",
        time_unit="hour",
        parameter_status="average_of_specified_profiles",
        note="四张设定网的参数平均：每个起点对每个终点取平均概率后重新归一化，速率对出现过该边的剖面取平均。",
        absorbing=absorbing,
        edges=tuple(edges),
    )


def generate_mixture(
    nets: dict[str, PetriNet],
    n_objects: int,
    seed: int,
    weights: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    names = list(weights)
    probs = [weights[name] for name in names]
    log_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    for index in range(n_objects):
        profile = rng.choices(names, weights=probs, k=1)[0]
        net = nets[profile]
        object_id = f"m{index:06d}"
        events, absorbing_action, censored = generate_trace(rng, net, horizon=None, max_events=10_000)
        for clock, action in events:
            log_rows.append({"objectId": object_id, "time": clock, "action": action})
        object_rows.append(
            {
                "objectId": object_id,
                "profile": profile,
                "censored": int(censored),
                "absorbing_action": absorbing_action,
                "n_events": len(events),
                "end_time": events[-1][0] if events else 0.0,
            }
        )
    return log_rows, object_rows


def discover_from_log(
    log_rows: list[dict[str, object]],
    absorbing: frozenset[str],
) -> PetriNet:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    waits: dict[tuple[str, str], float] = defaultdict(float)
    previous: dict[str, str] = {}
    previous_time: dict[str, float] = {}
    for row in log_rows:
        object_id = str(row["objectId"])
        clock = float(row["time"])
        action = str(row["action"])
        src = previous.get(object_id, "start")
        wait = clock - previous_time.get(object_id, 0.0)
        counts[(src, action)] += 1
        waits[(src, action)] += wait
        previous[object_id] = action
        previous_time[object_id] = clock
    outgoing: dict[str, int] = defaultdict(int)
    for (src, _dst), count in counts.items():
        outgoing[src] += count
    edges = []
    for (src, dst), count in sorted(counts.items()):
        edges.append(
            Edge(
                src=src,
                dst=dst,
                prob=count / outgoing[src],
                rate=count / waits[(src, dst)],
            )
        )
    return PetriNet(
        profile="mixed_discovered",
        family="exponential",
        time_unit="hour",
        parameter_status="estimated_from_mixture_log",
        note="从四类等权混合日志上用 DFG 恢复的总网：不分学生类型时会得到的那张网。",
        absorbing=absorbing,
        edges=tuple(edges),
    )


def dump_net_table(path: Path, net: PetriNet) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["src", "dst", "prob", "rate"])
        writer.writeheader()
        for edge in net.edges:
            writer.writerow(
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "prob": f"{edge.prob:.6f}",
                    "rate": f"{edge.rate:.6f}" if edge.rate is not None else "",
                }
            )


def main() -> None:
    nets = {name: load_net(HERE / f"{name}.json") for name in PROFILES}
    out = HERE / "output" / "mixed"
    out.mkdir(parents=True, exist_ok=True)

    avg_net = average_parameter_net(nets)
    write_pnml(out / "parameter_average_net.pnml", avg_net)
    dump_net_table(out / "parameter_average_edges.csv", avg_net)
    (out / "parameter_average_ground_truth.json").write_text(
        json.dumps(
            {
                "profile": avg_net.profile,
                "family": avg_net.family,
                "time_unit": avg_net.time_unit,
                "parameter_status": avg_net.parameter_status,
                "note": avg_net.note,
                "mixture_weights": MIXTURE_WEIGHTS,
                "absorbing": sorted(avg_net.absorbing),
                "edges": [
                    {
                        "src": edge.src,
                        "dst": edge.dst,
                        "prob": edge.prob,
                        "rate": edge.rate,
                        "shape": None,
                        "scale": None,
                    }
                    for edge in avg_net.edges
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log_rows, object_rows = generate_mixture(nets, N_OBJECTS, SEED, MIXTURE_WEIGHTS)
    write_outputs(
        out,
        PetriNet(
            profile="mixed_population",
            family="exponential",
            time_unit="hour",
            parameter_status="mixture_of_specified_profiles",
            note="四类学生等权混合生成的日志。objects.csv 里的 profile 列是每位学生的真实类型（ground truth）。",
            absorbing=frozenset({"active", "dropout"}),
            edges=avg_net.edges,
        ),
        log_rows,
        object_rows,
        seed=SEED,
        n_objects=N_OBJECTS,
        horizon=None,
    )
    # Override ground_truth with mixture metadata; keep parameter-average edges as reference.
    (out / "mixture_meta.json").write_text(
        json.dumps(
            {
                "n_objects": N_OBJECTS,
                "seed": SEED,
                "weights": MIXTURE_WEIGHTS,
                "profile_counts": {
                    name: sum(1 for row in object_rows if row["profile"] == name) for name in PROFILES
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    discovered = discover_from_log(log_rows, frozenset({"active", "dropout"}))
    write_pnml(out / "discovered_mixed_net.pnml", discovered)
    dump_net_table(out / "discovered_mixed_edges.csv", discovered)

    profile_counts = {name: sum(1 for row in object_rows if row["profile"] == name) for name in PROFILES}
    print("mixture profile counts:", profile_counts)
    print(f"mixed log: {len(log_rows)} events for {N_OBJECTS} objects")
    print(f"parameter-average edges: {len(avg_net.edges)}")
    print(f"discovered-from-mixture edges: {len(discovered.edges)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
