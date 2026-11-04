"""Task 3: fitness / precision of our Petri nets on a KDD Cup 2015 subset.

KDD events keep the six learning actions used in Step 1. page_close is dropped
because our nets do not generate it. Optional absorbing labels from truth_train
are appended as active (label 0) or dropout (label 1) so terminal edges can be
scored too.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "step1"))

from generate_log import load_net  # noqa: E402

LEARNING = frozenset({"problem", "video", "access", "wiki", "discussion", "navigate"})
ABSORBING = frozenset({"active", "dropout"})
N_COURSES = 5
OUT = Path(__file__).resolve().parent / "output"


def load_enrollments(path: Path) -> dict[str, str]:
    """enrollment_id -> course_id"""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["enrollment_id"]] = row["course_id"]
    return mapping


def pick_courses(enrollment_to_course: dict[str, str], n_courses: int) -> set[str]:
    counts: Counter[str] = Counter(enrollment_to_course.values())
    return {course for course, _ in counts.most_common(n_courses)}


def load_truth(path: Path) -> dict[str, str]:
    """enrollment_id -> active|dropout"""
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            enrollment_id, label = row[0], row[1]
            # file may or may not have a header
            if enrollment_id == "enrollment_id":
                continue
            labels[enrollment_id] = "dropout" if label.strip() == "1" else "active"
    return labels


def stream_subset_traces(
    log_path: Path,
    keep_ids: set[str],
    truth: dict[str, str],
) -> dict[str, list[str]]:
    """Return enrollment_id -> list of learning actions (+ optional absorbing)."""
    traces: dict[str, list[str]] = defaultdict(list)
    with log_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            enrollment_id = row["enrollment_id"]
            if enrollment_id not in keep_ids:
                continue
            event = row["event"]
            if event in LEARNING:
                traces[enrollment_id].append(event)
    for enrollment_id, actions in list(traces.items()):
        if not actions:
            continue
        label = truth.get(enrollment_id)
        if label is not None:
            actions.append(label)
    return {key: value for key, value in traces.items() if value}


def dfg_counts(traces: dict[str, list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for actions in traces.values():
        previous = "start"
        for action in actions:
            counts[(previous, action)] += 1
            previous = action
    return counts


def net_edge_set(net) -> set[tuple[str, str]]:
    return {(edge.src, edge.dst) for edge in net.edges}


def load_edges_csv(path: Path) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edges.add((row["src"], row["dst"]))
    return edges


def fitness(log_dfg: Counter[tuple[str, str]], model_edges: set[tuple[str, str]]) -> dict[str, float]:
    total = sum(log_dfg.values())
    matched = sum(count for edge, count in log_dfg.items() if edge in model_edges)
    unique_log = set(log_dfg)
    return {
        "fitness_event": matched / total if total else 0.0,
        "fitness_unique_edges": len(unique_log & model_edges) / len(unique_log) if unique_log else 0.0,
        "log_moves": total,
        "matched_moves": matched,
        "log_unique_edges": len(unique_log),
        "matched_unique_edges": len(unique_log & model_edges),
    }


def precision(log_dfg: Counter[tuple[str, str]], model_edges: set[tuple[str, str]]) -> dict[str, float]:
    """Precision = share of model edges that actually appear in the log DFG."""
    if not model_edges:
        return {"precision": 0.0, "model_edges": 0, "used_model_edges": 0}
    used = model_edges & set(log_dfg)
    return {
        "precision": len(used) / len(model_edges),
        "model_edges": len(model_edges),
        "used_model_edges": len(used),
        "unused_model_edges": len(model_edges - set(log_dfg)),
    }


def main() -> None:
    train = ROOT / "data" / "kdd2015" / "train" / "train"
    log_path = train / "log_train.csv"
    enrollment_path = train / "enrollment_train.csv"
    truth_path = train / "truth_train.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    enrollment_to_course = load_enrollments(enrollment_path)
    courses = pick_courses(enrollment_to_course, N_COURSES)
    keep_ids = {eid for eid, course in enrollment_to_course.items() if course in courses}
    truth = load_truth(truth_path)

    print(f"selected {len(courses)} courses, {len(keep_ids)} enrollments")
    print("streaming log_train.csv (this may take a minute)...")
    traces = stream_subset_traces(log_path, keep_ids, truth)
    log_dfg = dfg_counts(traces)
    n_events = sum(len(actions) for actions in traces.values())
    print(f"kept traces={len(traces)} learning+label events={n_events} dfg_edges={len(log_dfg)}")

    models: dict[str, set[tuple[str, str]]] = {}
    for name in ("browser", "explorer", "abandoner", "converter"):
        net = load_net(ROOT / "step1" / f"{name}.json")
        models[name] = net_edge_set(net)
    models["mixed_discovered"] = load_edges_csv(ROOT / "step1" / "output" / "mixed" / "discovered_mixed_edges.csv")

    rows = []
    for name, edges in models.items():
        fit = fitness(log_dfg, edges)
        prec = precision(log_dfg, edges)
        row = {
            "model": name,
            "fitness": round(fit["fitness_event"], 6),
            "fitness_unique_edges": round(fit["fitness_unique_edges"], 6),
            "precision": round(prec["precision"], 6),
            "log_moves": fit["log_moves"],
            "matched_moves": fit["matched_moves"],
            "model_edges": prec["model_edges"],
            "used_model_edges": prec["used_model_edges"],
        }
        rows.append(row)
        print(
            f"{name:18} fitness={row['fitness']:.4f} "
            f"precision={row['precision']:.4f} "
            f"edges={row['used_model_edges']}/{row['model_edges']}"
        )

    with (OUT / "kdd_fitness_precision.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data": str(log_path),
        "n_courses": N_COURSES,
        "courses": sorted(courses),
        "n_enrollments_in_courses": len(keep_ids),
        "n_traces_with_learning_events": len(traces),
        "n_events_scored": n_events,
        "dropped_event": "page_close",
        "absorbing_from_truth": True,
        "definitions": {
            "fitness": "在日志直接跟随中，能被模型边走通的次数占比",
            "precision": "模型边里，真正在该 KDD 子集日志中出现过的占比",
        },
        "results": rows,
    }
    (OUT / "kdd_fitness_precision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUT / "kdd_subset_dfg.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["src", "dst", "count"])
        writer.writeheader()
        for (src, dst), count in sorted(log_dfg.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"src": src, "dst": dst, "count": count})

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
