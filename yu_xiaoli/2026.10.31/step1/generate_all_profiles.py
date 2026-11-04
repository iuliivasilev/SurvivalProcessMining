"""Generate one event log for every behavior profile in step1/*.json."""

from __future__ import annotations

import json
from pathlib import Path

from generate_log import generate, load_net, write_outputs

HERE = Path(__file__).resolve().parent
PROFILES = ("browser", "explorer", "abandoner", "converter")
N_OBJECTS = 2000
SEED_BASE = 1


def main() -> None:
    summary = []
    for offset, name in enumerate(PROFILES):
        config = HERE / f"{name}.json"
        out = HERE / "output" / name
        net = load_net(config)
        seed = SEED_BASE + offset
        log_rows, object_rows = generate(net, N_OBJECTS, seed, horizon=None)
        write_outputs(out, net, log_rows, object_rows, seed=seed, n_objects=N_OBJECTS, horizon=None)

        ends = [row["absorbing_action"] for row in object_rows]
        active = sum(1 for end in ends if end == "active")
        dropout = sum(1 for end in ends if end == "dropout")
        mean_events = sum(int(row["n_events"]) for row in object_rows) / N_OBJECTS
        summary.append(
            {
                "profile": name,
                "n_objects": N_OBJECTS,
                "n_events": len(log_rows),
                "mean_events": round(mean_events, 2),
                "active_share": round(active / N_OBJECTS, 3),
                "dropout_share": round(dropout / N_OBJECTS, 3),
                "n_edges": len(net.edges),
                "seed": seed,
                "out": str(out.relative_to(HERE)),
            }
        )
        print(
            f"{name}: events={len(log_rows)} mean_len={mean_events:.2f} "
            f"active={active / N_OBJECTS:.1%} dropout={dropout / N_OBJECTS:.1%}"
        )

    (HERE / "output" / "profiles_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote output/profiles_summary.json")


if __name__ == "__main__":
    main()
