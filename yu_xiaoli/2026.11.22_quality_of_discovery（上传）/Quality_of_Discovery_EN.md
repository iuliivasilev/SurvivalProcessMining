# Generation Event Log → Quality of Discovery (Recovery)

Matches the sketch block:

**Ground Truth Petri Net → Generate Event Log → Discover Petri Net → Compare → Recovery Quality**

Constraint from the sketch: **Only one Petri Net**. We use **`converter`**.

## Pipeline

1. Ground-truth net: `step1/converter.json` (specified `p` and `λ`)
2. Generate log: 2000 enrollments → `output/generated/event_log.csv`
3. Discover net from the log only (DFG probabilities + MLE rates)
4. Compare true vs discovered edges/parameters
5. Report recovery quality

## Results

| Structural | Value |
| --- | ---: |
| edge precision | 1.0 |
| edge recall | 1.0 |
| fitness (replay) | 1.0 |

| Parameter | MAE | MAPE | WAPE |
| --- | ---: | ---: | ---: |
| probability | 0.0062 | 0.072 | 0.027 |
| lambda rate | 0.029 | 0.080 | 0.033 |

Run: `python recovery_quality/run_one_net_recovery.py`
