# Task 1. Four Petri nets for client patterns

## What this is

One Petri net = a state machine over learning actions:

- places = current action (`video`, `problem`, `access`, `wiki`, `discussion`, `navigate`)
- transitions = next action
- each edge has **probability** and exponential **rate λ** (mean wait ≈ 1/λ hours)
- absorbing ends: `active`, `dropout`

## Four patterns

| profile | behavior idea | synthetic result (2000 clients each) |
| --- | --- | --- |
| browser | mostly watch / access, rarely solve problems, often drop | mean length 5.7, dropout ~76% |
| explorer | navigate / access / wiki a lot, longer traces | mean length 7.5, dropout ~68% |
| abandoner | reaches problem/discussion, then drops | mean length 4.0, dropout ~85% |
| converter | video↔problem, stays active more often | mean length 6.8, active ~66% |

Parameters are **specified**, not estimated from KDD.

## Files

- `browser.json`, `explorer.json`, `abandoner.json`, `converter.json`
- `output/<profile>/` — synthetic log + ground truth
- `generate_all_profiles.py` — regenerate all logs
