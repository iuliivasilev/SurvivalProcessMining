# Tasks 1–3: Short English Report

## What was done

1. **Task 1.** Built **4 Petri nets** for client patterns: `browser`, `explorer`, `abandoner`, `converter`. Each net has places = learning actions, transitions = next actions with probability and exponential rate λ. Parameters are **specified** (hand-set from qualitative behavior), **not estimated from KDD**. Synthetic event logs with ground truth were generated.
2. **Task 2.** Built **one shared Petri net** as the average over all clients: mix the 4 patterns with equal weights, then discover a net from the mixed log **without using the profile label**.
3. **Task 3.** Compared **fitness** and **precision** of these nets with **real KDD Cup 2015** data (downloaded from Kaggle). Used a **subset** (top 5 courses by enrollment count), not the full log.

---

## Task 1 — algorithm (simple steps)

1. Fix 4 behavior patterns and the action set: `video`, `problem`, `access`, `wiki`, `discussion`, `navigate`, plus absorbing ends `active` / `dropout`.
2. For each pattern, write a parameter table: for every `src → dst`, set `prob` and `rate` (λ). Probabilities from the same `src` sum to 1.
3. Simulate many enrollments: from `start`, sample next action by probability, sample waiting time by Exp(λ), write `(objectId, time, action)` until absorption.
4. Save the parameter table as ground truth together with the synthetic log.

**Note:** Numbers in `*.json` are design choices that match the qualitative profile, not KDD frequencies.

---

## Task 2 — algorithm (simple steps)

1. Draw a profile for each synthetic client with weights 25% / 25% / 25% / 25%.
2. Generate the client’s trace from that profile’s net.
3. Put all traces into one mixed log.
4. Ignore the true profile label. Estimate one Petri net from the mixed log with a directly-follows graph (edge frequencies → probabilities; mean waiting time → λ).
5. Keep the true profile only in `objects.csv` for later Task 5.

---

## Task 3 — algorithm (simple steps)

1. Load KDD `log_train`, `enrollment_train`, `truth_train`.
2. Keep only the **5 largest courses** (subset allowed by the plan).
3. Drop `page_close` (not in our nets). Keep the 6 learning actions.
4. Append `active` or `dropout` at the end of each enrollment using the truth label (0 → active, 1 → dropout).
5. Build the log directly-follows graph (DFG).
6. For each model net, compute:
   - **fitness** = share of log DF-moves that exist as model edges  
   - **precision** = share of model edges that appear at least once in the log DFG  
7. These metrics judge **graph paths / edge presence**, **not** transition weights (probabilities or λ).

### Results (clean table)

| model | fitness | precision | model_edges | used_in_KDD_subset | unused_edge |
| --- | ---: | ---: | ---: | ---: | --- |
| browser | 1.000 | 0.981 | 54 | 53 | `start → problem` |
| explorer | 1.000 | 0.981 | 54 | 53 | `start → problem` |
| abandoner | 1.000 | 0.981 | 54 | 53 | `start → problem` |
| converter | 1.000 | 0.981 | 54 | 53 | `start → problem` |
| mixed_discovered | 1.000 | 0.981 | 54 | 53 | `start → problem` |

**Why the numbers look the same:** the 4 pattern nets (and the mixed net) share almost the **same edge set**. Fitness/precision therefore coincide. Pattern differences are mainly in **probabilities and λ**, which these two metrics do not evaluate.

**Why precision = 0.98:** one model edge, `start → problem`, is almost unused in this 5-course subset (after filtering). This is expected; no fix is required for Task 3.

**Data size used:** ~47,706 enrollments, ~2.69M scored DF-moves.

---

## Next (as suggested)

Think how to find the client types inside real KDD automatically, e.g. build a CountVectorizer / TfIdf vector per enrollment history and cluster enrollments; clusters may correspond to the behavior patterns.
