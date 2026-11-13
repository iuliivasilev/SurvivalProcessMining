# Task 2. Shared Petri net (average over all clients)

## Goal

Task 1 builds one net per pattern.  
Task 2 builds **one net for everyone**, as if profile labels were unknown.

## Algorithm

1. Sample each client’s profile with equal weight 0.25.
2. Generate the client trace from that profile net.
3. Merge all traces into one mixed log.
4. Discover one Petri net from the mixed log (DFG), without using the profile column.
5. Keep true profiles in `objects.csv` only as ground truth for later Task 5.

Main output: `output/mixed/discovered_mixed_net.pnml` and `discovered_mixed_edges.csv`.

Optional control: `parameter_average_net.pnml` = direct average of the four parameter tables (not estimated from the log).
