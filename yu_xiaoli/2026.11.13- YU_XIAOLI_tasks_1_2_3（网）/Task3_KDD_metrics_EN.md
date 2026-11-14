# Task 3. Fitness and precision on KDD

## Goal

Compare our Petri nets with real KDD Cup 2015 logs (Kaggle download). Use a subset if the full log is too large.

## Algorithm

1. Select the 5 courses with the most enrollments.
2. Keep learning actions only; drop `page_close`.
3. Append absorbing label from `truth_train` (`0 → active`, `1 → dropout`).
4. Build the log DFG.
5. Compute fitness and precision against each model net.

## Definitions

- **fitness** = fraction of log DF-moves allowed by the model edges  
- **precision** = fraction of model edges that appear in the log DFG  

These metrics evaluate **paths/edges in the graph**, not transition weights.

## Result

All five models: fitness = 1.0, precision ≈ 0.981 (53/54 edges used).  
Unused model edge in this subset: `start → problem`.  
Identical scores are expected because the nets share almost the same edge set.

Clean table: `output/kdd_fitness_precision_clean.csv`  
Full English summary: `../English_report_tasks_1_2_3.md`
