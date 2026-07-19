# Formal Recovery Experiment: Baseline B + DMS

- Protocol ID: `formal_recovery_bd_20apps_v2`
- Scale: 20 tasks x 5 rounds x 2 methods = 200 scored records
- Methods: Baseline B, then DMS; both memories start empty
- Baseline A: reused from the v1 RunRoot and disclosed as cross-run
- Frozen tasks, seeds, order, budgets, model, Prompt and evaluator: unchanged
- Task-level infrastructure retry: unchanged, at most once
- Infrastructure isolation: cold restart emulator before every round
- Prior failed Baseline B data: excluded from recovered formal results
