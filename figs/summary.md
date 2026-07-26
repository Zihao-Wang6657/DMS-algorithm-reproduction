# Main Experiment Results

Strict success requires both AndroidWorld success and completion within the official per-task `int(10 * complexity)` action budget.

## Overall

| Method | Tasks | Strict Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Runtime Errors | Final Memory Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 25 | 5 | 20.00% | 67594.0 | 13.88 | 0 | 0 |
| Baseline B | 25 | 9 | 36.00% | 80411.1 | 13.44 | 0 | 25 |
| DMS | 25 | 13 | 52.00% | 42119.9 | 12.20 | 0 | 12 |

## By Round

| Method | Round | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Runtime Errors | End Memory Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 1 | 1/5 | 20.00% | 55393.2 | 13.80 | 0 | 0 |
| Baseline A | 2 | 1/5 | 20.00% | 78873.4 | 13.80 | 0 | 0 |
| Baseline A | 3 | 1/5 | 20.00% | 70927.6 | 13.80 | 0 | 0 |
| Baseline A | 4 | 1/5 | 20.00% | 53105.2 | 14.00 | 0 | 0 |
| Baseline A | 5 | 1/5 | 20.00% | 79670.6 | 14.00 | 0 | 0 |
| Baseline B | 1 | 1/5 | 20.00% | 75239.8 | 13.80 | 0 | 5 |
| Baseline B | 2 | 2/5 | 40.00% | 77027.4 | 13.40 | 0 | 10 |
| Baseline B | 3 | 1/5 | 20.00% | 78593.0 | 13.80 | 0 | 15 |
| Baseline B | 4 | 2/5 | 40.00% | 81489.8 | 13.80 | 0 | 20 |
| Baseline B | 5 | 3/5 | 60.00% | 89705.6 | 12.40 | 0 | 25 |
| DMS | 1 | 2/5 | 40.00% | 47609.6 | 12.80 | 0 | 6 |
| DMS | 2 | 2/5 | 40.00% | 48256.8 | 12.60 | 0 | 6 |
| DMS | 3 | 3/5 | 60.00% | 50793.4 | 12.20 | 0 | 9 |
| DMS | 4 | 3/5 | 60.00% | 27555.6 | 11.60 | 0 | 10 |
| DMS | 5 | 3/5 | 60.00% | 36384.0 | 11.80 | 0 | 12 |
