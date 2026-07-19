# Main Experiment Results

Strict success requires both AndroidWorld success and completion within the official per-task `int(10 * complexity)` action budget.

## Overall

| Method | Tasks | Strict Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Infra Retries | Infra Failure After Retry | Final Memory Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 25 | 7 | 28.00% | 49526.0 | 11.32 | 0 | 0 | 0 |
| Baseline B | 25 | 6 | 24.00% | 45929.2 | 11.28 | 0 | 0 | 25 |
| DMS | 25 | 5 | 20.00% | 39473.8 | 10.76 | 2 | 1 | 13 |

## By Round

| Method | Round | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Infra Retries | Infra Failure After Retry | End Memory Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 1 | 2/5 | 40.00% | 54262.0 | 10.40 | 0 | 0 | 0 |
| Baseline A | 2 | 1/5 | 20.00% | 53471.4 | 11.60 | 0 | 0 | 0 |
| Baseline A | 3 | 1/5 | 20.00% | 52339.0 | 11.80 | 0 | 0 | 0 |
| Baseline A | 4 | 1/5 | 20.00% | 45888.0 | 11.80 | 0 | 0 | 0 |
| Baseline A | 5 | 2/5 | 40.00% | 41669.6 | 11.00 | 0 | 0 | 0 |
| Baseline B | 1 | 1/5 | 20.00% | 46031.0 | 11.40 | 0 | 0 | 5 |
| Baseline B | 2 | 1/5 | 20.00% | 42379.2 | 11.40 | 0 | 0 | 10 |
| Baseline B | 3 | 1/5 | 20.00% | 50037.0 | 11.60 | 0 | 0 | 15 |
| Baseline B | 4 | 1/5 | 20.00% | 44716.4 | 11.40 | 0 | 0 | 20 |
| Baseline B | 5 | 2/5 | 40.00% | 46482.6 | 10.60 | 0 | 0 | 25 |
| DMS | 1 | 1/5 | 20.00% | 50453.0 | 13.40 | 1 | 0 | 6 |
| DMS | 2 | 1/5 | 20.00% | 38238.8 | 11.60 | 1 | 1 | 9 |
| DMS | 3 | 1/5 | 20.00% | 33453.2 | 8.80 | 0 | 0 | 11 |
| DMS | 4 | 1/5 | 20.00% | 40233.4 | 10.40 | 0 | 0 | 12 |
| DMS | 5 | 1/5 | 20.00% | 34990.8 | 9.60 | 0 | 0 | 13 |
