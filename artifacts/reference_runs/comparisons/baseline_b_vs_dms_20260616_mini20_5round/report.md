# Baseline B vs DMS Report

## Setup

- Baseline B run:
  `/mnt/20T/shichangyue/jfz/ysb_task/Darwinian/working_space/runs/baseline_b_static_memory/20260616_mini20_5round_gpu6`
- DMS run:
  `/mnt/20T/shichangyue/jfz/ysb_task/Darwinian/working_space/runs/dms_hierarchical_memory/20260616_mini20_5round_resume`
- Comparison artifacts:
  `/mnt/20T/shichangyue/jfz/ysb_task/Darwinian/working_space/runs/comparisons/baseline_b_vs_dms_20260616_mini20_5round`
- Dataset:
  `mini_benchmark_20apps.yaml`
- Protocol:
  `5-round lifespan`, `100` task attempts total, identical mini-benchmark task list.

## Overall Result

| Method | Successes | Success Rate | Avg Tokens / Task | Avg Steps / Task | Final Memory Size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline B | 12 / 100 | 12.00% | 69689.32 | 14.26 | 100 |
| DMS | 12 / 100 | 12.00% | 61308.57 | 13.38 | 20 |

## Round-by-Round

| Method | R1 | R2 | R3 | R4 | R5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline B success rate | 15.00% | 10.00% | 5.00% | 15.00% | 15.00% |
| DMS success rate | 10.00% | 10.00% | 5.00% | 15.00% | 20.00% |

## Takeaways

1. On the primary success metric, this run is a tie. Baseline B and DMS both finished at `12 / 100 = 12.00%`.
2. DMS is more efficient at equal success:
   `-12.03%` average tokens per task and `-6.17%` average steps per task relative to Baseline B.
3. DMS is much more memory-bounded:
   final memory size `20` versus Baseline B `100`, an `80%` reduction.
4. DMS shows the stronger late-lifespan trend:
   round 5 reaches `20.00%`, while Baseline B stays at `15.00%`.
5. The statement "DMS is strictly better than Baseline B on overall task success" is **not supported** by this specific run.
   The defensible conclusion is:
   DMS matches Baseline B on success, while outperforming it on efficiency, memory control, and late-round behavior.

## Charts

- `success_rate_by_round.png`
- `avg_tokens_by_round.png`
- `avg_steps_by_round.png`
- `memory_size_timeline.png`
