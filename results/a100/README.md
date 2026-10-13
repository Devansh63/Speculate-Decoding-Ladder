# results/a100

A100-SXM4-40GB rows, produced on Sahil's allocation and copied here on
2026-10-13 so that every figure quoting an A100 number is checkable.

Source of truth at the time of copy: `/projects/bikd/ssashi/Speculative_Ladder/results/a100`.
Jobs: 22757104 22802262 22802261 22802309 22802308
Rows: 15
Per-request files: 9

Raw engine logs for these jobs are under `results/logs/a100/` (gzipped).
Job 22713396 is included deliberately: it is `COMPLETED` with exit code 0:0,
ran 28:42, and produced **zero rows** -- it predates the multi-architecture
rebuild and died on `cudaErrorNoKernelImageForDevice`. It is the clearest
example of this project's standing convention that COMPLETED does not mean
measured.

Known gaps, so nobody rediscovers them:
* `per_request_timing` is 0 on every row -- this vLLM build never populates
  engine request metrics. There is no per-request latency anywhere.
* `prefix_caching` reads `null` on every row, meaning the flag was not
  passed. The engine default was ON: the logs report prefix cache hit rates of
  4-8%. Cite the logs, not the rows, for caching state.
* Per-request filenames carry no job id, so where a cell was run twice the
  second run overwrote the first. Cross-node replication for those cells is
  aggregate-only.
