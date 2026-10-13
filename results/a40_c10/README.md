# results/a40_c10 -- the non-determinism floor

Three realizations of ONE cell: A40, 2K bucket, rung `off`, n=50,
concurrency 1, seed 1234, prefix caching on (read back off the engine).
Harness `v3-2026-10-13`.

* job 22875382, gpub015 -- two runs (REPEAT=2)
* job 22876140, gpub036 -- two runs (REPEAT=2)

Job 22875382 produced only ONE per-request file: both its runs shared a job id
and the first version of the v3 file-name fix still overwrote run 1. The fix
now picks the first free name (`_j<job>`, then `_r2`, `_r3`), which is why
22876140 kept both. Three surviving realizations is what the control needed.

Result: 150/150 sequences bit-identical across all three pairs, per-token
hazard 0.000%. Zero events in 12,800 token positions bounds the true hazard at
about 0.023% per token (rule of three, 95%). See C21 in `paper/claims.md`.

Throughput, same cell, four runs: 32.5698 and 32.5696 on gpub015, 32.5703 and
32.5644 on gpub036 -- a 0.018% total spread across two nodes.

**Still missing:** no speculative arm has been run twice at one configuration,
so only the BASELINE is shown reproducible. See threat 12.
