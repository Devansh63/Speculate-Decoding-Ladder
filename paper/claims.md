# Claims to evidence

Every claim the report will make, the experiment that supports it, and what
result would falsify it. Append-only: if a claim changes, add a dated line rather
than editing the old one, so the record shows what was believed when.

| ID | Claim | Evidence needed | Status | Falsifier |
|---|---|---|---|---|
| C1 | Draft acceptance decays with prompt context | Acceptance at 2K, 8K, 32K per rung, paired bootstrap over prompts | **Not measured.** Only 2K exists | D within noise of zero across buckets |
| C2 | Different mechanisms decay at different rates | Same, compared at a common draft truncation | Not measured | Rung by log-context interaction not significant |
| C3 | Speedup crosses 1.0x, and the crossing point differs per rung | Speedup against the no-speculation arm across the concurrency axis at fixed context | **Partial.** One uncontrolled high-load point where all rungs are below 1.0x | No rung ever crosses in the measurable range |
| C4 | The crossing point can be predicted on hardware not used for fitting | Cost model fitted on one platform, prediction frozen and tagged, then measured on A40 or A100 | Not started | Median error worse than a factor of two, or no better than naive transfer |
| C5 | n-gram and a trained head are of opposite type on this workload | Per-position acceptance curves at a common k | **Supported at 2K**, single run, no interval | Curves converge once intervals are drawn |
| C6 | Draft length matters: k=7 dominates k=15 for DFlash | Throughput and accepted length at both k, across load | **Suggestive at 2K**, one point | k=15 wins at low concurrency |

## Rules

1. No claim moves to "supported" from a single run without a confidence interval.
2. Acceptance is never compared across rungs at different k without truncating to
   a common length first.
3. Nothing is pooled across context regimes (native against YaRN) or across model
   runners. Both are separate populations.
4. A prediction counts only if it was committed and tagged before the measurement
   it predicts.
