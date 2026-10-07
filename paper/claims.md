# Claims to evidence

Every claim the report will make, the experiment that supports it, and what
result would falsify it. Append-only: if a claim changes, add a dated line rather
than editing the old one, so the record shows what was believed when.

## As of 30 Sep 2026

| ID | Claim | Evidence needed | Status | Falsifier |
|---|---|---|---|---|
| C1 | Draft acceptance decays with prompt context | Acceptance at 2K, 8K, 32K per rung, paired bootstrap over prompts | **Not measured.** Only 2K exists | D within noise of zero across buckets |
| C2 | Different mechanisms decay at different rates | Same, compared at a common draft truncation | Not measured | Rung by log-context interaction not significant |
| C3 | Speedup crosses 1.0x, and the crossing point differs per rung | Speedup against the no-speculation arm across the concurrency axis at fixed context | **Partial.** One uncontrolled high-load point where all rungs are below 1.0x | No rung ever crosses in the measurable range |
| C4 | The crossing point can be predicted on hardware not used for fitting | Cost model fitted on one platform, prediction frozen and tagged, then measured on A40 or A100 | Not started | Median error worse than a factor of two, or no better than naive transfer |
| C5 | n-gram and a trained head are of opposite type on this workload | Per-position acceptance curves at a common k | **Supported at 2K**, single run, no interval | Curves converge once intervals are drawn |
| C6 | Draft length matters: k=7 dominates k=15 for DFlash | Throughput and accepted length at both k, across load | **Suggestive at 2K**, one point | k=15 wins at low concurrency |

## Status as of 7 Oct 2026

Three context buckets measured on A40 (jobs 22598608, 22685911, 22713173), plus
a k sweep and replication (22714602).

| ID | Status now | Evidence |
|---|---|---|
| C1 | **Measured, strongly.** D at k=5, concurrency 1: DFlash 0.219 (8K), 0.402 (32K); n-gram 0.044 (8K), 0.016 (32K) | `notes/2026-10-06-8k-context-decay.md`, `notes/2026-10-06-32k-inversion.md` |
| C2 | **Measured.** DFlash decays 5x faster at 8K and 25x faster at 32K. First-position acceptance 0.601 -> 0.296 against 0.601 -> 0.552 | same |
| C3 | **Measured at all three buckets.** n-gram crosses at 14.4, 8.9, 3.7. DFlash crosses above 32, about 12, below 1 | same |
| C4 | **AT RISK.** A100 is blocked and H200 cannot run, so there is no second platform to predict onto. See below | `env/a100-sm80-failure.md` |
| C5 | **Holds at three buckets.** The curves diverge further with context rather than converging | `notes/2026-10-06-32k-inversion.md` |
| C6 | **Partly refuted.** k=7 does beat k=15 at 2K saturation. But optimal k does NOT shrink with context: at 32K speedup rises monotonically, 0.76x, 0.78x, 0.81x, 0.83x for k = 1, 2, 4, 7 | `notes/2026-10-07-k-sweep-refutation.md` |

### C4 needs saying plainly

C4 is the portability claim, and it is the reason the project separates tau from
R at all. It requires fitting a cost model on one platform and predicting on
another. **There is currently only one working platform.** The vLLM build has no
usable sm_80 kernels, so A100 fails outright, and H200 has no sm_90 cubins.

This is the price of the decision recorded in `TODO.md` to pursue the context
axis instead of rebuilding. It is a real cost and should appear in the paper's
limitations rather than being quietly dropped. A rebuild with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` would restore it; the root cause is known.

## New claims raised by the data

| ID | Claim | Status | Falsifier |
|---|---|---|---|
| C7 | Acceptance reported by serving engines is conditional on a draft being proposed, so comparing rungs on it is invalid without a coverage term | **Measured.** n-gram's coverage is 0.132 at 2K rising to 0.212 at 32K; corrected, its implied R falls below DFlash's, which matches what the hardware predicts | Coverage near 1.0 for all rungs, making the correction vacuous |
| C8 | The ranking of rungs inverts along the context axis | **Measured, three points.** At concurrency 1: DFlash 1.86x / 1.26x / 0.83x against n-gram 1.16x / 1.14x / 1.08x | The ordering holds at 32K, or reverses back at some longer context |
| C9 | Drafter cost is fixed per decode step, not per drafted token, and scales with context for model-based drafters | **Partial.** Measured at 32K: R(k) = 1.662 + 0.0148k, so a 66% fixed cost against 1.5% per position. At 2K only bounded (total R is 1.360 at k=7, so the fixed part is under 0.36) | R_draft flat across buckets, or marginal cost dominant at long context |
| C10 | Speculative decoding perturbs output less than the serving engine perturbs itself across batch sizes | **Measured.** Baseline against itself agrees on 12.7% of sequences; speculative against baseline on 19.0%. Per-token divergence 0.80% against 0.65% | The baseline reproduces itself exactly while speculative arms do not |
| C11 | Run-to-run variance is below 1% | **One replication.** off 14.9 -> 14.8, n-gram 16.1 -> 16.0, DFlash k=7 12.3 -> 12.2, across different jobs and nodes | A second repeat landing further out |

## Predictions that failed

Kept deliberately. A register that records only confirmations is not evidence of
anything.

| Date | Prediction | Outcome | What it taught |
|---|---|---|---|
| 6 Oct | DFlash at 32K is misconfigured; optimal k is 1, giving about 1.17x | **Refuted.** k=1 gives 0.76x, the worst in the sweep; speedup rises with k | A two-parameter cost model was fitted to one data point by assuming the intercept, so the answer came from the assumption. The sweep that tested it cost one hour |
| 6 Oct | DFlash at 32K near 0.95x, n-gram near 1.15x | Direction right, magnitudes off: 0.83x and 1.08x | Linear extrapolation of a two-point decay understates how fast a prediction-based drafter fails. The miss is what exposed C9 |

## Rules

1. No claim moves to "supported" from a single run without a confidence interval.
2. Acceptance is never compared across rungs at different k without truncating to
   a common length first.
3. Nothing is pooled across context regimes (native against YaRN) or across model
   runners. Both are separate populations.
4. A prediction counts only if it was committed and tagged before the measurement
   it predicts.

### Note on rule 1, added 7 Oct 2026

Per-request timing is unavailable on this vLLM build (`0/25` requests), so the
paired bootstrap rule 1 assumes is not possible. Intervals must come from
repeated runs instead.

One replication exists and bounds run-to-run variance below 1%, which is well
under the effects being claimed: the 32K inversion is a 30% gap, roughly thirty
times the noise. That is a bound, not an interval, so under rule 1 the claims
above are **measured** rather than **supported**. Two further repeats of the 32K
cell would close the gap properly.

The 32K n-gram row at concurrency 4 (0.99x) is within about one noise width of
1.0 and should be reported as indistinguishable from no speedup, not as below
it.
