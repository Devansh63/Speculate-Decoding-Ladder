# What the data says

The argument as a whole, rather than per experiment. Each section names the
notes and jobs behind it.

All measurements: Qwen3-8B on A40, LongBench-v2, greedy decoding, vLLM pinned to
the stable model runner. Jobs 22598608 (2K), 22685911 (8K), 22713173 (32K),
22714602 (k sweep and replication).

---

## The headline: the ranking of the rungs inverts along context

At concurrency 1, the operating point at which nearly all published speculative
decoding results are reported:

| context | DFlash k=7 | n-gram k=5 | winner |
|---|---|---|---|
| 2K | **1.86x** | 1.16x | trained head, by 60% |
| 8K | **1.26x** | 1.14x | trained head, by 11% |
| 32K | 0.83x | **1.08x** | **retrieval, by 30%** |

A trained drafting head beats prompt lookup by 60% at short context and *loses*
to it at long context, where it is also slower than not speculating at all.

The usable operating range collapses alongside it. The concurrency at which each
rung stops paying:

| rung | 2K | 8K | 32K |
|---|---|---|---|
| n-gram k=5 | 14.4 | 8.9 | 3.7 |
| DFlash k=7 | above 32 | about 12 | below 1 |

This is the claim the project exists to make, and it is stronger than the one in
the proposal. "The right rung depends on operating conditions" is unsurprising.
"The ranking reverses along an axis operators control, predictably from what each
mechanism does" is not.

---

## Why it inverts: two independent penalties

### Acceptance decays at very different rates

| | 2K | 8K | 32K |
|---|---|---|---|
| DFlash first-position acceptance | 0.601 | 0.451 | 0.296 |
| n-gram first-position acceptance | 0.601 | 0.563 | 0.552 |
| DFlash decay D at k=5 | | 0.219 | **0.402** |
| n-gram decay D at k=5 | | 0.044 | **0.016** |

Identical at 2K, unrecognisable at 32K. DFlash **predicts**, and prediction gets
harder as the context to condition on grows. n-gram **retrieves**, and a literal
repeat is a literal repeat however long the document is. n-gram's acceptance did
not move measurably across a sixteen-fold increase in context.

n-gram also fires more often as context grows, because a longer document holds
more matchable spans: coverage 0.132 -> 0.161 -> 0.212.

### Drafter cost scales with context for one mechanism and not the other

The k sweep at 32K gives

```
R(k) ~= 1.662 + 0.0148 * k
```

A **66% fixed cost** to run the drafter at all, and **1.5% per additional drafted
token**. DFlash's head attends over the full context, so its cost grows with
context while its accuracy falls. n-gram's drafter is a CPU string lookup, O(1)
in context, and its R stays at 1.290 against DFlash's 1.677 even at k=1.

Two penalties, compounding, in the same direction.

---

## The framework the data forced

The project began with

```
Speedup ~= tau / R
```

where tau is mean accepted length and R the verification cost ratio. Two
measurements showed that is too coarse.

**Coverage.** Serving engines report acceptance *conditional on a draft having
been proposed*. DFlash proposes on every step; n-gram fires on 13% of steps at
2K. Uncorrected, n-gram appears to out-accept DFlash (2.889 against 2.543) while
being 38% slower. Corrected:

```
tau_eff = cov * tau_cond + (1 - cov) * 1
```

n-gram's implied R drops to 1.082, below DFlash's 1.360, which is what the
hardware predicts for a CPU lookup verifying six tokens instead of eight. The
picture only becomes coherent with the coverage term.

**Drafter cost.** R bundles two things that behave differently: verification,
which scales with k, and drafting, which is fixed per decode step and scales with
context for model-based drafters.

```
R(k) = R_draft + R_verify(k)
```

So the working model is four quantities, not two:

```
Speedup ~= [cov * tau_cond + (1 - cov)] / (R_draft + R_verify(k))
```

Every added term is invisible at 2K and decisive at 32K, which is the recurring
shape of this project's results.

---

## Losslessness, honestly stated

Speculative decoding is supposed to return exactly what the target model would
have returned. On a continuous-batching server that is not testable.

Comparing the no-speculation arm **against itself** at different concurrency,
same prompts, same seed, no speculation anywhere: the outputs agree on only
**12.7%** of sequences. Speculative arms against the baseline agree on **19.0%**.

**Speculation perturbs the output less than the engine perturbs itself.** Implied
per-token divergence is 0.80% baseline-to-baseline and 0.65%
speculative-to-baseline. The causes are batch-dependent reduction order in GPU
kernels and prefix caching.

The defensible claim is therefore relative: speculative decoding does not
measurably perturb the greedy decoding path beyond the reproducibility floor of
the serving system it runs on. Any paper claiming byte-exact equality on a
vLLM-class server either disabled batching or did not check.

---

## What we got wrong

Both failures are more informative than the confirmations.

**The k prediction.** From the 32K acceptance profile, we inferred that DFlash
was misconfigured and k=1 would give about 1.17x. Measured: 0.76x, the worst
setting in the sweep. The error was fitting a two-parameter cost model to one
data point by assuming the intercept was 1.0, so the conclusion came from the
assumption rather than the data. Testing it cost one hour and produced the
drafter-cost finding above.

**The extrapolation.** Predicting 32K from 2K and 8K gave 0.95x and 1.15x
against a measured 0.83x and 1.08x. Direction right, DFlash worse than a linear
decay suggests. That miss is what exposed the fixed drafter cost.

---

## What the paper cannot claim

- **Hardware portability.** The project's original purpose was to show tau
  transfers across machines while R does not. The vLLM build has no usable sm_80
  kernels, so A100 fails outright and H200 cannot run. **Every number comes from
  one GPU.** The decomposition is demonstrated across load and context, never
  across hardware. This is the largest limitation and a rebuild would fix it.
- **Confidence intervals.** Per-request timing is unavailable on this build, so
  the planned paired bootstrap is impossible. One replication bounds run-to-run
  variance below 1%, far under the effects claimed, but a bound is not an
  interval.
- **Generality.** One target model, one drafter family, one workload. Whether
  prediction-based drafters generally decay faster than retrieval-based ones is a
  conjecture supported by three points on one curve.
- **Answer quality.** Losslessness was assumed, so task accuracy was never
  scored. Given the 0.65% token divergence, that assumption is weaker than
  planned. LongBench-v2 is multiple choice, so this is cheap to fix.
- **Prefix caching** is on throughout, at a hit rate that differs by bucket. It
  cancels from within-bucket ratios but confounds absolute throughput across
  buckets.

---

## In one paragraph

Speculative decoding is usually benchmarked at batch one on short prompts, and
reported as a single speedup. Decomposed into coverage, acceptance, drafting
cost and verification cost, and measured across serving load and context length,
that number stops being a property of a method and becomes a property of an
operating point. On short prompts a trained drafting head beats prompt lookup by
60%; at 32K it loses to it by 30% and is slower than not speculating at all,
because its acceptance falls while its drafting cost rises with the very context
it must attend over. Retrieval-based drafting suffers neither penalty. The
practical consequence is that a serving system should choose its drafter from
its context distribution and its load, and the measurement that supports that
choice has to report coverage and drafter cost, which current practice does not.
