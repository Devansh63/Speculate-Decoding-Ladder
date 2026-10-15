# What the data says

The argument as a whole, rather than per experiment. Each section names the
notes and jobs behind it.

All measurements: Qwen3-8B, LongBench-v2, greedy decoding, vLLM pinned to the
stable model runner. A40 jobs 22598608 (2K), 22685911 (8K), 22713173 (32K),
22714602 (k sweep and replication). A100 jobs 22757104 (2K, n=5), 22802261 (8K)
and 22802262 (32K). Determinism control: A40 jobs 22875382 and 22876140 under
harness `v3-2026-10-13`.

Updated 13 Oct 2026. The hardware axis is no longer empty, the losslessness
section below is rewritten, and two of the five "cannot claim" items have been
closed.

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

## Losslessness, in two halves

Speculative decoding is supposed to return exactly what the target model would
have returned. Measured, this build fails that at the bit level and passes it
at the task level, and the two halves point in opposite directions.

**At the bit level it is not lossless.** First the floor: three realizations of
the no-speculation arm at one configuration, across two nodes, are **150 of 150
sequences bit-identical**. Zero events in 12,800 token positions bounds the
true per-token hazard at about **0.023%** (rule of three, 95%). Against that
floor the speculative arms diverge at **0.556%** per token for n-gram k=5 and
**0.781%** for DFlash — 24 to 34 times the bound, and systematic in k
(0.738% at k=1 rising to 0.781% at k=2). Verifying k+1 positions runs different
kernel shapes from decoding one, so near-ties resolve differently. The
divergence is real, measurable and attributable to speculation (C21).

**At the task level it is lossless.** LongBench-v2 is multiple choice, so the
stored tokens can be scored against the answer key. At 8K **nine arms score
34.7%, identically**. Every difference at 2K and 32K is exactly one prompt —
and the baseline differs from itself by exactly one prompt across concurrency.
No rung deviates by more than the baseline's own variation, over 503 graded
questions (C22).

**This replaces the framing this section used to carry**, which was built on a
control comparing the baseline against itself at *different* concurrency: 12.7%
sequence agreement against 19.0% for speculative arms, read as the engine
perturbing itself more than speculation does. At matched concurrency the engine
does not disagree with itself at all, so **the entire control effect was
concurrency**, and the sentence this section ended on — that speculation does
not perturb the greedy path beyond the serving system's reproducibility floor —
is **withdrawn**. The floor is zero and speculation sits 24 to 34 times above
it.

What replaces it is sharper than what it replaces. Sequence identity, the
metric the entire losslessness argument was conducted in, **does not predict
task outcome**: speculation changes tokens at 24-34x the floor and changes no
answer. A paper claiming byte-exact equality on a vLLM-class server either
disabled batching or did not check. A paper claiming task-level equivalence can
support it — and should say which of the two it means, because they are not the
same claim and this data separates them.

**The control still missing.** No speculative arm has been run twice at one
configuration, so some part of the 0.556-0.781% could be that arm's own
run-to-run noise rather than a systematic offset. Until that job runs, the
supported statements are "the engine is deterministic" and "speculation
diverges from the baseline", but not yet "speculation diverges systematically"
(threats, 12).

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

- **Hardware portability — partly answered, and not in the direction planned.**
  The multi-architecture rebuild landed on 8 Oct, so A100 works and
  `results/a100` holds 15 rows; H200 is blocked only by `phase0.sh`'s
  `80|86` architecture guard. What the second platform showed is that the
  original framing was wrong on both halves: **tau does not transfer** (21 of 25
  sequences differ between machines at the same cell, even though the means
  agree to 0.8%), and **R does not scale by one factor** (the rungs differ at
  every bucket once acceptance is separated out). The hardware term is a result
  rather than a side quest, but it is **one platform pair**, the 2K cell is
  n=5, and a third point is needed before any of it is a trend.
- **Confidence intervals.** Still none, and the reason has changed twice.
  Per-request timing is not merely unavailable but **structurally absent**: the
  harness asks the engine for `first_token_time`, `last_token_time` and
  `finished_time`, this build populates none of them, so the planned paired
  bootstrap cannot be computed at all. Nor can repeats supply one — at fixed
  configuration the engine is bit-deterministic, so repeated whole runs measure
  zero. The only unmeasured variance left is **prompt sampling**, and the route
  to an interval is disjoint prompt subsets by document id.
- **Generality.** One target model, one drafter family, one workload. Whether
  prediction-based drafters generally decay faster than retrieval-based ones is a
  conjecture supported by three points on one curve.
- **Answer quality — closed.** Scored on 13 Oct: no rung's accuracy differs
  from the baseline by more than the baseline's own variation across
  concurrency, nine arms identical at 8K, every other difference exactly one
  prompt (C22). The caveat that remains is **n**: 24 to 49 prompts per bucket,
  where one flip moves 32K accuracy by 4.2 points, so the invariance has no
  interval and should not be quoted with one.
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
