# n-gram's drafter cost is not zero, and it grows with context

> **Partly superseded, 11 Oct 2026 (late).** The closing claim that optimal k
> "rises with context and shrinks with load" is half wrong: the load axis has
> no valid data at all, since its only source is the disqualified uncontrolled
> run 22569403. And the statement below that the inversion "is driven by the
> absolute gap, not the scaling" is withdrawn — a counterfactual on the
> measured numbers shows cost alone does not flip the ranking. **The R_draft
> measurements in this note are unaffected.** Register, C6 and C13.

**Jobs** 22757472 (32K, 01:01:22) and 22757473 (2K, 00:58:58), both A40,
concurrency 1, `KLIST="1 2 7"`, seed 1234, `vllm_env_multiarch`.
**Script** `kprobe_ngram.sh`, a variant of `kprobe_32k.sh` with the n-gram
call moved inside the k loop. The original runs n-gram once at k=5, which is
one point and cannot identify an intercept.
**Cost** 2.0 GPU-hours.

## Why this run existed

The three-term model says

```
Speedup ~= [cov * tau_cond + (1 - cov)] / R,    R = R_draft + c * k
```

and `notes/2026-10-08-drafter-cost-scaling.md` measured `R_draft` for DFlash
at every bucket: 0.296, 0.434, 0.662 for 2K, 8K, 32K. For n-gram it was never
measured. We *assumed* it was near zero on the reasoning that prompt lookup
runs no model forward pass, and that assumption was load-bearing in the
explanation of the inversion. An unmeasured assumption doing load-bearing
work is exactly the shape of the k=1 mistake
(`notes/2026-10-07-k-sweep-refutation.md`), so it was worth two GPU-hours to
test rather than to keep asserting.

## Measurement

Implied R, computed as `tau_eff / speedup`:

| bucket | k=1 | k=2 | k=7 |
|---|---|---|---|
| 2K | 1.074 | 1.077 | 1.092 |
| 32K | 1.156 | 1.141 | 1.189 |

Least-squares fit of `R(k) = (1 + R_draft) + c*k`:

| bucket | R_draft | c | fit quality |
|---|---|---|---|
| 2K | **0.071** | 0.0030 | monotonic, clean |
| 32K | **0.139** | 0.0068 | **non-monotonic, not resolved** |

## The assumption was wrong, but the conclusion it supported survives

n-gram's drafter cost is **not** zero. It is small, but it is real, and it
**grows with context at a rate close to DFlash's**:

| | 2K | 32K | growth |
|---|---|---|---|
| DFlash `R_draft` | 0.296 | 0.662 | 2.24x |
| n-gram `R_draft` | 0.071 | 0.139 | 1.96x |
| absolute gap | 0.225 | 0.523 | 2.32x |
| ratio | 4.2x | 4.8x | — |

So "context taxes the drafter" is **not** specific to model-based drafters.
Both drafters pay a context tax, at similar relative rates. This makes sense
once stated plainly: prompt lookup has to *search the prompt*, and a 32K
prompt is sixteen times more text to scan than a 2K one. One drafter's
context cost is a forward pass, the other's is a string search. The
difference is a constant factor of four to five, not a difference in kind.

**What drives the inversion is therefore the absolute gap, not the scaling.**
DFlash's drafting penalty exceeds n-gram's by 0.225 at 2K and by 0.523 at
32K. That doubling of the absolute penalty is what flips the ranking, and it
would still flip it even if the two costs grew at identical relative rates —
because DFlash starts from a base four times larger.

Note also that neither cost scales anything like linearly with context: a
16-fold increase in prompt length roughly doubles both. Sublinear in both
cases, by a similar factor, and we have no account of why. Three buckets
cannot distinguish log-like growth from a small power.

## The model predicts a point it was not fitted to

The 2K fit gives `R(5) = 1.071 + 0.0030*5 = 1.086`. The main results hold
n-gram at k=5 at 2K concurrency 1 from job 22598608, a different job on a
different day: **R = 1.083**. Off by 0.3 percent.

This is the first prediction this project has made that was not fitted to the
data it was tested against, and it is the direct counterpart to the k=1
failure, where an intercept was *assumed* rather than fitted and the
prediction missed by 50 percent. Fit the intercept, test out of sample.

The same test at 32K **fails**: the fit predicts `R(5) = 1.173` and the main
results measure 1.290, a 10 percent miss. That is not a failure of the model
but of the noise floor at that bucket — see
`notes/2026-10-11-noise-floor.md`, which came out of this job's re-measured
baseline. The 32K k coefficient above should be treated as unresolved.

## n-gram's acceptance is context-invariant

This was not the question the job asked, and it is the cleanest number in it:

| | 2K | 32K |
|---|---|---|
| tau_cond at k=1 | 1.670 | 1.670 |
| first-position acceptance, k=1 | 0.670 | 0.670 |
| AL@5, k=7 | 2.684 | 2.703 |

Decay `D = 1 - AL(long)/AL(2K)` is **-0.000, -0.008, -0.007** at k = 1, 2, 7.
Zero within noise, and slightly negative. Against DFlash's D of 0.219 at 8K
and 0.402 at 32K, n-gram's acceptance simply does not care how long the
prompt is.

Meanwhile coverage **rises** with context, 0.121 to 0.195 at k=7 — a longer
prompt offers more places for a lookup to hit. Flat acceptance plus rising
coverage plus a small drafting cost gives the result that n-gram is **faster
at 32K than at 2K**: 1.18x against 1.14x at k=7. The two rungs do not merely
converge as context grows; they move in opposite directions.

The acceptance-by-position profiles at 32K show why DFlash cannot compete
there:

```
dflash k=7   0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004
ngram  k=7   0.541, 0.382, 0.304, 0.256, 0.219, 0.193, 0.177
```

DFlash's drafts are worthless past position 2. n-gram's are still worth
having at position 7. So DFlash at long context pays four times more to draft
*and* gets drafts that collapse faster.

## Optimal k rises with context for n-gram too

Speedup by k, concurrency 1:

| bucket | k=1 | k=2 | k=7 |
|---|---|---|---|
| 2K | 1.06x | 1.09x | 1.14x |
| 32K | 1.04x | 1.13x | 1.18x |

Monotonic in k at both buckets, with no sign of a turn by k=7. Consistent
with the correction recorded on 8 Oct: **optimal k rises with context and
shrinks with load.** The two axes act in opposite directions, and the earlier
claim that they behaved alike is withdrawn.

## What this changes

1. The sentence "n-gram drafts for free" must not appear in the paper. It
   drafts for about a quarter of what DFlash pays, and its cost is
   context-dependent.
2. The mechanism behind the inversion is now measured on **both** rungs
   rather than measured on one and assumed on the other.
3. The explanation sharpens: it is not that one drafter has a context cost
   and the other does not. It is that a forward pass over a long prefix costs
   four to five times what scanning that prefix costs, and that multiple is
   what the ranking turns on.

## Open

- **The 32K k coefficient is unresolved.** The three points are
  non-monotonic. This needs repeats, not a longer k list.
- **Why both costs are sublinear in context** is unexplained. A 16x prompt
  gives a 2x drafter cost in both rungs. A fourth bucket would help separate
  candidate shapes.
- **n-gram at k > 7 has never been tried.** Speedup is still rising at k=7 at
  both buckets, so the optimum has not been bracketed from above at either.
  The only k=15 data is DFlash at 2K under saturation.
- **8K was not swept for n-gram**, so the middle point of the R_draft curve
  is missing for this rung.
