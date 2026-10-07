# The k hypothesis was wrong, and the reason is more interesting

Job 22714602. `COMPLETED`, 01:07:08, six arms, zero failures. 32K bucket,
concurrency 1, 25 prompts, seed 1234, A40.

## The prediction, and the refutation

The 32K note argued that DFlash was not failing at long context but running at
the wrong draft length. Its acceptance by position was `0.296, 0.087, 0.038,
...`, dead after position two, so drafting seven tokens looked wasteful. Fitting
`R = 1 + k*c` against the single measured point `R = 1.766 at k = 7` gave
`c = 0.109` per position, which exceeds the marginal acceptance at position 2.
Conclusion: optimal k is 1, predicted speedup about **1.17x**.

Measured:

| k | tok/s | speedup | tauC | R | acceptance by position |
|---|---|---|---|---|---|
| 1 | 11.3 | **0.76** | 1.278 | 1.677 | 0.278 |
| 2 | 11.6 | 0.78 | 1.336 | 1.704 | 0.265, 0.072 |
| 4 | 12.0 | 0.81 | 1.415 | 1.743 | 0.277, 0.084, 0.036, 0.018 |
| 7 | 12.2 | **0.83** | 1.459 | 1.766 | 0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004 |

**k=1 is the worst setting in the sweep**, and speedup rises monotonically with
k. The prediction was wrong by a wide margin and in the opposite direction.

## Why the model failed

`R = 1 + k*c` assumes speculation's cost is proportional to draft length, with
no fixed component. Fitting the four measured points instead:

```
R(k) ~= 1.662 + 0.0148 * k
```

| | assumed | measured |
|---|---|---|
| marginal cost per draft position | 0.109 | **0.0148** |
| fixed cost of speculating at all | 0 (intercept forced to 1.0) | **0.662** |

The two were backwards by roughly a factor of seven in one direction and
infinity in the other. Drafting additional tokens is nearly free. Running the
drafter **once** costs two-thirds of a target decode step before a single token
is verified.

The methodological error is worth naming: a one-parameter model was fitted to a
single data point by *assuming* the intercept. With two unknowns and one
observation, the answer was determined by the assumption rather than the data.
The sweep that tested it cost one hour of A40 time, which is the cheapest way
this could have been found out.

## What replaces it: drafter cost scales with context

DFlash's draft head attends over the full context. At 32K that attention is
expensive, so the drafter's cost grows with context **while its accuracy falls**.

| | DFlash | n-gram |
|---|---|---|
| drafter mechanism | trained head, attends over full context | prompt lookup, CPU string match |
| drafter cost at 32K | ~0.66 of a decode step | ~0 |
| R at 32K, conc 1 | 1.677 even at k=1 | 1.290 |
| scaling in context | **grows** | **constant** |
| first-position acceptance, 2K -> 32K | 0.601 -> 0.296 | 0.601 -> 0.552 |

n-gram survives long context for two independent reasons, not one. Its
acceptance holds up, which the 8K and 32K notes already established. And its
drafting cost does not scale with context at all, which this sweep establishes.
DFlash is penalised on both axes simultaneously.

## Finding 10: tau / R needs a drafter-cost term

Drafter cost is currently buried inside R, where it is indistinguishable from
verification cost. It should not be, because the two behave differently:

- **verification cost** scales with k, the number of tokens checked per pass
- **drafter cost** is fixed per decode step, independent of k, and for
  model-based drafters scales with context

```
R(k) = R_draft + R_verify(k)
```

At 2K with k=7, DFlash's total R is 1.360, so whatever `R_draft` is there, it
cannot exceed 0.36. At 32K it is 0.662 measured directly. The term roughly
doubles across the context range while acceptance halves.

This is invisible at short context, which is where essentially all published
speculative decoding results are reported, and dominant at long context. It is
also a reason to prefer drafters whose cost is context-independent when serving
long contexts, regardless of their acceptance.

**Testable next:** run the same k sweep at 2K and 8K to get `R_draft` per
bucket. Prediction: it grows with context, roughly with the attention cost of
the draft head. Three arms per bucket at concurrency 1, cheap.

## Second result: the noise floor is under 1%

The `off` and `ngram` arms were re-run deliberately, to give the k arms a
baseline from the same engine session. They also replicate job 22713173:

| arm | job 22713173 | job 22714602 | difference |
|---|---|---|---|
| off | 14.9 tok/s | 14.8 tok/s | 0.7% |
| ngram k=5 | 16.1 tok/s | 16.0 tok/s | 0.6% |
| dflash k=7 | 12.3 tok/s | 12.2 tok/s | 0.8% |

Different jobs, different nodes, hours apart. **Run-to-run variance is under
1%.**

This matters more than it looks. Per-request timing came back `0/25` on this
vLLM build, so the planned paired bootstrap is impossible and repeated runs are
the only route to intervals. This is the first bound on noise, and it sits far
below the effects being claimed:

| claim | margin | noise |
|---|---|---|
| 8K dflash c=12 at 1.02x | 2% | <1% |
| 8K ngram c=12 at 0.96x | 4% | <1% |
| 32K ngram c=4 at 0.99x | 1% | <1% |
| 32K inversion, 1.08x vs 0.83x | 30% | <1% |

The headline inversion is 30 times the noise floor. The marginal rows are
separable but close, and the 0.99x row is within about one noise width of 1.0,
so it should be reported as indistinguishable rather than as below.

Two further repeats would give a proper interval. One repeat already shows the
variance is not the problem.

## Caveats

- The linear fit to `R(k)` uses four points over k = 1 to 7 and is a
  description, not a mechanism. The intercept is an extrapolation to k=0, which
  is not a configuration that can be run.
- `R_draft` at 2K and 8K is bounded but not measured. The claim that drafter
  cost grows with context rests on one measured value and one upper bound.
- Single bucket, single concurrency, one drafter.
- The replication covers three arms in one bucket at one concurrency. It bounds
  noise there, not everywhere.
