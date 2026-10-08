# Drafter cost scales with context; verification cost does not

Jobs 22737716 (2K) and 22737717 (8K), completing the k sweep begun at 32K in
job 22714602. All at concurrency 1 on A40, 50 prompts at 2K and 8K, 25 at 32K.

This measures the term that `tau / R` was hiding, and it supplies the
mechanism behind the context inversion.

## The measurement

Fitting `R(k) = R_draft + c * k` in each bucket:

| bucket | R at k=1 | R at k=2 | R at k=7 | **R_draft** | **c**, per position |
|---|---|---|---|---|---|
| 2K | 1.305 | 1.322 | 1.360 | **0.296** | 0.0092 |
| 8K | 1.450 | 1.487 | 1.544 | **0.434** | 0.0157 |
| 32K | 1.677 | 1.704 | 1.766 | **0.662** | 0.0148 |

(32K also has k=4 at R = 1.743, consistent with the fit.)

**Drafter cost grows 2.24x from 2K to 32K. Marginal verification cost per
drafted token is flat**, between 0.009 and 0.016 with no trend.

The two components of R behave completely differently. That is what makes the
split worth having rather than a bookkeeping nicety.

## How lopsided it is

Share of DFlash's total overhead attributable to drafting, at k=7:

| bucket | total R | overhead (R - 1) | of which drafting | share |
|---|---|---|---|---|
| 2K | 1.360 | 0.360 | 0.296 | **82%** |
| 8K | 1.544 | 0.544 | 0.434 | **80%** |
| 32K | 1.766 | 0.766 | 0.662 | **86%** |

**Verification is nearly free. The drafter is almost the entire cost.**

This reframes the whole result. The intuitive story about speculative decoding
is that you gamble on cheap guesses and pay to verify them in bulk. On these
measurements the verification is the cheap part by a wide margin, and
producing the guesses is what costs. A trained head that must attend over the
full context pays that cost in proportion to the context; a retrieval drafter
pays nothing at all. That, and not acceptance decay alone, is why n-gram
overtakes DFlash at 32K.

## How R_draft scales

Neither linear in tokens nor linear in log tokens fits cleanly:

| model | predicts 32K from 2K and 8K | measured |
|---|---|---|
| linear in context length | 0.986 | 0.662 |
| linear in log2(context) | 0.572 | 0.662 |

The truth sits between, which is unsurprising for a head whose cost is part
fixed and part attention over the context. Three points cannot distinguish the
forms. **Report the measured values; do not fit a law to three points.**

## Optimal k rises with context. An earlier claim is retracted.

Speedup against k at concurrency 1:

```
2K    k=1 1.20x   k=2 1.47x   k=7 1.86x
8K    k=1 0.96x   k=2 1.10x   k=7 1.26x
32K   k=1 0.76x   k=2 0.78x   k=4 0.81x   k=7 0.83x
```

Monotonically increasing in every bucket. At 8K, k=1 is already **below 1.0x**
while k=7 reaches 1.26x.

The reason follows directly from the table above. Once the fixed drafter cost
is paid, each additional drafted token costs about 0.01 to 0.016 and returns
the acceptance at that position, which at early positions is far larger. So
draft more, not less.

**`notes/2026-10-06-32k-inversion.md` claimed the optimal k shrinks with
context and that "context does to k what load does to k". Both are wrong.**
What the 2K data actually shows is that optimal k shrinks with **load**: at
saturation, k=15 lost to k=7 (0.78x against 0.86x), because concurrency raises
the marginal cost per drafted position while context raises only the fixed
cost. Context and load are not interchangeable.

This is the second prediction in this project to fail by assuming a cost model
rather than measuring one, and both failures came from the same habit.

## Footnote: per-position acceptance is not comparable across k

First-position acceptance depends on k:

| bucket | k=1 | k=2 | k=7 |
|---|---|---|---|
| 2K | 0.570 | 0.589 | 0.601 |
| 8K | 0.395 | 0.426 | 0.451 |
| 32K | 0.278 | 0.265 | 0.296 |

The drafter's first guess should not depend on how many guesses it was asked
for. The likely explanation is sampling, not a bug: longer accepted runs change
where in the text subsequent drafting happens, so different k values visit
different distributions of context. The effect is real and must be noted
wherever per-position curves are compared across k.

## What this does to the decay numbers

Decay D at a common truncation of k=5, concurrency 1, now available at three
draft lengths:

```
dflash k=1: 2K 1.570 -> 8K 1.395 (D 0.112) -> 32K 1.278 (D 0.186)
dflash k=2: 2K 1.942 -> 8K 1.631 (D 0.160) -> 32K 1.336 (D 0.312)
dflash k=7: 2K 2.421 -> 8K 1.891 (D 0.219) -> 32K 1.448 (D 0.402)
ngram  k=5: 2K 2.889 -> 8K 2.762 (D 0.044) -> 32K 2.844 (D 0.016)
```

Decay is itself larger at larger k, which makes sense: later draft positions
are the ones that fail first as context grows, so a longer draft has more to
lose. n-gram remains flat at every k tested.

## Open

- **n-gram's R_draft is not measured.** Only k=5 was run, so its R cannot be
  decomposed. A k sweep for n-gram would confirm the expectation that its
  R_draft is near zero and context-independent, which is the load-bearing
  half of the comparison and currently rests on reasoning rather than data.
- Three points cannot identify how R_draft scales. More buckets, or a direct
  measurement of drafter forward time, would.
- All of this is one drafter on one model at one concurrency.
