# A40, 8K context: the rungs move in opposite directions

Job 22685911. `COMPLETED`, elapsed 01:03:12, exit 0. Partition `gpuA40x4`.
50 prompts per arm from `longbench_v2_8K.jsonl`, 256 max new tokens, greedy,
seed 1234, concurrency grid 1/4/12, `VLLM_USE_V2_MODEL_RUNNER=0`.

This is the first cell that tests the context axis, and it produced the most
interesting result in the project so far.

## The table

| conc | rung | k | tok/s | speedup | tauC | tauE | cov | R | AL@5 | pos1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | off | 0 | 27.6 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 | | |
| 1 | dflash | 7 | 34.8 | **1.26** | 1.953 | 1.947 | 1.000 | 1.546 | 1.891 | 0.451 |
| 1 | ngram | 5 | 31.5 | 1.14 | 2.762 | 1.284 | 0.161 | 1.124 | 2.762 | 0.563 |
| 4 | off | 0 | 68.3 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 | | |
| 4 | dflash | 7 | 75.9 | 1.11 | 1.941 | 1.939 | 1.000 | 1.744 | 1.884 | 0.451 |
| 4 | ngram | 5 | 73.2 | 1.07 | 2.857 | 1.315 | 0.169 | 1.227 | 2.857 | 0.577 |
| 12 | off | 0 | 99.7 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 | | |
| 12 | dflash | 7 | 102.1 | 1.02 | 1.996 | 1.990 | 1.000 | 1.944 | 1.927 | 0.457 |
| 12 | ngram | 5 | 95.3 | 0.96 | 2.813 | 1.322 | 0.178 | 1.383 | 2.813 | 0.571 |

## Finding 7: the two rungs decay in opposite directions

This is the result.

| quantity | 2K | 8K | direction |
|---|---|---|---|
| DFlash tau (conditional) | 2.543 | 1.953 | **collapses** |
| DFlash first-position acceptance | 0.601 | 0.451 | **collapses** |
| n-gram tau (conditional) | 2.889 | 2.762 | nearly flat |
| n-gram first-position acceptance | 0.601 | 0.563 | nearly flat |
| n-gram coverage | 0.132 | 0.161 | **rises** |

Decay at a common truncation of k=5, matched on concurrency 1:

```
dflash k=7 : 2K 2.421 -> 8K 1.891    D = +0.219
ngram  k=5 : 2K 2.889 -> 8K 2.762    D = +0.044
```

Five times the decay for the trained head.

The mechanism explains the sign of each. **DFlash predicts.** Its head has to
guess the next tokens from the hidden state, and that gets harder as the
context it must summarise grows. **n-gram retrieves.** It fires only when the
recent prefix matches an earlier span in the prompt, and a longer prompt
contains more spans to match, so its hit rate goes *up*. Acceptance given a
hit barely changes, because a literal repeat is a literal repeat regardless of
how long the document is.

The clearest single number: at 2K the two rungs had **identical**
first-position acceptance, 0.601 each. At 8K n-gram is ahead, 0.563 against
0.451. The cheap retrieval rung is now the more accurate one on the first
drafted token.

## Finding 8: the useful operating window shrinks with context

Crossing points, where speedup falls through 1.0x:

| rung | 2K | 8K |
|---|---|---|
| n-gram k=5 | concurrency 14.4 | concurrency 8.9 |
| DFlash k=7 | above 32 | about 12 |

Both rungs lose roughly two to three times their usable concurrency range
going from 2K to 8K. Two effects compound. R rises, because verifying k+1
tokens costs more attention work when the context is longer (DFlash R at
concurrency 1: 1.360 at 2K, 1.546 at 8K). And the baseline saturates earlier:
27.6 to 68.3 to 99.7 tok/s is 2.47x for 4x the requests and then 1.46x for 3x,
against 5.3x for 8x at 2K. Less headroom, and a more expensive trade to make
with it.

## The gap is closing

Speedup at concurrency 1:

| bucket | DFlash k=7 | n-gram k=5 | gap |
|---|---|---|---|
| 2K | 1.86x | 1.16x | 0.70 |
| 8K | 1.26x | 1.14x | **0.12** |

Ratio of effective tau fell from 2.03 to 1.52 over the same step.

Note what is *not* driving this. The hardware did not change. The batch size
did not change. Only the context did, and the ordering of the rungs is on its
way to inverting because the two mechanisms scale differently along that axis.

## Prediction for 32K

Extrapolating the per-bucket decay crudely (D applied again over the next two
doublings, coverage continuing its trend):

| | DFlash k=7 | n-gram k=5 |
|---|---|---|
| AL@5 | ~1.48 | ~2.64 |
| coverage | 1.00 | ~0.20 |
| tau_eff | ~1.5 | ~1.33 |
| R (carrying the 8K trend) | ~1.6 | ~1.2 |
| **implied speedup at conc 1** | **~0.95x** | **~1.15x** |

**Two falsifiable predictions, both testable by the 32K cell:**

1. **n-gram overtakes DFlash at 32K.** The ordering of the rungs inverts along
   the context axis.
2. **DFlash falls below 1.0x even at concurrency 1.** The trained head stops
   paying at long context under *any* load.

The extrapolation is crude: it assumes D compounds per doubling and that R
continues its trend, neither of which is established from two points. Treat
the numbers as a direction, not a forecast. The direction is what matters, and
it is the strongest form of the ladder claim: not "the right rung depends on
conditions" but "the ranking of the rungs reverses along an axis operators
control."

## Why this matters more than the 2K result

The 2K cell showed speculation winning and then losing as load rose. True, and
useful, but unsurprising: everyone expects speculation to stop paying once the
GPU saturates.

The 8K cell shows something a reader would not predict. Two drafting methods,
benchmarked at the same acceptance on short context, diverge sharply on long
context, in opposite directions, for reasons traceable to what each mechanism
actually does. That is a claim about *which rung to pick*, not about whether
speculation helps, and picking is what the paper is for.

It also makes the coverage term from the 2K analysis load-bearing rather than
a correction. Without it n-gram's tau *rises* relative to DFlash at 8K and the
ranking looks absurd. With it, the picture is coherent: n-gram's advantage at
long context comes from firing more often, not from accepting more per firing.

## Caveats

- Single runs, no intervals, same as every other cell so far.
- Prefix caching is on. At 8K prompts share less structure than at 2K, so the
  hit rate differs between buckets and is a confound for the absolute
  throughput numbers, though not for the within-bucket speedup ratios.
- Two points do not establish a trend. The 32K cell is what turns this from an
  observation into a curve.
- The decay is measured for one drafter on one model family. Whether
  prediction-based heads generally decay faster than retrieval is a
  conjecture, not a result.
