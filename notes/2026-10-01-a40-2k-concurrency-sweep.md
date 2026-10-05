# A40, 2K context, concurrency sweep

Job 22598608. Submitted 1 Oct 2026, `COMPLETED`, elapsed 00:52:39, exit 0.
Partition `gpuA40x4`, node `gpub036`, one A40 48 GB.
50 prompts per arm from `longbench_v2_2K.jsonl`, 256 max new tokens,
greedy (temperature 0), seed 1234, `VLLM_USE_V2_MODEL_RUNNER=0`.

This is the first measurement in the project that means anything. The run
before it put every prompt in flight at once and reported all three rungs
losing to no speculation, which measured our batching choice rather than the
rungs.

## The table

| bucket | conc | rung | k | n | tok/s | speedup | tau | AL@5 | pos1 |
|---|---|---|---|---|---|---|---|---|---|
| 2K | auto | off | 0 | 100 | 405.1 | 1.00 | 1.000 | | |
| 2K | auto | dflash | 7 | 100 | 348.5 | 0.86 | 2.520 | 2.407 | 0.601 |
| 2K | auto | dflash | 15 | 100 | 316.3 | 0.78 | 2.823 | 2.545 | 0.656 |
| 2K | auto | ngram | 5 | 100 | 289.0 | 0.71 | 2.979 | 2.979 | 0.608 |
| 2K | 1 | off | 0 | 50 | 32.6 | 1.00 | 1.000 | | |
| 2K | 1 | dflash | 7 | 50 | 60.7 | **1.86** | 2.543 | 2.421 | 0.601 |
| 2K | 1 | ngram | 5 | 50 | 37.6 | 1.16 | 2.889 | 2.889 | 0.601 |
| 2K | 8 | off | 0 | 50 | 172.1 | 1.00 | 1.000 | | |
| 2K | 8 | dflash | 7 | 50 | 255.7 | **1.49** | 2.507 | 2.393 | 0.600 |
| 2K | 8 | ngram | 5 | 50 | 183.6 | 1.07 | 2.834 | 2.834 | 0.601 |
| 2K | 32 | off | 0 | 50 | 335.5 | 1.00 | 1.000 | | |
| 2K | 32 | dflash | 7 | 50 | 344.7 | **1.03** | 2.470 | 2.369 | 0.597 |
| 2K | 32 | ngram | 5 | 50 | 273.3 | 0.81 | 2.871 | 2.871 | 0.605 |

`conc = auto` is the earlier uncontrolled run, kept deliberately. With 100
prompts handed to vLLM at once it batches as wide as memory allows, which on
this node is roughly 70 concurrent at 2K. It is the right-hand end of the
concurrency axis, not a mistake.

## Finding 1: tau is invariant, speedup is not

DFlash mean accepted length across concurrency 1, 8, 32 and saturation:

```
2.543   2.507   2.470   2.520
```

First-position acceptance across the same four:

```
0.601   0.600   0.597   0.601
```

The full acceptance-by-position vectors are near identical too. Meanwhile
speedup goes 1.86, 1.49, 1.03, 0.86.

So acceptance is a property of the drafter and the text, and is unaffected by
how many requests share the GPU. Every bit of the speedup variation lives in
the verification cost ratio. This is `Speedup ~= tau / R` demonstrated on real
hardware, and it is the reason the two quantities are worth separating at all:
tau transfers across machines, R does not.

Implied R, computed as tau / speedup:

| concurrency | off tok/s | DFlash R | n-gram R |
|---|---|---|---|
| 1 | 32.6 | 1.37 | 2.49 |
| 8 | 172.1 | 1.68 | 2.65 |
| 32 | 335.5 | 2.40 | 3.54 |
| saturated | 405.1 | 2.93 | 4.20 |

## Finding 2: the rungs cross 1.0x at different places

n-gram crosses between concurrency 8 (1.07x) and 32 (0.81x). DFlash is still
above 1.0x at 32 (1.03x) and only loses at saturation (0.86x). The gap is
roughly 3 to 4x in serving load.

That is a more useful claim than any single speedup figure. It says the choice
of rung is a function of how loaded the server is, and it gives the ladder an
ordering along an axis an operator actually controls.

## Finding 3: the baseline knee explains the crossing

No-speculation throughput across the concurrency axis:

```
c=1   32.6 tok/s
c=8  172.1 tok/s   5.3x for 8x the requests
c=32 335.5 tok/s   1.95x for 4x the requests
sat  405.1 tok/s   1.21x for roughly 2x the requests
```

The GPU saturates between 8 and 32, and that is exactly where speculation
stops paying. Speculation spends spare compute to save memory traffic, so its
benefit is bounded by the slack between achieved throughput and the roofline.
Once the slack is gone the drafted tokens are pure overhead. The baseline
column and the speedup column are two views of the same mechanism.

For context, 32.6 tok/s at concurrency 1 is about 78% of the bandwidth-bound
ceiling for a 16.4 GB bf16 model on an A40's 696 GB/s, so the serial baseline
is healthy and not an artifact of a badly configured engine.

## Open question: n-gram's tau is not what it delivers

n-gram reports higher tau than DFlash (2.889 against 2.543) and verifies fewer
tokens per step (k=5 against k=7), yet it is 38% slower at concurrency 1. Its
implied R of 2.49 against DFlash's 1.37 is not physically sensible: fewer
verified tokens cannot cost more.

The likely explanation is that vLLM's `mean_accepted_length` is **conditional
on a draft having been proposed**. The n-gram proposer only fires when prompt
lookup finds a matching 3- to 5-gram; on every other step the engine does a
plain single-token decode that never enters the counter. If coverage is
partial, the headline tau flatters a rung that is idle much of the time.

Working backwards from the measured throughput, and assuming n-gram's true R
is no worse than DFlash's, coverage should be near 27%. That is a falsifiable
prediction and the `drafts` counter already in `results.jsonl` settles it.

`summarize.py` now reconstructs it. Every step without a draft emits exactly
one token, so

```
steps_without_draft = gen_tokens - drafts * tau_conditional
total_steps         = drafts + steps_without_draft
coverage            = drafts / total_steps
tau_effective       = gen_tokens / total_steps
```

`tau_effective` is the quantity that belongs in `Speedup ~= tau / R`.
`tau_conditional` describes the drafter when it fires. Reporting the second
and calling it the first would overstate n-gram throughout the paper.

## Caveats to carry into the writeup

- **Single runs, no intervals.** Nothing here is a claim yet. The milestone
  needs a paired bootstrap over prompts.
- **Prefix caching is on**, hit rate 21 to 24%. Identical in every arm, so it
  cancels out of the speedup ratio, but it inflates absolute tok/s and will
  interact differently with longer buckets, where prompts share less
  structure. Decide before the 8K runs whether to disable it for the headline
  table or report it as configuration.
- **Losslessness is not yet verified.** The per-request token-id hashes are
  being written but have not been compared against the no-speculation arm.
  That comparison is free and should be done before any speedup is published.
- **k is not matched across rungs.** DFlash ran at 7, n-gram at 5. Acceptance
  is only ever compared at a common truncation, which is what the `AL@5`
  column is for.

## Cost

52 minutes wall on one A40. The reported balance did not visibly move from 8
hours, so either A40 billing is coarse or the sub-allocation accounting lags.
Worth one more observation before relying on it for budget planning.
