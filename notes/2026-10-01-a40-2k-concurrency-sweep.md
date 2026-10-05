# A40, 2K context, concurrency sweep

Job 22598608. Submitted 1 Oct 2026, `COMPLETED`, elapsed 00:52:39, exit 0.
Partition `gpuA40x4`, node `gpub036`, one A40 48 GB.
50 prompts per arm from `longbench_v2_2K.jsonl`, 256 max new tokens,
greedy (temperature 0), seed 1234, `VLLM_USE_V2_MODEL_RUNNER=0`.

Full summarizer output: `results/a40/summary-2026-10-01.txt`.

This is the first measurement in the project that means anything. The run
before it put every prompt in flight at once and reported all three rungs
losing to no speculation, which measured our batching choice rather than the
rungs.

## The table

| bucket | conc | rung | k | n | tok/s | speedup | tauC | tauE | cov | R |
|---|---|---|---|---|---|---|---|---|---|---|
| 2K | auto | off | 0 | 100 | 405.1 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 2K | auto | dflash | 7 | 100 | 348.5 | 0.86 | 2.520 | 2.513 | 1.000 | 2.921 |
| 2K | auto | dflash | 15 | 100 | 316.3 | 0.78 | 2.823 | 2.801 | 1.000 | 3.587 |
| 2K | auto | ngram | 5 | 100 | 289.0 | 0.71 | 2.979 | 1.271 | 0.137 | 1.781 |
| 2K | 1 | off | 0 | 50 | 32.6 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 2K | 1 | dflash | 7 | 50 | 60.7 | **1.86** | 2.543 | 2.536 | 1.000 | 1.360 |
| 2K | 1 | ngram | 5 | 50 | 37.6 | 1.16 | 2.889 | 1.250 | 0.132 | 1.082 |
| 2K | 8 | off | 0 | 50 | 172.1 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 2K | 8 | dflash | 7 | 50 | 255.7 | **1.49** | 2.507 | 2.498 | 1.000 | 1.681 |
| 2K | 8 | ngram | 5 | 50 | 183.6 | 1.07 | 2.834 | 1.245 | 0.134 | 1.167 |
| 2K | 32 | off | 0 | 50 | 335.5 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 2K | 32 | dflash | 7 | 50 | 344.7 | **1.03** | 2.470 | 2.465 | 1.000 | 2.399 |
| 2K | 32 | ngram | 5 | 50 | 273.3 | 0.81 | 2.871 | 1.247 | 0.132 | 1.531 |

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
the verification cost ratio. This is the decomposition demonstrated on real
hardware, and it is the reason the quantities are worth separating: tau
transfers across machines, R does not.

## Finding 2: the decomposition needs a third factor, coverage

vLLM's `mean_accepted_length` is **conditional on a draft having been
proposed**. DFlash proposes on every decode step, so for it the conditional
and unconditional numbers coincide (cov = 1.000). The n-gram proposer only
fires when prompt lookup finds a matching 3- to 5-gram, and it turns out to
fire on only **13.2%** of steps. On the other 87% the engine performs a plain
single-token decode that never enters the counter.

So the working model becomes

```
tau_eff  = cov * tau_cond + (1 - cov) * 1
Speedup ~= tau_eff / R
```

Checked against the measurement at concurrency 1:

```
0.132 * 2.889 + 0.868 * 1 = 1.249     measured tau_eff = 1.250
```

This reverses the apparent paradox. With coverage accounted for, n-gram's
implied R is **1.082**, lower than DFlash's 1.360, which is exactly what the
hardware predicts: n-gram's drafter is a CPU string lookup with no model
forward, and it verifies 6 tokens per pass instead of 8. n-gram is the
cheapest rung per verification pass. It loses because it is asleep on seven
steps out of eight.

Reconstruction, since vLLM does not expose total decode steps directly. Every
step without a draft emits exactly one token, so

```
steps_without_draft = gen_tokens - drafts * tau_cond
total_steps         = drafts + steps_without_draft
cov                 = drafts / total_steps
tau_eff             = gen_tokens / total_steps
```

**This is a methodological result, not a bug fix.** Published tau figures for
lookup-based proposers are conditional in the same way, so they are not
comparable to a model drafter's tau. A ladder that compares rungs on
conditional acceptance is comparing the wrong quantity, and would rank n-gram
above DFlash on acceptance (2.889 against 2.543) while n-gram is 38% slower.

## Finding 3: the rungs cross 1.0x at different places

n-gram crosses between concurrency 8 (1.07x) and 32 (0.81x), interpolated at
about 14. DFlash is still above 1.0x at 32 (1.03x) and only loses at
saturation (0.86x). The gap is roughly 3 to 4x in serving load.

That is a more useful claim than any single speedup figure. It says the choice
of rung is a function of how loaded the server is, along an axis an operator
actually controls.

## Finding 4: the baseline knee explains the crossing

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

## Finding 5: acceptance decay shape distinguishes mechanism

Per-position acceptance at concurrency 1:

```
DFlash k=7   0.601  0.360  0.219  0.140  0.101  0.072  0.051
n-gram k=5   0.601  0.440  0.345  0.277  0.226
```

Identical at the first position, then n-gram decays far more slowly. That is
the mechanism showing through: when n-gram fires it is copying a literal
repeated span, so if the first token is right the rest usually are too. DFlash
is predicting, so confidence falls off fast. The two rungs have the same
hit rate on token one and completely different continuation behaviour.

Worth a figure. It also means the two rungs want different k, see below.

## Predictions to test next

**1. n-gram needs coverage, not acceptance.** Its ceiling at cov = 0.132 is
1.25x even with free verification. To reach DFlash's 1.86x it would need

```
cov * 2.889 + (1 - cov) = 1.86 * 1.08   ->   cov ~= 0.54
```

a four-fold increase. `prompt_lookup_min` is currently 3. Dropping it to 2
fires more often at lower acceptance, tracing a coverage-acceptance frontier.
That frontier is a cheap figure and, as far as we have seen, not published.

**2. The optimal draft length shrinks as load grows.** Fitting R's slope per
draft position, roughly 0.051 per position at concurrency 1 and 0.20 at
concurrency 32, and setting marginal acceptance equal to marginal cost:

| concurrency | marginal cost per position | optimal k |
|---|---|---|
| 1 | 0.051 | about 7 |
| 8 | 0.097 | about 5 |
| 32 | 0.200 | about 3 |

The k=15 row already confirms the direction: at saturation it bought 0.29
effective tokens and cost 0.67 in R, positions 8 through 15 contributing
0.156 tokens between them. Add a k=3 arm at concurrency 32 to test it.

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
- **R here is implied**, computed as tau_eff / speedup, not measured
  independently. Label it as such.
- **k is not matched across rungs.** DFlash ran at 7, n-gram at 5. Acceptance
  is only ever compared at a common truncation, which is what `AL@5` is for.

## Cost

52 minutes wall on one A40. The reported balance did not visibly move from 8
hours, so either A40 billing is coarse or the sub-allocation accounting lags.
Worth one more observation before relying on it for budget planning.
