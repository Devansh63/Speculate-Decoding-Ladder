# The Speculation Ladder: project brief for external review

CS 598, Hardware and Software for AI, Fall 2026. Two students: Devansh Agrawal,
Sahil Sashi. Target output is a paper-level course project.

Repo: https://github.com/Devansh63/Speculate-Decoding-Ladder

This document is self-contained. It states the research question, the metric
framework, what has actually been measured, what the measurements showed, and
where the work is weak. It ends with specific questions for a reviewer.

Nothing here is padded. Where a number is uncertain or a claim is unsupported,
it says so.

---

## 1. The question

Speculative decoding makes a small, fast drafter propose k tokens, which the
large target model verifies in a single forward pass, keeping the longest
correct prefix. Output is supposed to be identical to what the target would
have produced alone.

There are now many drafting methods: n-gram prompt lookup, self-speculation,
trained heads such as EAGLE-3 and DFlash, and full small-model drafters. The
published speedups are not comparable to each other. Each is reported on its
own hardware, at its own batch size, on its own workload, usually at batch 1.

The project treats these methods as **rungs on a ladder** and asks which rung
is correct under which operating conditions, where conditions are:

- **serving load**, the number of concurrent requests
- **context length**, 2K to 32K tokens
- **hardware**, which fixes the compute-to-bandwidth ratio

The claim we want to support is that the right rung is a function of operating
point, not a global ranking, and that the choice can be predicted rather than
benchmarked for every new machine.

## 2. The metric framework

A single speedup number bundles together two unrelated things: how good the
drafter's guesses are, and how expensive verification is on this machine at
this load. Separating them is the point of the project.

- **tau**, mean accepted length. How many tokens come out of one verification
  pass. A property of the drafter and the text. Should not depend on hardware
  or batch size.
- **R**, verification cost ratio. What one verification pass costs relative to
  one ordinary decode step. A property of the machine and the load. Should not
  depend on the drafter's quality.

```
Speedup ~= tau / R
```

If this holds, tau measured once on cheap hardware predicts behaviour on
expensive hardware you do not have, provided R can be computed from the
hardware's roofline, the batch size and k.

**We found this decomposition is incomplete** and added a third term. See
Finding 2.

## 3. Setup

| | |
|---|---|
| Target model | Qwen3-8B, bf16, 36 layers, 8 KV heads, head dim 128 |
| Drafter | `z-lab/Qwen3-8B-DFlash-b16` (trained head), plus n-gram prompt lookup |
| Serving engine | vLLM `0.29.1rc1.dev345+gbf13ecc2f.cu132`, built from source on the cluster |
| Workload | LongBench-v2, bucketed at 2K, 8K, 32K context |
| Decoding | greedy, temperature 0, 256 max new tokens, fixed seed |
| Cluster | NCSA Delta |
| Hardware | A40 48 GB (ridge point 107), A100 40 GB (ridge point 201) |

**Rungs measured so far:** `off` (no speculation, the baseline), `ngram`
(prompt lookup, k=5), `dflash` (trained head, k=7 and k=15).

### Constraints that shaped the design

- **The vLLM build carries CUDA kernels for sm_80 and sm_86 only, with no
  embedded PTX**, verified with `cuobjdump`. A40 and A100 run. H200 (sm_90)
  cannot run at all, not slowly, and is deferred rather than rebuilt, because
  rebuilding would replace the binary that produced every existing
  measurement.
- **Both rungs are pinned to the stable model runner** via
  `VLLM_USE_V2_MODEL_RUNNER=0`. Without the pin, n-gram silently falls back to
  the stable runner while DFlash stays on the experimental one, and the two
  rungs are then being measured on different execution paths. The harness
  refuses to start without the pin.
- **The concurrency grid shrinks with context, and has to.** Qwen3-8B holds
  about 144 KiB of KV per token; measured cache on A40 is 160,912 tokens, so a
  32K sequence is about a fifth of it. Reachable concurrency is roughly 70 at
  2K, 19 at 8K, 5 at 32K. Asking for more does not fail: vLLM accepts
  `max_num_seqs=32` and then schedules fewer, so the row gets labelled with a
  concurrency it never ran at.
- **Budget is tight.** 10 GPU-hours per student on the current sub-allocation,
  roughly 8 remaining. More is available on request. A100 bills at about twice
  the A40 rate.

## 4. What has been measured

One complete cell: **A40, 2K context, three rungs, four levels of concurrency.**
50 prompts per arm (100 for the earliest run), same prompts, same seed.

| conc | rung | k | tok/s | speedup | tau_cond | tau_eff | coverage | R |
|---|---|---|---|---|---|---|---|---|
| 1 | off | 0 | 32.6 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 1 | dflash | 7 | 60.7 | **1.86** | 2.543 | 2.536 | 1.000 | 1.360 |
| 1 | ngram | 5 | 37.6 | 1.16 | 2.889 | 1.250 | 0.132 | 1.082 |
| 8 | off | 0 | 172.1 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 8 | dflash | 7 | 255.7 | **1.49** | 2.507 | 2.498 | 1.000 | 1.681 |
| 8 | ngram | 5 | 183.6 | 1.07 | 2.834 | 1.245 | 0.134 | 1.167 |
| 32 | off | 0 | 335.5 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 32 | dflash | 7 | 344.7 | **1.03** | 2.470 | 2.465 | 1.000 | 2.399 |
| 32 | ngram | 5 | 273.3 | 0.81 | 2.871 | 1.247 | 0.132 | 1.531 |
| sat | off | 0 | 405.1 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| sat | dflash | 7 | 348.5 | 0.86 | 2.520 | 2.513 | 1.000 | 2.921 |
| sat | dflash | 15 | 316.3 | 0.78 | 2.823 | 2.801 | 1.000 | 3.587 |
| sat | ngram | 5 | 289.0 | 0.71 | 2.979 | 1.271 | 0.137 | 1.781 |

`sat` is an uncontrolled run where all prompts were submitted at once and vLLM
batched as wide as memory allowed, roughly 70 concurrent. Kept deliberately as
the right-hand end of the load axis.

`R` here is **implied**, computed as `tau_eff / speedup`, not measured
independently. This is a known weakness.

Acceptance by draft position, concurrency 1:

```
dflash k=7   0.601  0.360  0.219  0.140  0.101  0.072  0.051
ngram  k=5   0.601  0.440  0.345  0.277  0.226
dflash k=15  0.656  0.401  0.240  0.148  0.100  0.069  0.052  0.039
                    0.031  0.024  0.019  0.016  0.011  0.009  0.007
```

## 5. Findings

### Finding 1: tau is invariant to load, speedup is not

DFlash mean accepted length across concurrency 1, 8, 32 and saturation:
2.543, 2.507, 2.470, 2.520. First-position acceptance: 0.601, 0.600, 0.597,
0.601. Meanwhile speedup goes 1.86, 1.49, 1.03, 0.86.

All of the variation in speedup lives in R. This is the decomposition
demonstrated rather than asserted, and it is the basis for the portability
claim.

### Finding 2: the decomposition needs a third term, coverage

vLLM's reported mean accepted length is **conditional on a draft having been
proposed**. DFlash proposes on every decode step. The n-gram proposer only
fires when prompt lookup finds a match, and measurement shows it fires on
**13.2%** of steps. On the other 87% the engine performs a plain single-token
decode that never enters the counter.

```
tau_eff  = cov * tau_cond + (1 - cov) * 1
Speedup ~= tau_eff / R
```

Verified: `0.132 * 2.889 + 0.868 = 1.249` against a measured 1.250.

This reverses the apparent ranking. On conditional acceptance n-gram looks
*better* than DFlash (2.889 vs 2.543). Corrected for coverage, n-gram's
implied R is **1.082**, lower than DFlash's 1.360, which is what the hardware
predicts, since n-gram's drafter is a CPU string lookup with no model forward
and it verifies 6 tokens per pass instead of 8. n-gram is the cheapest rung
per pass. It loses because it is idle seven steps out of eight.

Consequence we believe matters beyond this project: published tau figures for
lookup-based proposers are conditional in the same way and are therefore not
comparable to a model drafter's tau.

### Finding 3: rungs cross 1.0x at different load

n-gram crosses between concurrency 8 (1.07x) and 32 (0.81x), interpolated
around 14. DFlash is still above 1.0x at 32 and only loses at saturation. Gap
of roughly 3 to 4x in serving load.

### Finding 4: the baseline knee explains the crossings

No-speculation throughput: 32.6, 172.1, 335.5, 405.1 tok/s across the four
load levels. That is 5.3x for 8x the requests, then 1.95x for 4x, then 1.21x.
The GPU saturates between concurrency 8 and 32, which is exactly where
speculation stops paying. Speculation spends spare compute to save memory
traffic, so its benefit is bounded by the slack between achieved throughput and
the roofline.

For scale, 32.6 tok/s at concurrency 1 is about 78% of the bandwidth-bound
ceiling for a 16.4 GB bf16 model on an A40's 696 GB/s.

### Finding 5: acceptance decay shape distinguishes mechanism

n-gram and DFlash have **identical** first-position acceptance (0.601) and
completely different decay. n-gram decays slowly because when it fires it is
copying a literal repeated span. DFlash decays fast because it is predicting.
Same hit rate on token one, different continuation behaviour.

### Finding 6: speculation perturbs output less than the engine perturbs itself

We checked losslessness by hashing generated token ids per prompt. Initially
alarming: 0 of 9 speculative arms reproduced the baseline, roughly 80% of
prompts differing.

The control changes the interpretation. Comparing the `off` arm against the
`off` arm at different concurrency, same prompts, same seed, no speculation
anywhere:

```
off c=1  vs off c=8   :  7/50 identical
off c=1  vs off c=32  :  5/50 identical
off c=8  vs off c=32  :  7/50 identical
```

12.7% agreement with no speculation involved. The speculative arms against the
baseline at matching concurrency average **19.0%**. Speculation agrees with the
baseline more often than the baseline agrees with itself.

Treating first divergence as a hazard over 256 positions, implied per-token
divergence is **0.80%** baseline-to-baseline and **0.65%**
speculative-to-baseline. Causes are batch-dependent reduction order in GPU
kernels and prefix caching, which was on at a 21 to 24% hit rate.

Our reading: exact-match losslessness is not testable on a continuous-batching
server, and the claim should be stated relative to the engine's own
reproducibility floor.

## 6. Infrastructure built

- `run_benchmark.py`, one arm per invocation, records throughput, the vLLM
  spec-decode counters including acceptance by draft position, and per-request
  token ids. Refuses to start without the runner pin.
- `phase0.sh`, one job per bucket, sweeps rungs across a concurrency grid.
  Reads the GPU from `nvidia-smi` at run time rather than trusting the
  partition flag, so results cannot be filed under the wrong hardware. Aborts
  in seconds if the GPU's compute capability has no kernels in the build.
- `summarize.py`, computes speedup against the baseline **at matching
  concurrency**, reconstructs coverage and effective tau, reports implied R,
  truncates acceptance to a common draft length before any cross-rung
  comparison, and interpolates the 1.0x crossing.
- `check_lossless.py`, runs the baseline-against-itself control before
  comparing speculative arms.
- Everything versioned in git, with Slurm logs committed as provenance.

## 7. Known weaknesses, stated plainly

1. **No confidence intervals anywhere.** Every number is a single run. Two of
   the measured speedups (1.03x, 1.07x) sit close enough to 1.0 that a reader
   will ask whether they are distinguishable from noise. The harness has just
   been changed to record per-request latency so a paired bootstrap over
   prompts becomes possible; no data with it exists yet.
2. **R is implied, not measured.** It is computed as `tau_eff / speedup`, so
   any error in the throughput measurement lands in R by construction, and R
   cannot be used to independently predict speedup without circularity.
3. **Only one of nine cells is complete.** 8K and 32K buckets, and all A100
   data, are queued but not collected. Decay D across context is the number
   the project's milestone is written against and it does not exist yet.
4. **k is not matched across rungs.** DFlash ran at 7, n-gram at 5. Acceptance
   is only compared at a common truncation, but throughput is not controlled
   for k.
5. **Prefix caching is on** at a 21 to 24% hit rate. Identical across arms so
   it cancels from the speedup ratio, but it inflates absolute throughput and
   will interact differently with longer context, where prompts share less
   structure. A caching-off control has not been run.
6. **One target model, one workload family.** Qwen3-8B and LongBench-v2 only.
   No evidence the findings generalise across model scale or task type.
7. **No H200**, so the hardware axis has two points rather than three, and both
   are Ampere. The compute-to-bandwidth contrast is narrower than intended.
8. **No quality evaluation.** Because speculation is assumed lossless, no task
   accuracy was measured. Given Finding 6, that assumption is now weaker than
   assumed, and nobody has checked whether the 0.65% token divergence changes
   any answer.
9. **Optimal k is predicted but untested.** Fitting R's slope per draft
   position suggests the best k shrinks with load, roughly 7 at concurrency 1
   and 3 at concurrency 32. Only the k=15 point supports the direction.

## 8. Planned remaining work

Queued or assigned: A40 8K, A100 2K, A100 8K, A40 32K. Held: A100 32K.

Candidate additional experiments, none yet committed:

- n-gram coverage/acceptance frontier by varying `prompt_lookup_min`. n-gram's
  ceiling at 13% coverage is 1.25x even with free verification; it would need
  roughly 54% coverage to match DFlash.
- k sweep at high concurrency to test the shrinking-optimal-k prediction.
- Prefix-caching-off control to split the two causes of the non-determinism
  floor.
- Repeated runs for confidence intervals.

## 9. What we want from a review

Be adversarial. We would rather find the holes now.

1. **Is the tau / R / coverage decomposition sound?** Is the coverage term a
   real contribution or a restatement of something standard in the literature
   that we have simply not found?
2. **Is Finding 6 novel or known?** Does published work on speculative decoding
   address output non-determinism under continuous batching, and is our framing
   (deviation relative to the engine's own floor) defensible or a dodge?
3. **What is the single strongest objection a reviewer would raise?** We expect
   it is either the absence of confidence intervals or the single-model scope.
4. **Is "implied R" acceptable**, or does the paper need an independently
   measured R, and if so how would you measure it without circularity?
5. **Given roughly 8 GPU-hours and a semester deadline, what is the minimum
   set of additional experiments** that makes this defensible? What on the list
   in section 8 should be cut?
6. **What related work should we check against?** We are aware of SmartSpec,
   AdaServe, AdaSpec, Nightjar, EAGLE-3, and the DFlash line. We have not done
   a systematic survey of speculative decoding under batched serving.
7. **Is the central claim too weak?** "The right rung depends on operating
   point" may be true but unsurprising. Is there a sharper claim the data
   supports, for example a predictive rule that picks the rung from load,
   context and hardware without re-benchmarking?
8. **Does anything in sections 4 to 6 look like a measurement artifact** rather
   than a result? Specifically: is the coverage reconstruction in Finding 2
   sound, given it is derived from aggregate counters rather than per-step
   logging?

## 10. Requested output

A written evaluation covering:

- an honest assessment of whether this is paper-level work and what it is
  missing
- the three most serious methodological problems, ranked
- a prioritised experiment list that fits the budget
- related work we should read, with specific reasons
- a suggested sharper framing of the central claim, if one exists
- anything we appear to have gotten wrong
