# The Speculation Ladder: project brief for external review

CS 598, Hardware and Software for AI, Fall 2026. Two students: Devansh Agrawal,
Sahil Sashi. Target output is a paper-level course project.

Repo: https://github.com/Devansh63/Speculate-Decoding-Ladder
Last updated 6 Oct 2026.

This document is self-contained. It states the research question, the metric
framework, what has been measured, what the measurements showed, and where the
work is weak. It ends with specific questions for a reviewer.

Where a number is uncertain or a claim is unsupported, it says so.

---

## 1. The question

Speculative decoding makes a small, fast drafter propose k tokens, which the
large target model verifies in a single forward pass, keeping the longest
correct prefix. Output is supposed to be identical to what the target would
have produced alone.

There are now many drafting methods: n-gram prompt lookup, self-speculation,
trained heads such as EAGLE-3 and DFlash, and full small-model drafters. The
published speedups are not comparable. Each is reported on its own hardware,
at its own batch size, on its own workload, usually at batch 1.

The project treats these as **rungs on a ladder** and asks which rung is
correct under which operating conditions:

- **serving load**, the number of concurrent requests
- **context length**, 2K to 32K tokens
- **hardware**, which fixes the compute-to-bandwidth ratio

## 2. The metric framework

A single speedup number bundles two unrelated things: how good the drafter's
guesses are, and how expensive verification is on this machine at this load.

- **tau**, mean accepted length. Tokens yielded by one verification pass. A
  property of the drafter and the text.
- **R**, verification cost ratio. What one pass costs relative to one ordinary
  decode step. A property of the machine and the load.
- **coverage**, the fraction of decode steps on which a draft was proposed at
  all. Added after measurement showed it was not optional; see Finding 2.

```
tau_eff  = cov * tau_cond + (1 - cov) * 1
Speedup ~= tau_eff / R
```

## 3. Setup

| | |
|---|---|
| Target model | Qwen3-8B, bf16, 36 layers, 8 KV heads, head dim 128 |
| Drafter | `z-lab/Qwen3-8B-DFlash-b16` (trained head), plus n-gram prompt lookup |
| Serving engine | vLLM `0.29.1rc1.dev345+gbf13ecc2f.cu132`, built from source on the cluster |
| Workload | LongBench-v2, bucketed at 2K, 8K, 32K context |
| Decoding | greedy, temperature 0, 256 max new tokens, fixed seed |
| Cluster | NCSA Delta |
| Hardware | A40 48 GB (ridge point 107). **A100 is blocked, see below.** |

**Hardware status.** A40 works. **A100 does not run this build at all**: every
vLLM operation fails with `cudaErrorNoKernelImageForDevice`, including a
single `rms_norm` custom op with no torch.compile or CUDA graphs involved,
while plain torch runs a bf16 matmul on the same node without trouble.
`cuobjdump` lists sm_80 cubins in the extension, so the listing is not
sufficient evidence that the kernels run. H200 (sm_90) has no cubins and no
PTX and cannot run either. Full diagnosis in `env/a100-sm80-failure.md`.
Fixing it needs a from-source rebuild, which is hours of login-node compiling
and no GPU time, and which would also cover H200.

**Other constraints that shaped the design:**

- Both rungs are pinned to the stable model runner via
  `VLLM_USE_V2_MODEL_RUNNER=0`. Without the pin, n-gram silently falls back to
  it while DFlash stays on the experimental runner, and the two rungs are then
  measured on different execution paths. The harness refuses to start without
  it.
- **The concurrency grid shrinks with context, and has to.** Qwen3-8B stores
  about 144 KiB of KV per token; measured cache on A40 is 160,912 tokens.
  Reachable concurrency is roughly 70 at 2K, 19 at 8K, 5 at 32K. Asking for
  more does not fail: vLLM accepts `max_num_seqs=32` and then schedules fewer,
  so the row gets labelled with a concurrency it never ran at.
- **Budget.** 10 GPU-hours per student on the current sub-allocation, roughly
  6 remaining. More is available on request. A100 bills at about twice A40.

## 4. What has been measured

Two complete cells, both on A40: **2K** and **8K**, three rungs each across a
concurrency grid. 50 prompts per arm, same prompts, same seed.

### 2K

| conc | rung | k | tok/s | speedup | tauC | tauE | cov | R |
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

`sat` is an uncontrolled run, all prompts submitted at once, roughly 70
concurrent. Kept deliberately as the right end of the load axis.

### 8K

| conc | rung | k | tok/s | speedup | tauC | tauE | cov | R | pos1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | off | 0 | 27.6 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 | |
| 1 | dflash | 7 | 34.8 | **1.26** | 1.953 | 1.947 | 1.000 | 1.546 | 0.451 |
| 1 | ngram | 5 | 31.5 | 1.14 | 2.762 | 1.284 | 0.161 | 1.124 | 0.563 |
| 4 | off | 0 | 68.3 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 | |
| 4 | dflash | 7 | 75.9 | 1.11 | 1.941 | 1.939 | 1.000 | 1.744 | 0.451 |
| 4 | ngram | 5 | 73.2 | 1.07 | 2.857 | 1.315 | 0.169 | 1.227 | 0.577 |
| 12 | off | 0 | 99.7 | 1.00 | 1.000 | 1.000 | 0.000 | 1.000 | |
| 12 | dflash | 7 | 102.1 | 1.02 | 1.996 | 1.990 | 1.000 | 1.944 | 0.457 |
| 12 | ngram | 5 | 95.3 | 0.96 | 2.813 | 1.322 | 0.178 | 1.383 | 0.571 |

`R` is **implied**, computed as `tau_eff / speedup`, not measured
independently. A known weakness.

## 5. Findings

### 1. tau is invariant to load, speedup is not

DFlash mean accepted length across concurrency 1, 8, 32 and saturation at 2K:
2.543, 2.507, 2.470, 2.520. First-position acceptance: 0.601, 0.600, 0.597,
0.601. Speedup over the same: 1.86, 1.49, 1.03, 0.86.

All of the variation lives in R. The decomposition demonstrated rather than
asserted, and the basis for the portability claim.

### 2. The decomposition needs coverage

vLLM's reported mean accepted length is **conditional on a draft having been
proposed**. DFlash proposes on every step. The n-gram proposer fires only on a
prompt-lookup hit, measured at **13.2%** of steps at 2K.

Verified: `0.132 * 2.889 + 0.868 = 1.249` against a measured 1.250.

This reverses the apparent ranking. On conditional acceptance n-gram looks
better than DFlash (2.889 against 2.543). Corrected, n-gram's implied R is
**1.082**, below DFlash's 1.360, which is what the hardware predicts: a CPU
string lookup with no model forward, verifying 6 tokens instead of 8. n-gram
is the cheapest rung per pass and loses because it is idle seven steps in
eight.

Published tau figures for lookup-based proposers are conditional in the same
way and are therefore not comparable to a model drafter's tau.

### 3. Rungs cross 1.0x at different load

At 2K: n-gram crosses around concurrency 14, DFlash only at saturation. Gap of
roughly 3 to 4x in serving load.

### 4. The baseline knee explains the crossings

2K no-speculation throughput: 32.6, 172.1, 335.5, 405.1 tok/s. That is 5.3x
for 8x the requests, then 1.95x for 4x, then 1.21x. The GPU saturates between
8 and 32, exactly where speculation stops paying. Speculation spends spare
compute to save memory traffic, so its benefit is bounded by the slack between
achieved throughput and the roofline.

### 5. Acceptance decay shape distinguishes mechanism

At 2K, concurrency 1, the two rungs have **identical** first-position
acceptance (0.601) and completely different decay: n-gram 0.601, 0.440, 0.345,
0.277, 0.226 against DFlash 0.601, 0.360, 0.219, 0.140, 0.101. n-gram copies a
literal span, so a correct first token usually implies correct continuation.
DFlash predicts, so confidence falls fast.

### 6. Speculation perturbs output less than the engine perturbs itself

0 of 9 speculative arms reproduced the baseline token for token, about 80% of
prompts differing. But the control, comparing the `off` arm against itself at
different concurrency with no speculation anywhere, agrees on only 12.7% of
sequences. The speculative arms agree with the baseline on 19.0%, **more often
than the baseline agrees with itself.**

Implied per-token divergence, treating first divergence as a hazard over 256
positions: 0.80% baseline-to-baseline, 0.65% speculative-to-baseline. Causes
are batch-dependent reduction order in GPU kernels and prefix caching, on at a
21 to 24% hit rate.

Exact-match losslessness is not testable on a continuous-batching server. The
claim is framed relative to the engine's own reproducibility floor.

### 7. The rungs decay in OPPOSITE directions along the context axis

**The most important result.** Going from 2K to 8K:

| | 2K | 8K | direction |
|---|---|---|---|
| DFlash tau | 2.543 | 1.953 | collapses |
| DFlash first-position acceptance | 0.601 | 0.451 | collapses |
| n-gram tau | 2.889 | 2.762 | nearly flat |
| n-gram first-position acceptance | 0.601 | 0.563 | nearly flat |
| n-gram coverage | 0.132 | 0.161 | **rises** |

Decay at common truncation k=5, concurrency 1: **D = 0.219 for DFlash, 0.044
for n-gram.** Five times the decay for the trained head.

The mechanism gives the sign of each. DFlash **predicts**, and prediction gets
harder as the context to summarise grows. n-gram **retrieves**, and a longer
prompt holds more matchable spans, so it fires more often while acceptance
given a hit barely moves.

At 2K the two rungs had identical first-position acceptance. At 8K n-gram is
ahead, 0.563 against 0.451. The cheap retrieval rung is now the more accurate
one on the first drafted token.

The speedup gap at concurrency 1 fell from 0.70 (1.86 against 1.16) to 0.12
(1.26 against 1.14) in one bucket step. Nothing about the hardware or the
batch size changed.

### 8. The usable operating window shrinks with context

| rung | 2K crossing | 8K crossing |
|---|---|---|
| n-gram k=5 | concurrency 14.4 | concurrency 8.9 |
| DFlash k=7 | above 32 | about 12 |

Two compounding effects: R rises because verifying k+1 tokens costs more
attention work at longer context (DFlash R at concurrency 1: 1.360 at 2K,
1.546 at 8K), and the baseline saturates earlier (2.47x for 4x the requests at
8K, against 5.3x for 8x at 2K).

### Predictions for 32K, currently running

1. **n-gram overtakes DFlash.** The ranking of the rungs inverts along the
   context axis.
2. **DFlash falls below 1.0x even at concurrency 1**, ceasing to pay at long
   context under any load.

Both follow from extrapolating a two-point trend and should be read as
direction, not forecast.

## 6. Infrastructure

- `run_benchmark.py`: one arm per invocation. Records throughput, the vLLM
  spec-decode counters including acceptance by draft position, full output
  token ids, and per-request timing where exposed. Refuses to start without
  the runner pin.
- `phase0.sh`: one job per bucket, sweeps rungs across a concurrency grid.
  Reads the GPU from `nvidia-smi` at run time rather than trusting the
  partition flag, so results cannot be filed under the wrong hardware. Aborts
  in seconds on an uncovered compute capability.
- `summarize.py`: speedup against the baseline **at matching concurrency**,
  coverage and effective tau reconstructed, implied R, acceptance truncated to
  a common draft length before any cross-rung comparison, 1.0x crossing
  interpolated, decay D across buckets.
- `check_lossless.py`: runs the baseline-against-itself control before
  comparing speculative arms.
- `diag_a100.sh`: layer-by-layer isolation of the A100 failure.
- Everything in git, with Slurm logs committed as provenance.

## 7. Known weaknesses, ranked

1. **The hardware axis has one point.** A100 is blocked and H200 cannot run at
   all, so every number comes from a single A40. The central claim is about
   how the right rung depends on hardware, and there is currently no hardware
   variation at all. This is the most serious gap and it is new since the last
   revision of this brief.
2. **No confidence intervals anywhere.** Every number is a single run. Two
   measured speedups (1.02x, 1.07x) sit close enough to 1.0 that a reader will
   ask whether they are distinguishable from noise. The harness now records
   per-request latency so a paired bootstrap is possible; no data with it
   exists yet.
3. **R is implied, not measured.** Computed as `tau_eff / speedup`, so any
   error in the throughput measurement lands in R by construction, and R
   cannot independently predict speedup without circularity.
4. **The context trend rests on two points.** Findings 7 and 8, the most
   interesting results, are a line through 2K and 8K. The 32K cell is what
   turns them into a curve.
5. **One target model, one workload family, one trained drafter.** Whether
   prediction-based drafters generally decay faster than retrieval-based ones
   is a conjecture the data suggests but cannot establish.
6. **k is not matched across rungs.** DFlash at 7, n-gram at 5. Acceptance is
   compared only at a common truncation, but throughput is not controlled for
   k.
7. **Prefix caching is on** at 21 to 24% hit rate, and the rate differs
   between buckets because longer prompts share less structure. It cancels
   from within-bucket speedup ratios but confounds absolute throughput across
   buckets. A caching-off control has not been run.
8. **No quality evaluation.** Losslessness was assumed, so task accuracy was
   never measured. Finding 6 weakens that assumption, and nobody has checked
   whether the 0.65% token divergence changes any answer.
9. **Optimal k is predicted but untested.** Fitting R's slope per draft
   position suggests the best k shrinks with load, roughly 7 at concurrency 1
   and 3 at concurrency 32. Only the k=15 point supports the direction.

## 8. Remaining work

Running: A40 32K. Blocked: everything on A100 and H200, pending a rebuild.

Candidates, none committed:

- Rebuild vLLM with `8.0;8.6;9.0` and re-run the A40 cells so the whole table
  comes from one binary. Roughly 3 GPU-hours plus hours of compiling.
- n-gram coverage/acceptance frontier via `prompt_lookup_min`. Its ceiling at
  13% coverage is 1.25x even with free verification; matching DFlash at 2K
  would need about 54%. At 8K the required coverage is lower, which makes the
  frontier more interesting, not less.
- k sweep at high concurrency to test the shrinking-optimal-k prediction.
- Prefix-caching-off control.
- Repeated runs for confidence intervals.

## 9. What we want from a review

Be adversarial. We would rather find the holes now.

1. **Is the tau / R / coverage decomposition sound?** Is the coverage term a
   contribution or a restatement of something standard we have not found?
2. **Is Finding 7 known?** Does published work establish that retrieval-based
   and prediction-based drafters diverge in opposite directions as context
   grows? This is our strongest result and we most need it checked.
3. **Is Finding 6 known?** Does published work address output
   non-determinism under continuous batching, and is our framing defensible or
   a dodge?
4. **Given one working GPU and a tight budget, what should we spend on?**
   Rescue the hardware axis with a rebuild and re-runs, or abandon it and go
   deep on the context axis where the results are strongest? The proposal
   committed to a hardware comparison; the data is more interesting along
   context.
5. **What is the single strongest objection a reviewer would raise?**
6. **Is "implied R" acceptable**, or does the paper need an independently
   measured R, and if so how, without circularity?
7. **What related work should we check?** We know of SmartSpec, AdaServe,
   AdaSpec, Nightjar, EAGLE-3 and the DFlash line. No systematic survey of
   speculative decoding under batched serving has been done.
8. **Is the central claim too weak?** "The right rung depends on operating
   point" may be unsurprising. Finding 7 suggests a sharper one: the ranking
   of the rungs inverts along context, predictably from each mechanism. Is
   that strong enough to carry a paper?
9. **Does anything in sections 4 and 5 look like a measurement artifact?**
   Specifically, is the coverage reconstruction sound given it derives from
   aggregate counters rather than per-step logging?

## 10. Requested output

- an honest assessment of whether this is paper-level work and what it lacks
- the three most serious methodological problems, ranked
- a prioritised experiment list that fits one GPU and roughly 6 GPU-hours
- related work we should read, with reasons
- a sharper framing of the central claim, if one exists
- anything we appear to have gotten wrong
