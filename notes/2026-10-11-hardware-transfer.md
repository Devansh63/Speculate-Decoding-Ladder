# tau transfers across hardware. R does not.

> **Partly superseded, 11 Oct 2026 (late).** Two things in this note are
> withdrawn by the second review of the same day. The bandwidth mechanism
> offered for "speculation pays less on a faster machine" contradicts the ridge
> points — the A100's ~201 FLOP/byte exceeds the A40's ~107, so roofline
> predicts speculation should gain *more* there, and it gained less. And the
> cache-pressure item in the list of explanations for a miss is withdrawn: a
> concurrency-1 32K cell needs ~24 GB against 40 GB, so there is no capacity
> pressure at that concurrency. **The tau and R measurements themselves are
> unaffected — C16 and C17's observation stands; only C17's explanation is
> withdrawn.** Register, C17.

**Source** jobs 22757102 (A40, gpub067), 22757105 (A40, gpub085) and
**22757104 (A100-SXM4-40GB, gpua045)** — all run by ssashi on 8 Oct as the
`notes/teammate-quickstart.md` smoke test: 2K bucket, concurrency 1,
`LIMIT=5`, seed 1234, `vllm_env_multiarch`. Roughly 20 minutes each.
**The first A100 measurements this project has.**

This note exists because a smoke test accidentally answered the central
question. The run was meant only to prove a teammate's pipeline worked.

## The measurement

| | A40 | A100 | change |
|---|---|---|---|
| **DFlash tau_cond** | 2.594 | 2.572 | **-0.8%** |
| **n-gram tau_cond** | 2.533 | 2.525 | **-0.3%** |
| DFlash tau_eff | 2.570 | 2.560 | -0.4% |
| n-gram tau_eff | 1.223 | 1.199 | -2.0% |
| **DFlash R** | 1.358 | 1.433 | **+5.5%** |
| **n-gram R** | 1.075 | 1.132 | **+5.3%** |
| DFlash speedup, k=7 | 1.89x | 1.79x | -5.3% |
| n-gram speedup, k=5 | 1.14x | 1.06x | -7.0% |
| `off` throughput | 33.0 tok/s | 72.8 tok/s | A100 **2.21x faster** |

## Why this matters

The project's entire structure rests on a claim that had never been tested:
that speedup factors into an **acceptance** term and a **cost** term,

```
Speedup ~= [cov * tau_cond + (1 - cov)] / R
```

where `tau` belongs to the drafter and the text, and `R` belongs to the
machine and its load. If that split is real, you can measure acceptance once
and predict performance anywhere. If it is not, every number is local to the
GPU it was taken on and the study has no reach beyond one A40.

**It is real, at least between these two GPUs.** Acceptance agrees to under
one percent. The cost ratio moves by 5.5%, and — this is the part that makes
it a machine property rather than a coincidence — **it moves by the same
amount for both rungs**, 5.5% and 5.3%, despite those rungs differing by a
factor of four in drafting cost and by a factor of seven in coverage.

The direction is also right. The A100 is 2.21x faster at plain decoding,
which is the memory-bandwidth-bound regime, and speculation helps *less*
there. Speculation converts memory-bound decode steps into a compute-bound
verification pass, so its payoff is largest on the machine wasting the most
bandwidth. A faster machine has less waste to reclaim.

**We have not verified the spec sheets.** The bandwidth and FLOPS ratios that
would turn that story into arithmetic have not been checked against
NVIDIA's own numbers, and should be before the paper asserts a mechanism
rather than a direction.

## Caveats

**n = 5.** About 500 draft events and 1,280 generated tokens per arm. That is
plenty for `tau` and `R`, which average over many events, and thin for
anything per-position.

**First-position acceptance wobbles more than the theory allows.** DFlash
0.641 to 0.610, n-gram 0.559 to 0.547. Coverage for n-gram moves 0.145 to
0.130 on 152 against 139 draft events. At this sample size that is plausibly
noise — but a different GPU also means different kernels and therefore
different logits, which is the same nondeterminism seen between the eager and
compiled paths on 8 Oct. **Unresolved.** It needs n=50 to separate sampling
noise from a real numerics effect.

**The 5.5% factor rests on one A100 job.** It is simultaneously the most
useful number in this note and the least supported.

**The A40 side is itself a replication.** Jobs 22757102 and 22757105 ran the
identical configuration on gpub067 and gpub085 and agree to 0.03% on the
baseline and 0.03% on n-gram speedup (1.1375 against 1.1372). DFlash
differed by 1.4%. Four A40 nodes have now been measured and three agree
tightly, which retires the gpub073 anomaly as an outlier rather than typical
spread — see `notes/2026-10-11-noise-floor.md`.

## What it sets up: C4, properly

C4 has always been the portability claim: fit a cost model on one platform,
freeze a prediction, measure on hardware not used for fitting. Until now
there was no second platform. There is one now, and the single-factor
hypothesis gives a sharp, falsifiable prediction for the rest of the A100
grid.

**The prediction is frozen in `paper/claims.md` under "Frozen prediction, 11
Oct 2026", committed before any A100 cell at full n is run.** It is not
repeated here, so that there is exactly one authoritative copy with a
timestamp.

In outline: apply the measured factor to the A40 `R` values at concurrency 1
and predict A100 speedups at 8K and 32K. 2K is **in sample** — the factor was
derived there — so only 8K and 32K count as tests. The headline consequence
is that the inversion is predicted to survive on A100 and widen slightly.

Two cells at n=50, one at 8K and one at 32K, test all four numbers. That is
the highest-value GPU time available to this project, and it belongs to
Sahil's allocation, which is almost untouched.

## What a failure would teach

If the measured A100 numbers miss by much more than the factor predicts, the
single-factor model is wrong and the interesting question becomes *what else*
R depends on. Candidates, in the order we could test them:

1. **Context length.** R may scale differently with prompt length on a
   bandwidth-rich machine, since `R_draft` is a forward pass over the prefix.
   A miss that grows with bucket would point here.
2. **Coverage.** If the factor differs between rungs at long context but not
   short, the machine term is not separable from the drafter term after all.
3. **Cache pressure.** The A100-SXM4-40GB has 40 GB against the A40's 48, so
   at 32K the concurrency ceiling is lower and a long-context cell may hit a
   different regime entirely. This is the most likely cause of a large miss
   at 32K specifically, and it is a confound rather than a refutation.

Item 3 is worth saying loudly before the cells run: **a 32K miss on A100 may
be a memory-capacity artifact, not a failure of the cost model.** The
prediction should be judged primarily on 8K.
