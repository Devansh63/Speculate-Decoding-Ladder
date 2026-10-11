# Every cell measured

One table for the whole grid: model, hardware, context, batch size, draft
length, and the job and node that produced each number. Derived from
`results/*/results.jsonl` read **per record**, split on `env.slurm_job`.

Two things to state before any number:

1. **Target model is Qwen3-8B throughout.** Hardware is A40 for everything at
   n >= 25; the A100 column holds one smoke cell at n=5, which is the whole of
   C4's measured evidence so far.
2. **Every speedup is a within-job ratio** — each arm divided by the `off` arm
   measured in the same job, same engine session, same node. Ratios replicate
   across nodes to 0.03–0.08%; absolute throughput does not (8.8% against one
   node). Never compare throughput across jobs; compare ratios.

This file exists because the grid previously lived only in `summarize.py`
output, which keys on `(bucket, conc, rung, k)` and silently keeps one record
per key. Two jobs measuring the same cell were rendered as one row, which is
how a replicated result was recorded as a single run for four days. See
`notes/2026-10-11-noise-floor.md`.

---

## Models

| role | what | notes |
|---|---|---|
| target | Qwen3-8B | bf16, 256 max new tokens, seed 1234, greedy |
| `off` | no drafter | the baseline every speedup divides by |
| `ngram` | prompt lookup, no model | n-gram match against the prompt; drafts on a fraction of steps |
| `dflash` | Qwen3-8B-DFlash-b16 | trained draft head; drafts on ~100% of steps |

Workload: LongBench-v2, bucketed by prompt length into 2K / 8K / 32K.
Engine: vLLM `0.29.1rc1.dev345+gbf13ecc2f`, torch 2.13.0+cu130,
`VLLM_USE_V2_MODEL_RUNNER=0`, prefix caching on.

---

## Headline grid — A40, speedup against no speculation

| context | batch | prompts | `off` | `ngram` k=5 | `dflash` k=7 | job | node |
|---|---|---|---|---|---|---|---|
| 2K | 1 | 50 | 1.00 | 1.15 | **1.86** | 22598608 | — |
| 2K | 1 | 5 | 1.00 | 1.137 | 1.867 | 22757102 | gpub067 |
| 2K | 1 | 5 | 1.00 | 1.137 | 1.893 | 22757105 | gpub085 |
| 2K | 8 | 50 | 1.00 | 1.07 | 1.49 | 22598608 | — |
| 2K | 32 | 50 | 1.00 | 0.81 | 1.03 | 22598608 | — |
| 2K | unbounded | 100 | 1.00 | 0.71 | 0.86 | 22569403 | — |
| 8K | 1 | 50 | 1.00 | 1.15 | 1.26 | 22685911 | — |
| 8K | 4 | 50 | 1.00 | 1.07 | 1.11 | 22685911 | — |
| 8K | 12 | 50 | 1.00 | 0.96 | 1.02 | 22685911 | — |
| 32K | 1 | 25 | 1.00 | **1.08** | **0.83** | 22713173 | gpub019 |
| 32K | 1 | 25 | 1.00 | **1.08** | **0.83** | 22714602 | gpub047 |
| 32K | 2 | 25 | 1.00 | 1.05 | 0.91 | 22713173 | gpub019 |
| 32K | 4 | 25 | 1.00 | 0.99 | 0.88 | 22713173 | gpub019 |

## Headline grid — A100-SXM4-40GB

| context | batch | prompts | `off` | `ngram` k=5 | `dflash` k=7 | job | node |
|---|---|---|---|---|---|---|---|
| 2K | 1 | 5 | 1.00 | 1.06 | 1.79 | 22757104 | gpua045 |

Everything else on A100 is **predicted, not measured** — see "Frozen
prediction, 11 Oct 2026" in `paper/claims.md`.

---

## Replications

Three cells have been measured more than once. All are within-job ratios.

**32K, batch 1, n=25** — the inversion, three days apart on two nodes:

| | 22713173, gpub019 | 22714602, gpub047 |
|---|---|---|
| `off` | 14.90 tok/s | 14.83 tok/s |
| `ngram` k=5 | 16.07 | 15.98 |
| `dflash` k=7 | 12.31 | 12.25 |
| n-gram speedup | 1.0785 | 1.0776 |
| DFlash speedup | 0.8262 | 0.8260 |

**0.08% and 0.02% agreement.** This is the inversion, and it is the
best-replicated result in the project — replicated across **nodes and runs**,
not across prompt samples: both jobs used the same 25 prompts and the same
seed, so sampling variance is not bounded by this pair (register, C8, 11 Oct).

**2K, batch 1, n=5** — two A40 nodes, same hour:

| | 22757102, gpub067 | 22757105, gpub085 |
|---|---|---|
| `off` | 32.95 | 32.94 |
| `ngram` k=5 | 37.48 | 37.46 |
| `dflash` k=7 | 61.51 | 62.37 |
| n-gram speedup | 1.1375 | 1.1372 |
| DFlash speedup | 1.8668 | 1.8934 |

Baselines agree to **0.03%**. Four A40 nodes have now been measured; three
agree tightly, which makes gpub073 (job 22757472, 8.8% low) an outlier rather
than typical spread.

**2K, batch 1** across n — n=5 against n=50: `off` 32.95 vs 32.6 (1.1%),
n-gram 1.137 vs 1.15, DFlash 1.867 vs 1.86. The small prompt count holds up
better than expected for ratios.

"unbounded" in the A40 grid is the 30 Sep run with no concurrency control.
vLLM batched as wide as memory allowed, so those speedups measure the batching
decision rather than the drafter. Kept as the saturation end of the load axis,
not as a valid cell.

---

## Hardware comparison — A40 against A100, 2K, batch 1, n=5

The central test of the project's structure: does acceptance transfer across
machines while cost does not?

| | A40 | A100 | change |
|---|---|---|---|
| **DFlash tau_cond** | 2.594 | 2.572 | **-0.8%** |
| **n-gram tau_cond** | 2.533 | 2.525 | **-0.3%** |
| DFlash tau_eff | 2.570 | 2.560 | -0.4% |
| n-gram tau_eff | 1.223 | 1.199 | -2.0% |
| **DFlash R** | 1.358 | 1.433 | **+5.5%** |
| **n-gram R** | 1.075 | 1.132 | **+5.3%** |
| DFlash speedup k=7 | 1.89x | 1.79x | -5.3% |
| n-gram speedup k=5 | 1.14x | 1.06x | -7.0% |
| n-gram coverage | 0.145 | 0.130 | -10% |
| DFlash first-position acceptance | 0.641 | 0.610 | -4.8% |
| n-gram first-position acceptance | 0.559 | 0.547 | -2.1% |
| `off` throughput | 33.0 tok/s | 72.8 tok/s | A100 **2.21x faster** |

**Acceptance transfers to under one percent; the cost ratio rises 5.5%, and by
the same amount for both rungs** — rungs that differ by 4x in drafting cost
and 7x in coverage. That is what a machine property looks like.

The A100 is 2.21x faster at plain decoding and speculation pays *less* there.
The explanation that used to follow — "speculation reclaims wasted memory
bandwidth, and a faster machine wastes less" — is **withdrawn**: the A100's
ridge point (~201 FLOP/byte against the A40's ~107) is *higher*, so naive
roofline predicts speculation should gain *more* there. It gained less. The
observation stands; the mechanism is unexplained (register, C17, 11 Oct).

Two caveats on this table. **n=5**, so the per-position and coverage rows may
be sampling noise rather than real — about 500 draft events per arm. And a
different GPU means different kernels and therefore different logits, the same
nondeterminism seen between eager and compiled paths on 8 Oct, so a real
acceptance shift cannot be ruled out either. Needs n=50 to separate them.
Write-up in `notes/2026-10-11-hardware-transfer.md`.

---

## Draft-length sweeps, batch 1, A40

| context | rung | k=1 | k=2 | k=4 | k=5 | k=7 | k=15 | job |
|---|---|---|---|---|---|---|---|---|
| 2K | DFlash | 1.20 | 1.47 | — | — | **1.86** | — | 22737716 / 22598608 |
| 2K | DFlash, unbounded batch | — | — | — | — | 0.86 | 0.78 | 22569403 |
| 2K | n-gram | 1.06 | 1.09 | — | 1.15 | 1.14 | — | 22757473 / 22598608 |
| 8K | DFlash | 0.96 | 1.10 | — | — | **1.26** | — | 22737717 / 22685911 |
| 8K | n-gram | — | — | — | 1.15 | — | — | 22685911 |
| 32K | DFlash | 0.76 | 0.78 | 0.81 | — | **0.83** | — | 22714602 |
| 32K | n-gram | 1.04 | 1.13 | — | 1.08 | 1.18 ⚠ | — | 22757472 / 22713173 |

**Optimal k rises with context. The load axis is unmeasured.** DFlash improves
monotonically in k at every bucket, with no turn by k=7; the only place k=15
loses to k=7 is 2K under unbounded batching. That single point is the entire
evidence for the load axis — and it comes from job 22569403, the run
disqualified on this page two sections above as measuring the batching decision
rather than the drafter. **There is no valid load-varying k data at any
bucket.** So the picture is one measured axis and one unmeasured one, not two
axes acting in opposite directions. Corrected 11 Oct 2026; see
`paper/claims.md`, C6.

Two provenance defects in the table above. The 2K and 32K n-gram k = 1, 2 rows
cite jobs 22757472 and 22757473, whose records are **not in this repository** —
they were written to a separate output directory (`results/a40_ngramk`) that
was never pushed, so those cells survive only as numbers in prose and cannot be
re-derived from `results/*/results.jsonl`. And `LOG.md`'s recorded submission
commands disagree with the records for the same jobs: it logs `LIMIT=25` for
22737716/22737717 where the records carry `limit 50`, and `KLIST="1 2 7 15"`
for 22714602 where the records hold k = 1, 2, 4, 7 and no k=15.

⚠ The 32K n-gram k = 1, 2, 7 values come from job 22757472 on **gpub073**, the
one anomalous node. k=5 (1.08) comes from gpub019 and gpub047. Do not read the
k=5 to k=7 step as a real gain; n-gram's 32K k-coefficient is unresolved and
non-monotonic within its own job.

---

## Acceptance, A40

Conditional mean accepted length (`tau_cond`), batch 1, and decay
`D = 1 - AL(long)/AL(2K)` at a common k=5 truncation:

| context | rung | tau_cond | first-position acceptance | D |
|---|---|---|---|---|
| 2K | DFlash k=7 | 2.543 | 0.601 | — |
| 8K | DFlash k=7 | 1.953 | 0.451 | 0.219 |
| 32K | DFlash k=7 | 1.459 | 0.296 | 0.402 |
| 2K | n-gram k=5 | 2.889 | 0.601 | — |
| 8K | n-gram k=5 | 2.762 | 0.563 | 0.044 |
| 32K | n-gram k=5 | 2.844 | 0.552 | 0.016 |

Acceptance by draft position at 32K, batch 1 — the clearest single picture of
why DFlash cannot compete at long context:

```
dflash k=7   0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004
ngram  k=7   0.541, 0.382, 0.304, 0.256, 0.219, 0.193, 0.177
```

DFlash's drafts are worthless past position 2. n-gram's still pay at
position 7.

**Coverage** — the fraction of decode steps on which a draft was proposed at
all, which engine-reported acceptance rates are conditional on:

| context | `ngram` k=5 | `dflash` k=7 |
|---|---|---|
| 2K | 0.132 | ~1.000 |
| 8K | 0.161 | ~1.000 |
| 32K | 0.212 | ~0.998 |

n-gram's coverage *rises* with context — a longer prompt offers more places
for a lookup to hit — while its acceptance stays flat. DFlash drafts on
essentially every step. Comparing the two rungs on raw acceptance without this
correction overstates n-gram by more than a factor of two.

---

## Cost model

`R(k) = (1 + R_draft) + c·k`, fitted per cell at batch 1 on A40. `R_draft` is
the fixed cost of drafting per decode step; `c` is the marginal cost per
drafted token.

| context | rung | R_draft | c | fit |
|---|---|---|---|---|
| 2K | n-gram | **0.071** | 0.0030 | monotonic, clean |
| 2K | DFlash | **0.296** | 0.0092 | monotonic |
| 8K | DFlash | **0.434** | 0.0157 | monotonic |
| 32K | n-gram | ~0.139 | unresolved | non-monotonic, gpub073 |
| 32K | DFlash | **0.662** | 0.0148 | monotonic, 4 points |

Growth across the context range: DFlash 2.24x, n-gram 1.96x. The **absolute
gap** grows 0.225 to 0.523. **That gap is not what flips the ranking**, contra
C13 as first written: a counterfactual on these measured numbers shows cost
alone leaves DFlash ahead (2K acceptance with 32K cost = 1.44x against
n-gram's 1.08x) while acceptance alone brings it to a tie (1.07x). Both
channels are needed and acceptance dominates (register, C13, corrected 11
Oct). Both costs are sublinear in context: a 16x longer prompt roughly doubles
each, and nothing in the model explains why.

Out-of-sample checks:

- 2K n-gram fit predicts `R(5) = 1.086`; job 22598608 measured **1.083**, ten
  days earlier with k=5 excluded from the fit.
- Machine factor **1.054** measured between A40 and A100 at 2K. Applied to the
  A40 grid it predicts the unmeasured A100 cells; frozen in `claims.md`.

---

## Hardware coverage

| GPU | arch | status | data |
|---|---|---|---|
| A40 | sm_86 | working throughout | everything at n >= 25, plus two n=5 cells |
| A100-SXM4-40GB | sm_80 | **working** since the 8 Oct multi-architecture rebuild | one cell: 2K, batch 1, n=5 (job 22757104) |
| H200 | sm_90 | kernels present in the rebuilt venv, but `phase0.sh`'s guard allows only `80\|86` and aborts | none |

Any A100 or H200 job **must** set
`LADDER_VENV=/projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch`.
Without it the script falls back to a single-architecture build and the job
dies at kernel launch after reserving a node, as job 22685764 did.

The A100-SXM4-40GB has **40 GB against the A40's 48**, which was recorded as a
pre-registered confound at 32K. **That confound is withdrawn** (register, 11
Oct): a concurrency-1 32K cell needs about 24 GB — 16.4 weights + 4.9 target KV
+ 1.9 drafter + 0.7 drafter KV — so there is no capacity pressure at the
concurrency the prediction is made at. 8K is still weighted higher, on a
different basis: the 1.054 machine factor was fitted at 2K, and `R`'s
composition changes with context.

---

## What is not in this table

- **No confidence intervals.** Per-request timing is unavailable on this vLLM
  build, so intervals require repeated whole runs. Three cells have two
  realizations each; nothing has three.
- **Only one A100 cell, at n=5.** The 8K and 32K A100 predictions are frozen
  and untested. Two cells at n=50 would settle C4.
- **No 8K n-gram k sweep**, so the middle `R_draft` point is missing for that
  rung.
- **No k above 7 at batch 1** anywhere. Speedup is still rising at k=7 in
  every bucket, so the optimum is unbracketed from above.
- **No controlled k sweep at any load above 1.** Every k sweep in this table
  ran at concurrency 1. The one load-varying k observation, k=15 against k=7 at
  2K, comes from the disqualified unbounded run 22569403, so optimal k's
  behaviour with load is unmeasured — not merely unconfirmed.
- **No prefix-caching-off control.**
- **No quality evaluation.** Token ids are stored for every run; nothing has
  been scored against LongBench-v2 answers yet.
- **One slow node.** gpub073, job 22757472: `off` 8.8% low with identical KV
  cache allocation and zero preemption. Belongs in the limitations.
