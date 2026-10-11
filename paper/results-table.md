# Every cell measured

One table for the whole grid: model, hardware, context, batch size, draft
length, and the job and node that produced each number. Derived from
`results/*/results.jsonl` read **per record**, split on `env.slurm_job`.

Two things to state before any number:

1. **Everything here is Qwen3-8B on A40.** The hardware axis (C4) has no data
   in it. The single A100 attempt produced nothing, and the rebuild that fixed
   that has not yet been used for measurement.
2. **Every speedup is a within-job ratio** — each arm divided by the `off` arm
   measured in the same job, same engine session, same node. Ratios replicate
   across nodes to 0.08%; absolute throughput does not (8.8% against one
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
| `ngram` | prompt lookup, no model | n-gram match against the prompt; drafts on only a fraction of steps |
| `dflash` | Qwen3-8B-DFlash-b16 | trained draft head; drafts on ~100% of steps |

Workload: LongBench-v2, bucketed by prompt length into 2K / 8K / 32K.
Engine: vLLM `0.29.1rc1.dev345+gbf13ecc2f`, torch 2.13.0+cu130,
`VLLM_USE_V2_MODEL_RUNNER=0`, prefix caching on.

---

## Headline grid — speedup against no speculation

| context | batch | prompts | `off` | `ngram` k=5 | `dflash` k=7 | job | node |
|---|---|---|---|---|---|---|---|
| 2K | 1 | 50 | 1.00 | 1.15 | **1.86** | 22598608 | — |
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

The two bolded 32K rows at batch 1 are the same cell measured independently
three days apart on different machines. Underlying throughput:

| | job 22713173, gpub019 | job 22714602, gpub047 |
|---|---|---|
| `off` | 14.90 tok/s | 14.83 tok/s |
| `ngram` k=5 | 16.07 | 15.98 |
| `dflash` k=7 | 12.31 | 12.25 |
| n-gram speedup | 1.0785 | 1.0776 |
| DFlash speedup | 0.8262 | 0.8260 |

**0.08% and 0.02% agreement.** This is the inversion — the only place n-gram
beats DFlash, and the only place DFlash loses to doing nothing — and it is
the best-replicated result in the project.

"unbounded" is the 30 Sep run with no concurrency control. vLLM batched as
wide as memory allowed, so those speedups measure the batching decision
rather than the drafter. Kept as the saturation end of the load axis, not as
a valid cell.

---

## Draft-length sweeps, batch 1

| context | rung | k=1 | k=2 | k=4 | k=5 | k=7 | k=15 | job |
|---|---|---|---|---|---|---|---|---|
| 2K | DFlash | 1.20 | 1.47 | — | — | **1.86** | — | 22737716 / 22598608 |
| 2K | DFlash, unbounded batch | — | — | — | — | 0.86 | 0.78 | 22569403 |
| 2K | n-gram | 1.06 | 1.09 | — | 1.15 | 1.14 | — | 22757473 / 22598608 |
| 8K | DFlash | 0.96 | 1.10 | — | — | **1.26** | — | 22737717 / 22685911 |
| 8K | n-gram | — | — | — | 1.15 | — | — | 22685911 |
| 32K | DFlash | 0.76 | 0.78 | 0.81 | — | **0.83** | — | 22714602 |
| 32K | n-gram | 1.04 | 1.13 | — | 1.08 | 1.18 | — | 22757472 / 22713173 |

**Optimal k rises with context and shrinks with load.** DFlash improves
monotonically in k at every bucket, with no turn by k=7; the only place k=15
loses to k=7 is 2K under unbounded batching. The two axes act in opposite
directions, which corrects an earlier claim that they behaved alike.

**Caveat on the 32K n-gram row.** k = 1, 2, 7 come from job 22757472 on
**gpub073**, a node whose `off` throughput was 8.8% below the two other
nodes, and on which speculation appears to help ~7% more than elsewhere.
k=5 (1.08) comes from gpub019 and gpub047. Do not read the k=5 to k=7 step as
a real gain; the k-coefficient for n-gram at 32K is unresolved and
non-monotonic within its own job.

---

## Acceptance

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
essentially every step. Comparing the two rungs on raw acceptance without
this correction overstates n-gram by more than a factor of two.

---

## Cost model

`R(k) = (1 + R_draft) + c·k`, fitted per cell at batch 1. `R_draft` is the
fixed cost of drafting per decode step; `c` is the marginal cost per drafted
token.

| context | rung | R_draft | c | fit |
|---|---|---|---|---|
| 2K | n-gram | **0.071** | 0.0030 | monotonic, clean |
| 2K | DFlash | **0.296** | 0.0092 | monotonic |
| 8K | DFlash | **0.434** | 0.0157 | monotonic |
| 32K | n-gram | ~0.139 | unresolved | non-monotonic, gpub073 |
| 32K | DFlash | **0.662** | 0.0148 | monotonic, 4 points |

Growth across the context range: DFlash 2.24x, n-gram 1.96x. The **absolute
gap** grows 0.225 to 0.523, and that is what flips the ranking — not a
difference in how the two scale. Both costs are sublinear in context: a 16x
longer prompt roughly doubles each, and nothing in the model explains why.

Out-of-sample check: the 2K n-gram fit predicts `R(5) = 1.086`; job 22598608
measured **1.083**, ten days earlier and with k=5 excluded from the fit.

---

## Hardware coverage

| GPU | arch | status | data |
|---|---|---|---|
| A40 | sm_86 | working throughout | everything above |
| A100 | sm_80 | one attempt lost all 9 arms; unblocked 8 Oct by the multi-architecture rebuild, verified by job 22754950 | **none** |
| H200 | sm_90 | kernels present in the rebuilt venv, but `phase0.sh`'s guard allows only `80|86` and aborts | none |

Any A100 or H200 job **must** set
`LADDER_VENV=/projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch`.
Without it the script falls back to a single-architecture build and the job
dies at kernel launch after reserving a node, as job 22685764 did.

---

## What is not in this table

- **No confidence intervals.** Per-request timing is unavailable on this vLLM
  build, so intervals require repeated whole runs. The 32K batch-1 cell is
  the only one with two independent realizations.
- **No 8K n-gram k sweep**, so the middle `R_draft` point is missing for that
  rung.
- **No k above 7 at batch 1** anywhere. Speedup is still rising at k=7 in
  every bucket, so the optimum is unbracketed from above.
- **No prefix-caching-off control.**
- **No quality evaluation.** Token ids are stored for every run; nothing has
  been scored against LongBench-v2 answers yet.
- **One slow node.** gpub073, job 22757472: `off` 8.8% low with identical KV
  cache allocation and zero preemption. Belongs in the limitations.
