# Claims to evidence

Every claim the report will make, the experiment that supports it, and what
result would falsify it. Append-only: if a claim changes, add a dated line rather
than editing the old one, so the record shows what was believed when.

## As of 30 Sep 2026

| ID | Claim | Evidence needed | Status | Falsifier |
|---|---|---|---|---|
| C1 | Draft acceptance decays with prompt context | Acceptance at 2K, 8K, 32K per rung, paired bootstrap over prompts | **Not measured.** Only 2K exists | D within noise of zero across buckets |
| C2 | Different mechanisms decay at different rates | Same, compared at a common draft truncation | Not measured | Rung by log-context interaction not significant |
| C3 | Speedup crosses 1.0x, and the crossing point differs per rung | Speedup against the no-speculation arm across the concurrency axis at fixed context | **Partial.** One uncontrolled high-load point where all rungs are below 1.0x | No rung ever crosses in the measurable range |
| C4 | The crossing point can be predicted on hardware not used for fitting | Cost model fitted on one platform, prediction frozen and tagged, then measured on A40 or A100 | Not started | Median error worse than a factor of two, or no better than naive transfer |
| C5 | n-gram and a trained head are of opposite type on this workload | Per-position acceptance curves at a common k | **Supported at 2K**, single run, no interval | Curves converge once intervals are drawn |
| C6 | Draft length matters: k=7 dominates k=15 for DFlash | Throughput and accepted length at both k, across load | **Suggestive at 2K**, one point | k=15 wins at low concurrency |

## Status as of 7 Oct 2026

Three context buckets measured on A40 (jobs 22598608, 22685911, 22713173), plus
a k sweep and replication (22714602).

| ID | Status now | Evidence |
|---|---|---|
| C1 | **Measured, strongly.** D at k=5, concurrency 1: DFlash 0.219 (8K), 0.402 (32K); n-gram 0.044 (8K), 0.016 (32K) | `notes/2026-10-06-8k-context-decay.md`, `notes/2026-10-06-32k-inversion.md` |
| C2 | **Measured.** DFlash decays 5x faster at 8K and 25x faster at 32K. First-position acceptance 0.601 -> 0.296 against 0.601 -> 0.552 | same |
| C3 | **Measured at all three buckets.** n-gram crosses at 14.4, 8.9, 3.7. DFlash crosses above 32, about 12, below 1 | same |
| C4 | **AT RISK.** A100 is blocked and H200 cannot run, so there is no second platform to predict onto. See below | `env/a100-sm80-failure.md` |
| C5 | **Holds at three buckets.** The curves diverge further with context rather than converging | `notes/2026-10-06-32k-inversion.md` |
| C6 | **Partly refuted.** k=7 does beat k=15 at 2K saturation. But optimal k does NOT shrink with context: at 32K speedup rises monotonically, 0.76x, 0.78x, 0.81x, 0.83x for k = 1, 2, 4, 7 | `notes/2026-10-07-k-sweep-refutation.md` |

### C4 needs saying plainly

C4 is the portability claim, and it is the reason the project separates tau from
R at all. It requires fitting a cost model on one platform and predicting on
another. **There is currently only one working platform.** The vLLM build has no
usable sm_80 kernels, so A100 fails outright, and H200 has no sm_90 cubins.

This is the price of the decision recorded in `TODO.md` to pursue the context
axis instead of rebuilding. It is a real cost and should appear in the paper's
limitations rather than being quietly dropped. A rebuild with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` would restore it; the root cause is known.

## New claims raised by the data

| ID | Claim | Status | Falsifier |
|---|---|---|---|
| C7 | Acceptance reported by serving engines is conditional on a draft being proposed, so comparing rungs on it is invalid without a coverage term | **Measured.** n-gram's coverage is 0.132 at 2K rising to 0.212 at 32K; corrected, its implied R falls below DFlash's, which matches what the hardware predicts | Coverage near 1.0 for all rungs, making the correction vacuous |
| C8 | The ranking of rungs inverts along the context axis | **Measured, three points.** At concurrency 1: DFlash 1.86x / 1.26x / 0.83x against n-gram 1.16x / 1.14x / 1.08x | The ordering holds at 32K, or reverses back at some longer context |
| C9 | Drafter cost is fixed per decode step, not per drafted token, and scales with context for model-based drafters | **Partial.** Measured at 32K: R(k) = 1.662 + 0.0148k, so a 66% fixed cost against 1.5% per position. At 2K only bounded (total R is 1.360 at k=7, so the fixed part is under 0.36) | R_draft flat across buckets, or marginal cost dominant at long context |
| C10 | Speculative decoding perturbs output less than the serving engine perturbs itself across batch sizes | **Measured.** Baseline against itself agrees on 12.7% of sequences; speculative against baseline on 19.0%. Per-token divergence 0.80% against 0.65% | The baseline reproduces itself exactly while speculative arms do not |
| C11 | Run-to-run variance is below 1% | **One replication.** off 14.9 -> 14.8, n-gram 16.1 -> 16.0, DFlash k=7 12.3 -> 12.2, across different jobs and nodes | A second repeat landing further out |

## Predictions that failed

Kept deliberately. A register that records only confirmations is not evidence of
anything.

| Date | Prediction | Outcome | What it taught |
|---|---|---|---|
| 6 Oct | DFlash at 32K is misconfigured; optimal k is 1, giving about 1.17x | **Refuted.** k=1 gives 0.76x, the worst in the sweep; speedup rises with k | A two-parameter cost model was fitted to one data point by assuming the intercept, so the answer came from the assumption. The sweep that tested it cost one hour |
| 6 Oct | DFlash at 32K near 0.95x, n-gram near 1.15x | Direction right, magnitudes off: 0.83x and 1.08x | Linear extrapolation of a two-point decay understates how fast a prediction-based drafter fails. The miss is what exposed C9 |

## Rules

1. No claim moves to "supported" from a single run without a confidence interval.
2. Acceptance is never compared across rungs at different k without truncating to
   a common length first.
3. Nothing is pooled across context regimes (native against YaRN) or across model
   runners. Both are separate populations.
4. A prediction counts only if it was committed and tagged before the measurement
   it predicts.

### Note on rule 1, added 7 Oct 2026

Per-request timing is unavailable on this vLLM build (`0/25` requests), so the
paired bootstrap rule 1 assumes is not possible. Intervals must come from
repeated runs instead.

One replication exists and bounds run-to-run variance below 1%, which is well
under the effects being claimed: the 32K inversion is a 30% gap, roughly thirty
times the noise. That is a bound, not an interval, so under rule 1 the claims
above are **measured** rather than **supported**. Two further repeats of the 32K
cell would close the gap properly.

The 32K n-gram row at concurrency 4 (0.99x) is within about one noise width of
1.0 and should be reported as indistinguishable from no speedup, not as below
it.

---

## Status as of 11 Oct 2026

n-gram swept over k at 2K and 32K for the first time (jobs 22757473 and
22757472), and the A100 blocker cleared (job 22754950).

| ID | Status now | Evidence |
|---|---|---|
| C4 | **UNBLOCKED, not yet started.** Job 22754950 passed all four diagnostic steps on A100 under the rebuilt multi-architecture venv, including the compiled CUDA-graph path that originally failed. There is now a second platform. No measurement cells have been run on it, so the claim itself is still untested — but it is no longer at risk of being impossible | `LOG.md` 8 Oct, `env/gpu-support.md` |
| C6 | **Direction settled.** n-gram's speedup is also monotonically rising in k at both buckets (2K: 1.06x, 1.09x, 1.14x; 32K: 1.04x, 1.13x, 1.18x), with no turn by k=7. Optimal k rises with context and shrinks with load, for both rungs | `notes/2026-10-11-ngram-drafter-cost.md` |
| C9 | **Measured for both rungs, not just DFlash.** Fitting `R(k) = (1 + R_draft) + c*k`: DFlash R_draft 0.296 / 0.434 / 0.662 at 2K / 8K / 32K; n-gram 0.071 at 2K and 0.139 at 32K. The claim's qualifier "for model-based drafters" is **wrong** — see C12 | `notes/2026-10-08-drafter-cost-scaling.md`, `notes/2026-10-11-ngram-drafter-cost.md` |
| C11 | **AMENDED — bound withdrawn.** A third 32K baseline realization came in at 13.5 tok/s against the 14.9 and 14.8 recorded above, 8.8% low. Two of three agree to 0.67%; one does not. An occasional excursion, not a wide distribution. No 32K interval may be computed until the excursion is explained | `notes/2026-10-11-noise-floor.md` |

### New claims

| ID | Claim | Status | Falsifier |
|---|---|---|---|
| C12 | Every drafter pays a context tax, including one that runs no model. Prompt lookup must scan the prompt, so its cost grows with context too — at a similar *relative* rate to a trained head, and a factor of four to five smaller in absolute terms | **Measured at two buckets.** n-gram R_draft 0.071 (2K) to 0.139 (32K), a 1.96x growth against DFlash's 2.24x. Absolute gap 0.225 to 0.523 | n-gram's R_draft flat across buckets, which would restore the "free drafter" account |
| C13 | The inversion is driven by the absolute drafting-cost gap, not by a difference in how the two drafters scale | **Measured, two buckets.** Both costs roughly double across the context range; the gap doubles with them because DFlash starts 4.2x higher | The two costs scaling at materially different rates, making the mechanism a scaling difference rather than a level difference |
| C14 | n-gram's acceptance is invariant to context length | **Measured.** tau_cond at k=1 is 1.670 at both 2K and 32K; first-position acceptance 0.670 at both; D = -0.000, -0.008, -0.007 at k = 1, 2, 7 | Any bucket where n-gram's acceptance profile shifts outside noise |
| C15 | n-gram is *faster* at long context than at short, because coverage rises while acceptance holds flat | **Measured at k=7, single runs.** 1.14x at 2K against 1.18x at 32K; coverage 0.121 to 0.195 | The 32K advantage disappearing under repeats, which given C11 is entirely possible |

### Predictions that succeeded

First entry in this column. Added so the register is not only a list of errors.

| Date | Prediction | Outcome | Why it counts |
|---|---|---|---|
| 11 Oct | n-gram's R at 2K, k=5, should be 1.086, from the fit `R(k) = 1.071 + 0.0030k` over k = 1, 2, 7 | **Confirmed: 1.083**, off by 0.3% | k=5 was not in the fit, and the measurement comes from job 22598608, ten days earlier. Out of sample in both k and job. The same test at 32K misses by 10%, which is the excursion in C11, not the model |

### Note on rule 1, added 11 Oct 2026

Rule 1 is now **binding and unmet at 32K**, and the reason has changed. It is no
longer that intervals are merely absent; it is that the sample we would compute
them from is known to contain an excursion. An interval over a contaminated
sample is worse than an honest "single run", because it presents as rigour.

Order of operations for the 32K bucket, which is where every headline lives:

1. Explain the excursion. Free, from logs on disk. The leading candidate is KV
   cache allocation variance driving preemption, which predicts exactly the
   on/off pattern observed.
2. Fix the configuration if that is the cause, rather than averaging over it.
3. Then repeat the cell, each run to its own output directory, and compute
   intervals.

C15 is the claim most exposed to this, since a 1.14x against 1.18x comparison is
smaller than the excursion that has already been observed once.

---

## Correction, 11 Oct 2026, later the same day

The entry above was written from `summarize.py` output. Reading
`results/a40/results.jsonl` **per record** instead — every record carries
`env.slurm_job` and `env.host`, and has since harness v2 on 5 Oct — overturns
three statements in it.

### C8 is replicated, and has been since 7 Oct

The 15 records at 32K split cleanly by job:

| | job 22713173, gpub019 | job 22714602, gpub047 |
|---|---|---|
| `off` c1 | 14.90 | 14.83 |
| `ngram` k=5 c1 | 16.07 | 15.98 |
| `dflash` k=7 c1 | 12.31 | 12.25 |
| **n-gram speedup** | **1.0785** | **1.0776** |
| **DFlash speedup** | **0.8262** | **0.8260** |

Two independent jobs, two different physical nodes, three days apart, with
speedup ratios agreeing to **0.08%** and **0.02%**.

**C8 is therefore the best-supported claim in the register, not a single run.**
It has been replicated for four days. The reason nobody noticed is that
`summarize.py` keys on `(bucket, conc, rung, k)`, both jobs produce identical
keys, and it keeps one record and prints one row — so a genuine replication was
rendered as a single measurement and the register copied that reading forward.

### C11 is restated, not withdrawn

Withdrawing the bound was an overcorrection. The right statement separates two
quantities that had been conflated:

| quantity | agreement |
|---|---|
| **speedup ratios**, across two nodes | **0.08%** |
| absolute throughput, between those two nodes | 0.47% |
| absolute throughput, against a third node (gpub073, job 22757472) | 8.8% low |

**Ratios are stable to a tenth of a percent across nodes.** Absolute throughput
is stable between two nodes and not against a third. Since every speedup in this
project is a within-job ratio, the stable quantity is the one we report.

On gpub073 the ratio moved too — interpolating its n-gram k=2 to k=7 curve to
k=5 gives about 1.16 against 1.078 elsewhere. So that node was slower at plain
decoding **and** speculation helped more on it, which is the direction a
bandwidth-limited node produces, since speculation trades memory-bound decode
steps for a compute-bound verification pass. Consistent with the R model;
**not established** on one job with no visibility into co-tenancy.

Both earlier explanations for gpub073 are refuted from the logs: the `off` arm
was allocated 173,056 tokens in all three jobs, identical, and there are zero
preempt or recompute lines in any of them.

### C15 is demoted

1.14x against 1.18x is a 4% claim drawn from two different jobs, and its
long-context half is job 22757472 on gpub073 — the one node with a known
anomaly, in the direction that would inflate exactly this number. Demote to
**"n-gram's speedup does not degrade with context"**, which both measurements
support comfortably, and do not claim the improvement.

n-gram's 32K k coefficient is unresolved for the same reason, compounded by
being non-monotonic within its own job.

### Revised rule, replacing the 11 Oct note on rule 1

The order of operations given this morning — explain the excursion, fix the
config, then buy repeats — was built on the belief that the headline rested on
one run. It does not. Revised:

1. **Report ratios, never absolute throughput.** Two orders of magnitude
   difference in stability.
2. **Never compare values across `slurm_job` unless both are ratios** and the
   difference is well clear of a few percent.
3. **Split by `env.slurm_job` before analysing.** `summarize.py` should group by
   job, or at minimum warn on duplicate keys rather than silently keeping one.
   This is the one software fix this episode demands.
4. Repeats remain desirable for a formal interval, but they are no longer
   blocking: C8 has two independent realizations and they agree to 0.1%.

### What this episode cost, and what it teaches

No GPU hours, and three wrong write-ups in one day — two versions of
`notes/2026-10-11-noise-floor.md` and the status entry above. Every one argued
from `summarize.py`'s collapsed table while the per-record data sat in the same
directory with host and job id in it.

The dead-ends table in `LOG.md` says the recurring failure here is trusting a
proxy instead of testing the thing. A derived summary is a proxy for the data,
and it is a more seductive one than a file listing or a traceback, because it
looks like the measurement rather than like a clue.

---

## Second platform, 11 Oct 2026

Jobs 22757102 (A40 gpub067), 22757105 (A40 gpub085) and **22757104 (A100-SXM4-40GB
gpua045)**, run by ssashi on 8 Oct as the quickstart smoke test: 2K, concurrency
1, `LIMIT=5`, seed 1234, `vllm_env_multiarch`. **The project's first A100
measurements.** Write-up in `notes/2026-10-11-hardware-transfer.md`.

| | A40 | A100 | change |
|---|---|---|---|
| DFlash tau_cond | 2.594 | 2.572 | -0.8% |
| n-gram tau_cond | 2.533 | 2.525 | -0.3% |
| DFlash R | 1.358 | 1.433 | **+5.5%** |
| n-gram R | 1.075 | 1.132 | **+5.3%** |
| DFlash speedup k=7 | 1.89x | 1.79x | -5.3% |
| n-gram speedup k=5 | 1.14x | 1.06x | -7.0% |
| `off` throughput | 33.0 | 72.8 | A100 2.21x faster |

### New claims

| ID | Claim | Status | Falsifier |
|---|---|---|---|
| C16 | `tau` is a property of the drafter and the text, not of the machine, and transfers across hardware unchanged | **Measured, one pair.** tau_cond agrees to 0.8% (DFlash) and 0.3% (n-gram) between A40 and A100 at 2K | tau differing by more than a couple of percent on a third platform, or at a longer context |
| C17 | `R` is a property of the machine, and scales by a single factor independent of the rung | **Measured, one pair.** R rises 5.5% (DFlash) and 5.3% (n-gram) on A100 — the same factor for rungs that differ 4x in drafting cost and 7x in coverage | The factor differing materially between rungs, which would mean the machine term is not separable from the drafter term |

C4's status is unchanged — **unblocked, untested** — because a prediction has
not yet been measured. The ingredients now exist.

### Frozen prediction, 11 Oct 2026

Committed under rule 4, **before any A100 cell at full n has been run.** The
only A100 data in existence when this was written is job 22757104 at 2K,
`LIMIT=5`.

Method: take the A40 `R` values at concurrency 1 from the n=50 grid, multiply by
the measured machine factor **1.054** (mean of 1.0552 and 1.0530), and divide the
A40 `tau_eff` by the result.

| cell | A40 R | predicted A100 R | A40 tau_eff | **predicted A100 speedup** |
|---|---|---|---|---|
| 8K, DFlash k=7 | 1.544 | 1.627 | 1.947 | **1.20x** |
| 8K, n-gram k=5 | 1.120 | 1.181 | 1.284 | **1.09x** |
| 32K, DFlash k=7 | 1.766 | 1.861 | 1.459 | **0.78x** |
| 32K, n-gram k=5 | 1.290 | 1.360 | 1.390 | **1.02x** |

**2K is in sample** — the factor was derived there — so it is not a test. Only
8K and 32K count.

**Headline consequence: the inversion is predicted to survive on A100 and widen
slightly**, 0.78x against 1.02x, compared with 0.83x against 1.08x on A40.

**Falsification criterion**, fixed now rather than after the fact:

- the model **passes** if every predicted speedup is within 5% relative of
  measured, and the DFlash-below-1.0 / n-gram-above-1.0 ordering holds at 32K;
- it **fails** if the median absolute relative error across the four cells
  exceeds 10%, or if the 32K ordering does not reproduce.

**Pre-registered confound.** The A100-SXM4-40GB has 40 GB of memory against the
A40's 48. At 32K the concurrency ceiling is therefore lower and a long-context
cell may be in a different memory regime entirely. **A large miss at 32K may be
a capacity artifact rather than a failure of the cost model**, so the prediction
should be judged primarily on 8K. Recording this before the measurement so it
cannot be used as an excuse afterwards.

**Test:** two cells at n=50 on A100, 8K and 32K, concurrency 1, with
`LADDER_VENV` set to the multiarch venv. About 1.5 GPU-hours on Sahil's
allocation, which is almost untouched.

### What a failure would teach

If the single factor misses, the question becomes what else `R` depends on, in
this order of testability: context length (a miss growing with bucket would
point at `R_draft`, a forward pass over the prefix); coverage (a factor that
differs between rungs at long context but not short would mean the machine and
drafter terms are not separable); and memory capacity (the confound above).

---

## External review, 11 Oct 2026 — four corrections and what each caught

An external review of this register on 11 Oct 2026 found four errors, each in a
statement this file was making as established. They are recorded here on the
same principle as the "Predictions that failed" table: a register that keeps
only its confirmations is not evidence of anything. Every dated entry above is
left unchanged. Where an entry above and a correction below disagree, **the
correction governs**.

| # | correction | what it caught |
|---|---|---|
| 1 | C10 restated in the narrow form | a control that varies two things, compared against a treatment that varies one |
| 2 | C15's figures corrected | a demoted claim whose *replacement* wording was also unsupported |
| 3 | Frozen prediction's criterion clarified | a pass/fail gap: no mixed outcome was specified before the measurement it governs |
| 4 | C6 split; the k-versus-load axis is unmeasured | a claim supported only by a run this repository disqualifies |

Items 1, 2 and 4 also appear in `LOG.md` and `paper/results-table.md`, which are
live records rather than append-only registers, and are corrected in place
there. `HANDOFF.md` section 2.5 states item 4 as well and has deliberately
**not** been changed — flagged here so the discrepancy is not rediscovered
from scratch.

### 1. C10 — the strong form is not supported; restated in the narrow form

Claimed on 5 Oct: "Speculative decoding perturbs output less than the serving
engine perturbs itself across batch sizes", from comparing 12.7% (baseline
against itself) with 19.0% (speculative against baseline). **That comparison is
confounded, and the strong form is not established by it.**

`check_lossless.py` compares the `off` arm against the `off` arm **at
different concurrency** — its own output reads `off c=1 vs off c=8`, and it
takes every pair by `combinations(sorted(loaded), 2)` over the `off` arms. The
treatment compares a speculative arm against the baseline **at matching
concurrency** (its other output line). So:

| arm | what differs from its comparator |
|---|---|
| control, `off` vs `off` | engine non-determinism **plus** a change of batch size |
| treatment, spec vs `off` | engine non-determinism **plus** speculation, batch size fixed |

The control varies two things and the treatment varies one. The two rates are
therefore not comparable, and "speculation perturbs output less than the engine
perturbs itself" cannot be read off them.

**What is supported, and all that should be claimed:** changing *concurrency*
perturbs greedy output more than adding *speculation* at fixed concurrency
does. One variable is swapped, on the same prompts, seed and workload. This
says nothing about how much speculation perturbs output relative to a
**fixed-config** repeat, because no such repeat exists.

**The missing control** is two `off` runs at the same bucket, same n, same
concurrency, same seed, written to **separate output directories** so that both
sets of per-request token ids survive. On A40 at 2K n=50 that is two
submissions at ≈0.32 GPU-hours billed (floor: the `off` arm of job 22598608
took 393 s of generation, plus the ≈172 s per-arm engine init measured in job
22757104's log). Until it exists, C10 stays in the narrow form.

**A second defect in the same comparison, found while checking the first.**
The control's `off` per-request files are not from one job. There is exactly
one file per cell key — `perreq_off_2K_k0_n50_s1234_c1.json` — but two jobs
measured that key (22598608 on 1 Oct, 22737716 on 7 Oct), the filenames encode
no run, and the files themselves carry no job identifier (their keys are `id`,
`prompt_tokens`, `gen_tokens`, `sha1`, `finish_reason`, `token_ids`, `timing`).
By the convention in `HANDOFF.md` section 8.5 the later job's file overwrote
the earlier one, so the control's `c=1`-against-`c=8` pair very likely compares
arms from **two different jobs**, adding run-to-run and node variance on top of
the batch-size change already noted. *[Inference from the file inventory plus
the record set: the overwrite cannot be proved from inside the surviving file.]*

The replications this file leans on elsewhere have the same problem for their
**per-request token ids**: 32K 22713173/22714602, 2K 22598608/22737716 and 8K
22685911/22737717 each wrote twice into one output directory, and one file per
cell key survives. Their **aggregate** records do carry `env.slurm_job` and
`env.host` in `results.jsonl` — which is why the speedup-ratio replication
stands — but the token ids do not. The 2K A40 smoke pair (22757102 / 22757105)
is worse off than overwritten: **no per-request file for it exists in this
repository at all** (no `_n5_` file under `results/a40/`).

### 2. C15 — corrected wording, and the numbers

The demoted wording recorded above — "n-gram's speedup does not degrade with
context" — is **wrong**, and the 4%-improvement claim it was demoted *from* was
wrong for a different reason. At matched k=5, split per record on
`env.slurm_job`, with two realizations on each side:

| bucket | n | job | node | `off` tok/s | n-gram k=5 tok/s | speedup |
|---|---|---|---|---|---|---|
| 2K | 50 | 22598608 | gpub036 | 32.563 | 37.628 | **1.1556** |
| 2K | 50 | 22737716 | gpub014 | 32.554 | 37.581 | **1.1544** |
| 32K | 25 | 22713173 | gpub019 | 14.904 | 16.070 | **1.0782** |
| 32K | 25 | 22714602 | gpub047 | 14.832 | 15.981 | **1.0775** |

n-gram **degrades by 6.7%** from 2K to 32K at matched k=5 (1.1556 against
1.0782), and the decline is solid in the sense that the two realizations on
each side agree to ≈0.1%. It is still far less than DFlash, which falls 55.7%
at matched k=7 (1.8640 to 0.8260), and it is nowhere near the 4% *improvement*
claimed before.

**Corrected statement:** n-gram's speedup degrades mildly with context, far
less than DFlash's. Its `tau_cond` is context-invariant (1.670 at both 2K and
32K at k=1, C14) and its coverage rises (0.132 to 0.212), which is why a
drafter with a small fixed overhead degrades so much less than one with a large
fixed overhead does.

**Provenance, and a caveat found while checking.** The 2K second realization is
job 22737716, submitted from `kprobe_32k.sh` with `BUCKET=2K, KLIST="1 2 7"`;
that script's pre-loop is `run off 0` followed by `run ngram 5` before the
`KLIST` loop (lines 120-121), which is what produces an n-gram k=5 arm at every
bucket. Two of `LOG.md`'s recorded submission commands do **not** match the
records, however: it logs `LIMIT=25` for 22737716 and 22737717 while both
records carry `limit 50`, and it logs `KLIST="1 2 7 15"` for 22714602 while
that job's records hold k = 1, 2, 4, 7 and no k=15. The records are the
measurement; `LOG.md`'s replay commands are unreliable in at least these two
places and are corrected in place where the load axis is concerned.

### 3. Clarification of the frozen prediction's criterion, 11 Oct 2026

**Amends "Frozen prediction, 11 Oct 2026" above without editing it.** That text
was written before any A100 cell at full n existed — the only A100 data in
existence at the time was job 22757104 at 2K, `LIMIT=5` — and it is left
byte-for-byte intact for that reason; that is precisely what makes it a
prediction under rule 4 rather than a retrodiction.

As written the criterion has a gap: **pass** requires all four cells within 5%
relative, while **fail** requires the *median* absolute relative error across
the four cells to exceed 10%. One badly missed cell among four can satisfy
neither — three near-exact cells hold the median down while the pass condition
fails on the single miss. No mixed outcome was specified, so this entry
specifies one, before the measurement it governs:

1. **Report per cell, never as a median.** Four rows, each with predicted
   value, measured value, and signed relative error.
2. **8K is the primary judgement**, per the pre-registered confound (40 GB
   against 48 GB, a lower 32K concurrency ceiling, a possibly different memory
   regime). The 8K cells decide whether the single-factor transfer works.
3. **The 32K ordering is a separate pass/fail** in its own right: DFlash below
   1.0 and n-gram above 1.0. Ordering and magnitude are reported separately,
   because a correct ordering with a 20% magnitude miss is a different finding
   from a failed inversion.
4. **One cell outside 5% fails the pass condition.** Whether C4 then fails
   overall depends on the 8K cells and on whether the miss is systematic (both
   rungs, both buckets) or confined to the capacity-limited 32K cell.
5. **The confound is neither an excuse nor a refutation.** If only 32K misses,
   report a magnitude miss in a regime the prediction itself flagged as at
   risk, and say plainly that the 32K cell cannot separate capacity from
   cost-model error.
6. **The prediction is never revised.** The outcome goes in as a dated entry,
   pass or fail, whatever it is.

### 4. C6 — the k-versus-load axis has no valid data

C6's 11 Oct status above reads "Optimal k rises with context and shrinks with
load, for both rungs". **The second half has no valid evidence.** The only
load-varying k observation in the project is job 22569403 (30 Sep, 2K,
uncontrolled): k=15 at 0.78x against k=7 at 0.86x. That run is disqualified by
this repository's own account — `paper/results-table.md`: "vLLM batched as wide
as memory allowed, so those speedups measure the batching decision rather than
the drafter ... not as a valid cell", and `LOG.md` records it as "speedups
void, acceptance usable". A direction read off a disqualified run is not
evidence.

**Restated, splitting the claim in two:**

- **Optimal k rises with context — measured.** DFlash monotonic at 32K over
  k = 1, 2, 4, 7 (0.76x, 0.78x, 0.81x, 0.83x) and at 2K and 8K; n-gram
  monotonic at 2K and 32K; no turn by k=7 anywhere.
- **Optimal k shrinks with load — unmeasured.** There is no controlled
  k sweep at any concurrency above 1 at any bucket. The prescriptive
  consequence the paper wants — rung *and* draft length chosen per request —
  therefore rests on one axis, not two, and the missing one is exactly the
  axis a per-request controller would need.

### C10 addendum, 11 Oct 2026, later the same day — the overwrite is measured

Item 1 above recorded the cross-job overwrite as an inference from the file
inventory. It is now measured directly, from the per-request files themselves:

| `perreq_*_2K_k0_n50_s1234_*.json` | fields present | records with `token_ids` |
|---|---|---|
| `…_c1.json` | `finish_reason`, `timing`, `token_ids` | **50/50** |
| `…_c8.json` | v1 fields only | 0/50 |
| `…_c32.json` | v1 fields only | 0/50 |

The 2K concurrency grid was measured by job 22598608 on **1 Oct**, three days
before harness v2 (5 Oct) began writing `timing` and `token_ids`. Its `c8` and
`c32` arms still carry v1 fields only. Its `c1` arm carries v2 fields. The
surviving `c1` file was therefore written by a later job — 22737716, 7 Oct —
into the same output directory, as section 8.5 of `HANDOFF.md` says it would
be. So the control's `c1`-against-`c8` pair differs in **four** respects, not
two: process, batch size (1 against 8), node (gpub014 against gpub036), and
harness revision (v1 against v2). It is not a within-job comparison at all.

Consequences, all checkable from this repository:

- The recorded control figure of 12.7% is the mean of the three 2K pairwise
  rates, every one of them cross-job: **14%** (`c1` vs `c8`), **10%**
  (`c1` vs `c32`), **14%** (`c8` vs `c32`). Re-running `check_lossless.py` on
  `results/a40` on 11 Oct reproduces them exactly.
- The same tool gives 32K 32/16/24% and 8K 16/16/18%. Agreement is *better* at
  32K, where batch sizes are smaller — the opposite of what a batch-size story
  alone would predict, and further reason to run the clean control.
- Only `sha1` can be compared for these pairs, because the v1 files carry no
  token ids. Position-level divergence is unavailable for the confounded
  pairs even in principle.

`code/c10_control.py` implements the missing control, including the paired
bootstrap over prompts that rule 1 asks for — legitimate here because
agreement is a per-prompt binary and needs no per-request timing, unlike
speedup and tau. Written while testing it: a comparison whose two files have
different bases (one with token ids, one without) **silently returns 0%**,
because a token list never equals a hash string. The tool now refuses to mix
bases and reports which one it used, because a silent zero is exactly the kind
of proxy this project keeps being caught by.

---

## Second review, 11 Oct 2026 (late) — four claims withdrawn

Folds in `paper/claims-corrections-2026-10-11b.md`, which was committed
separately so that it would land before the A100 results were read. That file
stays as the record of what arrived when; this is the register's copy, and the
append-only rule means neither edits an earlier entry. **All four objections
are accepted**, and they concern entries made earlier the same day.

Timing, recorded deliberately: the A100 cells (jobs **22802261** 8K and
**22802262** 32K) were submitted before this was written, and **no A100 cell
at full n had been read**. Withdrawing these now is housekeeping. Withdrawing
them after seeing a 32K miss would have been moving the goalposts.

This entry also **supersedes two statements in the external-review section
above**: item 3's clause weighting 8K on the capacity confound, and item 2's
sentence saying `HANDOFF.md` section 2.5 was deliberately left unchanged. The
rules require the opposite — a withdrawn claim left standing elsewhere is
worse than one never withdrawn — so section 2.5 and the other repeat sites
were corrected in the same commit as this entry, and the list is at the end.

### 5. WITHDRAWN: the capacity confound on the frozen prediction

The frozen prediction carries a pre-registered confound saying that at 32K the
A100's 40 GB against the A40's 48 GB may put the cell in a different memory
regime. **Withdrawn. It fails its own arithmetic.** At concurrency 1 a 32K
cell needs approximately:

| item | size |
|---|---|
| Qwen3-8B weights, bf16 | ~16.4 GB |
| target KV, 33,792 tokens at 144 KiB/token | ~4.9 GB |
| DFlash-b16 drafter weights | ~1.9 GB |
| drafter KV | ~0.7 GB |
| **total** | **~24 GB** |

Against 40 GB. It fits with room, and the prediction is a concurrency-1
prediction. Arithmetic checked here, not taken on trust: 16.4 + 4.9 + 1.9 +
0.7 = 23.9 GB, and 33,792 × 144 KiB = 4.86 GiB.

An escape hatch written into a pre-registration without checking it was
physically possible is worse than no clause at all, because it pre-licences
explaining away a failure. Rule 11 now says so.

**Replacement reason to weight 8K.** The instruction to judge primarily on 8K
survives; the basis changes. The machine factor 1.054 was measured **only at
2K**, and C9 says `R`'s composition changes with context — `R_draft` grows
2.24x from 2K to 32K — so carrying a 2K-fitted factor to 32K is a materially
longer extrapolation than carrying it to 8K. 8K is the nearer test.

**A 32K-specific miss is informative, not excusable.** If 8K lands and 32K
misses, the reading is that the machine factor is **context-dependent** — that
`R` is not purely a machine property, because the drafter's cost over the
prefix may scale differently across GPUs. That weakens **C17**, not merely
C4. It is a result.

### 6. C13 is wrong as stated

C13 claims the inversion is driven by the absolute drafting-cost gap and
**not** by a difference in how the two drafters scale. Tested by holding one
factor at its 2K value and moving the other to its 32K value, on measured
`tau_eff` and `R` for DFlash k=7 at concurrency 1:

| counterfactual | speedup | against n-gram's 1.08x |
|---|---|---|
| 2K acceptance (tau_eff 2.536) with 32K cost (R 1.766) | **1.44x** | DFlash still wins — **no inversion** |
| 32K acceptance (tau_eff 1.459) with 2K cost (R 1.360) | **1.07x** | **tied — inversion on the cusp** |

Arithmetic checked here: 2.536/1.766 = 1.436 and 1.459/1.360 = 1.073.

**Moving cost alone does not flip the ranking. Moving acceptance alone roughly
does.** The cost arithmetic itself stands — `R_draft` does grow 2.24x against
n-gram's 1.96x and the gap does widen 0.225 to 0.523. The error was building a
cost-only causal story while C1 and C2 were already measuring a large
acceptance collapse: DFlash's `tau_cond` falls 2.543 to 1.459 and its
first-position acceptance 0.601 to 0.296, while n-gram's `tau_cond` is flat
and its coverage *rises*. Both channels were in the data; only one went into
the explanation.

**Corrected:** the inversion requires both channels, and the acceptance
channel is the larger contributor. The decomposition is order-dependent, as
any counterfactual is; the asymmetry survives either ordering.

### 7. C17's mechanism is withdrawn; the observation stands

The A100 result was explained in `HANDOFF.md` section 2.7, `LOG.md` and
`notes/2026-10-11-hardware-transfer.md` as a faster machine wasting less
memory bandwidth, so speculation has less to reclaim. That contradicts a
roofline reading:

| GPU | peak bf16 | bandwidth | ridge point |
|---|---|---|---|
| A40 | ~74.8 TFLOPS | ~696 GB/s | ~107 FLOP/byte |
| A100-SXM4-40GB | ~312 TFLOPS | ~1555 GB/s | ~201 FLOP/byte |

The A100 has the higher ridge point, so more arithmetic intensity is needed
before compute binds, leaving *more* headroom to convert bandwidth-bound
decoding into compute-bound verification. Naive roofline therefore predicts
speculation should gain **more** on A100. It gained **less** (DFlash 1.89x to
1.79x, n-gram 1.14x to 1.06x). Ratios check out: 74.8/0.696 = 107 and
312/1.555 = 201.

**The observation stands, the mechanism does not.** Direction measured,
mechanism unexplained. Candidate hypothesis, recorded as a hypothesis: at
batch size 1, fixed per-step overheads — kernel launch, CPU scheduling,
drafter setup — are a larger *fraction* of a faster GPU's step time, which
would inflate `R` on the faster machine with no bandwidth story. Testable by
profiling one step, not by more cells. **The spec-sheet figures above are from
the review and are unverified against NVIDIA's published numbers; that check
is a prerequisite for any mechanism claim in the paper, and it is free.**

### 8. C8's two realizations share one prompt set

Jobs 22713173 and 22714602 both used **the same 25 prompts with the same
seed**. They therefore bound **node variance and run-to-run engine variance**,
which is what they were cited for, and say **nothing about sampling variance
across prompts**. At n=25 on a heavy-tailed quantity like long-prompt latency
that is a real gap. The inversion's margin is ~30% and probably survives it,
but "C8 is replicated" must carry the qualifier: **replicated across nodes and
runs, not across prompt samples.**

**What would close it:** a 32K cell at concurrency 1 on a *different* prompt
subset — a different seed, or a different slice of the bucket. Roughly 1
GPU-hour, and it tests something none of the existing repeats test.

### Where each withdrawn claim was still standing, and what changed

Rule 8: a claim withdrawn in one file but left standing in another is worse
than not withdrawing it. Every repeat site, grepped rather than recalled:

| file | withdrawn claim | treatment |
|---|---|---|
| `HANDOFF.md` §2.1, §4.2 | C8 "replicated" without the prompt-set qualifier | qualifier added |
| `HANDOFF.md` §2.3 | C13 absolute-gap mechanism | corrected to both channels |
| `HANDOFF.md` §2.5, §4.2 | optimal k shrinks with load | marked unmeasured |
| `HANDOFF.md` §2.6, §4.2 | C10 strong form | restated in the narrow form |
| `HANDOFF.md` §2.7, §5, §6.4 | C17 mechanism; capacity confound | mechanism withdrawn, confound withdrawn |
| `paper/results-table.md` | mechanism, absolute gap, confound, "best-replicated" | all four corrected |
| `paper/conclusions.md` | C10 strong form | corrected |
| `TODO.md` | optimal k shrinks with load | corrected |
| `LOG.md` | mechanism, confound, absolute gap | dated entry + inline markers |
| `notes/2026-10-11-hardware-transfer.md` | mechanism, cache-pressure confound | dated banner |
| `notes/2026-10-11-ngram-drafter-cost.md` | C13, k-shrinks-with-load | dated banner |
| `notes/2026-10-11-noise-floor.md` | C8 "best-supported" | dated banner |
| `notes/2026-10-05-losslessness-...md` | C10 strong form | dated banner |

Dated notes keep their bodies: they are records of what was believed when, and
rewriting them would destroy the same information the register exists to keep.
Each carries a banner instead.
