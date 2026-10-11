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
