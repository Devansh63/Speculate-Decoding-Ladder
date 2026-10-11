# Decision tree for the A100 prediction test, and the plan to 20 Oct

Written 11 Oct 2026, **before any record from jobs 22802261 (A100 8K) or
22802262 (A100 32K) was read.** The point of writing it now is that the
response to each outcome is committed in advance rather than fitted to the
numbers. Where this note says "read as", that reading is binding; where it
says "then run", that is a plan and may be reordered for queue reasons.

## 0. What constrains the plan

GPU-hours are **not** the constraint. More are available on request, and a
request for about 30 should go in now (section 6). The constraints are:

1. **Wall-clock.** Mid-semester review is 20 or 22 Oct. **The last three
   days before 20 Oct are reserved for the write-up.** Measurement stops at
   the end of 17 Oct. Anything unmeasured by then is out of scope for the
   review, and the plan below is ordered so the highest-value cells land
   first.
2. **Queue wait.** Hours on A100; longer on H200 (8 nodes, 1 per job).
   Submit long-queue jobs first and in parallel.
3. **Analysis and bookkeeping.** Every job costs a `LOG.md` entry, a
   `results-table.md` row and per-record scoring. This is the real
   bottleneck once hours are not, so the per-record analysis tool (section
   5, item 2) is built before most of the jobs are submitted.

The deliverable is a document, not a grid.

## 1. Scoring, free, done first in every branch

From `results/a100/results.jsonl`, per record, split on `env.slurm_job`:

1. Confirm both jobs added records. `COMPLETED` is not evidence.
2. Four cells, each with predicted value, measured value and signed relative
   error. Never a median. Frozen values: 8K DFlash k=7 **1.20x**, 8K n-gram
   k=5 **1.09x**, 32K DFlash k=7 **0.78x**, 32K n-gram k=5 **1.02x**.
3. `tau_cond` and `tau_eff` on A100 against the A40 n=50 cells at 8K and 32K.
   This extends C16 beyond 2K.
4. `off` throughput ratio A100/A40 at 8K and 32K against the bandwidth ratio
   2.23. This extends C18 beyond 2K.
5. The machine factor `R_A100 / R_A40` per rung per bucket. The frozen
   prediction assumed 1.054 everywhere.

The outcome is appended to `paper/claims.md` as a dated entry. The
prediction is never edited.

## 2. Branches, and how each is read

Pass is within 5% relative per cell. The 32K ordering is DFlash below 1.0x
and n-gram above. These four cover every outcome.

| Branch | Definition | Read as (binding) |
|---|---|---|
| **A** | both 8K cells within 5%; 32K ordering holds; both 32K cells within 10% | C4 passes for speedup on one platform pair at 8K and 32K. The 16K and H200 tests (section 4) are then attempts on harder targets, not confirmations |
| **B** | both 8K cells within 5%; 32K ordering fails or either 32K cell misses by more than 10% | The single-factor transfer breaks with context. Split by the section 1 numbers: **B-tau**, A100 `tau_cond` at 32K differs from A40 by more than 3%, so C16 fails at long context and the miss is in acceptance; **B-R**, tau holds and the factor is the same for both rungs but is not 1.054, so the machine factor is context-dependent and C17's "one factor" is restated as "one factor per context"; **B-sep**, the factor differs between rungs, so C17 is falsified as stated and the machine and drafter terms do not separate |
| **C** | either 8K cell misses by more than 10% | **The frozen prediction failed.** Recorded as such and never rescored, even if a corrected factor would have hit. The A100 2K n=50 cell (section 5) decides whether the input (a factor from 5 prompts) or the model's structure was wrong. Any further test of C4 is a new cell with a new frozen prediction |
| **Grey** | any 8K cell between 5% and 10% | Partial pass. The 5-prompt factor is named as the likeliest cause, and the A100 2K n=50 cell moves to the front of the queue |

Within B-R the sign is read now: A100 **better** than predicted (factor
below 1.054 at long context) is consistent with the fixed-per-step-overhead
hypothesis, since a fixed overhead is a smaller share of a longer step. A100
**worse** than predicted counts against it.

## 3. Disjoint prompt subsets: intervals without per-request timing

Rule 1 of the register has been unmet because per-request timing is
unavailable on this build. It was never needed. An interval needs
independent samples, and the prompt dimension was unused: every run so far
reads the **first `LIMIT` lines** of the bucket file (`run_benchmark.py`,
`build_prompts`), and `SEED` reaches only `SamplingParams`, which is inert
under greedy decoding. A "different seed" repeat re-measures the same prompts.
The `SEED=5678` item in `TODO.md` is struck for that reason.

The buckets hold 503 documents at 2K and 8K and 387 at 32K. So:

- **Five disjoint subsets per bucket**, n=50 at 2K and 8K, n=25 at 32K.
  Subset 0 is the existing measurement (lines 1 to 50, or 1 to 25). Subsets
  1 to 4 are the next slices.
- Each subset is a directory holding the file under the canonical name,
  because `phase0.sh` builds the path as `$DATA/longbench_v2_${BUCKET}.jsonl`:
  `LADDER_DATA=/projects/bikd/$USER/Speculative_Ladder/data/sub1`, and so on.
  `results.jsonl` records the full `dataset` path, so provenance survives.
- Each subset gets its own `LADDER_OUT`, per RULES section 5.
- One job per (bucket, subset), concurrency 1, three arms, via `phase0.sh`.
  Cost floor per job from the nearest whole job: 2K 0h33 (22737716), 8K
  0h42 (22737717), 32K 1h16 (22713173). Twelve jobs, about 10 A40-hours.
- Interval: for each ratio (speedup, `tau_eff`, implied `R`), the spread
  across the five subsets. With five points report min, max and the
  percentile bootstrap over subsets; call it what it is, five samples.

**This is extensible.** A sixth subset is one more directory and one more
job; nothing already measured is redone, and the analysis tool groups by
`dataset` path, so a new subset appears as a new row.

What it moves: C8, C1, C2, C5, C14 from "measured" to "supported" under rule
1, if the intervals hold the ordering. C12 and C13 need k sweeps per subset
and stay "measured".

What it does not do: it does not test C10's strong form, which needs two
runs of the **same** prompts in the same configuration (section 5, item 3).
Different questions.

A confound the subsets also expose: the existing 2K, 8K and 32K cells
**share documents**. All 25 ids in the 32K cell are among the 2K cell's 50,
and the 2K and 8K lists are identical in order. The 2K and 8K figures
therefore average 25 documents that are absent from 32K. Recorded in
`paper/threats.md`.

## 4. Two serial chains with frozen predictions

### 4.1 The 16K bucket: locating DFlash's break-even

Under C17 the machine factor is shared by both rungs, so it cancels in the
ratio of their speedups: **the context where the rungs cross is predicted to
be hardware-invariant.** What moves between machines is each rung's
break-even, where its own speedup crosses 1.0x. That is the proposal's
primary metric, and DFlash's break-even on A40 is somewhere between 8K
(1.26x) and 32K (0.83x), a 4x interval.

1. Build a 16K bucket on a login node (free) from a copy of
   `build_workload.py` with `TARGET_LENGTHS` extended; own data directory.
2. A40 16K, concurrency 1, n=25, three arms. Floor 1h16 (22713173, the
   nearest bucket). Brackets the break-even to 2x and adds a fourth point to
   `R_draft` against context.
3. **Freeze** the A100 16K prediction in `paper/claims.md`: A40 16K `R`
   times the machine factor measured in section 1 (1.054 if branch A, the
   context-appropriate factor if B-R), `tau_eff` from A40. Commit.
4. A100 16K, same cell. Floor: 22802262's elapsed, times 2 for billing.

Not worth running if step 2 shows DFlash at 16K within noise of 1.0x on
A40, because then the break-even cannot be located more finely than the
bucket spacing on either machine and step 4 would not discriminate.

### 4.2 H200: the strongest test of C4, and it waits for the intervals

Bandwidth is the axis that exists (0.70, 1.56, 4.8 TB/s; the
compute-to-bandwidth ratios are all 201 to 215, register, "Ratio axis").
The model for `R` on a third machine is

    t_step = bytes / bandwidth + overhead

per rung, two unknowns fitted from A40 and A100. **Two parameters on two
machines has zero residual by construction**, so a fit from the point
estimates would extrapolate with no uncertainty and a miss would be
uninterpretable. Therefore:

1. The H200 fit **waits for the subset intervals** (section 3). Each subset
   gives one A40 and, once the A100 subsets exist, one A100 value; fit per
   subset, and the spread of fitted parameters propagates to a **predicted
   range** for H200 `R` per rung per bucket.
2. Freeze the range in `paper/claims.md`, with the rule: pass if the measured
   H200 `R` lies inside the range; fail if outside; and the range's width is
   reported with it, because a range wide enough to contain anything is not
   a prediction.
3. Only then run H200 cells.

Independent of the fit, two H200 steps can start now because the queue is
long: (a) a `phase0.sh` variant whose capability guard admits `90`, built
with `sed` and verified with `grep`, per RULES section 6; (b) an H200
diagnostic modelled on `diag_a100.sh`, because the multiarch venv contains
sm_90 kernels but has never executed on an H200 and "a listing is not proof
of execution" is already in the dead-ends table. Floor for (b): 11 min
(22754950), times 6 for billing.

## 5. Unconditional work, in order

Free, login node, before the GPU queue:

1. **Charge factor.** `jobcharge 22757104` and `jobcharge 22757102`: an A100
   and an A40 job of near-identical length (19:54 against 20:07). Their ratio
   is the billing ratio, and the A40 figure shows whether A40 bills at 0.5.
   If it does, every A40 cost in `LOG.md` is double the true charge.
2. **Per-record analysis tool** (`code/analyze.py`): groups by
   `env.slurm_job` and `dataset`, computes within-job speedups, `tau_eff`,
   coverage and implied `R` with the same arithmetic as `summarize.py`,
   scores a frozen prediction, and reports subset spreads. **And fix
   `summarize.py` itself** to group by job and refuse to silently collapse
   duplicate keys, per RULES section 8; a second script does not repair the
   first.
3. Slice the subset files and the 16K bucket; verify the subsets are
   disjoint by `_id` before any job is submitted.
4. **Quality evaluation, feasibility then scoring.** "Lossless by
   construction" is the assumption C10 measured and refuted: 19% sequence
   agreement, and eager against compiled disagreeing on the same seed. With
   outputs differing on four fifths of prompts, whether they differ in ways
   that change the answer is live. Token ids are stored for every v2 run and
   LongBench-v2 is multiple choice. Feasibility: decode the stored ids and
   check whether an answer letter appears before the 256-token cap, since
   the dead-ends table records that every generation hits the cap. If
   answers are extractable, score every arm and report accuracy per rung per
   bucket alongside the agreement rates. If none are, record that and the
   reason, and the item closes.
5. Push `kprobe_ngram.sh` and `diag_a100_multiarch.sh` from Delta.
6. Request about 30 GPU-hours (section 6).
7. The H200 guard variant (section 4.2).

GPU, value independent of the A100 outcome, submit in this order:

1. **H200 diagnostic** (longest queue; section 4.2b).
2. **A40 subsets**, twelve jobs (section 3). Queue them all; they are
   independent.
3. **A40 16K** (section 4.1, step 2).
4. **Same-config `off` against `off` control**, `code/c10_control.sh`. Floor
   0h20 (22757102). It is what C10's strong form requires, and the subsets do
   not answer it.
5. **A100 2K at n=50**, three arms. Floor 0h20 (22757104, same config at n=5)
   times 2. C16 and C17 rest on five prompts whatever 8K says; in branch C
   and the grey zone this moves to the front.
6. **A100 k sweeps** at 8K and 32K, exact copies of 22737717 and 22714602.
   Floors 0h42 and 1h07, times 2. C9 on a second machine, and inputs to the
   H200 fit.
7. **A100 subsets**, the same twelve cells on A100, needed for the per-subset
   H200 fit (section 4.2). Floors as the A40 subsets, times 2.
8. **32K n-gram k sweep off gpub073** and **8K n-gram k sweep**. Cheap in
   wall-clock in parallel, and they complete n-gram's `R_draft` curve. Low
   priority: C12 is secondary since the C13 correction.

Then, in branch order: the frozen 16K prediction and A100 16K (4.1), the
frozen H200 range and H200 cells (4.2), and in branch B only, a **step
profile** on A40 and A100 at 32K, `off` and DFlash k=7, to test the
fixed-overhead hypothesis directly. The profile needs a script that does not
exist; gate it on profiled and unprofiled `off` throughput agreeing to a few
percent, or the step ratios are meaningless.

## 6. Hours

| Block | A40-equivalent hours (A100 at 2x, H200 at 6x, pending item 5.1) |
|---|---|
| A40 subsets, 12 jobs | ~10 |
| A40 16K, C10 control | ~2 |
| A100 2K n=50, k sweeps, 16K | ~8 |
| A100 subsets, 12 jobs | ~20 |
| H200 diagnostic and 3 cells | ~25 |
| **Total** | **~65** |

Sahil has about 7 after the two in-flight jobs; Devansh about 4, held in
reserve. Request 30 now and more once the A100 subsets are scheduled; the
H200 block is the first to drop if the request is slow. Every floor above
is quoted from a whole job that ran; none is a per-arm extrapolation.

## 7. Not doing, and why

- **Repeats on the same prompts.** They re-measure node and engine noise,
  already shown at 0.1% on ratios. Information about prompt-sampling
  variance comes only from disjoint subsets.
- **Prefix-caching-off control.** Every prompt is distinct, so prefix
  caching barely touches speedup. It bears only on the cause of the
  non-determinism floor, which no headline claim depends on.
- **EAGLE-3, the controller, YaRN, the 27B target.** Each costs setup and
  attention before 17 Oct that the measurement cutoff does not allow. Hours
  do not fix that.
- **Fitting H200 from point estimates.** Section 4.2.
