# 16 Nov 2026: the proposed DeltaAI job set (dtai2), why each job exists, and how to verify it

**Status: proposed, not submitted. No GPU hours spent.** Everything here was
tested offline only, against stubs (`code/test_dtai2.sh`, 74 checks, all
passing). Nothing has run on DeltaAI.

**Revised 16 Nov 2026 after an outside review of commit `44e5a02`.** Section
10 lists what the review found and what changed. Where this note and section
10 disagree, section 10 is the later statement. The set waits on Sahil's smoke job
(`env/deltaai.md`, `HANDOFF.md`): submit nothing until that job's rows show
`resolved.use_v2_model_runner: true` and one `resolved.async_scheduling` state.

**Every DeltaAI number is a new baseline.** vLLM 0.33.0 on Model Runner V2,
torch 2.14.0, GH200 (aarch64 Grace CPU, H100 GPU). Never pool a dtai2 row with
a filed A40 or A100 row, and never form a ratio across them.

**This note makes no claims for `paper/claims.md`.** Section 2 lists the
evidence that motivated the design. Some of it is an estimate from a fit and
is labelled as such; none of it has been appended to the register.

---

## 0. Files

| file | status | what it is |
|---|---|---|
| `code/dtai2_bundle.sh` | new | one Slurm job = one bundle of k-sweep cells on one GH200 |
| `code/submit_dtai2.sh` | new | login node: checks, cost estimate, smoke gate, submits only with `GO=1` |
| `code/run_benchmark.v5b.py` | new, superseded | a variant of harness v5 (RULES section 6): `--k-schedule` and a `draft_config` record. No cell uses it now |
| `code/run_benchmark.v5c.py` | new | a variant of v5b: per-request engine timing under vLLM 0.33.0's field names, warm-up on prompts that are not measured, stricter schedule checks. Every dtai2 cell runs on it |
| `code/merge_domains.py` | new | turns `make_domains.py`'s per-domain files into one domain-balanced file per bucket |
| `code/test_dtai2.sh` | new | offline tests, stubs only, about 10 s |
| `code/nsys_v5.py` | new in the repo | runs a v5 harness under Nsight Systems (v5c here, through `LADDER_NSYS_HARNESS`); the profile bundle of `dtai2_bundle.sh` is its only user |

Unchanged, and relied on: `ksweep.sh`, `run_benchmark.v5.py`,
`make_domains.py`, `deltaai_env.sh`, `install_deltaai.sh` (all Sahil's,
commits `a35aba4`, `959187b`, `120feb3`). The bundles call `ksweep.sh` with
`bash`, so its Delta `#SBATCH` lines are inert comments here.

---

## 1. What the set is for

| question | answered by |
|---|---|
| Does adaptive k help at all? (the TA's question) | `c1-*` core: every rung at every k, concurrency 1 |
| What does each drafter cost per step, by length? (the cost model) | `c1-*` core + `pre` + `k0` |
| How does that change with batch size? | `conc-*` + their prefill cells |
| Does the best drafter depend on the kind of text? | `dom-*` |
| Is a per-request router worth building, and what would it learn from? | `c1-*` `loopA` and `acc`, `dom-*` |
| Why is DFlash expensive at 32K? | `profile` nsys cells, `k0` |
| Are the numbers stable? | `rep`, `nopc` (32K), GPU clock logs, DFlash metadata |

---

## 2. The evidence behind the design

Each item names its source so it can be rechecked (section 8 has commands).

**2.1 Prefill dilutes every filed ratio** (from the 15 Nov check of a proposed
C23 text; a fit, not a measurement). `elapsed_s` is prefill plus decode, and
prefill is the same in every arm of a cell. Eliminating prefill gives, for the
filed per-step ratio `R = tau / speedup`,

    R_naive = w * R_true + (1 - w) * tau,     w = decode share of the off arm

so the filed R is pulled toward tau, not toward 1, and its slope against k
mixes cost with acceptance growth. Fitting prefill P, plain step t_d, R_draft
and the per-token slope c jointly to A40 job 22714602 (32K, off plus DFlash
k = 1, 2, 4, 7; five observations, four unknowns, residuals under 0.07 s):

| quantity | joint fit | filed (naive) |
|---|---|---|
| P | about 193 s, 45% of the off arm | not separated |
| t_d | 37.3 ms | 67.4 ms (implied) |
| R_draft | 1.00 (0.96 to 1.01) | 0.673 |
| c | +0.0023 (0.0020 to 0.0044 leaving one arm out) | 0.0143 |

The plain step comes out at 81 to 82% of the A40's 696 GB/s in all three
buckets, and prefill at about 100 to 130 TFLOP/s, both plausible.

**How much the fit can bear (review, 16 Nov).** The numbers reproduce, but
the off row is fitted exactly, so four DFlash points fit three unknowns, and
P is their intercept extrapolated about eightfold. With `spec.drafts` as the
step count instead of generated minus accepted, the same fit gives P 176 s,
t_d 39.9 ms, R_draft 0.96, c 0.0034, with smaller residuals. P and t_d are
supported independently: the harness warms up on the first two prompts with
prefix caching on, so the second timed prompt is decode only, and the engine
log of job 22713173 (the same cell) gives t_d at most 37.4 ms and P about
190 s. Those 190 s cover 23 uncached prompts, about 8.3 s each, not 25.
R_draft and c are not identified: R_draft assumes DFlash's prefill equals the
off arm's (a 20 s gap would make it 0.84), and c changes sign within
residuals of 0.3 s. Quote neither. **Design consequence:** measure prefill directly instead of fitting it. Hence the
`pre` cells (max_tokens 1, so no decode step) at c=1, and `pre1`/`preTop` in
the concurrency jobs.

**2.2 DFlash's per-step cost grows with context far faster than its own reads.**
From the same fit: about 8.6 ms of drafting per step at 2K, about 37 ms at
32K, a slope of about 0.93 us per context token. The drafter's own KV read is
about 0.03 us per token (20 KiB per token at 696 GB/s; the 5 layers x 8 KV
heads x 128 x 2 x 2 bytes is from the C23 text, the checkpoint config was not
opened). So something linear in context runs at a few percent efficiency, or
is CPU work. "Drafting" is a label, not a finding: the quantity is everything
a speculative step costs beyond a plain one, and the n-gram arm, which has no
neural drafter, shows a context-growing excess too (about 2.3, 2.9 and 7.8 ms
per step at 2K, 8K and 32K, roughly 0.18 us per token). **Design consequence:** the `k0` cell splits DFlash's upkeep from
its draft pass (section 4.2), and the profile bundle runs DFlash under nsys at
2K and 32K to name the kernel or CPU gap.

**2.3 Acceptance at one k cannot be inferred from another for DFlash.** Job
22569403 (A40, the same 100 2K prompts, excluded from every c=1 table; the
DFlash arms peaked at 53 and 56 concurrent requests, the 77 was the off and
n-gram arms): at k=15 DFlash accepts more at each of positions 1 to 7 than
at k=7 (position 1: 0.656 against 0.601; positions 1 to 7 summed: 1.666 against
1.520). The filed c=1 rows show the same direction without that job:
position-1 acceptance at k = 1, 2, 7 is 0.570, 0.589, 0.601 at 2K and 0.395,
0.426, 0.451 at 8K; at 32K it is small and not monotone. **Design
consequence:** the acceptance passes run DFlash at k = 2, 4, 7
and 15 instead of extrapolating from k=7.

**2.4 The filed cells are mostly prose question answering.** Every filed cell
reads the first 50 (or 25) lines of the shared bucket file. Code Repository
Understanding is 3 of the 47 scored prompts at 2K and 2 of the 24 at 32K
(`paper/claims.md`, the C22 domain table, which counts scored prompts only). **Design consequence:** the `dom-*` jobs.

**2.5 The filed concurrency cells were short.** At c=16 with 50 prompts there
are about three rounds of requests, and `LOG.md` records sustained concurrency
below the request (c=32 asked, mean 23.5). **Design consequence:** 128 prompts
at 2K and 8K, 48 at 32K, the same prompt set at every concurrency of a bucket.

**2.6 Withdrawn 16 Nov: "no speculative arm has ever been run twice at one
setting" is false.** Ten speculative settings at concurrency 1 were run in two
jobs each, with identical draft counts and elapsed times within 0.03 to 0.8%
(A40 dflash 2K k=7: 210.88 s in job 22598608, 210.94 s in 22737716; A100
dflash 32K k=7: 296.42 s and 296.24 s). What has not been done is a repeat
inside one job, or any repeat on this build. **Design consequence:** `rep`,
last in each `c1` job, is kept for that reason.

**2.7 No filed row has per-request timing.** `per_request_timing` is 0 in 53
rows and absent in 13; `loop_timing` is never true. **Design consequence:**
`loopA`, per-prompt latency for every setting.

**2.8 GH200 shares one power budget between the Grace CPU and the GPU.**
**Design consequence:** every job logs GPU clocks and power every 5 s.

**2.9 On Model Runner V2, method `dflash` can build four different classes**
(section 4.4). **Design consequence:** every job records the checkpoint's
architectures and causal flags and the class vLLM's rule picks.

---

## 3. The jobs

Arm sets. `ksweep.sh` always runs `off` first in every cell, so every speedup
is within one cell of one job. On V2 the rungs are `ngram_gpu` and `dflash`.

| set | arms (plus off) |
|---|---|
| FULL | ngram_gpu k = 1 2 5 7 11; dflash k = 1 2 4 7 11 15 (12 arms) |
| REDUCED | ngram_gpu 5; dflash 7 15 (4) |
| CONCSET | ngram_gpu 2 5; dflash 2 4 7 15 (7) |
| PRESET | ngram_gpu 5; dflash 7 (3) |

### `c1-2K`, `c1-8K`, `c1-32K` (concurrency 1, 50 / 50 / 25 prompts)

Cells in run order. Every results directory below carries its bucket
(`gh200v2_ksweep_2K`, and so on): `ksweep.sh` counts the rows of
`results.jsonl` before and after a cell, so jobs sharing a file would each
report themselves incomplete.

| cell | arms | settings | results dir | answers |
|---|---|---|---|---|
| core | FULL | batched | `gh200v2_ksweep` | best k per bucket; the k sweep |
| pre | REDUCED | max_tokens 1 | `gh200v2_prefill` | prefill per arm, directly |
| loopA | FULL | one prompt at a time | `gh200v2_loopA` | per-prompt latency for every setting: router labels and the per-prompt oracle |
| acc | CONCSET | per-request acceptance | `gh200v2_acc` | per-prompt acceptance by position at the k a router could choose |
| k0 | off + dflash 7 | k schedule `1-1024:0` | `gh200v2_k0` | the cost of carrying an idle DFlash drafter (section 4.2) |
| nopc | REDUCED | prefix caching off, 32K only | `gh200v2_nopc` | caching control |
| rep | REDUCED | identical to core | `gh200v2_rep` | repeatability across the job; with `acc`, the cost of per-request logging |

- `rep` is last so that `rep` against `core` spans the whole job.
- `acc` against `rep` on their shared arms (off, ngram_gpu 5, dflash 7 and 15)
  is the cost of `--per-request-spec detailed`, never measured on any build.
  Until it is, do not read speedups from `acc`.
- Every FULL arm has both a batched (`core`) and a per-prompt (`loopA`)
  timing, so harness v5's rule (compare summed loop time with batched time
  before using loop numbers) can be applied arm by arm.
- **`loopB` is defined but off** (`LOOPB=1` turns it on). It repeats `loopA`
  so that each prompt's winner can be picked on one pass and scored on the
  other, which removes the bias of taking the minimum of noisy timings.
  Decided 16 Nov: run everything once first; add `loopB` only if `rep`
  against `core` differs by more than about 0.5% on some arm, the clock log is
  not flat, or a fifth or more of prompts have their top two settings within
  about 1%. It then runs on its own as three short jobs (section 7).
- The 2K prefix-caching-off cell of the first draft was dropped. The reason
  given at first, that there is nothing to reuse within an arm, was wrong for
  harness v5: it warms up on the first two measured prompts, which are then
  cache hits in the timed run. Harness v5c warms up on two prompts that are
  not measured, which makes the reason true. 32K keeps the cell as the
  control: with v5c, `nopc` and `core` should agree.
- Every prompt count in this section is the measured count. Each arm reads
  two more prompts from the file for its warm-up, so the shared files must
  hold 52 / 52 / 27 prompts for the `c1` jobs and 130 / 130 / 50 for `conc`.

### `conc-2K`, `conc-8K`, `conc-32K` (one prompt set per bucket)

| job | prompts | concurrencies | prefill cells |
|---|---|---|---|
| conc-2K | 128 | 1 4 8 16 32 | `pre1` at c=1, `preTop` at c=32 |
| conc-8K | 128 | 1 4 8 16 | c=1, c=16 |
| conc-32K | 48 | 1 2 4 8 | c=1, c=8 |

CONCSET arms at every concurrency (`gh200v2_conc`); PRESET at max_tokens 1
for the prefill cells (`gh200v2_conc_prefill`). Answers: where speculation
stops paying as the batch grows, the best fixed k at each batch size, and the
per-step cost curve against batch size once prefill is split out. c=16 is
left out at 32K: 16 sequences of about 33K tokens at 144 KiB of target KV per
token is about 78 GB of target KV alone, more than the roughly 65 GB left on
the GH200's 96 GB H100 at `gpu_memory_utilization 0.90` after weights.

Batch-size data here is for uniform batches (every request on the same
drafter). That is the cost model's input. Mixed batches cannot run until a
per-request drafter exists (section 6).

### `dom-2K`, `dom-8K`, `dom-32K` (the domain-balanced file, every prompt)

Built once on a login node (section 7): `make_domains.py --n 50 --n32 25`
(twice its defaults, for router training), then `merge_domains.py`, which
interleaves the domains round-robin into one file per bucket. Up to about 300
prompts at 2K and 8K and 150 at 32K if every domain is that large; a domain
with fewer documents contributes what it has, one with fewer than 10 at a
bucket is skipped there. The real counts are unknown until
`make_domains.py --check` runs. The last two prompts of each merged file are
the warm-up and are not measured.

| cell | arms | settings | results dir |
|---|---|---|---|
| domloop | REDUCED | one prompt at a time | `gh200v2_dom_loop` |
| domacc | CONCSET | per-request acceptance | `gh200v2_dom_acc` |

Every per-request record carries its prompt's `domain`, so the split is done
afterwards. One merged file costs 12 model loads per pass; one cell per domain
(`submit_ksweep.sh DOMAINS=each`) would cost about 72. The 2K and 8K merged
files hold the same documents in the same order, so the length effect can be
read within documents. **Limit:** every LongBench v2 prompt is the same task
(a multiple-choice question, "answer and explain", 256 tokens), so this tests
the kind of text, not the kind of task. These prompts overlap the all-domain
cells' prompts; never pool them.

### `profile` (concurrency 1)

| cell | settings | results dir |
|---|---|---|
| t64, t512 | 32K, 10 prompts, max_tokens 64 and 512; off, ngram_gpu 5, dflash 7 | `gh200v2_outlen_t64`, `_t512` |
| nsys2K | 2K, 3 prompts, max_tokens 64; off, dflash 7, under nsys | `gh200v2_nsys` |
| nsys32K | the same at 32K plus ngram_gpu 5 | `gh200v2_nsys` |

The output-length pair shows whether the 32K result depends on the 256-token
cap, and gives two equations per arm (time = prefill + tokens x per-token
time), a cross-check on `pre`. On harness v5 that cross-check would have
failed for a reason unrelated to the engine: two of the ten prompts, against
two of 25 in `pre`, had a cached prefill. The nsys reports split one step into target
forward, drafter context upkeep, draft pass, verification and CPU gaps. Rows
written under the profiler are never used for speedups. The module that
provides `nsys` on DeltaAI is unverified; the job tries `cudatoolkit`,
`nvhpc`, `cuda`, and if none works it skips the nsys cells and marks itself
incomplete.

### Written by every job

- `logs/dtai2-<bundle>-<job>-gpu.csv`: clocks, power, temperature, utilisation, every 5 s.
- `results/gh200v2_meta/<bundle>_j<job>.json`: the DFlash checkpoint's
  architectures, depth, KV heads, `layer_types`, causal flags, and the
  speculator class vLLM's dispatch rule picks.

---

## 4. vLLM 0.33.0 facts the design relies on

Read from the source at tag `v0.33.0` (commit `4ba787ec`) on 16 Nov. **None of
this has been observed in a run.**

**4.1 The batch-size k schedule.** `num_speculative_tokens_per_batch_size`
(`vllm/config/speculative.py:524`), entries `(start, end, k)`. The scheduler
builds a dense lookup (`vllm/v1/core/sched/scheduler.py:325`) and picks k by
the number of scheduled requests each step (`:1709` onward). On V2 the
speculator still drafts `num_speculative_tokens` and the runner masks the
surplus (`vllm/v1/worker/gpu/model_runner.py:2320`, `:2325`), so a smaller
scheduled k saves verification, not DFlash's draft pass. vLLM drops the
schedule with a warning on the V1 runner for some methods and with data
parallelism (`vllm/config/vllm.py:1088`, `:1110`).

**4.2 k = 0 for DFlash.** With zero scheduled drafts, DFlash stores the
context K/V to stay in sync and skips the query pass
(`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:546-548`). So a DFlash
arm with schedule `1-1024:0` keeps its drafter current and never drafts. That
is the `k0` cell. Its time over the off arm is **not** per-step upkeep alone
(review, 16 Nov). It contains:

- the drafter's context build over the whole prompt at prefill. This is
  `pre(dflash)` minus `pre(off)`; subtract it;
- the target's auxiliary hidden states, produced every step for DFlash;
- the per-step context write, which in `k0` runs eagerly for one row per
  step over every generated token, and in a k=7 arm runs inside the draft
  CUDA graph for 1+k rows over about a third as many steps.

Report it as the cost of carrying an idle DFlash drafter: an upper-bound
diagnostic, not the upkeep term of the cost model. The source accepts the
schedule (`vllm/v1/spec_decode/dynamic/utils.py:52-55` allows 0) and the k=0
branch is the one stock DFlash takes on non-final prefill chunks, so it
should run; no run has shown it. Harness v5c aborts the arm, writing no row,
if the engine drops the schedule, if the schedule cannot be read back, or if
any draft was counted.

**4.3 Not used as a baseline, and why.** In these offline runs the
concurrency is fixed, so a batch-size schedule amounts to one k per
concurrency, which the CONCSET grid already measures. It matters only when
load varies over time, which needs an online benchmark (section 6).
`enable_adaptive_verification` exists (`speculative.py:585`) but is
documented as "Currently only supported for method="dspark"" (`:587`), so it
is not used as a DFlash baseline at this tag. Nothing in the code rejects it
for DFlash; the statement is about what is documented, not what would run. (The documentation on vLLM's main branch lists
DFlash; that is main, not 0.33.0.)

**4.4 DFlash dispatch.** For method `dflash`, `init_speculator`
(`vllm/v1/worker/gpu/spec_decode/__init__.py:28` onward) builds
`XPressSpeculator` if any architecture is in `XPRESS_ARCHITECTURES`
(`speculative.py:98`), `LiLiCorrSpeculator` for `LiLiCorrDraftModel`,
`DFlash2Speculator` for `DFlash2DraftModel` or `DFlash2BeamDraftModel`, else
`DFlashSpeculator`.

**4.5 n-gram and suffix on V2.** `use_ngram()` is true for both `ngram` and
`ngram_gpu` (`speculative.py:2185`), and its only branch builds
`NgramGPUSpeculator` (`__init__.py:87-92`). There is no `suffix/` directory
under `vllm/v1/worker/gpu/spec_decode/` (it holds acceptance_estimator,
adaptive_verification, dflash, dflash2, dspark, eagle, extract_hidden_states,
gemma4, lilicorr, mtp, multi_module_mtp, ngram, standalone_ar,
target_dependent_ar, xpress), so suffix decoding is V1 only at this tag.

**4.6 Two more, from the review.** With `max_tokens=1` a request produces one
token from prefill and no decode step (`scheduler.py:801-815`), and DFlash
skips its first query pass there (`scheduler.py:317-321`, `:535-545`), so
`pre(dflash)` holds the drafter's context build but no draft; n-gram's
`pre` holds a full lookup. `RequestOutput.metrics` is a `RequestStateStats`
(`vllm/v1/metrics/stats.py:259-275`) with `scheduled_ts`, `first_token_ts`
and `last_token_ts`; harness v5 asks for other names and so reports no
timing, harness v5c reads these.

---

## 5. Design decisions

- **Bundles.** The number of jobs a user may run at once on `ghx4` is not
  published (NCSA lists it as TBD; assumed 2 per person). A bundle runs a
  whole bucket's cells back to back on one GH200. Every cell keeps its own off
  arm and its own results directory, so nothing that must not be pooled
  shares a `results.jsonl`.
- **`gh200v2_*` directories.** `submit_ksweep.sh` (the smoke job and its
  sweeps) writes `gh200_*`. Nothing else writes `gh200v2_*`.
- **A harness variant, not an edit.** `run_benchmark.v5b.py` is v5 plus
  `--k-schedule` (or `LADDER_KSCHED`) and a `draft_config` record. With no
  schedule it writes the same rows as v5 apart from the added fields (test
  T2). `run_benchmark.v5c.py` is v5b plus the review's three changes (section
  10), and every cell runs on it; it measures the same prompts as v5 and
  writes the same row fields plus `warmup` and `per_request_engine_timing`
  (test T2c). **The smoke job validated v5, not v5c.** Rerun it on v5c before
  submitting anything (section 7). The schedule is read from the environment because
  `ksweep.sh` has no knob for it, and `ksweep.sh` was left untouched.
- **Domain cells set `LADDER_DATA` themselves.** `ksweep.sh:126` lets an
  exported `LADDER_DATA` replace the per-domain path, and `deltaai_env.sh`
  exports it, so `submit_ksweep.sh DOMAINS=each` would abort on DeltaAI (the
  harness refuses prompts of another domain). The dtai2 domain cells point
  `LADDER_DATA` at the merged directory per cell and do not depend on that
  line. The line itself is not changed here.
- **Run once first.** `loopB` is off by default (section 3).
- **One billing unit per job.** Each job asks for 1 GPU, 16 cores and 64 GB.
  NCSA charges for the number of GH200s needed to cover the requested memory
  or cores, whichever needs more, with about 110 GB and 72 cores per GH200
  (DeltaAI job accounting page, read 15 Nov). Do not raise `--mem` above
  about 110 GB.
- **DeltaAI only.** Both scripts refuse a non-aarch64 machine (Delta),
  `LADDER_CAPS` other than 90, any job path under `/projects`, and a venv
  without `ladder_modules` (without it `ksweep.sh` would load Delta's
  `cray-python/3.12.12`). `LADDER_TEST_HOST=1` exists for the offline tests
  only.

---

## 6. What the set does not cover

- **Real task types.** Summarization, extraction, code editing: the tasks
  where n-gram should differ most. Needs a second workload (Spec-Bench for
  short prompts, LongBench v1 summarization and code completion for long
  ones), about a day to build plus GPU time.
- **Mixed-drafter batches.** Some requests on DFlash, some on n-gram in one
  batch. Needs a per-request drafter in vLLM (Design B) first.
- **Load that varies over time.** An online benchmark with arrivals; that is
  how a router is evaluated, not data to train it.
- **A per-step oracle**, warm against cold drafter state. Both need the
  router implementation.
- **A per-prompt oracle above batch 1.** Per-prompt times are not separable
  when requests share a step; above batch 1 the oracle comes from the cost
  model and must be labelled as modelled.

---

## 7. Cost and how to run

**Estimate.** `submit_dtai2.sh` prints it before submitting anything (RULES
section 7). It starts from the filed Delta A40 seconds per prompt, by bucket
and concurrency, slowest arm, plus 3 minutes of engine start per arm, and
scales by the GH200/A40 time ratio measured from any GH200 off row on disk
(the smoke job's, once it exists). With no GH200 row yet it is an A40-speed
ceiling; walltimes are always requested from that ceiling so that a ratio
measured on five smoke prompts cannot make a job time out. RULES section 7
asks for a floor from a whole job that ran: no GH200 job has run, so this is
the best available until the smoke job lands, and engine start time on GH200
is a guess.

| job | ceiling at A40 speed | walltime requested |
|---|---|---|
| c1-2K | 377 min | 11 h |
| profile | 79 min | 4 h (floor: the profiler's slowdown is unmeasured) |
| c1-8K | 436 min | 12 h |
| c1-32K | 570 min | 16 h |
| conc-2K | 342 min | 10 h |
| conc-8K | 411 min | 11 h |
| conc-32K | 513 min | 14 h |
| dom-2K | 470 min (if 300 prompts) | 13 h |
| dom-8K | 564 min (if 300 prompts) | 16 h |
| dom-32K | 666 min (if 150 prompts) | 18 h |
| **total** | **73.8 GPU-hours** | |

The table now counts each arm's two warm-up prompts. Engine start is 276
arms at an assumed 3 minutes, 13.8 of the 73.8 hours, and the GH200 factor
does not scale it: at 6 minutes the total is about 88 hours, and at GH200
generation speed engine start may be close to half of what is actually
spent. Pass `SMOKE_LOG=<the smoke job's log>` and the submitter measures it
from the arm timestamps. Whose allocation each job comes from (RULES section
7) is open question 3 and must be settled before `GO=1`.

Budget as of 16 Nov: about 200 DeltaAI GPU-hours between Devansh and Sahil,
more on request. `loopB`, if needed later, is about 6.6 more.

**Run** (DeltaAI login node, after the smoke job passes):

```bash
ssh <netid>@dtai-login.delta.ncsa.illinois.edu
cd /work/nvme/<proj>/$USER/Speculative_Ladder/repo/code && git pull
PROJ=<proj> ACCOUNT=<account> source deltaai_env.sh
# the workloads must be under /work/nvme (copy them once on Delta, env/deltaai.md)

"$LADDER_VENV/bin/python3" make_domains.py --src "$LADDER_DATA" --dst "$LADDER_ROOT/data" --n 50 --n32 25 --check
"$LADDER_VENV/bin/python3" make_domains.py --src "$LADDER_DATA" --dst "$LADDER_ROOT/data" --n 50 --n32 25
"$LADDER_VENV/bin/python3" merge_domains.py

# the smoke job again, on the harness the jobs will use (five prompts, minutes):
LADDER_RUN=$PWD/run_benchmark.v5c.py bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7 GO=1

S="SMOKE=<smoke results dir> SMOKE_LOG=<smoke job log>"
bash submit_dtai2.sh $S                                              # plan, cost, smoke check
# 1. the two riskiest cells alone (six arms), and the profile job:
bash submit_dtai2.sh $S BUNDLES=c1-2K CELLS="k0 pre" GO=1
bash submit_dtai2.sh $S BUNDLES=profile GO=1
# 2. read both logs (section 8.5), then the rest of c1-2K and the other c1 and conc jobs:
bash submit_dtai2.sh $S BUNDLES=c1-2K CELLS="core loopA acc rep" GO=1
bash submit_dtai2.sh $S BUNDLES="c1-8K c1-32K conc-2K conc-8K conc-32K" GO=1
# 3. the domain jobs last, once the core cells show which k to carry (their
#    arms are fixed at ngram_gpu 5, dflash 7 and 15; change REDUCED first if
#    the core results point elsewhere), and dom-32K only if some setting
#    beats off at 32K:
bash submit_dtai2.sh $S BUNDLES="dom-2K dom-8K dom-32K" GO=1

# only if section 3's rule calls for it:
bash submit_dtai2.sh SMOKE=<smoke results dir> BUNDLES="c1-2K c1-8K c1-32K" CELLS=loopB LOOPB=1 GO=1
```

Log every job id in `LOG.md`. `c1-2K` gives the first go/no-go reading: how
far a perfect per-prompt choice beats the best fixed setting at 2K. If it
barely does, hold the domain jobs and reconsider the router.

---

## 8. How to verify this (for a reviewing agent)

**8.1 Offline tests.** From `code/`: `bash test_dtai2.sh`. Expect
`passed 74, failed 0`. The tests build stubs (fake `vllm` and `torch`, venv,
`nvidia-smi` answering GH200 and 9.0, `sbatch`, `module`, `nsys`) and run the
real scripts. They cover: syntax; v5b rows equal to v5 rows; every k-schedule
refusal and the abort when the engine drops a schedule; the exact cell list of
every bundle; that no setting leaks from the parent environment or between
cells; every guard; `make_domains.py` plus `merge_domains.py` end to end; the
submitter's plan, smoke gate, GH200 factor and walltimes; and three cells end
to end through the real `ksweep.sh` and harness. They say nothing about vLLM
on a GH200. A useful sanity check is to break something (for example give
`acc` the REDUCED arms) and confirm the cell-list tests fail; they do.

**8.2 Read the code for these, in order.**
1. `dtai2_bundle.sh`, function `cell`: the `unset` list must cover every knob
   any cell sets (CONC LIMIT PC PERREQ LOOP MAXTOK K_NGRAM K_NGRAM_GPU
   K_DFLASH RUNGS LADDER_RUN LADDER_OUT LADDER_KSCHED), and each cell runs in
   a subshell.
2. Each cell's results directory: no two cells with identical run_ids share
   one (run_id carries `_c`, `_loop`, `_nopc`, `_prs`, `_ks`, but not
   max_tokens).
3. `submit_dtai2.sh`: the `sbatch` line (ghx4, 1 GPU, 16 cores, 64 GB), the
   walltime from the ceiling, the smoke check's conditions, the refusals.
4. `run_benchmark.v5b.py` against v5: `diff run_benchmark.v5.py
   run_benchmark.v5b.py` should show only the docstring, the version string,
   `--k-schedule`, `parse_k_schedule`, `k_schedule_tag`, `speculator_for`,
   `draft_config`, the split of `spec_kwargs`, the run_id tag, the
   `spec_k_schedule` read-back and abort, and two added row fields.

**8.3 The vLLM claims of section 4.** In a checkout of `v0.33.0`:

```bash
git clone --depth 1 --branch v0.33.0 https://github.com/vllm-project/vllm.git v033 && cd v033
sed -n 520,527p vllm/config/speculative.py; sed -n 583,588p vllm/config/speculative.py
sed -n 320,330p vllm/v1/core/sched/scheduler.py; sed -n 1709,1716p vllm/v1/core/sched/scheduler.py
sed -n 2305,2330p vllm/v1/worker/gpu/model_runner.py
sed -n 540,555p vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
sed -n 1085,1130p vllm/config/vllm.py
sed -n 14,95p vllm/v1/worker/gpu/spec_decode/__init__.py; ls vllm/v1/worker/gpu/spec_decode/
```

**8.4 The evidence of section 2**, from the filed rows (run in the repo root):

```bash
python3 - <<'EOF'
import json, numpy as np
rows = [json.loads(l) for l in open("results/a40/results.jsonl") if l.strip()]
# 2.3: job 22569403, DFlash k=15 against k=7, same prompts
for r in rows:
    if r["env"].get("slurm_job") == "22569403" and r["rung"] == "dflash":
        a = r["spec"]["acceptance_by_position"]; print(r["run_id"], round(sum(a[:7]), 3), [round(x, 3) for x in a[:3]])
# 2.1: joint fit at 32K, job 22714602. E_off = P + G*t ; E_k = P + N_k*t*(1 + Rd) + N_k*k*t*c
A = {(r["rung"], r["k"]): r for r in rows if r["env"].get("slurm_job") == "22714602"}
G = A[("off", 0)]["gen_tokens"]; X = [[1, G, 0, 0]]; y = [A[("off", 0)]["elapsed_s"]]
for k in (1, 2, 4, 7):
    r = A[("dflash", k)]; N = G - r["spec"]["accepted_tokens"]; X.append([1, 0, N, N * k]); y.append(r["elapsed_s"])
P, t, S, C = np.linalg.lstsq(np.array(X, float), np.array(y), rcond=None)[0]
print("P %.1f s  t_d %.2f ms  R_draft %.3f  c %.4f" % (P, 1000 * t, S / t - 1, C / t))
# 2.7: per-request timing in the filed rows
allr = rows + [json.loads(l) for f in ("results/a100/results.jsonl", "results/a40_c10/results.jsonl") for l in open(f) if l.strip()]
print("per_request_timing:", sorted({str(r.get("per_request_timing", "absent")) for r in allr}),
      " loop_timing true:", sum(1 for r in allr if r.get("loop_timing")))
EOF
```

Expected: k=15 sums to 1.666 and k=7 to 1.520 over positions 1 to 7; the fit
prints about `P 192.7 s  t_d 37.31 ms  R_draft 0.998  c 0.0023`;
per_request_timing is only `0` or `absent`, and loop_timing is never true.
2.4 is the C22 domain table in `paper/claims.md` (search for "Code Repository
Understanding, 2K"). 2.5 is in `LOG.md` (search for "mean 23.5").

**8.5 After the first real jobs (`c1-2K`, `profile`).** Check before
submitting the rest:
- every row: `resolved.use_v2_model_runner` true, one `resolved.async_scheduling` state;
- `gh200v2_meta/*.json`: `speculator_class` is `DFlashSpeculator`, and note `layer_types` and the causal flags;
- `gh200v2_k0_2K`: the dflash row exists, its `resolved.spec_k_schedule` is `[[1, 1024, 0]]`, `spec.drafts` is 0, and its per-request `sha1` values equal the off arm's in the same cell. If the row is missing, read the log: the engine refused the schedule or drafted anyway;
- the startup log's resolved CUDA graph mode is the same for off, k0 and dflash 7;
- every row: `warmup.measured_too` is empty, and `per_request_engine_timing` equals the prompt count. If it is 0, print `vars(outputs[0].metrics)` once and fix the field names before going on;
- per-arm wall time against the plan's engine start; rerun the plan with `LOAD_MIN=<measured>`;
- `gh200v2_prefill`: `gen_tokens` equals the prompt count, and elapsed is well under the core cell's;
- summed per-request `prefill_s` in `core` against the `pre` cell's elapsed, arm by arm: two measurements of the same prefill;
- the GPU clock log is flat;
- `rep` against `core` on the four shared arms, and `acc` against `rep` (logging cost);
- `loopA` summed per-prompt time against `core` elapsed, arm by arm;
- the nsys reports exist, if the module was found.

**8.6 Known unknowns.** Whether vLLM 0.33.0 accepts a schedule of all zeros
for DFlash at runtime; which module provides `nsys` on DeltaAI; whether
`per_request_spec_decode_metrics` costs anything; GH200 engine start time;
the real per-domain counts; how many jobs `ghx4` lets each person run at
once.

---

## 9. Open questions for Sahil

1. `ksweep.sh:126`: an exported `LADDER_DATA` overrides the per-domain path,
   so `submit_ksweep.sh DOMAINS=each` aborts on DeltaAI while
   `deltaai_env.sh` exports it. Fix there, or leave per-domain runs to the
   dtai2 domain jobs?
2. "Domain" here is LongBench v2's `domain` field (your Claude's 12 Nov
   interpretation in the handoff notes). Is that what you meant by
   "different domains"?
3. How to split the job slots: one suggestion is Devansh runs `c1-*` and
   `conc-*`, Sahil runs `dom-*` and, if needed, `loopB`.

---

## 10. Outside review of `44e5a02`, 16 Nov 2026, and what changed

An adversarial read-only review (the brief is
`EVALUATE-dtai2-proposed-jobs.md`) against the repository at `44e5a02`, vLLM
at tag `v0.33.0`, and NCSA's DeltaAI pages. Verdict: run with changes. Read
from source and reproduced against stubs; nothing here has been observed on
DeltaAI either.

**Found, and fixed in this revision.**

1. *Harness v5 discards timing vLLM 0.33.0 provides.* It reads
   `first_scheduled_time`, `first_token_time`, `finished_time`
   (`run_benchmark.v5.py:274`); the tag has `scheduled_ts`, `first_token_ts`,
   `last_token_ts` (`vllm/v1/metrics/stats.py:259-275`). Fixed in v5c:
   per-request `prefill_s`, `decode_s`, `queue_s`, `engine_s`, `ttft_s`.
2. *Warm-up on measured prompts with prefix caching on.* Two prompts per arm
   had a cached prefill in the timed run. Fixed in v5c: warm-up on the two
   prompts after the limit; the row records it.
3. *Three jobs per results directory.* Reproduced with stubs: two bundles at
   once both exit 1 ("expected 12 new rows, found 24"). Fixed: the bucket is
   in every directory name.
4. *`k0` unasserted and mislabelled.* Fixed: v5c aborts on a failed read-back
   and on any draft; sections 3 and 4.2 reworded.
5. *Settings leaking from the submitting shell.* `SEED`, `TEMP`, `PLMIN`,
   `PLMAX` reached every cell (reproduced: rows `..._s999_c1_T0p7_pl3-9`).
   Fixed: cleared per cell; `LADDER_RESULTS` joins the `/projects` check.
6. *Submitter.* Engine start measured from the smoke log; the smoke gate also
   checks the vLLM version and that speculative rows drafted; the profile
   job's walltime has a 4 h floor; the GH200 factor accepts a GPU named H100.
7. *Tests.* Two mutations had passed unnoticed (five names dropped from the
   `unset` list; the walltime multiplier). Both now fail a test.
8. *This note.* 2.1 (what the fit can bear), 2.2, 2.3, 2.4, 2.6 (withdrawn),
   4.2, 4.3, 4.6, the cost table and the submission order.

**Found, not fixed here.**

- `loopB`'s trigger compares `rep` with `core`, an aggregate over 50 prompts;
  it says nothing about per-prompt noise, and with 12 arms on a smooth k
  curve "top two within 1%" will almost always hold. A check that costs
  nothing: by `make_domains.py`'s selection rule every `loopA` prompt is
  also in the domain files, so `domloop` re-times them in another job at
  four settings. Compare per-prompt speedups there before paying for `loopB`.
  The off arm is also run three times per `c1` job (`core`, `k0`, `rep`).
- Above batch 1 the router's value is modelled from uniform batches. Enough
  for the course if stated; a paper needs arrivals and mixed batches.
- The domain jobs are 39% of the estimate and vary the text, not the task.
- 32K prompts beyond the first 25 have not been checked against
  `max_model_len` 33,792 (the longest of the first 25 is 33,189 tokens, so
  512 output tokens fit with 91 to spare).
- `env/deltaai.md` has no step for copying the workloads (only
  `deltaai_env.sh` prints one) and no checksum against the filed files.
- Which module provides `nsys` on DeltaAI; the bundle's `module load`
  attempts persist in the job shell.
- With `VLLM_USE_V2_MODEL_RUNNER=1` the config returns the environment value
  directly (`vllm/config/vllm.py:748-750`), so the harness's "engine resolved
  V1" abort cannot fire. Read from source by the review, not rechecked.

---

## 11. Addendum, 16 Nov evening: pieces of 1 h

The first two real jobs (3612400, `c1-2K` cells `k0 pre`, and the smoke job
3612174) passed every check of 8.5 that applies to them; `LOG.md` has the
numbers. Two unknowns of 8.6 are settled: vLLM 0.33.0 accepts an all-zero
schedule for DFlash, and engine start on the GH200 is about 3 minutes per arm.

The queue settled a third the wrong way. After 13 hours on `ghx4`, with 2747
jobs waiting, none of ours asking for 2 h or more had started, while both 1 h
jobs had started within 20 minutes. So the bundles can now run as pieces:

- `PIECE=<tag>` suffixes every results directory of the job with `_p<tag>`;
  `ONLY_NGRAM_GPU` and `ONLY_DFLASH` keep only some k values of each cell
  (`none` drops the rung). A cell left with no speculative arm is skipped.
- `WALL_H` sets the walltime in whole hours and warns when the estimate is
  over 75% of it.
- `dtai2_pieces.sh` lists 46 pieces covering every c1 and conc cell not yet
  run, each under 40 minutes at the 2K factor of 0.199.

What this changes in the design. Every piece runs its own off arm, so each
speedup still divides by an off arm from the same job and GPU (about 1.5
GPU-hours more in all). `rep` no longer shares a job with `core`, so "rep
against core" becomes a repeat across jobs and nodes, which is the harder
test. The analysis must read `gh200v2_<cell>_<bucket>` and
`gh200v2_<cell>_<bucket>_p<tag>` as one cell, joining each piece's arms to
its own off arm. The 32K factor is still unmeasured; the `prek0` piece of
`c1-32K` measures it, and at a factor of 0.35 the largest 32K piece is about
50 minutes.
