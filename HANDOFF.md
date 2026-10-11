# Handoff: The Speculation Ladder

Everything needed to pick this project up. Written 11 Oct 2026.

**If you are an AI agent being briefed on this project, read this file first,
then follow the read order in section 3. Do not start from the Google Drive
documents — see section 3.2 for why.**

---

## 1. The project in one page

**Course** CS 598, Hardware & Software for AI, UIUC, Fall 2026.
**Team** Devansh Agrawal (`dagraw2`), Sahil Sashi (`ssashi`).
**Compute** NCSA Delta, project `bikd`, account `bikd-delta-gpu`.

### The question

Speculative decoding makes a small "drafter" propose k tokens and has the
large target model verify them in one forward pass. When drafts are accepted,
you get several tokens for the price of one pass. The literature reports
speedups at batch size 1 on short prompts. **We are asking where those
speedups go when you change the operating point** — more concurrent requests,
longer prompts, different hardware.

### The model everything hangs on

```
Speedup  ~=  [ cov * tau_cond + (1 - cov) ]  /  R
```

- **tau_cond** — mean accepted draft length, *conditional on a draft being
  proposed*. A property of the drafter and the text.
- **cov** (coverage) — fraction of decode steps on which a draft was proposed
  at all. Serving engines report acceptance conditional on a draft existing,
  so without this term you overstate a low-coverage drafter by more than 2x.
  Discovering this was the first real finding (claim C7).
- **R** — verification cost ratio: the cost of one verification pass relative
  to one plain decode step. A property of the machine and its load.

`R` decomposes further:

```
R(k)  =  (1 + R_draft)  +  c * k
```

- **R_draft** — fixed cost of drafting, paid every decode step regardless of k.
- **c** — marginal cost per drafted token.

The whole study is an argument that **tau belongs to the drafter and R belongs
to the machine**, so you can measure acceptance once and predict performance
elsewhere. As of 11 Oct that split is measured, not assumed (section 4.3).

### The three "rungs"

| rung | what it is | drafts on |
|---|---|---|
| `off` | no speculation; the baseline every speedup divides by | — |
| `ngram` | prompt lookup: match the recent prefix against the prompt. No model. | ~13–21% of steps |
| `dflash` | Qwen3-8B-DFlash-b16, a small trained draft head | ~100% of steps |

Target model is **Qwen3-8B** throughout. Workload is **LongBench-v2**,
bucketed by prompt length into **2K / 8K / 32K**.

---

## 2. The headline results

### 2.1 The ranking of speculation methods inverts with context length

Speedup against no speculation, A40, concurrency 1:

| context | `ngram` | `dflash` |
|---|---|---|
| 2K | 1.15x | **1.86x** |
| 8K | 1.15x | 1.26x |
| 32K | **1.08x** | **0.83x** |

At short context the trained head wins by a wide margin. At 32K it is *worse
than not speculating at all*, while the cheap n-gram method still helps.
**This is the project's main result, and it is replicated**: two independent
jobs on two different physical nodes, three days apart, agree to 0.08% and
0.02% on the speedup ratios.

### 2.2 The mechanism: context taxes drafting, not verification

Fitting `R(k) = (1 + R_draft) + c*k` at concurrency 1 on A40:

| context | rung | R_draft | c |
|---|---|---|---|
| 2K | n-gram | 0.071 | 0.0030 |
| 2K | DFlash | 0.296 | 0.0092 |
| 8K | DFlash | 0.434 | 0.0157 |
| 32K | n-gram | ~0.139 | unresolved |
| 32K | DFlash | 0.662 | 0.0148 |

The fixed drafting cost grows 2.24x from 2K to 32K for DFlash; the marginal
cost per drafted token is flat. **Drafting is 82–86% of DFlash's total
overhead.** Long context makes the drafter's own forward pass over the prefix
expensive; it does not make verification expensive.

### 2.3 Even the "free" drafter is not free

We assumed n-gram's `R_draft` was ~0 because it runs no model. Measured, it is
0.071 at 2K and ~0.139 at 32K — it has to *scan the prompt*, and a 32K prompt
is 16x more text. Both drafters pay a context tax at a similar **relative**
rate (1.96x against 2.24x). **The inversion is driven by the absolute gap**
between the two drafting costs (0.225 at 2K growing to 0.523 at 32K), not by
different scaling.

### 2.4 Acceptance profiles are qualitatively different

Acceptance by draft position, 32K, concurrency 1:

```
dflash k=7   0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004
ngram  k=7   0.541, 0.382, 0.304, 0.256, 0.219, 0.193, 0.177
```

DFlash's drafts are worthless past position 2. n-gram's still pay at position
7. And n-gram's acceptance is **invariant to context length** (tau_cond 1.670
at both 2K and 32K at k=1) while its **coverage rises** (0.132 → 0.161 →
0.212), because a longer prompt offers more places for a lookup to hit.

### 2.5 Optimal k rises with context and shrinks with load

These two axes act in **opposite** directions. An earlier claim that they
behaved alike was retracted. DFlash's speedup rises monotonically in k at
every bucket with no turn by k=7; the only place k=15 loses to k=7 is 2K under
unbounded batching.

### 2.6 "Lossless" is only meaningful relative to the engine's noise floor

Comparing output token ids: the `off` arm against **itself** at different
concurrency, with no speculation anywhere, agrees on only **12.7%** of
sequences. Speculative arms agree with the baseline on **19.0%**. Per-token
divergence 0.80% against 0.65%. **Speculation perturbs output less than the
engine perturbs itself.** The eager and compiled execution paths also
disagree under greedy decoding with the same seed.

### 2.7 tau transfers across hardware; R does not

First paired A40/A100 measurement, 2K, concurrency 1:

| | A40 | A100 | change |
|---|---|---|---|
| DFlash tau_cond | 2.594 | 2.572 | **-0.8%** |
| n-gram tau_cond | 2.533 | 2.525 | **-0.3%** |
| DFlash R | 1.358 | 1.433 | **+5.5%** |
| n-gram R | 1.075 | 1.132 | **+5.3%** |
| `off` throughput | 33.0 tok/s | 72.8 tok/s | A100 2.21x faster |

Acceptance transfers to under 1%. The cost ratio shifts 5.5%, **and by the
same amount for both rungs** — rungs that differ 4x in drafting cost and 7x in
coverage. That is the signature of a machine property. The A100 is 2.2x faster
and speculation pays *less* there, which is the expected direction:
speculation reclaims wasted memory bandwidth, and a faster machine wastes
less. **Caveat: n=5.**

---

## 3. Read order

### 3.1 This repository — the live record

Read in this order. Everything here is current.

| # | file | what it is |
|---|---|---|
| 1 | `HANDOFF.md` | this file |
| 2 | `paper/results-table.md` | every cell ever measured, with the job id and node that produced it. Start here for "what do we know". |
| 3 | `paper/claims.md` | **the most important file.** Append-only register of every claim, its evidence, its falsifier, and its status. Includes failed predictions, successful ones, and the frozen prediction now under test. Read it start to finish; the corrections matter as much as the claims. |
| 4 | `LOG.md` | chronological activity record, newest first. What was run, what broke, the exact commands, the dead-ends table and the budget. |
| 5 | `notes/2026-10-11-hardware-transfer.md` | tau/R separation across GPUs |
| 6 | `notes/2026-10-11-ngram-drafter-cost.md` | n-gram's drafting cost |
| 7 | `notes/2026-10-11-noise-floor.md` | measurement reliability; read the "how this note was wrong twice" section |
| 8 | `notes/2026-10-08-drafter-cost-scaling.md` | R decomposition |
| 9 | `notes/2026-10-07-k-sweep-refutation.md` | a prediction that failed, and why |
| 10 | `notes/2026-10-06-32k-inversion.md`, `notes/2026-10-06-8k-context-decay.md` | the context axis |
| 11 | `notes/2026-10-05-losslessness-and-engine-nondeterminism.md` | output stability |
| 12 | `notes/2026-10-01-a40-2k-concurrency-sweep.md` | the load axis, and the founding result |
| 13 | `notes/teammate-quickstart.md` | how to run a job. Read before running anything. |
| 14 | `env/gpu-support.md`, `env/a100-sm80-failure.md` | the hardware situation and how it broke |
| 15 | `code/` | `run_benchmark.py` (harness), `phase0.sh` (the job script), `kprobe_32k.sh` (k sweeps), `summarize.py`, `check_lossless.py`, `diag_a100.sh`, `rebuild_multiarch.sh`, `submit_a100_predict.sh`. **Not in this repository:** `kprobe_ngram.sh` and `diag_a100_multiarch.sh` were both built with `sed` on Delta (section 8.6) and have never been pushed; they exist only at `/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts/`. Corrected 11 Oct 2026 — a read order that names files the reader cannot open sends them hunting for a file that is not there. |
| 16 | `results/` | raw JSONL plus per-request JSON files |
| 17 | `TODO.md` | task list |

### 3.2 The Google Drive folder — the planning archive

`https://drive.google.com/drive/folders/1A72SHdn9sv8FZ3iVWoH6YY6y9ImRwlIV`

**Every file in this folder is dated 18–30 September, which is before the
first valid measurement (1 October).** It records what we *intended* to do.
Where Drive and this repository disagree, **the repository is correct** — it
was written against measurements and Drive was not.

Read Drive for intent and for the graded proposal. Do not read it for state.

| file | date | what it is | status |
|---|---|---|---|
| `CS598_Proposal_SpeculationLadder_revisedFinal.pdf` | 30 Sep | **the authoritative proposal** — the approved scope | read this one |
| `CS598_Proposal_SpeculationLadder_v9_revised.pdf` | 27 Sep | earlier proposal draft | superseded |
| `CS598_Proposal_SpeculationLadder.pdf` | 18 Sep | original proposal | superseded |
| `speculation-ladder-plan-REALIGNED-partial.md` | 30 Sep | realigned experiment plan, ~776 KB, incomplete | skim; large |
| `speculation-ladder-implementation-plan-READABLE.md` | 22 Sep | implementation plan, ~561 KB | skim; predates all data |
| `speculation-ladder-implementation-plan-READABLE (1).md` | 30 Sep | byte-identical duplicate of the above | ignore |
| `speculation-ladder-HANDOFF_1.md` | 30 Sep | an earlier handoff document, 37 KB | **superseded by this file** |
| `speculation-ladder-delta-compute-memo.md` | 30 Sep | Delta allocation and compute planning, 17 KB | partly stale; see `LOG.md` budget section |
| `KEY NOTES SL` (Google Doc) | 22 Sep | early notes | historical |
| `WEEKLY LOG` (Google Doc) | 22 Sep | weekly log, not updated since | superseded by `LOG.md` |
| `Links For Docs` (Google Doc) | 27 Sep | index of links | may be stale |

The two large `.md` plans are hundreds of kilobytes. **Do not read them
wholesale into a context window.** Search them for a specific question if one
arises.

---

## 4. Current state

### 4.1 What is measured

A complete context x load grid on **A40** at n=25–50: three context buckets
(2K / 8K / 32K) x three concurrency levels each x three rungs, plus k sweeps
at every bucket for DFlash and at 2K and 32K for n-gram.

On **A100**: one cell, 2K concurrency 1 at n=5 (job 22757104).

On **H200**: nothing. The kernels exist in the rebuilt venv but `phase0.sh`'s
architecture guard allows only compute capability 80 and 86 and aborts.

### 4.2 Claim register status

See `paper/claims.md` for the authoritative version. Summary:

- **C1, C2, C3** (acceptance decays with context; rungs decay at different
  rates; speedup crosses 1.0x at rung-dependent points) — measured at three
  buckets.
- **C4** (the crossing point is predictable on hardware not used for fitting)
  — **unblocked and under test.** See section 5.
- **C5** (the two rungs are of opposite type) — holds at three buckets.
- **C6** (draft length matters) — partly refuted and then settled: optimal k
  rises with context, shrinks with load.
- **C7** (engine-reported acceptance is conditional and needs a coverage term)
  — measured.
- **C8** (the ranking inverts along the context axis) — **measured and
  replicated across two nodes to 0.08%.** The best-supported claim.
- **C9** (drafting cost is fixed per step and grows with context) — measured
  for both rungs.
- **C10** (speculation perturbs output less than the engine perturbs itself) —
  measured; needs an interval, which is obtainable for free (section 6).
- **C11** (run-to-run variance) — restated: **speedup ratios** replicate
  across nodes to 0.03–0.08%, **absolute throughput** does not (8.8% against
  one node). Report ratios.
- **C12, C13** (every drafter pays a context tax; the inversion is driven by
  the absolute gap) — measured at two buckets.
- **C14** (n-gram's acceptance is context-invariant) — measured.
- **C15** (n-gram is *faster* at long context) — **demoted.** A 4% cross-job
  claim whose long-context half came from an anomalous node.
- **C16, C17** (tau is machine-invariant; R scales by a single rung-independent
  factor) — measured, one pair, n=5.

### 4.3 What is NOT established

- **No confidence intervals anywhere.** Per-request timing is unavailable on
  this vLLM build (`timing` carries only `arrival_time`, and there are no
  per-request draft counts), so the paired bootstrap the project originally
  planned is impossible for speedup and tau. Intervals must come from repeated
  whole runs. Three cells have two realizations; nothing has three.
- The A100 comparison is **n=5**. Per-position acceptance and coverage
  differences in it may be sampling noise or may be a real cross-GPU numerics
  effect; they cannot be separated yet.
- **Why both drafting costs are sublinear in context** — a 16x longer prompt
  roughly doubles each. Unexplained, and three buckets cannot distinguish
  candidate shapes. This is the most interesting open scientific question.
- n-gram's `R_draft` at 8K was never measured, and its 32K value comes from the
  one anomalous node.
- No prefix-caching-off control.
- **No quality evaluation.** Output token ids are stored for every run, but
  nothing has been scored against LongBench-v2's answers. Free to do on a
  login node.

---

## 5. The live experiment: a frozen prediction

`paper/claims.md` contains, under **"Frozen prediction, 11 Oct 2026"**, a
prediction committed *before* the measurement that tests it, per the
register's rule 4.

Method: take A40 `R` at concurrency 1, multiply by the measured machine factor
**1.054**, divide A40 `tau_eff` by the result.

| cell | predicted A100 speedup |
|---|---|
| 8K, DFlash k=7 | **1.20x** |
| 8K, n-gram k=5 | **1.09x** |
| 32K, DFlash k=7 | **0.78x** |
| 32K, n-gram k=5 | **1.02x** |

The inversion is predicted to survive on A100 and widen slightly. 2K is in
sample and does not count as a test.

**Passes** if every cell is within 5% relative and the 32K ordering holds.
**Fails** if median absolute relative error exceeds 10%, or 32K does not
invert.

**Pre-registered confound:** the A100-SXM4 has 40 GB against the A40's 48, so
the 32K concurrency ceiling is lower and that cell may sit in a different
memory regime. A large 32K miss may be a capacity artifact rather than a
failure of the cost model. **Judge the prediction primarily on 8K.** This was
recorded before the measurement so it cannot be used as an excuse afterwards.

**The test** is `code/submit_a100_predict.sh`, one command, ~1.5 GPU-hours.
This is the highest-value GPU time available to the project.

---

## 6. Open items, in priority order

**Free — do these first, they cost nothing:**

1. **Fix `summarize.py` to group by `env.slurm_job`**, or at minimum warn on
   duplicate keys. It currently keys on `(bucket, conc, rung, k)` and silently
   keeps one record per key. This hid a genuine replication for four days and
   caused three wrong write-ups in one day. Highest-value change in the repo.
2. **Bootstrap the agreement result** (C10). Per-prompt, from `token_ids`
   already on disk. Paired construction: resample prompt ids with replacement,
   compute both agreement rates on the same resampled set, report an interval
   on the *difference*.
3. **Quality evaluation.** Decode stored token ids, score LongBench-v2
   multiple choice. Login node.
4. **Verify the A100 and A40 spec sheets.** The bandwidth story behind C17 is
   a direction, not arithmetic, until the numbers are checked.

**GPU hours required:**

5. **Run the frozen-prediction test** (section 5). ~1.5 h on Sahil's
   allocation.
6. **Re-run the 32K n-gram k sweep excluding gpub073.** ~1 h. Repairs a known
   defect: n-gram's 32K drafting cost and C15 both rest on that node.
7. **Re-run the A40/A100 2K comparison at n=50.** ~0.6 h billed. Separates
   sampling noise from a real cross-GPU acceptance effect.
8. **n-gram k sweep at 8K.** ~1.2 h. Third point on n-gram's cost curve.
9. **A fourth context bucket (e.g. 16K).** Needs re-bucketing LongBench-v2
   first (free, login node). Would begin to identify how `R_draft` scales.
10. **Repeated runs for formal intervals.** No longer blocking, since C8 has
    two agreeing realizations.
11. **Prefix-caching-off control.**

**Administrative:**

12. Tell the professor the scope grew — he approved a hardware comparison; the
    project is delivering a context-axis inversion *and* a hardware transfer.
13. Request additional GPU hours. ~15 across both students would finish the
    work; a breakdown is in `LOG.md`.
14. Reconcile the budget: `accounts` and `sreport` disagree, and the assumed 2x
    A100 billing multiplier has never been verified.
15. The architecture guard blocks H200 jobs the rebuilt venv could run.

---

## 7. Infrastructure

### 7.1 Delta paths

| what | where |
|---|---|
| shared project tree | `/projects/bikd/dagraw2/Speculative_Ladder` |
| scripts | `.../ladder/scripts/` |
| workloads | `.../ladder/data/longbench_v2/longbench_v2_{2K,8K,32K}.jsonl` |
| models | `/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub` |
| results (Devansh) | `.../Speculative_Ladder/results/{a40,a100,...}` |
| results (per user) | `/projects/bikd/<netid>/Speculative_Ladder/results/...` |
| Slurm logs (shared) | `.../Speculative_Ladder/logs/phase0-<jobid>.out` |
| git clone on Delta | `/projects/bikd/dagraw2/Speculative_Ladder/repo` |
| **vLLM, multi-arch** | `.../Speculative_Ladder/vllm_env_multiarch` — **use this** |
| vLLM, original | `.../Speculative_Ladder/vllm_env` and `~/vllm_env` — A40 only |
| vLLM build tree | `/work/nvme/bikd/dagraw2/vllm-build` (5.7 GB) |

### 7.2 Software versions

vLLM `0.29.1rc1.dev345+gbf13ecc2f`, torch `2.13.0+cu130`, CUDA 13.0, triton
`3.7.1`, python 3.12.12 (`module load cray-python/3.12.12`).

**Both venvs are identical in torch, triton and vLLM commit.** The only
difference is the architecture set compiled into vLLM's own extensions:
original has sm_86 (40 cubins) and a partial sm_80 (20); the rebuild has sm_80
71, sm_86 51, sm_90 82. This was verified, and it is what lets C16 and C17
claim "same binary, only architectures differ".

Note the multiarch venv is an **editable install** rooted at
`/work/nvme/bikd/dagraw2/vllm-build`. If that path is removed the venv breaks
and `code/rebuild_multiarch.sh` must be re-run (~4 h wall clock, zero GPU
hours, login node). No purge policy is known to apply.

### 7.3 Two environment variables are mandatory

```bash
LADDER_VENV=/projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch
LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/<gpu>
```

**`LADDER_VENV`**: `phase0.sh` resolves its venv as `LADDER_VENV`, then
`$HOME/vllm_env`, then the original shared venv. Both fallbacks are
single-architecture builds with no working sm_80 kernels, so **an A100 job
that omits this reserves a node and dies at kernel launch.** That is job
22685764: 17 minutes billed, nine arms lost, nothing written.

**`LADDER_OUT`**: `results/` in the shared tree is deliberately not
group-writable, and two jobs appending to one `results.jsonl` can interleave.

Do **not** override `LADDER_SHARED` — the data path, harness path and venv
fallback all derive from it.

`phase0.sh` also exports `VLLM_USE_V2_MODEL_RUNNER=0`, and the harness refuses
to start without it. This is deliberate: with the V2 runner, n-gram silently
falls back to the stable runner while DFlash stays on V2, so the two rungs
would be measured on different code paths.

### 7.4 Running a job

```bash
ssh <netid>@login.delta.ncsa.illinois.edu
module load cray-python/3.12.12
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
mkdir -p /projects/bikd/$USER/Speculative_Ladder/results/a40
cd $SHARED/ladder/scripts

sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,\
LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/a40,\
BUCKET=2K,LIMIT=50,CONC=1 phase0.sh
```

`BUCKET` is **case sensitive**: `2K`, `8K`, `32K`.

Concurrency grids must shrink as context grows — 2K: `1 8 32`; 8K: `1 4 12`;
32K: `1 2 4`. vLLM accepts an impossible `max_num_seqs` and quietly schedules
fewer, which mislabels the row rather than failing.

### 7.5 Budget

10 GPU-hours deposited per student; more available on request, confirmed by
the professor. As of 11 Oct: **`dagraw2` has ~4 h left; `ssashi` has ~8.5 h**
and his allocation is the project's main remaining GPU resource. A100 bills at
roughly 2x the A40 rate (assumed, unverified).

---

## 8. Working conventions, learned the hard way

These are not style preferences. Each one exists because its absence cost time.

1. **Report ratios, never absolute throughput across jobs.** Ratios replicate
   to 0.03–0.08% across nodes; absolute throughput varied 8.8% against one.
2. **Analyse `results.jsonl` per record, split on `env.slurm_job`.** Never
   trust `summarize.py`'s table for provenance — it collapses duplicate keys.
3. **A Slurm state of `COMPLETED` means the script finished its loop, not that
   measurements succeeded.** Job 22685764 is COMPLETED and produced nothing.
   Check that `results.jsonl` grew.
4. **`sbatch` snapshots the batch script but not the files it calls at run
   time.** Editing the harness after submitting changes what a queued job does.
5. **Every repeat gets its own `LADDER_OUT`.** Per-request filenames encode
   bucket, rung, k and concurrency but not the run, so a second run into the
   same directory is an overwrite, not a repeat.
6. **Build script variants rather than editing originals** (`kprobe_ngram.sh`,
   `diag_a100_multiarch.sh` were both made with `sed` from a working file), so
   existing results stay reproducible from the script that made them.
7. **Acceptance is never compared across different k** without truncating to a
   common draft length first. `summarize.py`'s `AL@5` column is the comparable
   number, not `tauC`.
8. **A prediction counts only if it was committed before the measurement.**
9. **Record failures.** `paper/claims.md` has a "Predictions that failed"
   table and `LOG.md` has a dead-ends table. A register of only confirmations
   is not evidence of anything.
10. **Diagnose by bisection, cheapest layer first.** `code/diag_a100.sh` is the
    template: four layers under `CUDA_LAUNCH_BLOCKING=1`, and the first failure
    names the culprit. It answered in 3m35s a question that three rounds of
    inference had got wrong.

---

## 9. This project's recurring failure mode

Read this. A new collaborator who reads only the results will repeat it.

**Almost every error in this project came from drawing a conclusion from a
proxy instead of testing the thing itself.** The specific instances:

| the proxy trusted | what was actually true |
|---|---|
| `cuobjdump` listed sm_80, so A100 must work | The listing was real and the kernels still would not launch. A listing is not proof of execution. Cost one failed job. |
| A traceback pointed at `output.fill_(0)` in FlashAttention | That line is a profiling shortcut that skips attention. The error was a *sticky asynchronous* CUDA failure from an earlier launch. Sent the diagnosis down the wrong path twice. |
| A group name (`grp_202`) explained a permission failure | It was the teammate's *primary* group. He had read access all along. |
| `group::rwx` meant the teammate could write | POSIX uses a matching **named user** ACL entry and ignores group entirely. Two days of his jobs dying at launch with no log written. |
| A single data point, with the intercept assumed | Fitting `R = 1 + k*c` to one point made the answer come from the assumption. Predicted 1.17x, measured 0.76x. |
| One replication bounding variance "under 1%" | A third realization came in 8.8% low. One observation is not a distribution. |
| **`summarize.py`'s table standing in for the records** | It collapses duplicate keys. A result replicated across two nodes read as a single run for four days, and nearly triggered a GPU spend to re-measure something already replicated. |
| `grep USER phase0.sh` was not run before theorising about `$USER` paths | The script has no `$USER` in it. The hypothesis was free to refute and should have been refuted first. |

Three corollaries:

- **Read the script, do not infer it.**
- **One observation is not a distribution.**
- **A derived summary is a proxy for the data, and the most seductive kind,
  because it looks like the measurement rather than like a clue.**

The counter-example worth imitating: `kprobe_32k.sh` re-measures the baseline
arm on every run, *on purpose*, so that run-to-run variance can be checked for
free. That one design decision produced the replication evidence for the
project's main claim.

---

## 10. Briefing prompt for a new agent

Copy everything between the lines.

---

You are joining an in-progress systems research project. Before doing anything
else, read `HANDOFF.md` at the root of this repository:

`https://github.com/Devansh63/Speculate-Decoding-Ladder`

Then read, in this order: `paper/results-table.md`, `paper/claims.md`,
`LOG.md`, and the dated notes in `notes/` from newest to oldest. `claims.md`
is the authoritative record of what is believed and how strongly; read it
start to finish including the corrections and the failed predictions.

There is also a Google Drive folder of planning documents. **Every file in it
predates the first valid measurement.** Read it only for the original proposal
and for intent. Where Drive and the repository disagree, the repository is
correct. Do not load the two large `.md` plans (561 KB and 776 KB) into
context; search them if a specific question arises.

The project measures how speculative decoding's speedup varies with serving
load, prompt length and GPU, using the decomposition
`Speedup ~= [cov*tau + (1-cov)] / R`. The main result is that the ranking of
two speculation methods **inverts** with context length, and the mechanism is
that long context taxes the drafter's forward pass rather than verification.

Four things to internalise before you touch anything:

1. **Section 9 of `HANDOFF.md` lists this project's recurring failure mode** —
   conclusions drawn from proxies rather than from the thing itself. It has
   happened eight times. Read the script, read the records, do not infer. When
   a cheap test exists, run it before theorising.
2. **GPU hours are scarce and metered.** About 4 remain on one student's
   allocation and 8.5 on the other's. Never submit a job without stating its
   cost first, and never estimate from per-arm extrapolation — that has
   underestimated by ~2x three times. Quote the nearest whole job that
   actually ran, and call it a floor.
3. **Two environment variables are mandatory on every job** (`LADDER_VENV`,
   `LADDER_OUT`). Omitting the first on an A100 job wastes a reservation and
   produces nothing. Section 7.3.
4. **Analyse `results.jsonl` per record, split on `env.slurm_job`.** The
   summary script collapses records that share a key, which has already caused
   three wrong write-ups in a day.

A frozen prediction is currently under test — see section 5 of `HANDOFF.md`
and "Frozen prediction, 11 Oct 2026" in `paper/claims.md`. It was committed
before the measurement, with a pass/fail criterion and a pre-registered
confound. Do not revise it; report whether it passed.

Priorities are in section 6, free work first. Tell me what you would do and
what it costs before you do it.

---

## 11. Contacts and access

- **Devansh Agrawal** — `dagraw2@illinois.edu`, NetID `dagraw2`, repo owner.
- **Sahil Sashi** — NetID `ssashi`. Has read/execute on the shared tree, write
  on `logs/` and `.vllm_cache`, and his own results tree. Holds the A100 axis.
- **Professor** — approved the project; has confirmed additional GPU hours are
  available on request for legitimate project use.

Shareable outright: this repository, and the Google Drive folder linked in
section 3.2.
