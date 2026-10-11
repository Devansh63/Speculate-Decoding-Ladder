# Work log

A record of what we ran, what broke, and how it was fixed.

**Newest first.** The common read of this file is "where are we now", so the
latest entry is at the top. Entries stay internally chronological. If you are
starting from scratch rather than catching up, read the replay block below,
then jump to the bottom and come forward.

For **what we learned**, see the dated notes in `notes/` and the claim
register in `paper/claims.md`. This file is the record of *activity*,
including the parts that did not work. Every entry that involved a command
records the command.

Four conventions, because each has bitten us:

- A Slurm state of `COMPLETED` means the shell script finished its loop, not
  that the measurements succeeded. Job 22685764 is `COMPLETED` and produced
  nothing. **Always check that `results.jsonl` grew before believing a job
  worked.**
- `sbatch` snapshots the batch script at submit time but **not** the files it
  calls at run time. Editing the harness after submitting changes what a
  queued job will do.
- **`BUCKET` is case sensitive**: `2K`, `8K`, `32K`. The dataset filename is
  built from it directly, so `2k` fails on a missing file.
- **Analyse `results.jsonl` per record, not through `summarize.py`.** Every
  record carries `env.slurm_job` and `env.host`. `summarize.py` keys on
  `(bucket, conc, rung, k)` and silently keeps one record per key, so two
  jobs measuring the same cell are rendered as one row. This cost three wrong
  write-ups on 11 Oct.

---

## Replaying a cell from scratch

The short version, for a teammate starting today. Paths assume
`PROJ=/projects/bikd/$USER/Speculative_Ladder`.

```bash
# 1. environment
module load cray-python/3.12.12
source $PROJ/vllm_env_multiarch/bin/activate

# 2. submit one cell. phase0.sh reads BUCKET, LIMIT, CONC, KLIST,
#    LADDER_VENV and LADDER_OUT from the environment, and reads the GPU from
#    nvidia-smi at run time rather than trusting the partition flag.
#    LADDER_VENV is MANDATORY: without it the script prefers
#    $HOME/vllm_env, which has no A100 kernels.
cd $PROJ/ladder/scripts
sbatch --export=ALL,LADDER_VENV=$PROJ/vllm_env_multiarch,BUCKET=2K,LIMIT=25,CONC=1 phase0.sh

# 3. watch it
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -X

# 4. read the results, per record
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open('results/a40/results.jsonl')]
for r in rows:
    print(r['env']['slurm_job'], r['env']['host'].split('.')[0],
          r['rung'], 'k=%s'%r['k'], r['bucket'],
          'c=%s'%r['max_num_seqs'], round(r['throughput_tok_s'],2))
PY
```

**A repeat gets its own `LADDER_OUT`.** The per-request filenames encode
bucket, rung, k and concurrency but not the run, so a second run into the
same directory overwrites the first's `perreq_*.json`.

Check each script's header comment before relying on a variable name; the
scripts are the source of truth and this file can drift.

---

## 11 Oct 2026 — n-gram's drafter cost, and a correction about replication

**Measured n-gram's R_draft for the first time.** It had been *assumed* near
zero on the reasoning that prompt lookup runs no forward pass, and that
assumption was load-bearing in the explanation of the inversion.

The existing k probe runs n-gram once at k=5, which is one point and cannot
identify an intercept. Built a variant rather than editing the original, so
the existing results stay reproducible from the script that made them:

```bash
cp kprobe_32k.sh kprobe_ngram.sh
sed -i 's|^#SBATCH --job-name=kprobe32k|#SBATCH --job-name=kprobengram|' kprobe_ngram.sh
sed -i 's|^run ngram 5  ""|# n-gram moved into the k loop below|' kprobe_ngram.sh
sed -i 's|^  run dflash "\$K" "--draft \$DRAFT"|  run ngram "$K" ""|' kprobe_ngram.sh
grep -n 'job-name\|^run \|run ngram\|run dflash' kprobe_ngram.sh   # verify all three took
```

```bash
sbatch --time=01:30:00 --export=ALL,LADDER_VENV=$V,LADDER_OUT=$P/results/a40_ngramk,BUCKET=32K,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_ngram.sh   # 22757472
sbatch --time=01:00:00 --export=ALL,LADDER_VENV=$V,LADDER_OUT=$P/results/a40_ngramk,BUCKET=2K,LIMIT=50,CONC=1,KLIST="1 2 7" kprobe_ngram.sh    # 22757473
```

**The assumption was wrong.** Fitting `R(k) = (1 + R_draft) + c*k`:

| bucket | rung | R_draft | c |
|---|---|---|---|
| 2K | n-gram | **0.071** | 0.0030 |
| 2K | DFlash | 0.296 | 0.0092 |
| 32K | n-gram | **0.139** | 0.0068, not resolved |
| 32K | DFlash | 0.662 | 0.0148 |

n-gram's drafting cost is small but real, and it **grows with context at
nearly DFlash's relative rate** — 1.96x against 2.24x across the range. Once
stated plainly it is obvious: prompt lookup has to *search the prompt*, and a
32K prompt is sixteen times more text to scan. Both drafters pay a context
tax; one pays it as a forward pass and the other as a string search, and the
difference is a constant factor of four to five.

**So the inversion is driven by the absolute gap, not the scaling.** DFlash's
drafting penalty exceeds n-gram's by 0.225 at 2K and 0.523 at 32K. The
ranking turns on that level difference, and would turn on it even if both
costs grew at identical rates. Write-up in
`notes/2026-10-11-ngram-drafter-cost.md`; new claims C12 and C13.

**n-gram's acceptance is context-invariant**, which was not the question
asked and is the cleanest number in the job: tau_cond at k=1 is 1.670 at both
buckets, first-position acceptance 0.670 at both, D = -0.000 / -0.008 /
-0.007. Coverage meanwhile *rises*, 0.121 to 0.195 at k=7. Claims C14, C15.

The acceptance-by-position profiles at 32K show why DFlash cannot compete
there at all:

```
dflash k=7   0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004
ngram  k=7   0.541, 0.382, 0.304, 0.256, 0.219, 0.193, 0.177
```

DFlash's drafts are worthless past position 2; n-gram's still pay at
position 7. It pays four times more to draft and gets drafts that collapse
faster.

**First successful out-of-sample prediction.** The 2K fit gives
`R(5) = 1.086`. Job 22598608, ten days earlier, measures **1.083** — off by
0.3%. k=5 was not in the fit and the measurement is from a different job.
Recorded in `claims.md` under a new "Predictions that succeeded" heading.

### The 32K baselines, and a correction to this morning's reading

The n-gram sweep re-measured the 32K baseline, which is what the k probe's
baseline arm exists for. It came in at **13.5 tok/s** against the 14.9 and
14.8 in C11. My first reading was "32K variance is 9%", my second was "an
occasional excursion". Both were written from `summarize.py` output. Reading
the records directly overturns both.

Splitting `results/a40/results.jsonl` on `env.slurm_job`:

| | job 22713173, gpub019 | job 22714602, gpub047 |
|---|---|---|
| `off` c1 | 14.90 | 14.83 |
| `ngram` k=5 c1 | 16.07 | 15.98 |
| `dflash` k=7 c1 | 12.31 | 12.25 |
| **n-gram speedup** | **1.0785** | **1.0776** |
| **DFlash speedup** | **0.8262** | **0.8260** |

**The inversion has been replicated since 7 Oct.** Two independent jobs, two
physical nodes, three days apart, ratios agreeing to **0.08%** and **0.02%**.
It was never a single run. Nobody noticed because `summarize.py` collapses
the duplicate keys and prints one row, and the register copied that reading
forward for four days.

The right statement separates two quantities we had conflated:

| quantity | agreement |
|---|---|
| speedup ratios, across two nodes | **0.08%** |
| absolute throughput, those two nodes | 0.47% |
| absolute throughput, against gpub073 | 8.8% low |

**Ratios are the stable quantity**, by two orders of magnitude, and every
speedup here is a within-job ratio. Report ratios; never report absolute
throughput across jobs.

gpub073 is genuinely different, and both of my explanations for it are
refuted from the logs:

```bash
for J in 22713173 22714602 22757472; do
  L=$(ls $P/logs/*-$J.out | head -1)
  grep -i "GPU KV cache size" "$L" | head -3
  echo "   preempt/recompute: $(grep -ic 'preempt\|recompute' "$L")"
done
sacct -j 22713173,22714602,22757472 --format=JobID,NodeList%22 -X
```

The `off` arm was allocated **173,056 tokens in all three jobs**, identical,
and there are **zero** preempt or recompute lines anywhere. So it is neither
cache allocation nor preemption. That also clears the separate worry that the
32K concurrency-4 cells were preempted — they were not.

What is left is the node. And on gpub073 the *ratio* moved too:
interpolating its n-gram k=2 to k=7 curve to k=5 gives about 1.16 against
1.078 on both other nodes. Slower at plain decoding **and** speculation
helping more is the direction a bandwidth-limited node produces, since
speculation trades memory-bound decode steps for a compute-bound
verification pass. **Consistent with the R model, not established** — one
job, and `--gpus-per-node=1` on a 4-GPU node means co-tenancy is possible
but invisible to us.

**C15 is demoted**: "n-gram is faster at 32K than at 2K", 1.14x against
1.18x, is a 4% cross-job claim whose long-context half is gpub073. Say
"n-gram's speedup does not degrade with context" instead. Write-up in
`notes/2026-10-11-noise-floor.md`, now in its third version.

### Sahil is running, and the ACL diagnosis is confirmed

```
-rw-rw----+ 1 ssashi  delta_bikd  59066 Oct  8 14:21 phase0-22757102.out
-rw-rw----+ 1 ssashi  delta_bikd  59066 Oct  8 14:21 phase0-22757105.out
-rw-rw----+ 1 ssashi  delta_bikd  56883 Oct  8 14:21 phase0-22757104.out
```

Three jobs, ~59 KB of log each, written into the shared `logs/` directory
hours after the `setfacl` on 8 Oct. Not 7-second deaths. **The write-access
diagnosis is confirmed**: he was failing because Slurm could not create his
output file, and `phase0.sh` line 10 hardcodes that path so no variable could
have saved him. `sreport` shows 24 CPU-hours against his name, roughly 1.5
GPU-hours, so the work was real.

22757102 and 22757105 have byte-identical log sizes, which may mean the same
configuration ran twice. Not yet analysed.

### Cost, and an estimate that was wrong again

22757472 took 01:01:22 and 22757473 took 00:58:58 — **2.0 GPU-hours against
my estimate of 1.1**. Third time a per-arm extrapolation from a previous job
has underestimated by roughly 2x. Arms are not interchangeable units of time:
model load, cold compile and bucket length all vary. Quote the nearest whole
job that actually ran, and call it a floor.

`accounts` reports **4 hours remaining of 10 deposited**. `sreport` reports
159 CPU-hours for dagraw2, which divided by 16 cores per GPU would be ~9.9
GPU-hour equivalents — not reconcilable with 6 charged. The divisor is an
assumption about Delta's node layout and GPU allocations may not be charged
through CPU-hours at all. **Treat 4 as operative**, since that is the number
that gates submission.

## 8 Oct 2026 — drafter cost isolated, A100 unblocked, teammate unblocked

**The rebuild finished.** `Successfully installed vllm`. Verified by counting
cubins again:

| arch | before | after |
|---|---|---|
| sm_80 (A100) | 20 | **71** |
| sm_86 (A40) | 40 | **51** |
| sm_90 (H200) | — | **82** |

**A100 diagnostic with the new venv:**

```bash
# diag_a100.sh hardcodes ~/vllm_env on line 40 and ignores LADDER_VENV,
# so a variant was produced with sed rather than editing the original.
sbatch diag_a100_multiarch.sh                                      # 22754950
```

**All four steps passed** in 11m06s, including step 4 — the compiled,
CUDA-graph configuration that lost all nine arms on 22685764. Plain torch,
vLLM's `rms_norm` custom op under `CUDA_LAUNCH_BLOCKING=1`, eager generation
and compiled generation all work on sm_80. **The A100 blocker is resolved and
needs no `enforce_eager` workaround.**

The log shows `import vllm._C` raising `ModuleNotFoundError`. That is a defect
in the diagnostic probe, not the build: this vLLM ships stable-ABI extensions
named `_C_stable_libtorch`, `_moe_C_stable_libtorch` and `_vllm_fa2_C`, and
has no module named `vllm._C` at all. The same probe fails identically on the
A40, where everything works.

**Unplanned finding: the eager and compiled paths disagree under greedy
decoding.** Same prompt, same seed, `temperature=0.0`:

```
EAGER OK:    ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
COMPILED OK: ' Paris. The capital of Italy is Rome. The capital of Germany is Berlin.'
```

A second, independent instance of the 5 Oct result, and a cleaner one: no
speculation, no batching difference, nothing but eager versus compiled.
Belongs in `notes/2026-10-05-losslessness-and-engine-nondeterminism.md`.

**C9 confirmed: R decomposes.** With the 2K and 8K sweeps in, fitting
`R(k) = R_draft + c*k` over k = 1, 2, 7 at each bucket:

| bucket | R(1) | R(2) | R(7) | R_draft | c |
|---|---|---|---|---|---|
| 2K  | 1.305 | 1.322 | 1.360 | **0.296** | 0.0092 |
| 8K  | 1.450 | 1.487 | 1.544 | **0.434** | 0.0157 |
| 32K | 1.677 | 1.704 | 1.766 | **0.662** | 0.0148 |

The fixed per-step drafter cost grows **2.24x** from 2K to 32K. The marginal
verification cost per drafted token is flat within noise. Drafting is 82 / 80
/ 86 percent of DFlash's total overhead. Write-up in
`notes/2026-10-08-drafter-cost-scaling.md`.

*Footnote worth not forgetting:* first-position acceptance varies with k
(0.570 / 0.589 / 0.601 at 2K for k = 1 / 2 / 7). Longer accepted runs sample
a different context distribution, so "first-position acceptance" is not
independent of k.

**The paired bootstrap is settled, and the answer is mixed.** Inspecting a
post-v2 per-request file:

```bash
python3 -c "
import json
d=json.load(open('results/a40/perreq_dflash_32K_k7_n25_s1234_c1.json'))
print(sorted(d[0].keys()))"
# ['finish_reason', 'gen_tokens', 'id', 'prompt_tokens', 'sha1', 'timing', 'token_ids']
```

`timing` holds **only `arrival_time`**, so per-request latency is unavailable
and speedup intervals can only come from repeated whole runs. There are also
**no per-request draft or accept counts**, so tau and coverage are
run-aggregate and cannot be resampled over prompts.

But `token_ids` *is* per-prompt, so the losslessness quantities can be
bootstrapped from data already on disk, for free. 12.7% against 19.0% at
n = 25 to 100 is a difference of a handful of sequences, quoted to three
significant figures. The construction has to be paired: resample prompt ids
with replacement, compute both agreement rates on the same prompt set, and
report an interval on the difference.

### Teammate access, and why Sahil's jobs died in seven seconds

POSIX ACL precedence: when a **named user entry** matches, it is used and the
group entries are never consulted. The project tree carries
`user:ssashi:r-x`, so his `delta_bikd` membership and the `group::rwx` bits
were both irrelevant to him — read and execute, no write, everywhere.

`phase0.sh` line 10 hardcodes
`#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-%j.out`,
and an `#SBATCH` directive cannot take a variable. So Slurm could not create
his output file and the job died at launch, on any partition, with **no log
written** — which is why we spent two days unable to find a log of his, and
read its absence as him not sending it. **Confirmed 11 Oct.**

```bash
P=/projects/bikd/$USER/Speculative_Ladder
setfacl -R -m u:ssashi:rwX -m d:u:ssashi:rwX "$P/logs" "$P/.vllm_cache"
getfacl -p "$P/logs" "$P/.vllm_cache" | grep -E 'file:|ssashi|mask'
```

`logs/` because Slurm must write there; `.vllm_cache` because the multiarch
build is a different build and his first run has to write roughly four
minutes of compiled kernels. `results/` is deliberately **not** granted; he
uses `LADDER_OUT`.

Also tightened the shared venv, which was group-writable:

```bash
chmod -R g+rX,g-w "$P/vllm_env_multiarch"    # 4m07s, ~100k files
```

The ACL mask became `r-x`, clamping `group::rwx` to `r-x` effective, so nobody
in the group can modify the venv. This protects the premise the hardware axis
rests on — same commit, same binary, only the architecture list differs.

**Two traps recorded while reading `phase0.sh` properly for the first time.**

The venv fallback is `LADDER_VENV`, then `$HOME/vllm_env` if it exists, then
`$SHARED/vllm_env`. Both fallbacks are single-architecture builds. Since
`~/vllm_env` exists and is executable, **any A100 job submitted without
`LADDER_VENV` reproduces 22685764 exactly.** Now mandatory on every job in
`notes/teammate-quickstart.md`.

And the architecture guard is `case "$CAP" in 80|86)`, written when the shared
build had only those two. The multiarch build carries sm_90, so the guard now
wrongly aborts an H200 job the venv could actually run.

### KV cache varies by arm within a job

From 22685911's log the arms got 173,056 / 173,104 / 136,160 / 160,336 /
160,032 / 125,456 tokens — a 28 percent spread. The small ones are the
speculative arms, which reserve memory for the drafter, so DFlash runs with a
smaller cache than the baseline it is compared against. Job 22713173's
smallest arm reports a ceiling of 3.72x for full-length requests while that
bucket ran at concurrency up to 4, which looked like a preemption risk.
**Checked 11 Oct: zero preempt or recompute lines. Not an issue.**

The `Maximum concurrency` audit the script asks for does not answer the
question anyway — vLLM computes that figure for a full `max_model_len`
request, 33,792 tokens in every log including the 2K runs, so it is a
worst-case ceiling rather than a statement about what got scheduled.

## 7 Oct 2026 — the k sweep refutes a prediction

**32K landed, and the ranking inverted.** At concurrency 1: DFlash 0.83x,
n-gram 1.08x. At 2K those were 1.86x and 1.16x. Write-up in
`notes/2026-10-06-32k-inversion.md`.

**Predicted that k=1 would rescue DFlash at 32K, and was wrong.** Predicted
1.17x, measured **0.76x** — the worst cell in the sweep.

*Why the prediction failed:* we fitted `R = 1 + k*c` to a single data point by
*assuming* the intercept was 1. It is not. The true fit at 32K is
`R(k) = 1.662 + 0.0148k`. There is a large fixed cost that does not shrink
when k shrinks, so cutting k cannot help. Write-up in
`notes/2026-10-07-k-sweep-refutation.md`.

**Also retracted:** "context does to k what load does to k". Wrong.
Optimal k **rises** with context and **shrinks** with load.

```bash
sbatch --export=ALL,BUCKET=32K,LIMIT=25,CONC=1,KLIST="1 2 7 15" \
       kprobe_32k.sh                                               # 22714602
```

This job also re-measured the full 32K concurrency-1 row, which is what made
the 11 Oct replication check possible.

**Submitted the same probe at 2K and 8K:**

```bash
sbatch --export=ALL,BUCKET=2K,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_32k.sh  # 22737716
sbatch --export=ALL,BUCKET=8K,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_32k.sh  # 22737717
```

**Started the vLLM rebuild**, zero GPU hours, in tmux:

```bash
tmux new -s build
bash rebuild_multiarch.sh          # then Ctrl-b then d to detach
```

`rebuild_multiarch.sh` builds the *same commit* `bf13ecc2f` with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` into `$PROJ/vllm_env_multiarch`, and
never touches `~/vllm_env`.

*Build failure:* `ModuleNotFoundError: No module named 'setuptools_rust'`.
*Cause:* the script looked for `requirements/build.txt`, but
`requirements/build` is a **directory**. *Fix:*

```bash
pip install -r requirements/build/cuda.txt -r requirements/build/rust.txt
```

*Operational note:* tmux survives an SSH drop, but Delta login nodes are
round-robin, so you must reattach **on the same login node**. The build ran
about four hours (432 object files, 451 FlashAttention-3 `.cu` sources,
`ninja -j 4`).

## 6 Oct 2026 — 8K decays, A100 collapses

**8K completed** (22685911, 63 minutes). The rungs decay in *opposite*
directions. Decay `D = 1 - AL(long)/AL(2K)` at the common k=5 truncation:
DFlash 0.219, n-gram 0.044. n-gram's coverage *rises*, 0.132 to 0.161. At 2K
the two had near-identical first-position acceptance; at 8K n-gram is ahead,
0.563 to 0.451. Write-up in `notes/2026-10-06-8k-context-decay.md`.

**A100 lost all nine arms** (22685764) to
`cudaErrorNoKernelImageForDevice`. Seventeen minutes billed, nothing written.

Three wrong diagnoses before the right one — see the dead-ends table. What
finally worked was bisection instead of inference:

```bash
sbatch diag_a100.sh                                                # 22713175
```

`diag_a100.sh` runs four layers cheapest-first under
`CUDA_LAUNCH_BLOCKING=1`, and the first failure names the culprit:

1. plain torch bf16 matmul and `fill_` — no vLLM anywhere
2. vLLM's own `rms_norm` custom op — no torch.compile, no CUDA graphs
3. vLLM eager, `enforce_eager=True`
4. vLLM compiled, the configuration that failed

`CUDA_LAUNCH_BLOCKING=1` is the point. The original traceback pointed at
`output.fill_(0)` and `.contiguous()`, both trivial ops, which is what an
asynchronous CUDA error looks like when it reaches a sync point.

Result, in 3m35s: step 1 passed, step 2 failed. Node, driver and torch are
healthy; **vLLM's extension does not execute on sm_80.** Confirmed by
counting cubins:

```bash
find $VD -name '*.so' -exec sh -c 'echo "== $1"; cuobjdump --list-elf "$1"' _ {} \;
```

sm_86: 40 cubins. sm_80: 20. Write-up in `env/a100-sm80-failure.md`, support
matrix in `env/gpu-support.md`.

**Submitted 32K** (22713173), grid 1/2/4, limit 25, 3h wall.

**Set up team access.**

```bash
bash share_with_team.sh
```

Copies the venv into project space with paths rewritten, then `chgrp -R`
*after* `cp -a` (the original did it before, so new files inherited the wrong
group). Group corrected from `grp_202` to `delta_bikd`. Sahil's jobs 22713193
and 22713194 still failed in seven seconds each with ExitCode `0:53` —
diagnosed two days later as the ACL write problem, confirmed 11 Oct.

**Cerebras.** The professor sent `sdk.cerebras.ai`. That is the CSL kernel
development SDK for the Wafer-Scale Engine, not an inference API — no models,
no endpoints, no decoding parameters. Its value is as a *prediction target*:
wafer-scale keeps weights in on-chip SRAM, so the ridge point sits roughly an
order of magnitude below a GPU's, and our R model predicts speculation is
worthless there even at batch 1. Discussion section, not experiment plan.
Specs need verifying from Cerebras's own sheet first.

## 5 Oct 2026 — losslessness, and harness v2

**Harness v2** (`HARNESS_VERSION = "v2-2026-10-05"`). Additive changes, no
default behaviour altered, installed before either queued job started:

- full output token ids, because a SHA-1 cannot distinguish one flipped token
  from a different answer, and that distinction *is* the losslessness result
- per-request timing, intended for a paired bootstrap
- a prefix-caching switch (`--prefix-caching` / `--no-prefix-caching`)
- **`env.host` and `env.slurm_job` in every record** — which is what made the
  11 Oct replication check possible, four days later

Every addition is guarded, so an instrumentation failure records a null
rather than losing the run.

**Investigated losslessness.**

```bash
python3 check_lossless.py $PROJ/results/a40/results.jsonl
```

Version 1 compared *generation lengths*, which was useless: every generation
hits the 256-token cap, so all lengths are equal by construction. Rewrote it
to compare token ids, and — the part that mattered — to run the baseline
against *itself* first as a control.

The control settled it. The `off` arm against itself at different
concurrency, with no speculation anywhere, agrees on only 12.7% of sequences.
Speculative arms agree with the baseline on 19.0%. Per-token divergence 0.80%
against 0.65%. **Speculation perturbs the output less than the engine
perturbs itself.** The question is only answerable relative to the engine's
own noise floor. Write-up in
`notes/2026-10-05-losslessness-and-engine-nondeterminism.md`.

```bash
sbatch --export=ALL,BUCKET=8K,LIMIT=25,CONC="1 8" phase0.sh        # 22685911
sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2K,LIMIT=25 \
       phase0.sh                                                   # 22685764
```

## 1 Oct 2026 — first valid measurement, and the coverage correction

```bash
sbatch --export=ALL,BUCKET=2K,LIMIT=25,CONC="1 8 32" phase0.sh    # 22598608
```

52 minutes on one A40. Three rungs at concurrency 1, 8 and 32.

**The result that set up the whole project:** tau is flat across load (DFlash
2.543, 2.507, 2.470) while speedup collapses (1.86x, 1.49x, 1.03x). All of
the variation lives in R, not in acceptance.

**Found a bug in how we were reading vLLM's numbers.** n-gram proposes a
draft on only 13.2% of decode steps, but vLLM's reported acceptance rate is
conditional on a draft having been proposed. Taken at face value it
overstates what n-gram delivers by more than a factor of two. Added a third
term:

```
Speedup ~= [cov * tau_cond + (1 - cov)] / R
```

Corrected, n-gram's R is 1.082, *below* DFlash's 1.360 — which is what the
hardware predicts. `summarize.py` now reconstructs coverage from the draft
and generation counters rather than printing vLLM's acceptance directly.

## 30 Sep 2026 — environment, and six jobs that failed in five seconds

**Built the environment.** Qwen3-8B and the DFlash drafter head downloaded to
`/work/nvme/bikd/$USER/Speculative_Ladder/hf/hub`, LongBench-v2 bucketed into
2K / 8K / 32K by prompt length, vLLM built from source at commit `bf13ecc2f`
into `~/vllm_env`.

**Verified the runner pin by reading the source**, not by trusting a flag:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

With the V2 runner, n-gram speculation silently falls back to the stable
runner while DFlash stays on V2, so the rungs would be measured on different
code paths. The harness refuses to start without this set.

**Six jobs failed in about five seconds each**, all with
`ModuleNotFoundError: No module named 'vllm'`.

*Cause:* the scripts ran `module load cray-python` but never activated the
venv. *Fix:* added `source "$VENV/bin/activate"`, with the venv resolved from
`LADDER_VENV`, then `$HOME/vllm_env`, then the shared copy.

**First real run** (22569403). All prompts submitted at once.

*What went wrong:* nothing crashed, which is worse. vLLM batched as wide as
memory allowed, so every rung lost to no speculation at all — 0.86x, 0.78x,
0.71x — while acceptance looked healthy. The run measured vLLM's batching
decision, not the drafter.

*Fix:* explicit concurrency control via `max_num_seqs`, exposed as `CONC`.
The speedups are void; the acceptance numbers are still usable. Kept in the
results because it is the right-hand end of the load axis, not a mistake.

---

## Job index

| Job | Date | GPU / node | Cell | State | Outcome |
|---|---|---|---|---|---|
| 22757473 | 8 Oct | A40 | 2K n-gram k sweep | COMPLETED 00:58:58 | n-gram R_draft = 0.071 |
| 22757472 | 8 Oct | A40 gpub073 | 32K n-gram k sweep | COMPLETED 01:01:22 | R_draft ~0.139; the slow node |
| 22757105 | 8 Oct | ? (ssashi) | ? | log written, ~59 KB | **parked, unanalysed** |
| 22757104 | 8 Oct | ? (ssashi) | ? | log written, ~57 KB | **parked, unanalysed** |
| 22757102 | 8 Oct | ? (ssashi) | ? | log written, ~59 KB | **parked, unanalysed** |
| 22754950 | 8 Oct | A100 | diagnostic, multiarch venv | COMPLETED 00:11:06 | **all 4 steps pass, A100 unblocked** |
| 22737717 | 7 Oct | A40 | 8K k sweep | COMPLETED 00:42:01 | R_draft = 0.434 |
| 22737716 | 7 Oct | A40 | 2K k sweep | COMPLETED 00:32:42 | R_draft = 0.296 |
| 22714602 | 6-7 Oct | A40 gpub047 | 32K k sweep | COMPLETED 01:07:08 | refuted k=1; **replicates the inversion** |
| 22713396 | 6 Oct | A100 | 8K (ssashi) | PENDING | should have been cancelled |
| 22713194 | 6 Oct | A40 | ? (ssashi) | FAILED 00:00:07 | ACL: no write for Slurm output |
| 22713193 | 6 Oct | A100 | ? (ssashi) | FAILED 00:00:07 | ACL: no write for Slurm output |
| 22713175 | 6 Oct | A100 | diagnostic | COMPLETED 00:03:35 | **answered the A100 failure** |
| 22713173 | 6 Oct | A40 gpub019 | 32K | COMPLETED 01:16:07 | **the inversion** |
| 22685911 | 5 Oct | A40 | 8K | COMPLETED 01:03:12 | opposite-direction decay |
| 22685764 | 6 Oct | A100 | 2K | COMPLETED 00:17:03 | **all 9 arms failed**, nothing written |
| 22598608 | 1 Oct | A40 | 2K concurrency sweep | COMPLETED 00:52:39 | first valid measurement |
| 22569403 | 30 Sep | A40 | 2K, uncontrolled | COMPLETED | speedups void, acceptance usable |
| (six jobs) | 30 Sep | A40 | — | FAILED ~00:00:05 | `ModuleNotFoundError: vllm` |

Elapsed times are from `sacct`; nodes are in `env.host` in every record.

```bash
sacct -S 2026-09-30 -u $USER -X \
  --format=JobID,JobName%20,Partition,NodeList%12,State,Elapsed,End
```

---

## Decision register

| Decision | Why | Date |
|---|---|---|
| Pin `VLLM_USE_V2_MODEL_RUNNER=0`, refuse to start without it | n-gram silently falls back to the stable runner while DFlash stays on V2 | 30 Sep |
| Control concurrency explicitly rather than letting vLLM batch freely | Otherwise speedup measures the batching choice, not the rung | 1 Oct |
| Keep the uncontrolled 30 Sep run in the results | It is the right-hand end of the load axis, not a mistake | 1 Oct |
| Shrink the concurrency grid per context bucket | vLLM accepts an impossible `max_num_seqs` and quietly schedules fewer | 5 Oct |
| Read the GPU from `nvidia-smi` at run time, not from the partition flag | A mislabelled result is harder to notice than a failed job | 5 Oct |
| Keep prefix caching on | It is what a real server does, and it cancels out of within-bucket ratios | 5 Oct |
| Store full token ids, not just a hash | A hash cannot distinguish one flipped token from a different answer | 5 Oct |
| Do **not** switch A100 to `TRITON_ATTN` to make it run | Mixing backends would put the difference into R, the quantity being measured | 6 Oct |
| Rebuild the **same commit** with three architectures, into a **new** venv | A rebuild in place would replace the binary that produced every existing measurement | 7 Oct |
| Run A100 cells compiled, not with `enforce_eager` | 22754950 step 4 passes, so there is no reason to accept a code-path difference | 8 Oct |
| Make the shared venv read-only to the group | The hardware axis rests on "same binary" | 8 Oct |
| Write on `logs/` and `.vllm_cache` for teammates, **not** `results/` | Slurm must write a log and vLLM a cache; two writers on one `results.jsonl` can corrupt it | 8 Oct |
| `LADDER_VENV` is mandatory on every job | The fallbacks are single-architecture builds | 8 Oct |
| Build script **variants** rather than editing originals | `kprobe_ngram.sh`, `diag_a100_multiarch.sh` — so existing results stay reproducible from the script that made them | 11 Oct |
| Every repeat gets its own `LADDER_OUT` | Per-request filenames do not encode the run | 11 Oct |
| **Report ratios, never absolute throughput across jobs** | Ratios agree to 0.08% across nodes; absolute throughput to 8.8% | 11 Oct |
| **Analyse per record, split by `env.slurm_job`** | `summarize.py` collapses duplicate keys and hid a replication for four days | 11 Oct |
| Repeats are desirable but **not blocking** | C8 already has two independent realizations agreeing to 0.1% | 11 Oct |

---

## Dead ends, and what they cost

Recording these is cheaper than rediscovering them.

| What was believed | Why it was wrong | Cost |
|---|---|---|
| A100 is supported, because `cuobjdump` lists sm_80 | `ls $VD/*.so` is not recursive and missed the FlashAttention extension in a subdirectory. Worse: sm_80 *was* listed and the kernels still did not run. A listing is not proof of execution | one failed A100 job, ~17 min billed |
| The A100 failure is compile-cache poisoning from the A40 runs | Per-config and `torch_aot_compile` hashes are disjoint across GPUs, and the A100 run logged nine saves and zero loads | ~20 min of analysis, no GPU |
| The failure is in FlashAttention, per the traceback | `output.fill_(0)` at `flash_attn.py:1211` is the profiling shortcut that skips attention. The error was sticky. CUDA errors surface where they are noticed, not where they happen | wrong path twice |
| Sahil could not read the venv because its group was `grp_202` | `id ssashi` shows `grp_202` is his *primary* group | a `chgrp` over 100k files |
| Sahil *could* write to the tree, because `group::rwx` and he is in `delta_bikd` | POSIX uses a matching **named user** ACL entry and ignores group entirely | two days of his jobs failing, and ours hunting a log that was never created |
| `same length` would distinguish a token flip from a different answer | Every generation hits the 256-token cap | one useless diagnostic, rewritten |
| k=1 would rescue DFlash at 32K | Fitted `R = 1 + k*c` to a single point by assuming the intercept. The real intercept is 1.662 | one wrong prediction, caught by the sweep that tested it |
| Optimal k shrinks with context, as it does with load | It **rises** with context. k=7 beat k=15 only at 2K *saturation* | one retracted claim |
| `requirements/build.txt` holds the build dependencies | `requirements/build` is a directory | one failed build start |
| Per-request timing would give us a paired bootstrap | `timing` carries only `arrival_time`, and there are no per-request draft or accept counts | one harness revision |
| `import vllm._C` tests whether the extension loads | This build ships stable-ABI extensions with different names | a false alarm on a good A100 log |
| `phase0.sh` derives its paths from `$USER` | `grep USER phase0.sh` returns nothing | one hypothesis, refuted for free |
| `BUCKET=2k` works | The dataset filename is built from it and is case sensitive | a wrong command in this file for a day |
| n-gram drafts for free, because it runs no forward pass | Prompt lookup must scan the prompt: R_draft 0.071 at 2K, 0.139 at 32K | two GPU-hours, well spent |
| Run-to-run variance is under 1% everywhere, from one 2K-derived replication | A third 32K realization came in 8.8% low on absolute throughput | nearly bought contaminated repeats |
| 32K is irreproducible to ~10% | C11 already held two 32K realizations agreeing to 0.67% | one wrong note |
| The 32K inversion is a single unreplicated run, and whether it is within-job cannot be established from stored data | `env.slurm_job` and `env.host` are in every record since v2. Two jobs measured it; ratios agree to 0.08%. **It had been replicated for four days** | three wrong write-ups in one day, and a near-miss on spending the last GPU hours re-measuring something already replicated |
| The harness should record node and job id | It already does, and has since 5 Oct | one wrong recommendation |
| Per-arm cost extrapolation estimates a job's runtime | Wrong by ~2x three times running | repeated bad planning |

**The pattern in most of these:** a conclusion drawn from a proxy — a file
listing, a directory name, a traceback line, a group name, a module name, a
mode string, a single point with an assumed intercept, a variance measured at
one operating point, **a summary table standing in for the records** —
instead of testing the thing itself. The fix that works is bisection,
cheapest layer first, as in `diag_a100.sh`.

Three corollaries earned the hard way: **read the script, do not infer it**;
**one observation is not a distribution**; and **a derived summary is a proxy
for the data, and the most seductive kind, because it looks like the
measurement rather than like a clue.**

---

## Budget

Sub-allocation: 10 GPU-hours deposited per student, more available on request.

`accounts` reports **4 hours remaining of 10 deposited** as of 11 Oct, which
is the operative number since it gates submission. `sreport` reports 159
CPU-hours for dagraw2; dividing by an assumed 16 cores per GPU gives ~9.9
GPU-hour equivalents, which does not reconcile with 6 charged. The divisor is
an assumption and GPU allocations may not be charged through CPU-hours at
all. **Do not trust the table below to better than a couple of hours.**

| Spent on | Elapsed | GPU-h (A100 at assumed 2x) |
|---|---|---|
| 30 Sep first run and six failures | not recorded | ~2 (est.) |
| 22598608, 2K concurrency sweep | 00:52:39 | 0.88 |
| 22685911, 8K sweep | 01:03:12 | 1.05 |
| 22685764, A100 2K, produced nothing | 00:17:03 | 0.57 |
| 22713175, A100 diagnostic | 00:03:35 | 0.12 |
| 22713173, 32K | 01:16:07 | 1.27 |
| 22714602, 32K k probe (4 values) | 01:07:08 | 1.12 |
| 22737716, 2K k probe (3 values) | 00:32:42 | 0.55 |
| 22737717, 8K k probe (3 values) | 00:42:01 | 0.70 |
| 22754950, A100 diagnostic | 00:11:06 | 0.37 |
| 22757472, 32K n-gram k probe | 01:01:22 | 1.02 |
| 22757473, 2K n-gram k probe | 00:58:58 | 0.98 |
| vLLM multi-architecture rebuild | ~4 h wall | **0** — login node |

Sahil has used ~24 CPU-hours (~1.5 GPU-hours) of his own 10, so his
allocation is the project's main remaining GPU resource.

---

## Open items

- **Fix `summarize.py` to group by `env.slurm_job`**, or at minimum warn on
  duplicate keys. This is the one software change this week demands: the
  current behaviour hid a replication for four days.
- **Analyse Sahil's 22757102 / 22757104 / 22757105.** Logs are in the shared
  directory. Two have identical sizes, so he may have run the same
  configuration twice.
- **Reconcile the budget** against `sreport`, and confirm or kill the 2x A100
  billing assumption.
- **Triton divergence between the two venvs.** The multiarch install pulled
  `tokenspeed-triton-3.8.10` and may have displaced `triton-3.7.1`. If the
  versions differ, any A40-vs-A100 gap is confounded by more than the
  architecture list, which breaks the "same binary" premise of C4.
- **Bootstrap the agreement result.** Per-prompt, free, and the claim most in
  need of an interval.
- **A100 measurement cells.** Sahil was assigned that axis; coordinate before
  spending your own hours on it.
- **gpub073.** One slow node, with the ratio moving in the direction a
  bandwidth-limited machine would produce. Worth one line in the limitations
  rather than a campaign, unless it recurs.
- **n-gram at 8K was never swept**, so the middle R_draft point is missing
  for that rung.
- **n-gram beyond k=7** has never been tried; speedup is still rising at k=7
  at both buckets, so the optimum is unbracketed from above.
- **Why both drafter costs are sublinear in context** — a 16x prompt gives a
  2x cost in both rungs, unexplained. A fourth bucket would help.
- **The architecture guard allows only 80 and 86**, so it aborts H200 jobs the
  multiarch build could run.
- **Professor not yet told about the scope change:** approved for a hardware
  comparison, delivering a context-axis inversion.
- **Prefix-caching-off control not run.**
- **Quality evaluation not run** — free, login node.
- **Eager-vs-compiled divergence** (8 Oct) not yet folded into the
  losslessness note.
