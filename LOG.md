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

Two conventions, because both have bitten us:

- A Slurm state of `COMPLETED` means the shell script finished its loop, not
  that the measurements succeeded. Job 22685764 is `COMPLETED` and produced
  nothing. **Always check that `results.jsonl` grew before believing a job
  worked.**
- `sbatch` snapshots the batch script at submit time but **not** the files it
  calls at run time. Editing the harness after submitting changes what a
  queued job will do.

---

## Replaying a cell from scratch

The short version, for a teammate starting today. Paths assume
`PROJ=/projects/bikd/$USER/Speculative_Ladder`.

```bash
# 1. environment
module load cray-python/3.12.12
source $PROJ/vllm_env_multiarch/bin/activate     # or ~/vllm_env, see 8 Oct

# 2. submit one cell. phase0.sh reads BUCKET, LIMIT, CONC, KLIST and
#    LADDER_VENV from the environment, and reads the GPU from nvidia-smi
#    at run time rather than trusting the partition flag.
cd $PROJ/ladder/scripts
sbatch --export=ALL,BUCKET=2k,LIMIT=25,CONC=1 phase0.sh

# 3. watch it
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -X

# 4. read the results
python3 summarize.py $PROJ/results/a40/results.jsonl
```

Check each script's header comment before relying on a variable name; the
scripts are the source of truth and this file can drift.

---

## 8 Oct 2026 — drafter cost isolated, A100 unblocked

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

Step 1 passed. Step 2 reported `rms_norm OK` under
`CUDA_LAUNCH_BLOCKING=1` on an A100. That is a real sm_80 kernel launch, so
**the blocker is resolved and the hardware axis (C4) is reachable again.**

The same log shows `import vllm._C` raising `ModuleNotFoundError`. That is a
defect in the diagnostic probe, not the build: this vLLM ships stable-ABI
extensions named `_C_stable_libtorch`, `_moe_C_stable_libtorch` and
`_vllm_fa2_C`, and has no module named `vllm._C` at all. The same probe fails
identically on the A40, where everything works. The line that matters is the
`rms_norm` dispatch on the next line.

**C9 confirmed: R decomposes.** With the 2K and 8K sweeps in, fitting
`R(k) = R_draft + c*k` over k = 1, 2, 7 at each bucket:

| bucket | R(1) | R(2) | R(7) | R_draft | c |
|---|---|---|---|---|---|
| 2K  | 1.305 | 1.322 | 1.360 | **0.296** | 0.0092 |
| 8K  | 1.450 | 1.487 | 1.544 | **0.434** | 0.0157 |
| 32K | 1.677 | 1.704 | 1.766 | **0.662** | 0.0148 |

The fixed per-step drafter cost grows **2.24x** from 2K to 32K. The marginal
verification cost per drafted token is flat within noise. Drafting is 82 / 80
/ 86 percent of DFlash's total overhead at 2K / 8K / 32K.

**This is the mechanism behind the inversion.** Context taxes the drafter's
own forward pass, not verification. n-gram has no forward pass to tax, which
is why it barely decays and eventually wins. Write-up in
`notes/2026-10-08-drafter-cost-scaling.md`.

*Footnote worth not forgetting:* first-position acceptance varies with k
(0.570 / 0.589 / 0.601 at 2K for k = 1 / 2 / 7). Longer accepted runs sample
a different context distribution, so "first-position acceptance" is not
independent of k.

**Confirmed the paired bootstrap is not available on this build.**
`request_timing(o)` is populated on **0 of 25** requests — this vLLM does not
emit per-request metrics. There is one aggregate number per run and nothing to
resample, so wall-clock speedup intervals can only come from repeated whole
runs. Acceptance-side quantities may still be bootstrappable from data already
on disk if the per-record counters are per-request rather than run-aggregate;
unchecked.

**Decision:** nothing is promoted from `paper/claims.md` to
`paper/conclusions.md` without an uncertainty interval. The inversion is
currently recorded as a measurement, not a claim.

## 7 Oct 2026 — the k sweep refutes a prediction

**32K landed, and the ranking inverted.** At concurrency 1: DFlash 0.83x,
n-gram 1.08x. At 2K those were 1.86x and 1.16x. The ordering of the two rungs
reverses with context length. Write-up in `notes/2026-10-06-32k-inversion.md`.

**Predicted that k=1 would rescue DFlash at 32K, and was wrong.** Predicted
1.17x, measured **0.76x** — the worst cell in the sweep.

*Why the prediction failed:* we fitted `R = 1 + k*c` to a single data point by
*assuming* the intercept was 1. It is not. The true fit at 32K is
`R(k) = 1.662 + 0.0148k`. There is a large fixed cost that does not shrink
when k shrinks, so cutting k cannot help. Write-up in
`notes/2026-10-07-k-sweep-refutation.md`.

**Also retracted:** "context does to k what load does to k". Wrong.
Optimal k **rises** with context and **shrinks** with load.

The k probe, reused unmodified at every bucket because it takes its grid from
the environment:

```bash
sbatch --export=ALL,BUCKET=32k,LIMIT=25,CONC=1,KLIST="1 2 7 15" kprobe_32k.sh
```

**Submitted the same probe at 2K and 8K**, to see whether the fixed cost is
what context is taxing:

```bash
sbatch --export=ALL,BUCKET=2k,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_32k.sh  # 22737716
sbatch --export=ALL,BUCKET=8k,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_32k.sh  # 22737717
```

**Started the vLLM rebuild**, zero GPU hours, in tmux so a closed laptop
could not kill it:

```bash
tmux new -s build
bash rebuild_multiarch.sh          # then Ctrl-b then d to detach
```

`rebuild_multiarch.sh` builds the *same commit* `bf13ecc2f` with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` into `$PROJ/vllm_env_multiarch`, and
never touches `~/vllm_env`. Keeping the old venv intact matters: it produced
every measurement we already have.

*Build failure:* `ModuleNotFoundError: No module named 'setuptools_rust'`.

*Cause:* the script looked for `requirements/build.txt`, but
`requirements/build` is a **directory**.

*Fix:*

```bash
pip install -r requirements/build/cuda.txt -r requirements/build/rust.txt
```

*Operational note for anyone doing this:* tmux survives an SSH drop, but Delta
login nodes are round-robin, so you must reattach **on the same login node**.
`tmux ls` on `dt-login01` will not show a session started on `dt-login02`.
The build ran about four hours (432 object files, 451 FlashAttention-3 `.cu`
sources, `ninja -j 4`).

## 6 Oct 2026 — 8K decays, A100 collapses

**8K completed** (22685911, 63 minutes). The two rungs decay in *opposite*
directions. Decay `D = 1 - AL(long)/AL(2K)` at the common k=5 truncation:
DFlash 0.219, n-gram 0.044. n-gram's coverage actually *rises*, 0.132 to
0.161. At 2K the two had near-identical first-position acceptance; at 8K
n-gram is ahead, 0.563 to 0.451. The speedup gap at concurrency 1 fell from
0.70 to 0.12. Write-up in `notes/2026-10-06-8k-context-decay.md`.

**A100 lost all nine arms** (22685764) to
`cudaErrorNoKernelImageForDevice`. Seventeen minutes billed, nothing written.

Three wrong diagnoses before the right one — see the dead-ends table below.
What finally worked was bisection instead of inference:

```bash
sbatch diag_a100.sh                                                # 22713175
```

`diag_a100.sh` runs four layers cheapest-first under
`CUDA_LAUNCH_BLOCKING=1`, and the first failure names the culprit:

1. plain torch bf16 matmul and `fill_` — no vLLM anywhere
2. vLLM's own `rms_norm` custom op — no torch.compile, no CUDA graphs
3. vLLM eager, `enforce_eager=True`
4. vLLM compiled, the configuration that failed

`CUDA_LAUNCH_BLOCKING=1` is the point of the exercise. The original traceback
pointed at `output.fill_(0)` and `.contiguous()`, both trivial ops, which is
what an asynchronous CUDA error looks like when it finally reaches a sync
point. With blocking launches the traceback lands on the kernel that actually
failed.

Result, in 3m35s: step 1 passed, step 2 failed. Node, driver and torch are
healthy; **vLLM's extension does not execute on sm_80.** Confirmed by
counting cubins per architecture:

```bash
find $VD -name '*.so' -exec sh -c 'echo "== $1"; cuobjdump --list-elf "$1"' _ {} \;
```

sm_86: 40 cubins. sm_80: 20. The binary ships an sm_80 half that is missing
half its kernels. Write-up in `env/a100-sm80-failure.md`, support matrix in
`env/gpu-support.md`.

**Submitted 32K** (22713173), grid 1/2/4, limit 25, 3h wall — the decisive
cell for whether the ranking inverts.

**Set up team access.**

```bash
bash share_with_team.sh
```

The script copies the venv into project space with paths rewritten, then
`chgrp -R` *after* `cp -a` (the original did it before, so new files inherited
the wrong group), sweeps the whole tree, and verifies by printing group and
mode. Group corrected from `grp_202` to `delta_bikd`. Sahil's jobs 22713193
and 22713194 still failed in seven seconds each with ExitCode `0:53`, on
different partitions. **Still undiagnosed.** His job 22713396 was queued on
A100 and should have been cancelled, since no A100 job could have worked that
day.

**Cerebras.** The professor sent `sdk.cerebras.ai`. That is the CSL kernel
development SDK for the Wafer-Scale Engine, not an inference API — no models,
no endpoints, no decoding parameters. Porting a serving stack to it is not a
semester task. Its value is as a *prediction target*: wafer-scale keeps
weights in on-chip SRAM, so the ridge point sits roughly an order of magnitude
below a GPU's, and our R model predicts speculation is worthless there even at
batch 1. Discussion section, not experiment plan. Specs need verifying from
Cerebras's own sheet first.

## 5 Oct 2026 — losslessness, and harness v2

**Harness v2** (`HARNESS_VERSION = "v2-2026-10-05"`). Three additive changes,
no default behaviour altered, installed before either queued job started so
both picked it up without a cancel:

- full output token ids, because a SHA-1 cannot distinguish one flipped token
  from a different answer, and that distinction *is* the losslessness result
- per-request timing, intended for a paired bootstrap
- a prefix-caching switch (`--prefix-caching` / `--no-prefix-caching`)

Every addition is guarded, so an instrumentation failure records a null rather
than losing the run.

**Investigated losslessness.**

```bash
python3 check_lossless.py $PROJ/results/a40/results.jsonl
```

Version 1 of this script compared *generation lengths*, which was useless:
every generation hits the 256-token cap, so all lengths are equal by
construction. Rewrote it to compare token ids, and — the part that mattered —
to run the baseline against *itself* first as a control.

The control settled it. The `off` arm compared against itself at different
concurrency, with no speculation anywhere, agrees on only 12.7% of sequences.
Speculative arms agree with the baseline on 19.0%. Per-token divergence is
0.80% baseline-vs-baseline against 0.65% speculative-vs-baseline.
**Speculation perturbs the output less than the engine perturbs itself.** The
question "is speculation lossless" is not answerable in absolute terms on this
engine; it is only answerable relative to the engine's own noise floor.
Write-up in `notes/2026-10-05-losslessness-and-engine-nondeterminism.md`.

**Submitted two cells:**

```bash
sbatch --export=ALL,BUCKET=8k,LIMIT=25,CONC="1 8" phase0.sh       # 22685911
sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2k,LIMIT=25 \
       phase0.sh                                                   # 22685764
```

## 1 Oct 2026 — first valid measurement, and the coverage correction

```bash
sbatch --export=ALL,BUCKET=2k,LIMIT=25,CONC="1 8 32" phase0.sh   # 22598608
```

52 minutes on one A40. Three rungs (`off`, `ngram`, `dflash`) at concurrency
1, 8 and 32.

**The result that set up the whole project:** tau is flat across load (DFlash
2.543, 2.507, 2.470) while speedup collapses (1.86x, 1.49x, 1.03x). All of
the variation lives in R, the verification cost ratio, not in acceptance.

**Found a bug in how we were reading vLLM's numbers.** n-gram proposes a draft
on only 13.2% of decode steps, but vLLM's reported acceptance rate is
conditional on a draft having been proposed. Taken at face value it overstates
what n-gram delivers by more than a factor of two. Added a third term:

```
Speedup ~= [cov * tau_cond + (1 - cov)] / R
```

Corrected, n-gram's R is 1.082, *below* DFlash's 1.360 — which is what the
hardware predicts, since n-gram does no model forward pass. `summarize.py`
now reconstructs coverage from the draft and generation counters rather than
printing vLLM's acceptance directly.

## 30 Sep 2026 — environment, and six jobs that failed in five seconds

**Built the environment.** Qwen3-8B and the DFlash drafter head downloaded to
`/work/nvme/bikd/$USER/Speculative_Ladder/hf/hub`, LongBench-v2 bucketed into
2K / 8K / 32K by prompt length, vLLM built from source at commit `bf13ecc2f`
into `~/vllm_env`.

**Verified the runner pin by reading the source**, not by trusting a flag:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

This matters more than it looks. With the V2 runner, n-gram speculation
silently falls back to the stable runner while DFlash stays on V2, so the two
rungs would be measured on different code paths and the comparison would be
meaningless. The harness now refuses to start without this set.

**Six jobs failed in about five seconds each**, all with
`ModuleNotFoundError: No module named 'vllm'`.

*Cause:* the scripts ran `module load cray-python` but never activated the
venv. Loading the Python module is not the same as entering the environment.

*Fix:* added `source "$LADDER_VENV/bin/activate"` to the job script, with
`LADDER_VENV` defaulting to `~/vllm_env`.

**First real run** (22569403). All prompts submitted at once.

*What went wrong:* nothing crashed, which is worse. vLLM batched as wide as
memory allowed, so every rung lost to no speculation at all — 0.86x, 0.78x,
0.71x — while acceptance looked healthy. The run measured vLLM's batching
decision, not the drafter.

*Fix:* explicit concurrency control via `max_num_seqs`, exposed as `CONC`.
The speedups from this run are void; the acceptance numbers are still usable.
We kept the run in the results because it is the right-hand end of the load
axis, not a mistake.

---

## Job index

| Job | Date | GPU | Cell | State | Outcome |
|---|---|---|---|---|---|
| 22754950 | 8 Oct | A100 | diagnostic, multiarch venv | steps 1-2 pass | **A100 unblocked** |
| 22737717 | 7 Oct | A40 | 8K k sweep | COMPLETED | R_draft = 0.434 |
| 22737716 | 7 Oct | A40 | 2K k sweep | COMPLETED | R_draft = 0.296 |
| (32K k probe) | 7 Oct | A40 | 32K k sweep | COMPLETED | refuted the k=1 prediction |
| 22713396 | 6 Oct | A100 | 8K (ssashi) | PENDING | should have been cancelled |
| 22713194 | 6 Oct | A40 | ? (ssashi) | FAILED 00:00:07 | undiagnosed, exit 0:53 |
| 22713193 | 6 Oct | A100 | ? (ssashi) | FAILED 00:00:07 | undiagnosed, exit 0:53 |
| 22713175 | 6 Oct | A100 | diagnostic | COMPLETED 00:03:35 | **answered the A100 failure** |
| 22713173 | 6 Oct | A40 | 32K | COMPLETED | **the inversion** |
| 22685911 | 5 Oct | A40 | 8K | COMPLETED 01:03:12 | opposite-direction decay |
| 22685764 | 6 Oct | A100 | 2K | COMPLETED 00:17:03 | **all 9 arms failed**, nothing written |
| 22598608 | 1 Oct | A40 | 2K concurrency sweep | COMPLETED 00:52:39 | first valid measurement |
| 22569403 | 30 Sep | A40 | 2K, uncontrolled | COMPLETED | speedups void, acceptance usable |
| (six jobs) | 30 Sep | A40 | — | FAILED ~00:00:05 | `ModuleNotFoundError: vllm` |

---

## Decision register

| Decision | Why | Date |
|---|---|---|
| Pin `VLLM_USE_V2_MODEL_RUNNER=0`, refuse to start without it | n-gram silently falls back to the stable runner while DFlash stays on V2, so the rungs would run on different code paths | 30 Sep |
| Control concurrency explicitly rather than letting vLLM batch freely | Otherwise speedup measures the batching choice, not the rung | 1 Oct |
| Keep the uncontrolled 30 Sep run in the results | It is the right-hand end of the load axis, not a mistake | 1 Oct |
| Shrink the concurrency grid per context bucket | vLLM accepts an impossible `max_num_seqs` and quietly schedules fewer, mislabelling the row | 5 Oct |
| Read the GPU from `nvidia-smi` at run time, not from the partition flag | A mislabelled result is harder to notice than a failed job, and poisons the cross-hardware table | 5 Oct |
| Keep prefix caching on | It is what a real server does, and it cancels out of within-bucket speedup ratios. Revisit before the headline table | 5 Oct |
| Store full token ids, not just a hash | A hash cannot distinguish one flipped token from a different answer, and that distinction is the losslessness result | 5 Oct |
| Do **not** switch A100 to `TRITON_ATTN` to make it run | A40 ran on FlashAttention; mixing backends would put the difference into R, the quantity being measured | 6 Oct |
| Rebuild the **same commit** with three architectures, into a **new** venv | A rebuild in place would replace the binary that produced every existing measurement | 7 Oct |
| No claim reaches `conclusions.md` without an interval | Every number so far is a single run | 8 Oct |

---

## Dead ends, and what they cost

Recording these is cheaper than rediscovering them.

| What was believed | Why it was wrong | Cost |
|---|---|---|
| A100 is supported, because `cuobjdump` lists sm_80 | `ls $VD/*.so` is not recursive and missed the FlashAttention extension in a subdirectory. Worse: sm_80 *was* listed in the main extension and the kernels still did not run. A listing is not proof of execution | one failed A100 job, ~17 min billed |
| The A100 failure is compile-cache poisoning from the A40 runs | Per-config and `torch_aot_compile` hashes are disjoint across GPUs, and the A100 run logged nine saves and zero loads, so it compiled everything fresh | ~20 min of analysis, no GPU |
| The failure is in FlashAttention, per the traceback | `output.fill_(0)` at `flash_attn.py:1211` is the profiling shortcut that skips attention entirely. The error was sticky, from an earlier failed launch. CUDA errors surface where they are noticed, not where they happen | sent the diagnosis down the wrong path twice |
| Sahil could not read the venv because its group was `grp_202` | `id ssashi` shows `grp_202` is his *primary* group. He could read it all along | a `chgrp` over 100k files; his failures are still undiagnosed |
| `same length` would distinguish a token flip from a different answer | Every generation hits the 256-token cap, so all lengths are equal by construction | one useless diagnostic, rewritten |
| k=1 would rescue DFlash at 32K | Fitted `R = 1 + k*c` to a single point by assuming the intercept. The real intercept is 1.662 | one wrong prediction, caught by the sweep that tested it |
| Optimal k shrinks with context, as it does with load | It **rises** with context. k=7 beat k=15 only at 2K *saturation*. The two axes act in opposite directions | one retracted claim |
| `requirements/build.txt` holds the build dependencies | `requirements/build` is a directory | one failed build start |
| Per-request timing would give us a paired bootstrap | This vLLM build populates it on 0 of 25 requests | one harness revision |

**The pattern in most of these:** a conclusion drawn from a proxy — a file
listing, a directory name, a traceback line, a group name, a single data point
with an assumed intercept — instead of testing the thing itself. The fix that
works is bisection, cheapest layer first, as in `diag_a100.sh`. Ten minutes of
GPU time, and it should have been the first move every time.

---

## Budget

Sub-allocation: 10 GPU-hours deposited per student. More is available on
request; the professor has confirmed this.

| Spent on | Approx. |
|---|---|
| 30 Sep first run and failures | ~2 h |
| 22598608, 2K sweep | ~0.9 h |
| 22685911, 8K sweep | ~1.1 h |
| 22685764, A100, produced nothing | ~0.6 h (A100 bills ~2x) |
| 22713175, diagnostic | ~0.1 h |
| 22713173, 32K | ~2 h |
| 32K k probe | ~0.7 h |
| 22737716 + 22737717, 2K and 8K k sweeps | ~1.3 h |
| 22754950, A100 diagnostic | ~0.2 h (A100 bills ~2x) |
| vLLM rebuild | **0 h** — login node, no GPU |

The balance is effectively exhausted before any top-up. Delta's `accounts`
balance has lagged visibly behind actual usage, so treat these as estimates.
**Request a top-up before submitting anything else**, not after a job is
refused.

---

## Open items

- **Triton divergence between the two venvs.** The multiarch install pulled
  `tokenspeed-triton-3.8.10` and may have displaced `triton-3.7.1`. If the
  versions differ, any A40-vs-A100 gap is confounded by more than the
  architecture list, which breaks the "same binary, only architectures
  differ" premise of C4. Free to check:
  `for V in ~/vllm_env $PROJ/vllm_env_multiarch; do "$V/bin/pip" list | grep -Ei 'triton|^torch |vllm'; done`
- **No confidence intervals anywhere.** Repeated runs are the only route for
  speedup. Check the stored JSONL schema first to see whether tau and
  coverage can be bootstrapped from data already on disk.
- **n-gram's R_draft has never been measured**, only assumed near zero.
- **Three buckets cannot identify how R_draft scales** with context.
- **Sahil's 7-second failures** (22713193, 22713194, exit `0:53`) remain
  undiagnosed. His group access was never the problem.
- **A100 cells not yet submitted** against the new venv.
- **Allocation top-up not requested.**
- **Professor not yet told about the scope change:** approved for a hardware
  comparison, delivering a context-axis inversion.
- **Prefix-caching-off control not run.**
- **Quality evaluation not run** — free, login node, decode the stored token
  ids and score LongBench-v2 multiple choice.
