# Work log

A record of what we ran, what broke, and how it was fixed.

**Newest first.** The common read of this file is "where are we now", so the
latest entry is at the top. Entries stay internally chronological. If you are
starting from scratch rather than catching up, read the replay block below,
then jump to the bottom and come forward.

For **what we learned**, see the dated notes in `notes/`, the claim register
in `paper/claims.md`, and the full grid in `paper/results-table.md`. This file
is the record of *activity*, including the parts that did not work.

Four conventions, because each has bitten us:

- A Slurm state of `COMPLETED` means the shell script finished its loop, not
  that the measurements succeeded. Job 22685764 is `COMPLETED` and produced
  nothing. **Always check that `results.jsonl` grew before believing a job
  worked.**
- `sbatch` snapshots the batch script at submit time but **not** the files it
  calls at run time. Editing the harness after submitting changes what a
  queued job will do.
- **`BUCKET` is case sensitive**: `2K`, `8K`, `32K`.
- **Analyse `results.jsonl` per record, not through `summarize.py`.** Every
  record carries `env.slurm_job` and `env.host`. `summarize.py` keys on
  `(bucket, conc, rung, k)` and silently keeps one record per key, so two jobs
  measuring the same cell are rendered as one row. This cost three wrong
  write-ups on 11 Oct.

---

## Replaying a cell from scratch

Paths assume `PROJ=/projects/bikd/$USER/Speculative_Ladder` and the shared
tree at `/projects/bikd/dagraw2/Speculative_Ladder`.

```bash
module load cray-python/3.12.12
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
cd $SHARED/ladder/scripts

# LADDER_VENV is MANDATORY: without it the script prefers $HOME/vllm_env,
# which has no A100 kernels. LADDER_OUT keeps your results out of someone
# else's results.jsonl.
sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,\
LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/a40,\
BUCKET=2K,LIMIT=50,CONC=1 phase0.sh

squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -X
```

Read results per record, which is the only way to see which job produced what:

```bash
python3 - <<'PY'
import json
for l in open('results/a40/results.jsonl'):
    r = json.loads(l); e = r['env']
    print(e['slurm_job'], e['host'].split('.')[0], e['gpu'],
          r['rung'], 'k=%s'%r['k'], r['bucket'],
          'c=%s'%r['max_num_seqs'], 'n=%s'%r['limit'],
          round(r['throughput_tok_s'],2))
PY
```

**A repeat gets its own `LADDER_OUT`.** Per-request filenames encode bucket,
rung, k and concurrency but not the run, so a second run into the same
directory overwrites the first.

---

## 11 Oct 2026 (latest) — decision tree committed; analysis tools; subsets

Still before any record from 22802261 or 22802262 was read.

- `notes/2026-10-11-a100-decision-tree.md`: how each outcome of the A100
  test is read, fixed in advance; the plan to the 20 Oct review under
  wall-clock rather than hours; measurement stops end of 17 Oct.
- **`SEED` is inert.** `run_benchmark.py` passes it only to `SamplingParams`
  and reads the first `LIMIT` lines of the bucket file, so the planned
  `SEED=5678` repeat would have re-measured the same prompts. Struck. Found by
  reading the script, not inferring it (RULES section 13).
- **Intervals come from disjoint prompt subsets.** `code/make_subsets.py`
  slices the shared bucket files into `sub1..sub4` directories under
  `/projects/bikd/$USER/Speculative_Ladder/data/`, each holding the file under
  the canonical name so `phase0.sh`'s `$DATA/longbench_v2_${BUCKET}.jsonl`
  resolves; refuses to write on any `_id` overlap. Subset 0 is the existing
  measurement. Extensible: more subsets later, nothing redone.
- **`summarize.py` now groups by job** and announces every cell measured in
  more than one job; speedups are within-job. Open item 1 closed. Verified on
  `results/a40`: nine cells listed as repeated, which is the replication that
  the old table hid.
- **`code/analyze.py`**: per-record table with job, host, GPU and subset;
  `--predict BUCKET:RUNG:K=VALUE` scores a frozen prediction per cell with
  signed errors and scores the 32K ordering separately; spreads across jobs,
  labelled as prompt-subset intervals or same-prompt noise. Checked against
  the register: 32K DFlash k=7 0.8260 / 0.8259, n-gram 1.0782 / 1.0775.
- The 2K, 8K and 32K cells share documents (all 25 of 32K's ids are in 2K's
  first 50); recorded in `paper/threats.md`.

Commands for the subsets, login node:

```bash
module load cray-python/3.12.12
cd /projects/bikd/dagraw2/Speculative_Ladder/repo && git pull
python3 code/make_subsets.py            # writes sub1..sub4 for 2K, 8K, 32K
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
D=/projects/bikd/$USER/Speculative_Ladder
cd $SHARED/ladder/scripts
for s in 1 2 3 4; do
  for b in 2K:50 8K:50 32K:25; do
    B=${b%%:*}; N=${b##*:}; mkdir -p $D/results/a40_sub$s
    sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_DATA=$D/data/sub$s,LADDER_OUT=$D/results/a40_sub$s,BUCKET=$B,LIMIT=$N,CONC=1 phase0.sh
  done
done
python3 $SHARED/repo/code/analyze.py $D/results/a40_sub1 $D/results/a40_sub2 ...   # after they land
```

---

## 11 Oct 2026 (later) — spec sheets checked; the A40 ridge point was wrong

The second review withdrew C17's bandwidth mechanism on the strength of two
ridge points it marked unverified. Checked against NVIDIA's own datasheets
(no GPU, login-node work only):

| GPU | dense bf16 | bandwidth | ridge | source |
|---|---|---|---|---|
| A40 | 149.7 TFLOPS | 696 GB/s | **~215** | A40 datasheet, Mar22 (299.4 is the sparse figure) |
| A100-SXM4-40GB | 312 TFLOPS | 1,555 GB/s | ~201 | A100 datasheet, Jun21 r4, 40GB SXM column |
| H200 SXM | ~989 TFLOPS | 4.8 TB/s | ~206 | nvidia.com/h200: 1,979 TFLOPS "with sparsity", halved |

**The A40's 107 was wrong.** The 27 Sep compute memo took 74.8 TFLOPS by
halving 149.7 on the belief that it was the sparse figure. It is the dense
one. What followed:

1. **New claim C18: plain decode is bandwidth-bound at batch 1.** A100/A40
   `off` throughput 72.8/33.0 = 2.21x against a bandwidth ratio of 2.23x and a
   compute ratio of 2.08x. Both machines reach ~78% of peak bandwidth on the
   target's weights. First spec-sheet number checked against a measurement.
2. **C17's mechanism stays unexplained, with an elimination argument.** At
   batch 1 every pass in a decode step runs at about 8 to 16 FLOP/byte,
   far below every ridge here, so roofline predicts `R` identical on all
   three GPUs. The measured +5.4% is therefore non-roofline. The review's
   "roofline predicts more gain on A100" is withdrawn along with the 107.
3. **There is no compute-to-bandwidth ratio axis across A40 / A100 / H200.**
   All three sit at 201 to 215. C4 and C17 are restated as transfer across a
   **bandwidth** axis (0.70 to 4.8 TB/s, ~7x). The proposal's stated reason
   for holding out the A40 traces to the halved figure. The Drive compute
   memo still carries it and is outside this repository; `HANDOFF.md`'s Drive
   table now warns.

Swept per RULES section 8: `HANDOFF.md` (2.7, 4.2, open item 4, Drive table),
`paper/results-table.md`, `env/gpu-support.md`, a further banner on
`notes/2026-10-11-hardware-transfer.md`, and inline markers in this file.
`notes/PROJECT-BRIEF-for-review.md` keeps its 107 as a dated snapshot.
`paper/claims-corrections-2026-10-11b.md` deleted: it was folded into the
register in 597d4e3.

---

## 11 Oct 2026 (late) — second review, and four claims withdrawn

A second independent review, briefed only from `HANDOFF.md` and the
repository, produced four objections, all backed by arithmetic. **All four
accepted**, and three concern entries written earlier the same day. Full text
in `paper/claims-corrections-2026-10-11b.md`, folded into `paper/claims.md`.
None needed a GPU: a memory budget, a counterfactual on two measured numbers,
and a ratio of two spec-sheet figures.

1. **The capacity confound is withdrawn.** A concurrency-1 32K cell needs
   ~24 GB — 16.4 weights + 4.9 target KV + 1.9 drafter + 0.7 drafter KV —
   against the A100's 40 GB. No capacity pressure exists at the concurrency
   the prediction is made at, so the clause was an escape hatch that had never
   been checked. Withdrawn **before** any A100 cell at full n was read: jobs
   22802261 (8K) and 22802262 (32K) were submitted and unread. 8K is still
   weighted higher, on a better basis — the 1.054 factor was fitted at 2K and
   `R`'s composition changes with context.
2. **C13 is wrong as stated.** Cost alone does not flip the ranking: 2K
   acceptance with 32K cost leaves DFlash at 1.44x, ahead of n-gram's 1.08x.
   Acceptance alone roughly does: 32K acceptance with 2K cost gives 1.07x, a
   tie. Both channels are needed; acceptance dominates.
3. **C17's mechanism is withdrawn.** "Speculation reclaims wasted memory
   bandwidth and a faster machine wastes less" contradicts the ridge points:
   A100 ~201 FLOP/byte against A40 ~107, so naive roofline predicts
   speculation should gain *more* on A100, and it gained less. Observation
   stands, mechanism unexplained. Candidate hypothesis recorded — fixed
   per-step overheads are a larger fraction of a faster GPU's step time.
   **The spec sheets are now a prerequisite for any mechanism claim.**
   **[Superseded 11 Oct (later): the A40's ridge point is ~215, not ~107, so
   this roofline argument is withdrawn too. Mechanism still unexplained; see
   the newer entry.]**
4. **C8's two realizations share one prompt set** — same 25 prompts, same
   seed. They bound node and run-to-run variance, not prompt-sample variance.
   Closing it needs a 32K concurrency-1 cell on a different prompt subset,
   ~1 GPU-hour.

Swept every file that repeated a withdrawn claim, per RULES section 8:
`HANDOFF.md` (eight sites), `paper/results-table.md` (four),
`paper/conclusions.md`, `TODO.md` and this file. Dated notes under `notes/`
keep their bodies and carry superseded banners, because rewriting a dated note
destroys the record of what was believed when.

Also added, not yet submitted: `code/c10_control.sh` and `code/c10_control.py`
— the same-config control C10 needs.

---

## 11 Oct 2026 — first A100 data, and tau transfers

### A smoke test answered the central question

Sahil ran the quickstart smoke test on 8 Oct: 2K, concurrency 1, `LIMIT=5`,
`vllm_env_multiarch`, three jobs, ~20 minutes each.

| job | GPU | node |
|---|---|---|
| 22757102 | A40 | gpub067 |
| 22757105 | A40 | gpub085 |
| **22757104** | **A100-SXM4-40GB** | **gpua045** |

**The project's first A100 measurements**, produced by a run intended only to
prove a teammate's pipeline worked.

```bash
sacct -j 22757102,22757104,22757105 \
  --format=JobID,Partition%14,NodeList%12,State,Elapsed,ExitCode -X
python3 $SHARED/ladder/scripts/summarize.py \
  /projects/bikd/ssashi/Speculative_Ladder/results/a100
```

**tau transfers; R does not.**

| | A40 | A100 | change |
|---|---|---|---|
| DFlash tau_cond | 2.594 | 2.572 | **-0.8%** |
| n-gram tau_cond | 2.533 | 2.525 | **-0.3%** |
| DFlash R | 1.358 | 1.433 | **+5.5%** |
| n-gram R | 1.075 | 1.132 | **+5.3%** |
| DFlash speedup k=7 | 1.89x | 1.79x | -5.3% |
| n-gram speedup k=5 | 1.14x | 1.06x | -7.0% |
| `off` throughput | 33.0 | 72.8 | A100 2.21x faster |

Acceptance agrees to under one percent across two different GPUs. The cost
ratio rises 5.5%, **and by the same amount for both rungs** — rungs that
differ 4x in drafting cost and 7x in coverage. That is the signature of a
machine property rather than a drafter property, and it is the split the whole
study assumes. Until now it was an assumption.

Direction is right too: the A100 is 2.21x faster at plain decoding and
speculation pays *less* there. Speculation reclaims wasted memory bandwidth,
and a faster machine wastes less. **Spec sheets not yet verified**, so that is
a direction rather than arithmetic. **[Superseded 11 Oct (late): the mechanism
is withdrawn. The A100's ridge point is *higher* than the A40's, so roofline
predicts speculation should gain more there. See the latest entry.]**
**[And that reason is itself withdrawn, 11 Oct (later): the A40's ridge
point is ~215, not ~107. Spec-sheet entry, top of this file.]**

Write-up in `notes/2026-10-11-hardware-transfer.md`; new claims C16 (tau
invariant) and C17 (R scales by one factor, rung-independent).

### The node-variance question answered itself, for free

22757102 and 22757105 ran the identical configuration on two different A40
nodes:

| | gpub067 | gpub085 |
|---|---|---|
| `off` | 32.95 | 32.94 |
| n-gram speedup | 1.1375 | 1.1372 |
| DFlash speedup | 1.8668 | 1.8934 |

Baselines agree to **0.03%**. Four A40 nodes have now been measured and three
agree tightly, so **gpub073 is an outlier, not typical spread** — and the
variance study we were about to buy GPU hours for is unnecessary.

### The prediction, frozen

With a second platform and a rung-independent factor, C4 becomes constructible:
fit on A40, apply the factor, predict A100. Recorded in `paper/claims.md`
under "Frozen prediction, 11 Oct 2026", **before any A100 cell at full n was
run** (rule 4).

| cell | predicted A100 R | predicted speedup |
|---|---|---|
| 8K DFlash k=7 | 1.627 | **1.20x** |
| 8K n-gram k=5 | 1.181 | **1.09x** |
| 32K DFlash k=7 | 1.861 | **0.78x** |
| 32K n-gram k=5 | 1.360 | **1.02x** |

The inversion is predicted to survive on A100 and widen slightly. 2K is in
sample and does not count. Pass is every cell within 5% and the 32K ordering
holding; fail is median absolute relative error above 10%, or 32K not
inverting.

**Pre-registered confound**, recorded so it cannot be used as an excuse later:
the A100-SXM4 has 40 GB against the A40's 48, so the 32K concurrency ceiling
is lower and that cell may sit in a different memory regime. A large 32K miss
may be capacity rather than model. **Judge the prediction on 8K.**
**[Withdrawn 11 Oct (late): a concurrency-1 32K cell needs ~24 GB against the
A100's 40 GB, so the clause was never physically possible. See the latest
entry.]**

Wrote `code/submit_a100_predict.sh` to run the test — a login-node wrapper
that calls the verified `phase0.sh` twice rather than reimplementing anything,
and that enforces `LADDER_VENV`, a per-user `LADDER_OUT`, and the per-bucket
prompt counts the prediction was derived from (50 at 8K, 25 at 32K). About
1.5 GPU-hours on Sahil's allocation, which has ~8.5 left.

### Three things I got wrong about Sahil's jobs

Worth recording because the pattern is the same each time.

I predicted his results would be missing because he had not set `LADDER_OUT`,
and that `records found: 0` in the shared tree would be that failure. He had
set it. Zero records in *our* tree was the correct behaviour. Checking the
clock properly: his logs are stamped Delta local time, so 14:21 there is 19:21
UTC, and the quickstart that made those variables mandatory was pushed at
18:56 UTC — he ran it **25 minutes after it landed** and followed it exactly.

I also assumed I would need him to grant read access to his tree. The
directories are `drwxrws---+` under `delta_bikd`, so it was readable already.

And I recommended spending the last four GPU hours on a node-variance study
that his smoke test had already settled.

### Cost

22757472 took 01:01:22 and 22757473 took 00:58:58 — **2.0 GPU-hours against
my estimate of 1.1.** Third time a per-arm extrapolation has underestimated by
roughly 2x. Quote the nearest whole job that actually ran, and call it a floor.

`accounts` reports **4 hours remaining of 10 deposited**. `sreport` reports
159 CPU-hours for dagraw2 and 24 for ssashi; dividing by an assumed 16 cores
per GPU gives ~9.9 and ~1.5 GPU-hour equivalents, which does not reconcile
with 6 charged. The divisor is an assumption. **Treat 4 as operative.**

### The n-gram sweep, same day

**Measured n-gram's R_draft for the first time.** It had been *assumed* near
zero because prompt lookup runs no forward pass, and that assumption was
load-bearing in the explanation of the inversion.

```bash
cp kprobe_32k.sh kprobe_ngram.sh
sed -i 's|^#SBATCH --job-name=kprobe32k|#SBATCH --job-name=kprobengram|' kprobe_ngram.sh
sed -i 's|^run ngram 5  ""|# n-gram moved into the k loop below|' kprobe_ngram.sh
sed -i 's|^  run dflash "\$K" "--draft \$DRAFT"|  run ngram "$K" ""|' kprobe_ngram.sh
grep -n 'job-name\|^run \|run ngram\|run dflash' kprobe_ngram.sh   # verify

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

n-gram's drafting cost is small but real, and **grows with context at nearly
DFlash's relative rate** — 1.96x against 2.24x. Prompt lookup has to *search
the prompt*, and a 32K prompt is sixteen times more text to scan. Both
drafters pay a context tax; one pays it as a forward pass, the other as a
string search, and the difference is a constant factor of four to five.

**So the inversion is driven by the absolute gap, not the scaling.** DFlash's
penalty exceeds n-gram's by 0.225 at 2K and 0.523 at 32K. Claims C12, C13.
**[Corrected 11 Oct (late): C12 stands, C13 was wrong as stated — cost alone
does not flip the ranking. See the latest entry.]**

**n-gram's acceptance is context-invariant** — tau_cond 1.670 at both buckets
at k=1, first-position acceptance 0.670 at both, D = -0.000 / -0.008 / -0.007.
Coverage *rises*, 0.121 to 0.195 at k=7. Claims C14, C15.

Acceptance by position at 32K, the clearest picture of why DFlash cannot
compete there:

```
dflash k=7   0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004
ngram  k=7   0.541, 0.382, 0.304, 0.256, 0.219, 0.193, 0.177
```

**First successful out-of-sample prediction.** The 2K fit gives
`R(5) = 1.086`; job 22598608 measured **1.083**, ten days earlier with k=5
excluded from the fit.

Write-up in `notes/2026-10-11-ngram-drafter-cost.md`.

### The 32K baselines, and a correction

The n-gram sweep re-measured the 32K baseline at 13.5 tok/s against the 14.9
and 14.8 in C11. My first reading was "32K variance is 9%", my second "an
occasional excursion". Both were written from `summarize.py` output. Reading
the records overturns both.

| | job 22713173, gpub019 | job 22714602, gpub047 |
|---|---|---|
| `off` c1 | 14.90 | 14.83 |
| `ngram` k=5 c1 | 16.07 | 15.98 |
| `dflash` k=7 c1 | 12.31 | 12.25 |
| **n-gram speedup** | **1.0785** | **1.0776** |
| **DFlash speedup** | **0.8262** | **0.8260** |

**The inversion has been replicated since 7 Oct**, ratios agreeing to 0.08%
and 0.02%. It was never a single run. `summarize.py` collapsed the duplicate
keys and printed one row, and the register copied that reading forward for
four days.

Ratios are the stable quantity: 0.08% across nodes, against 8.8% for absolute
throughput on gpub073. Both explanations for that node are refuted from the
logs — the `off` arm got 173,056 tokens in all three jobs, identical, and zero
preempt or recompute lines anywhere. That also clears the separate worry that
the 32K concurrency-4 cells were preempted.

**C15 demoted**: 1.14x against 1.18x is a 4% cross-job claim whose
long-context half is gpub073. Say "n-gram's speedup does not degrade with
context". Write-up in `notes/2026-10-11-noise-floor.md`, third version.

## 8 Oct 2026 — drafter cost isolated, A100 unblocked, teammate unblocked

**The rebuild finished.** Verified by counting cubins:

| arch | before | after |
|---|---|---|
| sm_80 (A100) | 20 | **71** |
| sm_86 (A40) | 40 | **51** |
| sm_90 (H200) | — | **82** |

```bash
# diag_a100.sh hardcodes ~/vllm_env and ignores LADDER_VENV, so a variant
# was produced with sed rather than editing the original.
sbatch diag_a100_multiarch.sh                                      # 22754950
```

**All four steps passed** in 11m06s, including step 4 — the compiled,
CUDA-graph configuration that lost all nine arms on 22685764. **The A100
blocker is resolved and needs no `enforce_eager` workaround.**

The log shows `import vllm._C` raising `ModuleNotFoundError`. That is a defect
in the probe, not the build: this vLLM ships stable-ABI extensions named
`_C_stable_libtorch`, `_moe_C_stable_libtorch` and `_vllm_fa2_C`. The probe
fails identically on the A40.

**Unplanned finding: eager and compiled disagree under greedy decoding.** Same
prompt, same seed, `temperature=0.0`:

```
EAGER OK:    ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
COMPILED OK: ' Paris. The capital of Italy is Rome. The capital of Germany is Berlin.'
```

A cleaner instance of the 5 Oct result: no speculation, no batching
difference, nothing but eager versus compiled.

**C9 confirmed: R decomposes.**

| bucket | R(1) | R(2) | R(7) | R_draft | c |
|---|---|---|---|---|---|
| 2K  | 1.305 | 1.322 | 1.360 | **0.296** | 0.0092 |
| 8K  | 1.450 | 1.487 | 1.544 | **0.434** | 0.0157 |
| 32K | 1.677 | 1.704 | 1.766 | **0.662** | 0.0148 |

Fixed drafting cost grows 2.24x from 2K to 32K; marginal verification cost per
token is flat. Drafting is 82 / 80 / 86 percent of DFlash's overhead.
`notes/2026-10-08-drafter-cost-scaling.md`.

*Footnote:* first-position acceptance varies with k (0.570 / 0.589 / 0.601 at
2K for k = 1 / 2 / 7), so it is not independent of k.

**The paired bootstrap is settled, and mixed.** `timing` holds only
`arrival_time`, and there are no per-request draft or accept counts, so
speedup and tau intervals require repeated whole runs. But `token_ids` *is*
per-prompt, so the losslessness quantities can be bootstrapped from data on
disk, for free, paired over prompts.

### Teammate access, and the seven-second failures

POSIX ACL precedence: when a **named user entry** matches, it is used and
group entries are never consulted. The tree carried `user:ssashi:r-x`, so his
`delta_bikd` membership and `group::rwx` were both irrelevant — read and
execute, no write, everywhere. `phase0.sh` line 10 hardcodes
`#SBATCH --output=.../logs/phase0-%j.out` and an `#SBATCH` directive cannot
take a variable, so Slurm could not create his output file and the job died at
launch with **no log written**. **Confirmed 11 Oct** when his jobs wrote 59 KB
logs.

```bash
setfacl -R -m u:ssashi:rwX -m d:u:ssashi:rwX "$P/logs" "$P/.vllm_cache"
chmod -R g+rX,g-w "$P/vllm_env_multiarch"    # 4m07s, ~100k files
```

`logs/` because Slurm must write there; `.vllm_cache` because the multiarch
build must write ~4 minutes of compiled kernels. `results/` deliberately not
granted. The venv chmod set the ACL mask to `r-x`, clamping `group::rwx`, so
nobody can modify the binary the hardware axis depends on.

**Two traps found reading `phase0.sh` properly for the first time.** The venv
fallback chain means **any A100 job without `LADDER_VENV` reproduces
22685764**. And the architecture guard is `case "$CAP" in 80|86)`, written when
the build had only those two, so it now wrongly aborts H200 jobs the multiarch
venv could run.

### KV cache varies by arm within a job

22685911's arms got 173,056 / 173,104 / 136,160 / 160,336 / 160,032 / 125,456
tokens — a 28% spread, the small ones being speculative arms reserving drafter
memory. Looked like a preemption risk at 32K concurrency 4. **Checked 11 Oct:
zero preempt lines. Not an issue.** The `Maximum concurrency` figure vLLM
prints is computed for a full `max_model_len` request in every log, so it is a
worst-case ceiling, not what got scheduled.

## 7 Oct 2026 — the k sweep refutes a prediction

**32K landed, and the ranking inverted.** DFlash 0.83x, n-gram 1.08x at
concurrency 1, against 1.86x and 1.16x at 2K.
`notes/2026-10-06-32k-inversion.md`.

**Predicted k=1 would rescue DFlash at 32K, and was wrong.** Predicted 1.17x,
measured **0.76x** — the worst cell in the sweep. We had fitted
`R = 1 + k*c` to a single point by *assuming* the intercept was 1; the true
32K fit is `R(k) = 1.662 + 0.0148k`. A large fixed cost does not shrink when k
shrinks. `notes/2026-10-07-k-sweep-refutation.md`.

**Also retracted:** "context does to k what load does to k". Optimal k
**rises** with context.

> **Corrected 11 Oct 2026.** This entry previously ended "and **shrinks** with
> load". The load half has no valid data: its only evidence is 22569403, the
> uncontrolled 30 Sep run recorded below as speedups-void, and no controlled k
> sweep exists at any concurrency above 1 at any bucket. The load axis is
> **unmeasured**, not established. See `paper/claims.md`, C6.

```bash
sbatch --export=ALL,BUCKET=32K,LIMIT=25,CONC=1,KLIST="1 2 7 15" \
       kprobe_32k.sh                                               # 22714602
sbatch --export=ALL,BUCKET=2K,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_32k.sh  # 22737716
sbatch --export=ALL,BUCKET=8K,LIMIT=25,CONC=1,KLIST="1 2 7" kprobe_32k.sh  # 22737717
```

22714602 also re-measured the full 32K concurrency-1 row, which is what made
the 11 Oct replication check possible.

**Started the vLLM rebuild**, zero GPU hours, in tmux:

```bash
tmux new -s build
bash rebuild_multiarch.sh          # then Ctrl-b then d to detach
```

Same commit `bf13ecc2f`, `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`, into a **new**
venv — the old one produced every existing measurement.

*Build failure:* `ModuleNotFoundError: No module named 'setuptools_rust'`,
because the script looked for `requirements/build.txt` and
`requirements/build` is a **directory**. Fix:

```bash
pip install -r requirements/build/cuda.txt -r requirements/build/rust.txt
```

*Operational note:* tmux survives an SSH drop, but Delta login nodes are
round-robin, so reattach **on the same login node**. Four hours, 432 object
files, 451 FlashAttention-3 `.cu` sources, `ninja -j 4`.

## 6 Oct 2026 — 8K decays, A100 collapses

**8K completed** (22685911, 63 min). The rungs decay in *opposite* directions:
D at k=5 is DFlash 0.219, n-gram 0.044, and n-gram's coverage *rises* 0.132 to
0.161. `notes/2026-10-06-8k-context-decay.md`.

**A100 lost all nine arms** (22685764) to
`cudaErrorNoKernelImageForDevice`. Seventeen minutes billed, nothing written.
Three wrong diagnoses before bisection worked:

```bash
sbatch diag_a100.sh                                                # 22713175
```

Four layers cheapest-first under `CUDA_LAUNCH_BLOCKING=1`: plain torch, then
vLLM's `rms_norm` custom op, then eager generation, then compiled. The
blocking launches matter — the original traceback pointed at `output.fill_(0)`
and `.contiguous()`, which is what an async CUDA error looks like at a sync
point.

Step 1 passed, step 2 failed in 3m35s: **vLLM's extension does not execute on
sm_80.** Confirmed by counting cubins — sm_86 40, sm_80 20:

```bash
find $VD -name '*.so' -exec sh -c 'echo "== $1"; cuobjdump --list-elf "$1"' _ {} \;
```

`env/a100-sm80-failure.md`, `env/gpu-support.md`.

**Submitted 32K** (22713173), grid 1/2/4, limit 25.

**Set up team access** with `share_with_team.sh`, which copies the venv into
project space with paths rewritten and runs `chgrp -R` *after* `cp -a` — the
original did it before, so new files inherited the wrong group. Group
corrected from `grp_202` to `delta_bikd`. Sahil's jobs still failed in seven
seconds; diagnosed 8 Oct, confirmed 11 Oct.

**Cerebras.** `sdk.cerebras.ai` is the CSL kernel SDK for the Wafer-Scale
Engine, not an inference API — no models, no endpoints, no decoding
parameters. Its value is as a *prediction target*: wafer-scale keeps weights
in on-chip SRAM, so the ridge point sits an order of magnitude below a GPU's
and our R model predicts speculation is worthless there even at batch 1.
Discussion section, not experiment plan. Specs need verifying first.

## 5 Oct 2026 — losslessness, and harness v2

**Harness v2** (`HARNESS_VERSION = "v2-2026-10-05"`), additive and guarded:
full output token ids; per-request timing; a prefix-caching switch; and
**`env.host` and `env.slurm_job` in every record** — which is what made the
11 Oct replication check possible four days later.

**Investigated losslessness.** Version 1 of `check_lossless.py` compared
*generation lengths*, which is useless because every generation hits the
256-token cap. Rewrote it to compare token ids and — the part that mattered —
to run the baseline against *itself* first as a control.

The control settled it: `off` against itself at different concurrency, no
speculation anywhere, agrees on only **12.7%** of sequences; speculative arms
agree with the baseline on **19.0%**. Per-token divergence 0.80% against
0.65%. **Speculation perturbs the output less than the engine perturbs
itself.** `notes/2026-10-05-losslessness-and-engine-nondeterminism.md`.

```bash
sbatch --export=ALL,BUCKET=8K,LIMIT=25,CONC="1 8" phase0.sh        # 22685911
sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2K,LIMIT=25 \
       phase0.sh                                                   # 22685764
```

## 1 Oct 2026 — first valid measurement, and the coverage correction

```bash
sbatch --export=ALL,BUCKET=2K,LIMIT=25,CONC="1 8 32" phase0.sh    # 22598608
```

**The result that set up the project:** tau is flat across load (DFlash 2.543,
2.507, 2.470) while speedup collapses (1.86x, 1.49x, 1.03x). All the variation
lives in R.

**Found a bug in how we read vLLM's numbers.** n-gram proposes a draft on only
13.2% of decode steps, but the reported acceptance rate is conditional on a
draft existing — so it overstates delivery by more than 2x. Added a third
term:

```
Speedup ~= [cov * tau_cond + (1 - cov)] / R
```

Corrected, n-gram's R is 1.082, *below* DFlash's 1.360, which is what the
hardware predicts. `summarize.py` now reconstructs coverage from the counters.

## 30 Sep 2026 — environment, and six jobs that failed in five seconds

**Built the environment.** Qwen3-8B and the DFlash head downloaded to
`/work/nvme/bikd/$USER/Speculative_Ladder/hf/hub`, LongBench-v2 bucketed into
2K / 8K / 32K, vLLM built from source at `bf13ecc2f` into `~/vllm_env`.

**Verified the runner pin by reading the source**, not by trusting a flag:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

With the V2 runner, n-gram silently falls back to the stable runner while
DFlash stays on V2, so the rungs would run on different code paths.

**Six jobs failed in ~5 seconds**, all `ModuleNotFoundError: No module named
'vllm'`: the scripts ran `module load cray-python` but never activated the
venv. Fixed by adding `source "$VENV/bin/activate"`.

**First real run** (22569403). All prompts at once. *Nothing crashed, which is
worse* — vLLM batched as wide as memory allowed and every rung lost to no
speculation (0.86x, 0.78x, 0.71x) while acceptance looked healthy. The run
measured vLLM's batching decision. Fixed with explicit `max_num_seqs` as
`CONC`. Speedups void, acceptance usable, kept as the saturation end of the
load axis.

---

## Job index

| Job | Date | GPU / node | Cell | State | Outcome |
|---|---|---|---|---|---|
| 22757473 | 8 Oct | A40 | 2K n-gram k sweep | COMPLETED 00:58:58 | n-gram R_draft = 0.071 |
| 22757472 | 8 Oct | A40 gpub073 | 32K n-gram k sweep | COMPLETED 01:01:22 | R_draft ~0.139; the slow node |
| 22757105 | 8 Oct | A40 gpub085 | 2K smoke, n=5 (ssashi) | COMPLETED 00:20:07 | **A40 replication pair** |
| 22757104 | 8 Oct | **A100 gpua045** | 2K smoke, n=5 (ssashi) | COMPLETED 00:19:54 | **first A100 data** |
| 22757102 | 8 Oct | A40 gpub067 | 2K smoke, n=5 (ssashi) | COMPLETED 00:20:07 | **A40 replication pair** |
| 22754950 | 8 Oct | A100 | diagnostic, multiarch venv | COMPLETED 00:11:06 | **all 4 steps pass** |
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
| Rebuild the **same commit** with three architectures, into a **new** venv | A rebuild in place would replace the binary behind every existing measurement | 7 Oct |
| Run A100 cells compiled, not with `enforce_eager` | 22754950 step 4 passes, so there is no reason to accept a code-path difference | 8 Oct |
| Make the shared venv read-only to the group | The hardware axis rests on "same binary" | 8 Oct |
| Write on `logs/` and `.vllm_cache` for teammates, **not** `results/` | Slurm must write a log and vLLM a cache; two writers on one `results.jsonl` can corrupt it | 8 Oct |
| `LADDER_VENV` is mandatory on every job | The fallbacks are single-architecture builds | 8 Oct |
| Build script **variants** rather than editing originals | `kprobe_ngram.sh`, `diag_a100_multiarch.sh` — existing results stay reproducible from the script that made them | 11 Oct |
| Every repeat gets its own `LADDER_OUT` | Per-request filenames do not encode the run | 11 Oct |
| **Report ratios, never absolute throughput across jobs** | Ratios agree to 0.03–0.08% across nodes; absolute throughput to 8.8% | 11 Oct |
| **Analyse per record, split by `env.slurm_job`** | `summarize.py` collapses duplicate keys and hid a replication for four days | 11 Oct |
| Repeats desirable but **not blocking** | C8 has two independent realizations agreeing to 0.1% | 11 Oct |
| **Freeze the A100 prediction before measuring it** | Rule 4. A prediction tested after the fact is not a prediction | 11 Oct |
| Submit the A100 test with a wrapper, not a new batch script | `submit_a100_predict.sh` calls the verified `phase0.sh` twice rather than reimplementing the harness | 11 Oct |
| **Pre-commit the response to the A100 test** as a decision tree, before its records are read | So the reading of each outcome is fixed in advance, not fitted. `notes/2026-10-11-a100-decision-tree.md` | 11 Oct |
| **Intervals from disjoint prompt subsets**, five per bucket, each its own job and directory; same-prompt repeats do not count | Per-request timing is unavailable; independent samples were available all along in the unused prompt dimension. `SEED` is inert under greedy | 11 Oct |
| **Measurement stops end of 17 Oct**; the last three days before the review are the write-up | The deliverable is a document | 11 Oct |
| **H200 fit waits for the subset intervals** and freezes a predicted range, not a point | Two parameters on two machines has zero residual, so a point fit would carry no uncertainty and a miss would be uninterpretable | 11 Oct |

---

## Dead ends, and what they cost

| What was believed | Why it was wrong | Cost |
|---|---|---|
| A100 is supported, because `cuobjdump` lists sm_80 | `ls $VD/*.so` is not recursive and missed the FlashAttention extension. Worse: sm_80 *was* listed and the kernels still did not run. A listing is not proof of execution | one failed A100 job, ~17 min billed |
| The A100 failure is compile-cache poisoning from the A40 runs | Hashes are disjoint across GPUs, and the A100 run logged nine saves and zero loads | ~20 min of analysis |
| The failure is in FlashAttention, per the traceback | `output.fill_(0)` is the profiling shortcut that skips attention. The error was sticky. CUDA errors surface where they are noticed | wrong path twice |
| Sahil could not read the venv because its group was `grp_202` | `grp_202` is his *primary* group | a `chgrp` over 100k files |
| Sahil *could* write to the tree, because `group::rwx` and he is in `delta_bikd` | POSIX uses a matching **named user** ACL entry and ignores group entirely | two days of his jobs failing, hunting a log that was never created |
| `same length` would distinguish a token flip from a different answer | Every generation hits the 256-token cap | one useless diagnostic |
| k=1 would rescue DFlash at 32K | Fitted `R = 1 + k*c` to a single point by assuming the intercept. The real intercept is 1.662 | one wrong prediction, caught by the sweep that tested it |
| Optimal k shrinks with context, as it does with load | It **rises** with context. The load half was never measured in a valid cell — 22569403 is void | one retracted claim, one unmeasured claim |
| `requirements/build.txt` holds the build dependencies | `requirements/build` is a directory | one failed build start |
| Per-request timing would give us a paired bootstrap | `timing` carries only `arrival_time`; no per-request draft counts | one harness revision |
| `import vllm._C` tests whether the extension loads | This build ships stable-ABI extensions with different names | a false alarm on a good log |
| `phase0.sh` derives its paths from `$USER` | `grep USER phase0.sh` returns nothing | one hypothesis, refuted for free |
| `BUCKET=2k` works | The dataset filename is case sensitive | a wrong command in this file for a day |
| n-gram drafts for free, because it runs no forward pass | Prompt lookup must scan the prompt: R_draft 0.071 at 2K, 0.139 at 32K | two GPU-hours, well spent |
| Run-to-run variance is under 1% everywhere, from one 2K-derived replication | A third 32K realization came in 8.8% low on absolute throughput | nearly bought contaminated repeats |
| 32K is irreproducible to ~10% | C11 already held two realizations agreeing to 0.67% | one wrong note |
| The 32K inversion is a single unreplicated run, and its within-job status cannot be established from stored data | `env.slurm_job` and `env.host` are in every record since 5 Oct. **It had been replicated for four days** | three wrong write-ups in one day, and a near-miss on spending the last hours re-measuring it |
| The harness should record node and job id | It already does | one wrong recommendation |
| Sahil's results are missing because he did not set `LADDER_OUT` | He set it. Zero records in *our* tree was correct behaviour | one wrong diagnosis |
| Sahil must grant read access to his results tree | `drwxrws---+` under `delta_bikd`, readable already | — |
| Per-arm cost extrapolation estimates a job's runtime | Wrong by ~2x three times running | repeated bad planning |
| The A40's ridge point is ~107, half the H200's, so it is the held-out platform on the ratio axis | The 27 Sep memo halved NVIDIA's dense bf16 figure (149.7) believing it was the sparse one. The real ridge is ~215; A40, A100 and H200 all sit at 201 to 215 | the proposal's platform rationale, one withdrawn mechanism argument, and a review built on it; caught by reading the datasheet |

**The pattern in most of these:** a conclusion drawn from a proxy — a file
listing, a directory name, a traceback line, a group name, a module name, a
mode string, a single point with an assumed intercept, a variance measured at
one operating point, **a summary table standing in for the records** — instead
of testing the thing itself. The fix that works is bisection, cheapest layer
first, as in `diag_a100.sh`.

Three corollaries earned the hard way: **read the script, do not infer it**;
**one observation is not a distribution**; and **a derived summary is a proxy
for the data, and the most seductive kind, because it looks like the
measurement rather than like a clue.**

---

## Budget

Sub-allocation: 10 GPU-hours per student, more available on request.
`accounts` reports **4 hours remaining of 10** as of 11 Oct, which is the
operative number. `sreport` reports 159 CPU-hours for dagraw2 and 24 for
ssashi; divided by an assumed 16 cores per GPU that is ~9.9 and ~1.5
GPU-hour equivalents, which does not reconcile with 6 charged. The divisor is
an assumption. **Do not trust the table below to better than a couple of
hours.**

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

Sahil has spent ~1.5 of his own 10 GPU-hours (jobs 22757102/104/105), so
**his allocation is the project's main remaining GPU resource** and the A100
axis is assigned to it.

---

## Open items

- **Run `submit_a100_predict.sh`** — the two A100 cells that test the frozen
  prediction. ~1.5 GPU-hours on Sahil's allocation. This is the highest-value
  GPU time available to the project.
- **Fix `summarize.py` to group by `env.slurm_job`**, or at minimum warn on
  duplicate keys. The current behaviour hid a replication for four days.
- **Re-run the A40/A100 2K comparison at n=50.** The current pair is n=5, so
  the first-position acceptance wobble (0.641 vs 0.610) cannot be separated
  from sampling noise, and a real cross-GPU numerics effect would matter.
- **Reconcile the budget** against `sreport`; confirm or kill the 2x A100
  assumption.
- **Triton divergence between the two venvs.** The multiarch install pulled
  `tokenspeed-triton-3.8.10` and may have displaced `triton-3.7.1`. If so, any
  A40-vs-A100 gap is confounded by more than the architecture list, which
  breaks the "same binary" premise behind C16 and C17.
- ~~**Verify the A100 and A40 spec sheets.**~~ **Done 11 Oct (later).**
  A100 ~201 holds; A40 is ~215, not ~107; H200 ~206. Register, "Spec-sheet
  check".
- **Bootstrap the agreement result.** No longer free: the same-config control
  has to be run first, because the token ids for the existing pairs were
  overwritten. `code/c10_control.sh` (~0.32 GPU-hours) then
  `code/c10_control.py` (free). The claim most in need of an interval.
- **gpub073.** One slow node, ratio moving as a bandwidth-limited machine
  would. One line in the limitations unless it recurs.
- **n-gram at 8K was never swept**, so the middle R_draft point is missing.
- **n-gram beyond k=7** never tried; speedup still rising at k=7 everywhere.
- **Why both drafter costs are sublinear in context** — a 16x prompt gives a
  2x cost in both rungs, unexplained. A fourth bucket would help.
- **The architecture guard allows only 80 and 86**, so it aborts H200 jobs the
  multiarch build could run.
- **Professor not yet told about the scope change:** approved for a hardware
  comparison, delivering a context-axis inversion *and* a hardware transfer.
- **Prefix-caching-off control not run.**
- **Quality evaluation not run** — free, login node.
- **Eager-vs-compiled divergence** (8 Oct) not yet folded into the losslessness
  note.
