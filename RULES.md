# Project rules

Standing instructions for anyone — human or agent — working on The
Speculation Ladder. **These are rules, not guidance.** Every one exists
because its absence already cost us a job, a day, or a claim, and each says
which.

Read `HANDOFF.md` for what the project *is*. This file is how to work on it.

---

## 1. Git attribution

**All commits are authored as Devansh Agrawal.** This is a graded course
project; the contribution history must reflect its author.

- Author and committer: `Devansh Agrawal <dagraw2@illinois.edu>`
- **No** `Co-Authored-By` lines naming an AI, assistant or agent.
- **No** "Generated with", "Co-authored with" or similar footers.
- Commit messages describe the change and why it was made. They do not
  refer to the agent that wrote them, and they do not say "as requested".

Pushing from Delta, per commit, so it works whether or not the clone has an
identity configured:

```bash
git -c user.name="Devansh Agrawal" -c user.email="dagraw2@illinois.edu" \
    commit -m "<subject>"
```

Pushing through a GitHub API or tool: set the author explicitly to the same
name and address.

**Write real commit messages.** A subject line plus a body explaining the
reasoning. This repository's history is part of the deliverable — the
register of failed predictions and dead ends is deliberate — so a commit that
says only "update" destroys information.

---

## 2. Delta access

```bash
ssh <netid>@login.delta.ncsa.illinois.edu
module load cray-python/3.12.12
```

NCSA Kerberos password, then Duo (`1` sends a push).

Three things about login nodes:

- **They are free.** Analysis, downloads, builds, `pip`, and reading results
  all belong here, never inside a job.
- **They are round-robin.** `dt-login01` through `dt-login04` are different
  machines. A `tmux` session lives on one node only — `tmux ls` elsewhere
  shows nothing. Note which node you are on before detaching.
- **Long work goes in tmux.** Anything over a few minutes (builds, large
  `chmod`/`setfacl` walks) should survive a dropped connection.

---

## 3. Delta file structure

| what | path |
|---|---|
| shared project tree | `/projects/bikd/dagraw2/Speculative_Ladder` |
| scripts | `.../ladder/scripts/` |
| workloads | `.../ladder/data/longbench_v2/longbench_v2_{2K,8K,32K}.jsonl` |
| models (HF cache) | `/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub` |
| **your** results | `/projects/bikd/$USER/Speculative_Ladder/results/<gputag>/` |
| Slurm logs (shared) | `.../Speculative_Ladder/logs/phase0-<jobid>.out` |
| git clone on Delta | `.../Speculative_Ladder/repo` |
| compile cache (shared) | `.../Speculative_Ladder/.vllm_cache` |
| **vLLM, multi-arch** | `.../Speculative_Ladder/vllm_env_multiarch` ← use this |
| vLLM, original | `.../Speculative_Ladder/vllm_env`, `~/vllm_env` — A40 only |
| vLLM build tree | `/work/nvme/bikd/dagraw2/vllm-build` (5.7 GB) |

Permissions are **not** uniform. Readable and executable: the whole tree.
Writable by teammates: `logs/` and `.vllm_cache/` only. **Not** writable:
`results/`, deliberately, and the venvs, deliberately.

### Do not touch

- **The venvs.** Both are read-only to the group on purpose. A `pip install`
  into `vllm_env_multiarch` would invalidate C16 and C17, which rest on "same
  commit, same torch, same triton, only the architecture list differs."
- **`/work/nvme/bikd/dagraw2/vllm-build`.** The multiarch venv is an
  *editable* install rooted there. Deleting or moving it breaks the venv and
  costs a four-hour rebuild.
- **Anyone else's `results/`.** Two writers appending to one `results.jsonl`
  can interleave partial lines.
- **Originals of working scripts.** See rule 6.

---

## 4. Two environment variables are mandatory on every job

```bash
LADDER_VENV=/projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch
LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/<gputag>
```

**`LADDER_VENV`** — `phase0.sh` resolves its venv as `LADDER_VENV`, then
`$HOME/vllm_env`, then the original shared venv. Both fallbacks are
single-architecture builds with no working sm_80 kernels. *Why it is a rule:*
job 22685764 omitted it on A100, reserved a node, lost all nine arms to
`cudaErrorNoKernelImageForDevice`, and wrote nothing. Set it on A40 jobs too,
so it is one habit rather than a special case.

**`LADDER_OUT`** — without it, results append to the shared tree, which you
cannot write to and should not share anyway.

**Never override `LADDER_SHARED`.** The data path, harness path and venv
fallback all derive from it.

`phase0.sh` exports `VLLM_USE_V2_MODEL_RUNNER=0` for you, and the harness
refuses to start without it. Leave it alone: with the V2 runner, n-gram
silently falls back to the stable runner while DFlash stays on V2, so the two
rungs would be measured on different code paths.

---

## 5. Where results go

- **Your own tree**, by GPU: `/projects/bikd/$USER/Speculative_Ladder/results/{a40,a100}/`
- **Every repeat gets its own directory.** `results/a40_rep1`, `_rep2`, and so
  on. *Why:* per-request filenames encode rung, bucket, k, n, seed and
  concurrency but **not the run**, so a second run into the same directory
  overwrites the first's `perreq_*.json`. This already destroyed the token ids
  of two identical-config pairs and is why C10 cannot be established.
- **Logs are shared** and land in `dagraw2`'s `logs/` regardless of who runs
  the job, because `#SBATCH --output` cannot take a variable. This is a
  feature: either teammate can diagnose the other's failures without asking.

---

## 6. How to run and verify a job

```bash
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
mkdir -p /projects/bikd/$USER/Speculative_Ladder/results/a40
cd $SHARED/ladder/scripts

sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,\
LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/a40,\
BUCKET=8K,LIMIT=50,CONC=1 phase0.sh
```

- **Always `sbatch`. Never `srun --pty`** — an interactive session bills while
  it sits at a prompt.
- **`BUCKET` is case sensitive**: `2K`, `8K`, `32K`. The dataset filename is
  built from it directly, so `8k` fails on a missing file.
- **Concurrency grids shrink with context**: 2K `1 8 32`, 8K `1 4 12`, 32K
  `1 2 4`. vLLM accepts an impossible `max_num_seqs` and quietly schedules
  fewer, which mislabels the row instead of failing.
- **Match `LIMIT` to what you are comparing against.** 50 at 2K and 8K, 25 at
  32K. A repeat at a different prompt count is not a repeat.
- **Never ask for more cores or memory than one GPU's share** (16 cores,
  62.5 GB on A40 and A100 nodes). `phase0.sh` already asks for exactly that;
  raising it silently charges for extra GPUs.

### Modifying a script

**Build a variant; do not edit a working original.** `kprobe_ngram.sh` and
`diag_a100_multiarch.sh` were both made with `sed` from a file that worked, so
that existing results stay reproducible from the script that produced them.
After any `sed`, **verify the patch took** before submitting:

```bash
grep -n '^run \|job-name\|KLIST' <new_script>
```

### After the job

```bash
sacct -j <jobid> --format=JobID,Partition,NodeList%12,State,Elapsed,ExitCode -X
```

- **`COMPLETED` does not mean the measurements succeeded.** It means the shell
  script finished its loop. Job 22685764 is `COMPLETED` and produced nothing.
  **Check that `results.jsonl` grew.**
- `sbatch` snapshots the batch script at submit time but **not** the files it
  calls at run time. Editing the harness after submitting changes what a
  queued job will do.

---

## 7. GPU budget discipline

10 GPU-hours per student, more available on request. As of 11 Oct: `dagraw2`
~4 h, `ssashi` ~8.5 h. A100 bills at roughly 2x the A40 rate (assumed, never
verified).

- **State the cost before running anything.** Not after.
- **Quote the cost from the nearest whole job that actually ran, and call it a
  floor.** *Why:* per-arm extrapolation has underestimated by about 2x three
  separate times. Model load, cold compile and prompt length are not
  interchangeable units of time.
- **Say whose allocation it comes from.** The two are separate and only one
  has room.
- **Do nothing speculative while a decisive job is in flight.** If it fails,
  the hours are needed to diagnose it.

---

## 8. Which repository file to update, and when

| when this happens | update |
|---|---|
| any job submitted or finished | `LOG.md` — job id, cell, node, outcome, and the command |
| a result changes what we believe | `paper/claims.md` (append) **and** the relevant dated note |
| a new measurement lands | `paper/results-table.md` — with its job id and node |
| a finding is worth explaining | a new `notes/YYYY-MM-DD-topic.md` |
| a prediction is made | `paper/claims.md`, **before** the measurement (rule 10) |
| a prediction is tested | `paper/claims.md` (append) — never edit the prediction |
| a claim is weakened or withdrawn | `paper/claims.md` (append), and fix every file that repeats it |
| the environment changes | `env/gpu-support.md` |
| raw data is produced | `push_results.sh`, from Delta |
| a script is added or changed | `code/`, and check `HANDOFF.md`'s file list still matches |
| how to run something changes | `notes/teammate-quickstart.md` |
| anything in this list changes | this file |

**A claim withdrawn in one file but left standing in another is worse than
not withdrawing it.** When a claim changes, grep the repository for it.
`HANDOFF.md` carried three superseded statements for exactly this reason.

---

## 9. `LOG.md` is mandatory, including the failures

Newest first. Every entry that involved a command records the command.

Log: every job and its outcome, every decision and its reason, every dead end
and what it cost, and every cost estimate that was wrong.

The dead-ends table is not decoration. It is the most re-read part of the
repository, and it exists because eight separate errors in this project came
from the same cause — a conclusion drawn from a proxy instead of from the
thing itself. Adding to it is a contribution.

---

## 10. Analysis rules

- **Read `results.jsonl` per record and split on `env.slurm_job`.** Never take
  provenance from `summarize.py`: it keys on `(bucket, conc, rung, k)` and
  silently keeps one record per key (line 104, "Last run wins"). *Why:* it
  rendered a result replicated across two nodes as a single run, and the
  register repeated that for four days.
- **Report ratios, never absolute throughput across jobs.** Ratios replicate
  across nodes to 0.03–0.08%; absolute throughput varied 8.8% against one
  node.
- **Never compare two numbers from different jobs** unless both are ratios and
  the difference is well clear of a few percent.
- **Truncate to a common draft length before comparing acceptance.**
  `summarize.py`'s `AL@5` column is the comparable number, not `tauC`.
- **Coverage is not optional.** Engine-reported acceptance is conditional on a
  draft being proposed. Without the coverage term, a low-coverage drafter is
  overstated by more than 2x.
- **Check the arithmetic before you accept a mechanism.** Three claims were
  withdrawn on 11 Oct by a memory budget, a counterfactual on two measured
  numbers, and a ratio of two spec-sheet figures. None needed a GPU.

---

## 11. Claim discipline

- **`paper/claims.md` is append-only.** Corrections are dated additions.
  Never edit an earlier entry — the record of what was believed when is the
  point.
- **A prediction counts only if it was committed and tagged before the
  measurement**, with a pass/fail criterion that cannot return "undefined".
- **Record failed predictions and errors found from outside.** A register
  containing only surviving claims is not evidence about the process that
  produced them.
- **Do not write an escape hatch you have not checked.** A pre-registered
  confound that turns out to be physically impossible is worse than none,
  because it pre-licenses explaining away a failure.
- **No claim is "supported" from a single run without an interval.** Per-request
  timing is unavailable on this build, so intervals come from repeated whole
  runs.

---

## 12. Team

| who | NetID | role |
|---|---|---|
| Devansh Agrawal | `dagraw2` | repo owner, cost model, context axis. All commits are authored as him. |
| Sahil Sashi | `ssashi` | hardware axis (A100). Read/execute on the shared tree; write on `logs/` and `.vllm_cache`; own results tree. |
| Professor | — | approved the project; has confirmed more GPU hours are available on request for legitimate project use. |

Coordinate before submitting. Message the other person which cell you are
taking, so two people do not measure the same thing. The A100 axis belongs to
`ssashi`; `dagraw2`'s remaining hours are held in reserve.

---

## 13. When you are not sure

- **Read the script. Do not infer it.** Two wrong hypotheses on 11 Oct were
  refuted by one `grep` that should have been run first.
- **If a cheap test exists, run it before theorising.** `diag_a100.sh` is the
  template: four layers, cheapest first, and the first failure names the
  culprit. It answered in 3m35s what three rounds of inference got wrong.
- **One observation is not a distribution.**
- **Say what you verified, what failed verification, and what you are
  inferring.** Mark inference as inference.
- **Ask before anything irreversible**: deleting data, editing a venv, editing
  a working script in place, or spending GPU hours.
