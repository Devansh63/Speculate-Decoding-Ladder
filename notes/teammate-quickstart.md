# Running a Speculation Ladder job on Delta

For Sahil, or anyone else in the `bikd` project. Everything is already set up
and shared: the models, the workload files, the scripts, and a working vLLM.
You do not need to download a model, build vLLM, or create any directories
beyond your own results folder.

If you just want to run a job, read **Steps 1 to 3** and stop.

---

## Step 1. Log in

```bash
ssh <your-netid>@login.delta.ncsa.illinois.edu
```

NCSA Kerberos password, then a Duo prompt (`1` sends a push).

## Step 2. Run the smoke test first

Five prompts, one concurrency level, on the cheaper GPU. It proves the whole
pipeline — venv, model, data, output path, permissions — for a few minutes of
A40 time instead of failing a real cell an hour in.

```bash
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
mkdir -p /projects/bikd/$USER/Speculative_Ladder/results/a40
cd $SHARED/ladder/scripts

sbatch --time=00:30:00 --export=ALL,\
LADDER_VENV=$SHARED/vllm_env_multiarch,\
LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/a40,\
BUCKET=2K,LIMIT=5,CONC=1 phase0.sh
```

Watch it, then read the log:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -X
grep -n 'venv:\|out=\|### rung\|FAILED\|ABORT' $SHARED/logs/phase0-<jobid>.out
```

You want to see `venv:` naming `vllm_env_multiarch`, `out=` naming *your*
results directory, three `### rung=` lines, and no `FAILED` or `ABORT`.

**If no log file exists at all**, that is a permissions problem, not a code
problem — Slurm could not create the output file. Tell Devansh rather than
resubmitting.

## Step 3. Submit a real cell

Pick one row. Do not invent combinations: the concurrency grid has to shrink
as context grows or the rows get mislabelled (see "Why the grid shrinks").

All of these assume you have exported `SHARED` and `OUT` first:

```bash
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
OUT=/projects/bikd/$USER/Speculative_Ladder/results
mkdir -p $OUT/a40 $OUT/a100
cd $SHARED/ladder/scripts
```

| What | Command |
|---|---|
| A40, 2K | `sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_OUT=$OUT/a40,BUCKET=2K,LIMIT=50 phase0.sh` |
| A40, 8K | `sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_OUT=$OUT/a40,BUCKET=8K,LIMIT=50,CONC="1 4 12" phase0.sh` |
| A40, 32K | `sbatch --time=03:00:00 --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_OUT=$OUT/a40,BUCKET=32K,LIMIT=25,CONC="1 2 4" phase0.sh` |
| A100, 2K | `sbatch --partition=gpuA100x4 --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_OUT=$OUT/a100,BUCKET=2K,LIMIT=50 phase0.sh` |
| A100, 8K | `sbatch --partition=gpuA100x4 --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_OUT=$OUT/a100,BUCKET=8K,LIMIT=50,CONC="1 4 12" phase0.sh` |
| A100, 32K | `sbatch --partition=gpuA100x4 --time=03:00:00 --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,LADDER_OUT=$OUT/a100,BUCKET=32K,LIMIT=25,CONC="1 2 4" phase0.sh` |

A full cell runs nine arms (three rungs at three concurrency levels) and takes
40 to 80 minutes. You can close your laptop; the job is detached from your
session.

**`BUCKET` is case sensitive.** It is `2K`, `8K`, `32K` — uppercase. The
dataset filename is built from it directly, so `2k` fails on a missing file.

**Never submit to `gpuH200x8`.** The guard aborts on any compute capability
other than 80 and 86, so the job dies after reserving a node even though the
multiarch build does carry sm_90 kernels. See `env/gpu-support.md`.

**Coordinate before submitting.** Message Devansh with which cell you are
taking, so the two of you do not spend GPU hours measuring the same thing.

## Step 4. Read the results

```bash
module load cray-python/3.12.12
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
python3 $SHARED/ladder/scripts/summarize.py \
  /projects/bikd/$USER/Speculative_Ladder/results/a40     # or a100
```

The job prints this command at the end of its log with the paths filled in, so
you can copy it from there instead.

---

## The two variables you must always set, and why

**`LADDER_VENV`.** `phase0.sh` picks a venv in this order: `LADDER_VENV` if
set, then `$HOME/vllm_env` if it exists, then the shared `vllm_env`. Both
fallbacks are **single-architecture builds with no A100 kernels and no PTX**,
so an A100 job that omits this variable reserves a node and dies at kernel
launch with `cudaErrorNoKernelImageForDevice`. That is exactly what happened
to job 22685764: seventeen minutes billed, nine arms lost, nothing written.
Set it on every job, A40 included, so it is one habit rather than a special
case you have to remember.

**`LADDER_OUT`.** Without it, results are appended to Devansh's
`results/<gpu>/results.jsonl`. Two things are wrong with that: you may not
have write permission there, and two people appending to one file is a way to
corrupt a dataset that nobody notices until the analysis disagrees with
itself. Point it at your own tree and `summarize.py` reads either.

Do **not** override `LADDER_SHARED`. The data path, the harness path and the
venv fallback all derive from it, and your tree has none of those.

---

## Everything below is background

### What the job is measuring

Three rungs of the speculation ladder, at three levels of serving load:

- `off`, no speculation, the baseline every speedup is computed against
- `ngram`, draft tokens by looking the prefix up in the prompt, no model
- `dflash`, draft tokens with a small trained head

For each it records throughput, mean accepted length, acceptance broken down
by draft position, and the raw draft counters. The headline numbers are:

- **tau**, mean accepted length, how many tokens one verification pass yields
- **R**, verification cost ratio, how much one pass costs relative to a plain
  decode step
- **coverage**, what fraction of decode steps proposed a draft at all

`Speedup ~= [cov * tau + (1 - cov)] / R`

Two findings so far. tau is essentially constant as load changes while speedup
collapses, so all of the variation is in R
(`notes/2026-10-01-a40-2k-concurrency-sweep.md`). And R itself splits into a
fixed per-step drafter cost and a flat marginal cost per drafted token, with
only the first growing with context — which is why the ranking of `ngram` and
`dflash` reverses between 2K and 32K
(`notes/2026-10-08-drafter-cost-scaling.md`).

### Why the concurrency grid shrinks with context

Qwen3-8B holds about 144 KiB of key-value cache per token, and an A40 fits
about 160,912 tokens of it. A single 32K sequence is therefore about a fifth
of the whole cache, and 32 of them do not fit.

The trap is that asking for too much does not fail. vLLM accepts
`--max-num-seqs 32`, then quietly schedules fewer sequences, and the result
row is labelled with a concurrency it never ran at. Always check:

```bash
grep -i "maximum concurrency" /projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-<jobid>.out
```

Note that the number vLLM prints there is computed for a full `max_model_len`
request, not for the bucket you actually ran, so it is a worst-case ceiling
rather than a direct answer. Also note that the available cache **differs by
arm within one job** — speculative arms reserve memory for the drafter, and
have been observed with up to 28 percent less cache than the `off` arm in the
same job. At 32K that can push a speculative arm below the concurrency the
baseline reached.

### Where everything lives

| What | Path |
|---|---|
| Scripts | `/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts` |
| Workloads | `.../ladder/data/longbench_v2/longbench_v2_{2K,8K,32K}.jsonl` |
| Models | `/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub` |
| Devansh's results | `.../Speculative_Ladder/results/{a40,a100}/results.jsonl` |
| Your results | `/projects/bikd/<your-netid>/Speculative_Ladder/results/{a40,a100}` |
| Slurm logs | `.../Speculative_Ladder/logs/phase0-<jobid>.out` (shared) |
| vLLM, multi-arch | `.../Speculative_Ladder/vllm_env_multiarch` — **use this one** |
| vLLM, original | `.../Speculative_Ladder/vllm_env` — A40 only, kept for the record |

The paths say `dagraw2` because that is where the project tree was created.
Permissions are not uniform across it, so do not assume you can write
anywhere:

- **readable and executable**: the whole tree, including both venvs
- **writable by you**: `logs/` and `.vllm_cache/`, which is what lets Slurm
  create your job's output file and lets vLLM cache compiled kernels
- **not writable by you**: `results/`, deliberately — use `LADDER_OUT`

If you have a named ACL entry on the tree, note that POSIX uses that entry and
**ignores your group membership entirely**. Being in `delta_bikd` does not
grant you what the group bits say if `getfacl` lists you by name. Check with:

```bash
getfacl -p /projects/bikd/dagraw2/Speculative_Ladder/logs | grep -E "$USER|mask"
```

Your jobs charge your own allocation regardless of whose tree they read.

### If you would rather use your own vLLM

Build it for all three architectures, or it will only run on some of them:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
```

Verify afterwards, because a reinstall can silently reuse a cached wheel, and
scan recursively — the FlashAttention extension lives in a subdirectory and a
non-recursive glob misses it:

```bash
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
module load cuda
find "$VD" -name '*.so' -exec sh -c \
  'echo "== $1"; cuobjdump --list-elf "$1" | sed -n "s/.*\(sm_[0-9]\+\).*/\1/p" | sort -u' _ {} \;
```

You want `sm_80`, `sm_86` and `sm_90` in every extension that has any. A
listing is not proof of execution, though — the original A100 failure had
sm_80 listed in the main extension and still could not launch a kernel,
because that half of the fat binary was missing kernels the sm_86 half had.
The only real test is running `diag_a100.sh`.

### Pushing results to GitHub

Optional, and it needs your own token so the commits are attributed to you.
See `notes/github-from-delta.md`. Once set up:

```bash
bash /projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts/push_results.sh \
  "A40 8K sweep, job <id>"
```

### Things that will cost you GPU hours for nothing

1. **Forgetting `LADDER_VENV` on an A100 job.** A reserved node and a kernel
   launch failure. The most expensive mistake available.
2. **Interactive sessions bill while idle.** An `srun --pty` that sits at a
   prompt is charged the whole time. Use `sbatch`.
3. **Downloads belong on the login node**, which is free. Never inside a job.
4. **Asking for more cores or memory than one GPU's share** silently charges
   for extra GPUs. One GPU's share on A40 and A100 nodes is 16 cores and
   62.5 GB. `phase0.sh` already asks for exactly that; do not raise it.
5. **A100 bills at roughly twice the A40 rate.** Prove a configuration on A40
   first. That is what step 2 is for.

### Known gotchas

- The scripts refuse to start unless `VLLM_USE_V2_MODEL_RUNNER=0`. This is
  deliberate. Without it, n-gram silently runs on a different model runner
  than DFlash and the comparison between them is meaningless. `phase0.sh`
  exports it for you.
- `sbatch` snapshots the batch script at submit time but **not** the files it
  calls at run time. Editing `run_benchmark.py` after submitting changes what
  a queued job will do.
- A Slurm state of `COMPLETED` means the shell script finished its loop, not
  that the measurements succeeded. Check that your `results.jsonl` grew.
- `diag_a100.sh` hardcodes `$HOME/vllm_env` and ignores `LADDER_VENV`. If you
  run it, you are testing your own build, not the shared one.
- Acceptance is never compared across different draft lengths without
  truncating to a common one first. `summarize.py` does this; its `AL@5`
  column is the comparable number, not `tauC`.
