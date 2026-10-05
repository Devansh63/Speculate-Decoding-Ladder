# Running a Speculation Ladder job on Delta

For Sahil, or anyone else in the `bikd` project. Everything is already set up
and shared: the models, the workload files, the scripts, and a working vLLM.
You do not need to download a model, build vLLM, or create any directories.

If you just want to run a job, read **Step 1 and Step 2** and stop.

---

## Step 1. Log in

```bash
ssh <your-netid>@login.delta.ncsa.illinois.edu
```

NCSA Kerberos password, then a Duo prompt (`1` sends a push).

## Step 2. Submit a job

```bash
cd /projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts
sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC="1 4 12" phase0.sh
squeue -u $USER
```

That is the whole thing. It runs nine arms (three rungs at three concurrency
levels), takes 40 to 60 minutes, and appends to the shared results file. You
can close your laptop; the job is detached from your session.

**Pick your job from this table.** Do not invent combinations; the concurrency
grid has to shrink as context grows or the rows get mislabelled (see "Why the
grid shrinks" below).

| What | Command |
|---|---|
| A40, 2K | `sbatch --export=ALL,BUCKET=2K,LIMIT=50 phase0.sh` |
| A40, 8K | `sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC="1 4 12" phase0.sh` |
| A40, 32K | `sbatch --time=03:00:00 --export=ALL,BUCKET=32K,LIMIT=25,CONC="1 2 4" phase0.sh` |
| A100, 2K | `sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2K,LIMIT=50 phase0.sh` |
| A100, 8K | `sbatch --partition=gpuA100x4 --export=ALL,BUCKET=8K,LIMIT=50,CONC="1 4 12" phase0.sh` |
| A100, 32K | `sbatch --partition=gpuA100x4 --time=03:00:00 --export=ALL,BUCKET=32K,LIMIT=25,CONC="1 2 4" phase0.sh` |

**Never submit to `gpuH200x8`.** The shared vLLM has no H200 kernels and the
job will abort after reserving a node. See `env/gpu-support.md`.

**Coordinate before submitting.** Message Devansh with which cell you are
taking, so the two of you do not spend GPU hours measuring the same thing.

## Step 3. When it finishes

```bash
module load cray-python/3.12.12
SHARED=/projects/bikd/dagraw2/Speculative_Ladder
python3 $SHARED/ladder/scripts/summarize.py $SHARED/results/a40    # or a100
```

The job prints these two commands at the end of its log, with the right paths
already filled in, so you can copy them from there instead.

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

The finding so far is that tau is essentially constant as load changes while
speedup collapses, so all of the variation is in R. See
`notes/2026-10-01-a40-2k-concurrency-sweep.md`.

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

### Where everything lives

| What | Path |
|---|---|
| Scripts | `/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts` |
| Workloads | `.../ladder/data/longbench_v2/longbench_v2_{2K,8K,32K}.jsonl` |
| Models | `/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub` |
| Results | `.../Speculative_Ladder/results/{a40,a100}/results.jsonl` |
| Slurm logs | `.../Speculative_Ladder/logs/phase0-<jobid>.out` |
| Shared vLLM | `.../Speculative_Ladder/vllm_env` |

The paths say `dagraw2` because that is where the project tree was created.
It is group readable and writable by `delta_bikd`, so it is shared space, not
personal space. Your jobs charge your own allocation.

### If you would rather use your own vLLM

`phase0.sh` prefers `$HOME/vllm_env` when it exists and falls back to the
shared copy otherwise, so if you have already built one it will be used with
no configuration. To force a specific one:

```bash
sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC="1 4 12",LADDER_VENV=$HOME/my_env phase0.sh
```

If you build your own, build it for both architectures or it will only run on
one of them:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
```

Verify afterwards, because a reinstall can silently reuse a cached wheel:

```bash
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
module load cuda
cuobjdump --list-elf "$VD"/_C_stable_libtorch.abi3.so \
  | sed -n 's/.*\(sm_[0-9]\+\).*/\1/p' | sort -u
```

You want `sm_80` and `sm_86`.

### Pushing results to GitHub

Optional, and it needs your own token so the commits are attributed to you.
See `notes/github-from-delta.md`. Once set up:

```bash
bash /projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts/push_results.sh \
  "A40 8K sweep, job <id>"
```

### Things that will cost you GPU hours for nothing

1. **Interactive sessions bill while idle.** An `srun --pty` that sits at a
   prompt is charged the whole time. Use `sbatch`.
2. **Downloads belong on the login node**, which is free. Never inside a job.
3. **Asking for more cores or memory than one GPU's share** silently charges
   for extra GPUs. One GPU's share on A40 and A100 nodes is 16 cores and
   62.5 GB. `phase0.sh` already asks for exactly that; do not raise it.
4. **A100 bills at roughly twice the A40 rate.** Prove a configuration on A40
   first.

### Known gotchas

- The scripts refuse to start unless `VLLM_USE_V2_MODEL_RUNNER=0`. This is
  deliberate. Without it, n-gram silently runs on a different model runner
  than DFlash and the comparison between them is meaningless.
- `results.jsonl` is appended to by both of us. Records are small single-line
  writes, so interleaving is not a practical risk, but avoid running two jobs
  on the same GPU type at the same moment if you can help it.
- Acceptance is never compared across different draft lengths without
  truncating to a common one first. `summarize.py` does this; its `AL@5`
  column is the comparable number, not `tauC`.
