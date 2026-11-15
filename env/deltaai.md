# DeltaAI (GH200, aarch64): building and running the pinned vLLM

Written 15 Nov 2026. **Nothing in this file has been run yet.** The scripts
were checked for syntax and exercised against stubs on a laptop. Replace the
"expected" lines below with what actually happened, including the failures.

## Why a separate build

DeltaAI nodes are NVIDIA GH200 superchips: an H100 (96 GB, compute capability
9.0) with a 72-core Grace ARM CPU. Both Delta venvs are x86_64 and cannot run
there. `code/build_deltaai.sh` builds the same commit (`bf13ecc2f`) with the
same torch (`2.13.0+cu130`) and the same triton pins, for `sm_90` only, into
`/work/nvme/<proj>/$USER/Speculative_Ladder/vllm_env_gh200`.

Every DeltaAI number is a new column. It comes from a different binary, CPU
and compiler than every filed result, so its ratios are never pooled with or
scored against the A40 and A100 ratios without saying so.

## Facts from NCSA's documentation, read 15 Nov 2026

| | |
|---|---|
| login | `ssh <netid>@dtai-login.delta.ncsa.illinois.edu` (round robin over `gh-login01` to `04`; for tmux, ssh to one node by name) |
| home | separate from Delta's, because the architectures differ |
| `/work/nvme`, `/work/hdd` | shared with Delta |
| `/projects` | sharing with Delta not stated |
| partitions | `ghx4` (48 h max, factor 1.0), `ghx4-interactive` (2 h max, factor 2.0, one job per user) |
| one GPU's share | about 72 cores and 110 GB of RAM |
| default Python | `cray-python` 3.11.7; the Delta venvs are 3.12 |
| project code and account | printed by `accounts`; not assumed to be `bikd` |

The scripts request 16 cores and 64 GB with one GPU, which is NCSA's own
sample request and sits inside one GH200's share. The TA question about
whether a larger `--mem` charges a second unit is still open; do not raise it.

Confirmed the same day from the package indexes: aarch64 wheels exist for
`torch 2.13.0+cu130` (cp311, cp312), `triton 3.7.1` and
`tokenspeed-triton 3.8.10.post20260920`. Not confirmed: every other line of
vLLM's `requirements/cuda.txt`. The build script checks each one before the
compile and names the ones without a wheel.

## Steps

```bash
ssh <netid>@gh-login01.delta.ncsa.illinois.edu
accounts                                   # note the project code and account
mkdir -p /work/nvme/<proj>/$USER/Speculative_Ladder && cd $_
git clone https://github.com/Devansh63/Speculate-Decoding-Ladder.git repo
cd repo/code

tmux new -s build
PROJ=<proj> bash build_deltaai.sh          # one to three hours, no GPU hours

PROJ=<proj> ACCOUNT=<account> source deltaai_env.sh    # lists anything missing
bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7        # prints
bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7 GO=1   # submits
```

The smoke job is five arms on five prompts: off, off with async disabled,
CPU n-gram, GPU n-gram and DFlash. It is the first run of this build, of
harness v4, of `ngram_gpu` and of an explicit `async_scheduling=False`
anywhere. Read the log before anything larger.

## What to record here after the first build

- the module that provided `nvcc`, and its version;
- gcc version, Python version, whether `uv` was available;
- any dependency reported as having no wheel, and what was done about it;
- `ladder_build_info.txt` and `ladder_freeze.txt` from the venv, copied into
  this directory;
- the smoke job id, the GPU name `nvidia-smi` reports, and which arms wrote
  a row.
