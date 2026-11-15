# DeltaAI (GH200, aarch64): building and running the pinned vLLM

Written 15 Nov 2026. **Nothing in this file has been run yet.** The scripts
were checked for syntax and exercised against stubs on a laptop. Replace the
"expected" lines below with what actually happened, including the failures.

## Which vLLM, and why it is not the old pin

DeltaAI nodes are NVIDIA GH200 superchips: an H100 (96 GB, compute capability
9.0) with a 72-core Grace ARM CPU. Both Delta venvs are x86_64 and cannot run
there, so DeltaAI needs its own environment whatever the version.

**Decision, 15 Nov 2026: DeltaAI runs vLLM 0.33.0 on Model Runner V2**, not
`bf13ecc2f` on the stable runner. Reasons, each read from the source at the
`v0.33.0` tag and none yet observed in a run:

- On V2, `ngram` and `ngram_gpu` both resolve to `NgramGPUSpeculator`
  (`vllm/v1/worker/gpu/spec_decode/__init__.py`), and DFlash has its own V2
  speculator. Both rungs are on one runner, which is the thing rule 1 of the
  README has been enforcing by hand since September.
- The async-scheduling fallback for n-gram is now scoped to the V1 runner
  (`v1_async_sched_incompat` in `vllm/config/vllm.py`, from PR #62521). On V2
  every arm defaults to async on, so the cross-mode problem found on 2 Nov
  does not arise and no second baseline is needed.
- PyPI carries an aarch64 wheel for 0.33.0, so there is no three-hour
  compile. 0.33.0 still ships both runners; current `main` has removed V1
  (PR #63373, merged 7 Nov), which is why a tagged release is pinned and not
  `main`.
- The options the harness uses are unchanged at the tag: `async_scheduling`,
  `per_request_spec_decode_metrics`, `scheduler_cls`,
  `num_speculative_tokens_per_batch_size`, `CompletionOutput.spec_decode_metrics`,
  the `vllm:spec_decode_*` counters, and `LLM.get_metrics()`.

What it costs: 0.33.0 pins torch 2.14.0 (the old pin is 2.13.0), and there is
no CPU n-gram on V2, so the "weight-free CPU lookup" arm of the filed results
has no counterpart here; the n-gram arm is the GPU proposer.

**Every DeltaAI number is a new baseline.** Different vLLM, torch, runner,
CPU and GPU from every filed row. Its ratios are never pooled with or scored
against the A40 and A100 ratios. If the cross-GPU story is wanted on this
version, the A40 and A100 cells have to be re-run on 0.33.0 as well; the
x86_64 wheel exists.

`code/install_deltaai.sh` installs it. `code/run_benchmark.v5.py` is the
harness for it: it requires `VLLM_USE_V2_MODEL_RUNNER=1`, records the
resolved runner, and aborts if the engine fell back to V1.
`code/build_deltaai.sh` (a source build, by default of the old pin) is kept
as the fallback if the wheel does not work.

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

Confirmed the same day from the package indexes: PyPI has
`vllm-0.33.0-cp38-abi3-manylinux_2_28_aarch64.whl`, and PyTorch's index has
`torch 2.14.0+cu130` for aarch64 (cp312). Not confirmed: which CUDA build of
torch the PyPI vLLM wheel was compiled against, and that its kernels cover
`sm_90`. The smoke job is the test of both. For the old pin, aarch64 wheels
also exist for `torch 2.13.0+cu130`, `triton 3.7.1` and
`tokenspeed-triton 3.8.10.post20260920`.

## Steps

```bash
ssh <netid>@gh-login01.delta.ncsa.illinois.edu
accounts                                   # note the project code and account
mkdir -p /work/nvme/<proj>/$USER/Speculative_Ladder && cd $_
git clone https://github.com/Devansh63/Speculate-Decoding-Ladder.git repo
cd repo/code

PROJ=<proj> bash install_deltaai.sh        # minutes, no GPU hours

PROJ=<proj> ACCOUNT=<account> source deltaai_env.sh    # lists anything missing
bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7        # prints
bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7 GO=1   # submits
```

The smoke job is three arms on five prompts: off, GPU n-gram and DFlash,
all on Model Runner V2. It is the first run of this install, of harness v5
and of `ngram_gpu` anywhere. Read the log before anything larger: check that
every row has `resolved.use_v2_model_runner: true` and the same
`resolved.async_scheduling`.

If the install or the smoke job fails on a torch or CUDA mismatch, the
fallback is a source build of the same tag:

```bash
COMMIT=v0.33.0 TORCH_SPEC=torch==2.14.0 TRITON_SPEC= TS_TRITON_SPEC= \
  NEWVENV=/work/nvme/<proj>/$USER/Speculative_Ladder/vllm033_gh200 \
  PROJ=<proj> bash build_deltaai.sh
```

## What to record here after the first build

- the module that provided `nvcc`, and its version;
- gcc version, Python version, whether `uv` was available;
- any dependency reported as having no wheel, and what was done about it;
- `ladder_build_info.txt` and `ladder_freeze.txt` from the venv, copied into
  this directory;
- the smoke job id, the GPU name `nvidia-smi` reports, and which arms wrote
  a row.
