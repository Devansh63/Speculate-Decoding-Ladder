# DeltaAI: one shared vLLM venv for both of us

Written 16 Nov 2026. Sahil's DeltaAI install is shared with Devansh instead of
each of us building our own, so every DeltaAI row comes from one environment.
Part A is what Sahil runs once to grant access. Part B is what Devansh runs to
check it and to submit jobs.

| | |
|---|---|
| project / account | `bikd` / `bikd-dtai-gh` |
| shared tree | `/work/nvme/bikd/ssashi/Speculative_Ladder` |
| venv | `.../vllm033_gh200` (vLLM 0.33.0, torch 2.14.0+cu130, Model Runner V2) |
| venv Python | `/opt/cray/pe/python/3.11.7` (system `cray-python`, readable by everyone) |
| models | `/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub` (Devansh's cache; both pinned snapshots) |
| workloads | `/work/nvme/bikd/dagraw2/Speculative_Ladder/data/longbench_v2` (sha256 matches the `/projects` copy: 2K `3faddda8...`, 8K `ba1a7534...`, 32K `5704c985...`) |
| login node | `gh-login01.delta.ncsa.illinois.edu` |

Sahil's NCSA username is `ssashi`, not his campus NetID `ssashi2`. Check your
own with `whoami` after logging in.

## How the sharing works

- **Shared, read by jobs:** the venv in Sahil's tree, and the model
  snapshots (`hf/hub`) and workloads (`data/longbench_v2`) in Devansh's tree.
  One copy of each, so both of us read identical files. Sahil's tree has no
  models or data; do not copy them there.
- **Per person, written by jobs:** `LADDER_ROOT` (logs, results) and
  `LADDER_CACHE` (vLLM compile cache). `deltaai_env.sh` puts these under
  `/work/nvme/bikd/$USER/Speculative_Ladder`, so each of us writes to our own
  folder and parallel jobs never collide.
- **Charges:** every job charges the `bikd-dtai-gh` hours of whoever submits it.
- **Never edit `vllm033_gh200` in place.** A job running against it can crash
  or mix old and new code. For a patched engine (plan section C), build a new
  venv beside it under a new name and point only those jobs' `LADDER_VENV` at
  it.

## Part A: Sahil grants access (once)

Log in:

```bash
ssh -o PubkeyAuthentication=no ssashi@gh-login01.delta.ncsa.illinois.edu
```

Without `PubkeyAuthentication=no`, DeltaAI disconnects with "Too many
authentication failures" before the password prompt. Then password, then `1`
for a Duo push.

Set paths:

```bash
export PROJ=bikd; export R=/work/nvme/$PROJ/$USER/Speculative_Ladder; export V=$R/vllm033_gh200
```

Check that Devansh is on the project. `getent group` often lists no members on
NCSA systems, so use `id`:

```bash
stat -c '%G %A' /work/nvme/$PROJ; id dagraw2
```

The group in the first line must appear in his `id` list. If it does not, the
TA has to add him to `bikd` on DeltaAI before anything below helps.

Grant full access (traverse on Sahil's user folder, read and write on the
tree, and the same for anything created later):

```bash
setfacl -m u:dagraw2:x /work/nvme/$PROJ/$USER
```

```bash
setfacl -R -m u:dagraw2:rwX,d:u:dagraw2:rwX,d:u:$USER:rwX,m::rwx,d:m::rwx $R
```

`d:u:$USER` keeps Sahil able to edit files Devansh's jobs create. `m::rwx`
stops the ACL mask from cutting access back.

Verify:

```bash
getfacl $V/bin 2>/dev/null | grep -E "dagraw2|mask"
```

Expected: `user:dagraw2:rwx`, `mask::rwx`, `default:user:dagraw2:rwx`,
`default:mask::rwx`.

## Part B: Devansh checks access and runs jobs

### B1. Log in

```bash
ssh -o PubkeyAuthentication=no dagraw2@gh-login01.delta.ncsa.illinois.edu
```

If `dagraw2` is refused, use your NCSA username instead.

### B2. Confirm the project

```bash
accounts
```

`bikd-dtai-gh` must be listed. If not, ask the TA to add you to `bikd`.

### B3. Set paths and test access

```bash
export PROJ=bikd ACCOUNT=bikd-dtai-gh; export S=/work/nvme/bikd/ssashi/Speculative_Ladder; export V=$S/vllm033_gh200
```

```bash
ls $S && touch $S/logs/.wtest_$USER && rm $S/logs/.wtest_$USER && echo ACCESS_OK
```

Expected: the folder listing, then `ACCESS_OK`.

### B4. Check the venv loads (login node, no GPU, no hours)

```bash
env -u PYTHONPATH $V/bin/python3 -c "import sys,torch,vllm,transformers;print(sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('vllm',vllm.__version__)"
```

Expected: `3.11.7`, `torch 2.14.0+cu130 cuda 13.0`, `vllm 0.33.0`.

### B5. Your own folder and clone

```bash
mkdir -p /work/nvme/bikd/$USER/Speculative_Ladder && cd /work/nvme/bikd/$USER/Speculative_Ladder && git clone https://github.com/Devansh63/Speculate-Decoding-Ladder.git repo && cd repo/code
```

### B6. Check the job environment

```bash
unset LADDER_HF LADDER_DATA; export LADDER_VENV=$V && source deltaai_env.sh
```

Leave `LADDER_HF` and `LADDER_DATA` unset: `deltaai_env.sh` finds your own
cache and workloads. Expected: `LADDER_ROOT` is your own folder,
`LADDER_VENV` is Sahil's venv, `LADDER_HF` and `LADDER_DATA` are under
`/work/nvme/bikd/dagraw2/`, and the last line is `all present`.

### B7. GPU check (optional, about 10 minutes, roughly 0.2 GPU-hours)

```bash
srun --account=$ACCOUNT --partition=ghx4 --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g --time=00:10:00 $V/bin/python3 -c "import torch,vllm;print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0));x=torch.randn(1024,1024,device='cuda');print('matmul ok',(x@x).sum().item()!=0)"
```

Expected: a GH200 or H100 name, `(9, 0)`, `matmul ok True`.

### If a check fails

| What you see | Meaning | Fix |
|---|---|---|
| B2: no `bikd-dtai-gh` | Not on the project | TA adds you |
| B3: `Permission denied` | ACL missing | Sahil re-runs Part A's `setfacl` lines |
| B3: `ls` works, no `ACCESS_OK` | Read but no write (mask) | Sahil runs `setfacl -R -m m::rwx,d:m::rwx /work/nvme/bikd/ssashi/Speculative_Ladder` |
| B4: `ModuleNotFoundError` | A different Python is being picked up | Use the full `$V/bin/python3` path, `module unload cray-python`, `unset PYTHONPATH` |
| B4: error about `libpython` | Cray Python library not loaded | `module load cray-python`, rerun B4 |
| B6: `MISSING ...` | Model or data path wrong | Check `LADDER_HF` and `LADDER_DATA` are unset and point under `/work/nvme/bikd/dagraw2/` |
| B7: `no kernel image` or CUDA error | Wheel or GPU problem, not permissions | Tell Sahil; his jobs fail the same way |
| B7: waits in the queue | `ghx4` is busy | Wait, or use `ghx4-interactive` (charged at 2x) |
| Sahil cannot edit a file you created | Default ACL for Sahil missing | Run `setfacl -m u:ssashi:rwX <file>` |

## Every session after setup

Both of us:

```bash
export PROJ=bikd ACCOUNT=bikd-dtai-gh LADDER_VENV=/work/nvme/bikd/ssashi/Speculative_Ladder/vllm033_gh200
```

Sahil only. `LADDER_HF` finds Devansh's cache by itself, but `LADDER_DATA`
would fall back to Sahil's empty folder, so set it:

```bash
export LADDER_DATA=/work/nvme/bikd/dagraw2/Speculative_Ladder/data/longbench_v2
```

Devansh only: leave `LADDER_HF` and `LADDER_DATA` unset.

```bash
unset LADDER_HF LADDER_DATA
```

Then both:

```bash
cd /work/nvme/bikd/$USER/Speculative_Ladder/repo/code && git pull && source deltaai_env.sh
```

Check that `LADDER_HF` and `LADDER_DATA` print paths under
`/work/nvme/bikd/dagraw2/` and the last line is `all present`.

Then submit as usual. The submit scripts print the plan and submit only with
`GO=1`:

```bash
bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7
```

Do not install, upgrade or edit anything inside `vllm033_gh200`.
