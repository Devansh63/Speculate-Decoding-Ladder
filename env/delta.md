# Delta cluster notes

Facts established from a live session, 30 Sep 2026.

## Allocation

| Item | Value |
|---|---|
| Login host | `login.delta.ncsa.illinois.edu` |
| Auth | NCSA Kerberos password, then a separate Duo prompt (`1` sends a push). SSH keys are disabled for individual users |
| Project | `bikd`, shared by the whole class |
| Charge account | `--account=bikd-delta-gpu` |
| Balance | 10 hours deposited on a sub-allocation, which is far short of what the measurement matrix needs |

## Storage

| Path | Quota | Use |
|---|---|---|
| `/u/<user>` | 100 GB | Home. Code and configs only, never the Hugging Face cache |
| `/projects/bikd` | 500 GB | Shared project space, persistent. Repo and results |
| `/work/nvme/bikd` | 500 GB | Fast NVMe. Model weights, caches, run outputs |
| `/work/hdd/bikd` | 1 TB | Bulk and archive |

`$WORK` is unset on Delta, so use explicit paths. `/projects/bikd` is visible to
the entire class, so share with a teammate using an ACL rather than group
permissions, and never write credentials there.

## Partitions

| Partition | GPU | Memory | Nodes | Walltime |
|---|---|---|---|---|
| `gpuA40x4` | A40 | 48 GB | 100 | 48 h |
| `gpuA100x4` | A100 | 40 GB | 100 | 48 h |
| `gpuA100x8` | A100 | 40 GB | 6 | 48 h |
| `gpuH200x8` | H200 | 141 GB | 8, max 1 node per job | 48 h |

## Charging traps

1. **You pay for what you reserve, not what you use.** An idle interactive session
   bills at the same rate as a running job. This is what consumed the first two
   hours of the allocation.
2. **Cores or memory, whichever implies more GPUs.** One GPU's share is 16 cores
   and 62.5 GB of host RAM on the A40 and A100 nodes, 12 cores and 250 GB on the
   H200 node. Asking for more silently charges for extra GPUs.
3. **An A100 hour costs twice an A40 hour** (`billing=1024` against `billing=512`).
   Acceptance is hardware-independent, so acceptance runs belong on the A40.
4. **Batch beats interactive.** A batch job releases the allocation the moment the
   work ends; an interactive session keeps billing while you read the output.

## Build note

vLLM compiled from source targets only the architecture it detects. A build made
on an A40 (`sm_86`) fails on an A100 (`sm_80`) with "no kernel image is available",
and a plain reinstall reuses the cached wheel rather than rebuilding. Build once
for all three with `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` and
`pip install --no-cache-dir --force-reinstall`.
