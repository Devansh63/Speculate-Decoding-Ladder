# Which Delta GPUs this vLLM build can actually run on

Recorded 1 Oct 2026, on `dt-login02`, venv `~/vllm_env`.

## The build

| Item | Value |
|---|---|
| vLLM version | `0.29.1rc1.dev345+gbf13ecc2f.cu132` |
| PyTorch | `2.13.0+cu130` |
| PyTorch arch list | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` |
| vLLM cubins, `_C_stable_libtorch.abi3.so` | **`sm_80 sm_86` only** |
| vLLM cubins, `_moe_C_stable_libtorch.abi3.so` | **`sm_80 sm_86` only** |
| vLLM embedded PTX | **none** |

The local version label carries a git hash and the arch list is exactly two
targets, so this was built from source with `TORCH_CUDA_ARCH_LIST="8.0;8.6"`.

PyTorch's arch list is not the binding constraint. vLLM ships its own CUDA
extensions, built separately, and those are what decide whether a job runs.

## What follows

| Partition | GPU | Compute cap | Runs on this build |
|---|---|---|---|
| `gpuA40x4` | A40 48 GB | 8.6 | yes |
| `gpuA100x4` | A100 40 GB | 8.0 | yes |
| `gpuA100x8` | A100 40 GB | 8.0 | yes |
| `gpuH200x8` | H200 141 GB | 9.0 | **no** |

With no PTX embedded there is no JIT fallback, so an H200 job fails at kernel
launch with "no kernel image is available for execution on the device". It
does not run slowly, it does not run at all.

## How this was checked

```bash
source ~/vllm_env/bin/activate
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
module load cuda
cuobjdump --list-elf "$VD"/_C_stable_libtorch.abi3.so \
  | sed -n 's/.*\(sm_[0-9]\+\).*/\1/p' | sort -u
cuobjdump --list-ptx "$VD"/_C_stable_libtorch.abi3.so \
  | sed -n 's/.*\(compute_[0-9]\+\).*/\1/p' | sort -u
```

Repeat this check after any reinstall. A plain `pip install --force-reinstall`
can silently reuse a cached wheel and leave the arch set unchanged.

## Decision on H200

Deferred, deliberately. Adding sm_90 means a from-source rebuild, and a
rebuild would replace the exact binary that produced the A40 and A100
measurements, making every earlier number unreproducible. If H200 is attempted
later it goes in a **second venv**, built from the same commit `bf13ecc2f`
with `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`, so `~/vllm_env` stays frozen.

H200 was a stretch rung in the proposal, not a must-do. A40 and A100 already
give two separated hardware points, which is what the R side of the
`Speedup ~= tau / R` decomposition needs. They are separated in **memory
bandwidth** (696 against 1,555 GB/s), not in compute-to-bandwidth ratio: dense
bf16 ridge points are ~215 and ~201 FLOP/byte per NVIDIA's datasheets, and the
H200 is ~206. This paragraph previously gave the A40 as 107, which halved the
dense figure (corrected 11 Oct 2026; register, "Spec-sheet check").

## The second venv, and the triton question (resolved 11 Oct 2026)

`vllm_env_multiarch` is the second venv the H200 decision above calls for: the
rebuild went there rather than into `~/vllm_env`, so the binary that produced
the A40 measurements stays frozen. Both venvs are therefore expected to differ
in exactly one thing — the architecture list compiled into vLLM's own
extensions:

| | `vllm_env` (original) | `vllm_env_multiarch` |
|---|---|---|
| vLLM | `0.29.1rc1.dev345+gbf13ecc2f` | same commit |
| torch | `2.13.0+cu130` | `2.13.0+cu130` |
| triton | `3.7.1` | `3.7.1` |
| tokenspeed-triton | `3.8.10.post20260920` | `3.8.10.post20260920` |
| cubins | sm_80, sm_86 | sm_80 71, sm_86 51, sm_90 82 |

**The question, and the answer.** The multiarch install also pulled
`tokenspeed-triton-3.8.10.post20260920`, and `LOG.md` carried an open item
asking whether it had *displaced* `triton 3.7.1`. Had it, every A40-vs-A100
comparison would be confounded by more than the architecture list and the
"same binary, only architectures differ" premise behind C16 and C17 would be
false. **It had not.** The two venvs carry the same triton and
tokenspeed-triton versions, the same torch and the same vLLM commit; the
compiled architecture set is the only difference. The premise holds.

**Provenance, stated plainly.** This was established in conversation on 11 Oct
and left standing as an open item in this file and in `LOG.md` for a day. That
is a `RULES.md` section 8 violation by whoever resolved it — an environment
fact belongs here — and the dead-ends table in `LOG.md` now carries the row so
the cost of the omission is visible rather than absorbed.

Re-check with, after any reinstall on either venv:

```bash
for V in ~/vllm_env /projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch; do
  echo "$V"
  "$V"/bin/python3 -c "import importlib.metadata as m; [print('  ', p, m.version(p)) for p in ('vllm','torch','triton','tokenspeed-triton')]"
done
```

## Guard now in both job scripts

```bash
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$CAP" in
  80|86) echo "compute capability $CAP is covered by this build" ;;
  *) echo "ABORT: compute capability $CAP has no cubin in this vLLM build"; exit 1 ;;
esac
```

This turns a confusing mid-run CUDA error into an immediate, labelled abort,
and costs a few seconds of GPU time instead of a wasted reservation.

## Memory ceiling on concurrency, same topic, different cause

Qwen3-8B stores about 144 KiB of KV per token: 36 layers, 8 KV heads,
head dim 128, two tensors, bf16. One 32K sequence is therefore about 4.6 GB.

| | A40 48 GB | A100 40 GB |
|---|---|---|
| usable at `gpu_memory_utilization=0.90` | ~43 GB | ~36 GB |
| less weights and DFlash head | ~25 GB for KV | ~18 GB for KV |
| KV capacity in tokens | ~182,000 | ~131,000 |
| reachable concurrency at 2K | ~88 | ~64 |
| reachable concurrency at 8K | ~22 | ~16 |
| reachable concurrency at 32K | ~5 | ~4 |

So a concurrency grid of 1, 8, 32 is honest only at 2K. At 8K and 32K vLLM
will accept `--max-num-seqs 32` and then schedule fewer sequences than asked,
which silently mislabels the row. Hence the per-bucket grids:

| Bucket | Grid | Limit |
|---|---|---|
| 2K | 1, 8, 32 | 50 |
| 8K | 1, 4, 12 | 50 |
| 32K | 1, 2, 4 | 25 |

Verify the achieved value from the vLLM startup line in every log:

```bash
grep -i "maximum concurrency" /projects/bikd/$USER/Speculative_Ladder/logs/<log>
```

**Open methodological question for the cross-hardware table.** A40 and A100
end up with different KV budgets, 25 GB against 18 GB, because the cards
differ in size. Batch size is controlled explicitly by `--max-num-seqs`, so
the comparison at a fixed concurrency is still sound. But if a later claim
depends on the two GPUs having identical KV capacity, pin it with
`--num-gpu-blocks-override` instead of a utilization fraction, and say so in
the paper.
