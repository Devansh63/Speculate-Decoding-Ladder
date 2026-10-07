# A100 does not work with this vLLM build

Status as of 6 Oct 2026: **root cause found, requires a from-source rebuild.**

Evidence: job 22685764 (nine arms, all failed), job 22713175 (the diagnostic
that isolated the layer), and a cubin count that identified the cause.

## What happens

Every vLLM operation on A100 fails with:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution
on the device  (cudaErrorNoKernelImageForDevice)
```

No arm of the 2K sweep produced a result and `results.jsonl` was never
created. The "done" line at the end of the job log is the shell script
finishing its loop, not success. Worth knowing: the loop structure makes total
failure look like a COMPLETED job.

## Root cause

**Half the kernels in the main extension were never built for sm_80.**

```
_C_stable_libtorch      sm_80: 20 cubins    sm_86: 40 cubins
_moe_C_stable_libtorch  sm_80: 15 cubins    sm_86:  7 cubins
```

vLLM's CMake applies per-kernel architecture filters, so
`TORCH_CUDA_ARCH_LIST="8.0;8.6"` does not guarantee every kernel group emits
both. Some groups built for sm_86 only; `rms_norm`, which the diagnostic
exercises directly, is evidently among them.

Note the MoE extension is lopsided the *other* way, with more sm_80 than
sm_86. That rules out a global dropped architecture and confirms the filters
differ per kernel group.

Reproduce the count:

```bash
source <venv>/bin/activate
CUOBJ=$(command -v cuobjdump || echo /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/bin/cuobjdump)
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
for so in _C_stable_libtorch _moe_C_stable_libtorch; do
  echo "== $so =="
  for a in sm_80 sm_86; do
    printf '  %s: %s cubins\n' "$a" "$("$CUOBJ" --list-elf "$VD/$so.abi3.so" | grep -c "$a")"
  done
done
```

## How it was narrowed

`code/diag_a100.sh`, run on `gpua027`, four steps:

| step | what it exercises | result |
|---|---|---|
| 1 | plain torch: bf16 matmul, `fill_` | **OK** |
| 2 | vLLM's own `rms_norm`, no compile, no CUDA graphs | **FAIL** |
| 3 | vLLM eager (`enforce_eager=True`) | **FAIL** |
| 4 | vLLM compiled, the failing configuration | **FAIL** |

Node: `NVIDIA A100-SXM4-40GB`, compute capability `8.0`, driver `595.71.05`,
CUDA `13.2`. torch `2.13.0+cu130` reports `(8, 0)`, lists `sm_80` in its arch
list, and runs a bf16 matmul correctly.

Step 2 is the decisive one: a single custom op with nothing else involved.

## Ruled out, with the evidence

| hypothesis | how it was eliminated | cost |
|---|---|---|
| FlashAttention missing sm_80 | `cuobjdump` shows `sm_80` in `_vllm_fa2_C.abi3.so`; the run selected FLASH_ATTN v2 | free |
| compile cache poisoned by A40 runs | per-config and `torch_aot_compile` hashes are disjoint across GPUs; the A100 run logged nine saves and zero loads | free |
| node or driver fault | step 1 passes | 4 min |
| torch.compile or CUDA graph path only | steps 2 and 3 fail without either | 4 min |

## The fix

Rebuild vLLM from source at commit `bf13ecc2f` with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`. Hours of login-node compiling, zero GPU
hours, and the same rebuild covers H200.

**After any rebuild, verify with the cubin count above, not just the
architecture listing**, and then run `code/diag_a100.sh`. The count should be
comparable across architectures. Only step 2 passing proves the kernels run.

The cost is comparability. Every A40 measurement came from the current binary.
Two options, recorded in `TODO.md`:

1. Rebuild and re-run the A40 cells, roughly 3 GPU-hours. Every number then
   comes from one binary.
2. Keep the current venv for A40, build a second for A100 and H200. No
   re-runs, but the hardware axis spans two builds and that must be disclosed.

With the root cause known, a rebuild is now a well-founded fix rather than a
guess, which strengthens option 1.

## Do not do this instead

Forcing `VLLM_ATTENTION_BACKEND=TRITON_ATTN` would run today, because Triton
compiles at runtime. It would also void the cross-hardware comparison: A40 ran
on FlashAttention and A100 would run on Triton, and R is precisely the
quantity that would absorb the difference. A fast wrong number is worse than
no number.

## Two red herrings worth recording

**The traceback line.** The first failure pointed at
`vllm/v1/attention/backends/flash_attn.py:1211`, `return output.fill_(0)`.
That line is the profiling shortcut: when `attn_metadata is None` during a
dummy warmup, attention is skipped and the output tensor is zeroed. No
attention kernel runs there. The error was a **sticky CUDA error** from an
earlier failed launch surfacing at the next synchronisation point. CUDA errors
are asynchronous, so the traceback names where the error was noticed, not
where it happened. `CUDA_LAUNCH_BLOCKING=1` helps but does not cover CUDA
graph capture, which is why even the blocking run still pointed at `fill_`.

**The architecture listing.** `cuobjdump --list-elf` reporting `sm_80` means
only that *some* kernel was built for it. Counting per architecture is equally
cheap and would have pointed straight here. The general rule: a listing is
evidence about the build, not about whether the code runs. The only proof is
executing one custom op on the target GPU, which costs four minutes.

## How to re-check the whole dependency stack

```bash
CUOBJ=$(command -v cuobjdump || echo /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/bin/cuobjdump)
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
SP=$(dirname "$VD")
for f in $(find "$SP" -name '*.so' -size +1M | sort); do
  archs=$("$CUOBJ" --list-elf "$f" 2>/dev/null | sed -n 's/.*\(sm_[0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ')
  [ -n "$archs" ] && printf '%-72s %s\n' "${f#$SP/}" "$archs"
done
```

`find`, not `ls *.so`. A non-recursive glob misses the FlashAttention
extension in a subdirectory of the vllm package, and that omission produced a
confident wrong conclusion earlier in this project.
