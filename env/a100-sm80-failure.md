# A100 does not work with this vLLM build

Status as of 6 Oct 2026: **blocked, requires a from-source rebuild.**

Evidence: job 22685764 (nine arms, all failed) and job 22713175 (the
diagnostic that isolated the cause).

## What happens

Every vLLM operation on A100 fails with:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution
on the device  (cudaErrorNoKernelImageForDevice)
```

No arm of the 2K sweep produced a result. The `results.jsonl` file was never
created. The "done" line at the end of the job log is the shell script
finishing its loop, not success; it is worth knowing that the loop structure
makes total failure look like completion.

## What the diagnostic established

`code/diag_a100.sh`, run on `gpua027`, four steps:

| step | what it exercises | result |
|---|---|---|
| 1 | plain torch: bf16 matmul, `fill_` | **OK** |
| 2 | vLLM's own `rms_norm` custom op, no compile, no CUDA graphs | **FAIL** |
| 3 | vLLM eager (`enforce_eager=True`) | **FAIL** |
| 4 | vLLM compiled, the failing configuration | **FAIL** |

Node identity: `NVIDIA A100-SXM4-40GB`, compute capability `8.0`, driver
`595.71.05`, CUDA `13.2`. torch `2.13.0+cu130` reports capability `(8, 0)` and
an arch list containing `sm_80`, and runs a bf16 matmul correctly.

**The node, the driver and torch are all healthy. vLLM's own CUDA extension
does not work on sm_80.** Step 2 is the decisive one: a single custom op, with
no torch.compile and no CUDA graphs anywhere near it.

## Ruled out, with the evidence

| hypothesis | how it was eliminated | cost |
|---|---|---|
| FlashAttention missing sm_80 | `cuobjdump` shows `sm_80` in `_vllm_fa2_C.abi3.so`; the run selected FLASH_ATTN v2 | free |
| compile cache poisoned by A40 runs | per-config hashes and `torch_aot_compile` hashes are disjoint across the two GPUs; the A100 run logged nine saves and zero loads | free |
| node or driver fault | diagnostic step 1 passes | 4 min |
| torch.compile or CUDA graph path only | diagnostic steps 2 and 3 fail without either | 4 min |

## The remaining puzzle

`cuobjdump --list-elf` reports `sm_80` and `sm_86` cubins in
`_C_stable_libtorch.abi3.so`, yet sm_80 kernels do not run. Most likely the
build emitted sm_80 for only a subset of kernels: vLLM's CMake applies
per-kernel architecture filters, so `TORCH_CUDA_ARCH_LIST="8.0;8.6"` does not
guarantee every kernel is built for both. Counting cubins per architecture
would confirm it.

**The practical lesson stands regardless: `cuobjdump` listing an architecture
does not prove the kernels run on it.** The only reliable test is to execute
one custom op on the target GPU, which costs about four minutes.

## The fix, and the decision it forces

A from-source rebuild of vLLM at commit `bf13ecc2f` with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`. Hours of login-node compiling, zero GPU
hours. The same rebuild unlocks H200, so it is one job for both.

The cost is comparability. Every A40 measurement so far came from the current
binary. Two options:

1. **Rebuild and re-run the A40 cells.** Roughly 3 GPU-hours. Cleanest: every
   number in the paper comes from one binary.
2. **Keep the current venv for A40, build a second for A100 and H200.** No
   re-runs, but the hardware axis then spans two builds, and that has to be
   disclosed. Defensible if both come from the same commit and differ only in
   the architecture list, but a reviewer is entitled to ask whether the sm_86
   code path changed.

Current lean is option 1, conditional on the allocation being topped up.

## Do not do this instead

Forcing `VLLM_ATTENTION_BACKEND=TRITON_ATTN` would run today, because Triton
compiles at runtime for whatever GPU it finds. It would also void the
cross-hardware comparison, since A40 ran on FlashAttention and A100 would run
on Triton. R is precisely the quantity that would absorb the difference, and R
is what the study is measuring. A fast wrong number is worse than no number.

## Red herring worth recording

The first traceback pointed at:

```
vllm/v1/attention/backends/flash_attn.py:1211   return output.fill_(0)
```

That line is the profiling shortcut: when `attn_metadata is None` during a
dummy warmup, attention is skipped and the output tensor is just zeroed. No
attention kernel runs there at all. The error was a **sticky CUDA error** from
an earlier failed launch, surfacing at the next synchronisation point.

This cost real time. CUDA errors are asynchronous, so the line in the
traceback is where the error was *noticed*, not where it happened.
`CUDA_LAUNCH_BLOCKING=1` helps but does not cover CUDA graph capture, which is
why even the blocking run still pointed at `fill_`. The reliable approach is
to bisect by layer, which is what the diagnostic does.

## How to re-check after any rebuild

```bash
source <venv>/bin/activate
CUOBJ=$(command -v cuobjdump || echo /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/bin/cuobjdump)
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
SP=$(dirname "$VD")
for f in $(find "$SP" -name '*.so' -size +1M | sort); do
  archs=$("$CUOBJ" --list-elf "$f" 2>/dev/null | sed -n 's/.*\(sm_[0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ')
  [ -n "$archs" ] && printf '%-72s %s\n' "${f#$SP/}" "$archs"
done
```

Note `find`, not `ls *.so`. A non-recursive glob misses the FlashAttention
extension, which lives in a subdirectory of the vllm package, and that
omission produced a confident wrong conclusion earlier in this project.

Then run `code/diag_a100.sh` on the target partition. The architecture listing
is necessary but not sufficient; only step 2 passing proves the kernels run.
