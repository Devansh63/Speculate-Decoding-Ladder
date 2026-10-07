#!/bin/bash
#SBATCH --job-name=diaga100
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/diag-a100-%j.out

# Why job 22685764 failed on A100 with cudaErrorNoKernelImageForDevice.
#
# Already ruled out, for free, from the committed logs:
#   - missing sm_80 in the shipped extensions. cuobjdump shows sm_80 in
#     _C_stable_libtorch, _moe_C_stable_libtorch and _vllm_fa2_C, and the run
#     selected FLASH_ATTN v2, which is the one that has it.
#   - compile cache poisoning from the A40 runs. The per-config hashes and the
#     torch_aot_compile hashes are disjoint across the two GPUs, and the A100
#     run logged nine saves and zero loads, so it compiled everything fresh.
#
# Still open, and this job separates them:
#   A. the node, its driver or its toolkit cannot run the cubins at all
#   B. the eager path works and only the torch.compile / CUDA graph path fails
#   C. the vLLM extension is a fat binary whose sm_80 half is missing kernels
#      that its sm_86 half has
#
# CUDA_LAUNCH_BLOCKING=1 is the point of the exercise. The original failure
# surfaced at output.fill_(0) and at .contiguous(), both trivial ops, which is
# what an asynchronous error looks like when it finally hits a sync point. With
# blocking launches the traceback lands on the kernel that actually failed.
#
#   sbatch diag_a100.sh
#
# Costs roughly ten minutes on one A100.

set -uo pipefail

module load cray-python/3.12.12
source "$HOME/vllm_env/bin/activate"

export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_LAUNCH_BLOCKING=1

ROOT=/projects/bikd/$USER/Speculative_Ladder
# Separate cache so this diagnostic cannot disturb the measurement cache.
export VLLM_CACHE_ROOT=$ROOT/.vllm_cache_diag
mkdir -p "$VLLM_CACHE_ROOT"

HF=/work/nvme/bikd/$USER/Speculative_Ladder/hf/hub
MODEL=$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

echo "############ node identity ############"
hostname
date
nvidia-smi
echo
echo "driver / toolkit:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version 2>/dev/null | tail -2 || echo "nvcc not on PATH"
echo

echo "############ step 1: plain torch, no vLLM ############"
echo "If this fails, the node or its driver cannot run this torch build at all,"
echo "and nothing about vLLM is relevant."
python3 - <<'PY'
import torch
print("torch        ", torch.__version__)
print("device       ", torch.cuda.get_device_name(0))
print("capability   ", torch.cuda.get_device_capability(0))
print("arch_list    ", torch.cuda.get_arch_list())
print("cuda runtime ", torch.version.cuda)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
print("bf16 matmul OK, mean |x| =", float(b.float().abs().mean()))
c = torch.empty(1024, device="cuda")
c.fill_(0)
torch.cuda.synchronize()
print("fill_ OK")
PY
echo "step 1 exit: $?"
echo

echo "############ step 2: vLLM custom ops directly ############"
echo "Exercises vLLM's own extension, which is the fat binary in question,"
echo "without torch.compile or CUDA graphs anywhere near it."
python3 - <<'PY'
import torch
try:
    import vllm._C  # noqa: F401
    print("vllm._C import OK")
except Exception as e:
    print("vllm._C import FAILED:", repr(e))
try:
    from vllm import _custom_ops as ops
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, 1e-6)
    torch.cuda.synchronize()
    print("rms_norm OK, mean |x| =", float(out.float().abs().mean()))
except Exception as e:
    print("custom op FAILED:", repr(e))
PY
echo "step 2 exit: $?"
echo

echo "############ step 3: vLLM eager, no compile, no cudagraphs ############"
echo "If this works and step 4 fails, the fault is in the compiled path only,"
echo "and enforce_eager is a workaround (at a cost to comparability)."
python3 - "$MODEL" <<'PY'
import sys
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], dtype="bfloat16", seed=1234,
          max_model_len=2048, gpu_memory_utilization=0.85,
          max_num_seqs=1, trust_remote_code=True,
          enforce_eager=True, disable_log_stats=False)
o = llm.generate(["The capital of France is"],
                 SamplingParams(temperature=0.0, max_tokens=16))
print("EAGER OK:", repr(o[0].outputs[0].text))
PY
echo "step 3 exit: $?"
echo

echo "############ step 4: vLLM compiled, the configuration that failed ############"
python3 - "$MODEL" <<'PY'
import sys
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], dtype="bfloat16", seed=1234,
          max_model_len=2048, gpu_memory_utilization=0.85,
          max_num_seqs=1, trust_remote_code=True,
          disable_log_stats=False)
o = llm.generate(["The capital of France is"],
                 SamplingParams(temperature=0.0, max_tokens=16))
print("COMPILED OK:", repr(o[0].outputs[0].text))
PY
echo "step 4 exit: $?"
echo

date
echo "done"
