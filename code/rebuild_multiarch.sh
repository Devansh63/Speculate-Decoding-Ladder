#!/bin/bash
# Rebuild vLLM with sm_80, sm_86 and sm_90, into a SEPARATE virtualenv.
#
# Why: the current build in ~/vllm_env fails on A100 with
# cudaErrorNoKernelImageForDevice. Root cause in env/a100-sm80-failure.md:
# half the kernels in the main extension have no sm_80 build, because vLLM's
# CMake applies per-kernel architecture filters and setting
# TORCH_CUDA_ARCH_LIST is necessary but evidently was not sufficient.
#
# Restoring A100 unblocks claim C4, the portability claim that the whole
# tau/R separation exists to support. Right now every measurement in the
# project comes from one GPU.
#
# ---------------------------------------------------------------- safety
#
# This NEVER touches ~/vllm_env. A failed build costs disk and time, nothing
# else, and every existing measurement stays reproducible. Nothing in the
# project uses the new venv until someone passes LADDER_VENV explicitly.
#
# It pins the same commit, bf13ecc2f, so the new binary differs from the old
# in architecture coverage and nothing else. That is what would make it
# defensible to compare numbers across the two.
#
# ------------------------------------------------------------- how to run
#
# On a login node, inside tmux, because this takes one to three hours and an
# SSH drop would kill it:
#
#   ssh dagraw2@dt-login02.delta.ncsa.illinois.edu   # pin the node, tmux is per-node
#   tmux new -s build
#   bash rebuild_multiarch.sh
#   # Ctrl-B then D to detach, close the laptop
#   # later: ssh back to the SAME node, tmux attach -t build
#
# Or without tmux:
#   nohup bash rebuild_multiarch.sh > ~/rebuild.out 2>&1 &
#
# Either way everything is logged to $LOGFILE below.
#
# MAX_JOBS defaults to 8, not the core count. This is a shared login node and
# the account has no CPU allocation, so a compute node would spend GPU-hours
# on a CPU-bound job. Slower, but not antisocial.

set -uo pipefail

COMMIT=${COMMIT:-bf13ecc2f}
ARCHS=${ARCHS:-"8.0;8.6;9.0"}
MAX_JOBS=${MAX_JOBS:-8}

SHARED=${LADDER_SHARED:-/projects/bikd/$USER/Speculative_Ladder}
SRCDIR=${SRCDIR:-/work/nvme/bikd/$USER/vllm-build}
NEWVENV=${NEWVENV:-$SHARED/vllm_env_multiarch}
LOGFILE=${LOGFILE:-$SHARED/logs/rebuild-$(date +%Y%m%d-%H%M%S).log}

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================="
echo " vLLM multi-architecture rebuild"
echo "============================================================="
echo "commit    : $COMMIT"
echo "archs     : $ARCHS"
echo "max jobs  : $MAX_JOBS"
echo "source    : $SRCDIR"
echo "new venv  : $NEWVENV"
echo "log       : $LOGFILE"
echo "started   : $(date)"
echo "host      : $(hostname)"
echo
echo "This does NOT touch \$HOME/vllm_env. That environment stays frozen."
echo

if [ -e "$NEWVENV" ]; then
  echo "ABORT: $NEWVENV already exists."
  echo "       Remove it or set NEWVENV to a different path, then re-run."
  exit 1
fi

module load cray-python/3.12.12
module load cudatoolkit 2>/dev/null || true

echo "--- toolchain ---"
which python3 && python3 --version
which gcc && gcc --version | head -1
which nvcc && nvcc --version | tail -2
which cmake && cmake --version | head -1
echo

# ------------------------------------------------------------- 1. source
echo "=== [1/5] fetching vLLM source at $COMMIT ==="
mkdir -p "$(dirname "$SRCDIR")"
if [ -d "$SRCDIR/.git" ]; then
  echo "reusing existing clone"
  cd "$SRCDIR" && git fetch --all --tags
else
  git clone https://github.com/vllm-project/vllm.git "$SRCDIR" || exit 1
  cd "$SRCDIR"
fi
git checkout "$COMMIT" || { echo "ABORT: commit $COMMIT not found"; exit 1; }
git submodule update --init --recursive
echo "HEAD now at: $(git rev-parse --short HEAD)"
echo

# --------------------------------------------------------------- 2. venv
echo "=== [2/5] creating $NEWVENV ==="
python3 -m venv "$NEWVENV" || exit 1
# shellcheck disable=SC1091
source "$NEWVENV/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel ninja cmake packaging || exit 1
echo

# -------------------------------------------------------------- 3. torch
# Install the SAME torch the working environment has, so the only difference
# between the two builds is architecture coverage.
echo "=== [3/5] installing torch to match the frozen environment ==="
TORCH_SPEC=${TORCH_SPEC:-"torch==2.13.0"}
echo "target: $TORCH_SPEC (frozen env has $("$HOME/vllm_env/bin/python3" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo unknown))"
python3 -m pip install "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cu130 \
  || python3 -m pip install "$TORCH_SPEC" \
  || { echo "ABORT: could not install torch"; exit 1; }
python3 -c "import torch;print('torch', torch.__version__, torch.version.cuda)"
echo

# -------------------------------------------------------------- 4. build
echo "=== [4/5] building vLLM, this is the long part ==="
echo "started at $(date). Expect one to three hours at MAX_JOBS=$MAX_JOBS."
cd "$SRCDIR"
export TORCH_CUDA_ARCH_LIST="$ARCHS"
export MAX_JOBS="$MAX_JOBS"
export NVCC_THREADS=${NVCC_THREADS:-2}
export VLLM_TARGET_DEVICE=cuda
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"

python3 -m pip install -r requirements/build.txt 2>/dev/null \
  || python3 -m pip install -r requirements-build.txt 2>/dev/null \
  || echo "note: no build requirements file found, continuing"

python3 -m pip install -e . --no-build-isolation
BUILD_RC=$?
echo "build exit code: $BUILD_RC"
echo "finished at $(date)"
if [ $BUILD_RC -ne 0 ]; then
  echo
  echo "BUILD FAILED. \$HOME/vllm_env is untouched and still works on A40."
  echo "Full log: $LOGFILE"
  exit $BUILD_RC
fi
echo

# ------------------------------------------------------------- 5. verify
# Count cubins per architecture. Do NOT just check that an architecture
# appears: the old build listed sm_80 and still failed, because a listing
# proves some kernel was built for it, not the one you need.
echo "=== [5/5] verifying architecture coverage ==="
CUOBJ=$(command -v cuobjdump || echo /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/bin/cuobjdump)
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
if [ -z "$VD" ]; then
  echo "WARNING: cannot import vllm from the new venv. Build may be incomplete."
  exit 1
fi
echo "vllm package: $VD"
python3 -c "import vllm;print('vllm version', vllm.__version__)"
echo

FAIL=0
for so in _C_stable_libtorch _moe_C_stable_libtorch _C _moe_C; do
  f="$VD/$so.abi3.so"
  [ -f "$f" ] || continue
  echo "== $so =="
  declare -A COUNTS=()
  for a in sm_80 sm_86 sm_90; do
    n=$("$CUOBJ" --list-elf "$f" 2>/dev/null | grep -c "$a")
    COUNTS[$a]=$n
    printf '  %-7s %s cubins\n' "$a" "$n"
  done
  ref=${COUNTS[sm_86]:-0}
  for a in sm_80 sm_90; do
    n=${COUNTS[$a]:-0}
    if [ "$ref" -gt 0 ] && [ "$n" -lt $((ref / 2)) ]; then
      echo "  PROBLEM: $a has $n cubins against sm_86's $ref."
      echo "  This is the same asymmetry that broke A100 before."
      FAIL=1
    fi
  done
  echo
done

echo "============================================================="
if [ $FAIL -eq 0 ]; then
  echo "Architecture counts look balanced."
else
  echo "ARCHITECTURE COVERAGE STILL ASYMMETRIC. Do not trust this build yet."
fi
echo
echo "Counting is necessary but NOT sufficient. The only proof that the"
echo "kernels run is executing one on the target GPU:"
echo
echo "  sbatch --export=ALL,LADDER_VENV=$NEWVENV \\"
echo "    $SHARED/ladder/scripts/diag_a100.sh"
echo
echo "Step 2 of that diagnostic (a bare rms_norm, no torch.compile, no CUDA"
echo "graphs) is the one that matters. If it passes, A100 is back."
echo
echo "Then a real cell, still without touching the frozen environment:"
echo
echo "  sbatch --partition=gpuA100x4 \\"
echo "    --export=ALL,BUCKET=2K,LIMIT=50,LADDER_VENV=$NEWVENV \\"
echo "    $SHARED/ladder/scripts/phase0.sh"
echo
echo "log: $LOGFILE"
echo "============================================================="
exit $FAIL
