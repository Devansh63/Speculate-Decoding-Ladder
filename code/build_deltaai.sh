#!/bin/bash
# Build vLLM at the project's pinned commit on DeltaAI (aarch64, GH200), into
# a project-owned virtualenv. A variant of rebuild_multiarch.sh, which is
# x86_64-only and is left untouched (RULES section 6).
#
# Plan items E19 and E20. Nothing on Delta is touched: DeltaAI has its own
# home directory, and this writes only under LADDER_ROOT.
#
# Same commit (bf13ecc2f), same torch (2.13.0+cu130), same triton pins as the
# Delta build. What differs is the CPU architecture, the compiler, and the
# single GPU architecture compiled in (sm_90, the H100 inside a GH200). That
# makes every DeltaAI number a NEW COLUMN, measured from scratch. Ratios from
# this binary are never mixed with the filed A40 and A100 ratios.
#
# ------------------------------------------------------------- how to run
#
# On a DeltaAI login node, inside tmux (the build is one to three hours and
# an SSH drop would kill it; tmux lives on ONE login node, note which):
#
#   ssh <netid>@dtai-login.delta.ncsa.illinois.edu
#   accounts                      # find your four-letter project code
#   tmux new -s build
#   PROJ=<code> bash build_deltaai.sh
#
# Building uses no GPU hours. MAX_JOBS defaults to 8 because this is a shared
# login node.
#
# PROJ is required: it is the four-letter code `accounts` prints, the one in
# /work/nvme/<code>. It may or may not be bikd on DeltaAI; do not guess.
#
# What has NOT been verified, because nobody has run this yet:
#   - which module provides nvcc on DeltaAI (the script tries several and
#     stops with instructions if none works);
#   - that every runtime dependency in requirements/cuda.txt has an aarch64
#     wheel. Step 3 checks each one and lists the ones that do not, BEFORE the
#     long compile. torch 2.13.0+cu130, triton 3.7.1 and tokenspeed-triton
#     3.8.10.post20260920 were confirmed to have aarch64 wheels on 15 Nov 2026;
#   - that vLLM's CMake builds cleanly with the Cray gcc on aarch64.

set -uo pipefail

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ABORT: this host is $(uname -m). build_deltaai.sh is for DeltaAI login"
  echo "       nodes. On Delta use rebuild_multiarch.sh."
  exit 1
fi
if [ -z "${PROJ:-}" ]; then
  echo "ABORT: set PROJ to your DeltaAI project code (run: accounts)."
  exit 1
fi

COMMIT=${COMMIT:-bf13ecc2f}
ARCHS=${ARCHS:-"9.0"}
MAX_JOBS=${MAX_JOBS:-8}
PYVER=${PYVER:-3.12}
TORCH_SPEC=${TORCH_SPEC:-"torch==2.13.0"}
TRITON_SPEC=${TRITON_SPEC:-"triton==3.7.1"}
TS_TRITON_SPEC=${TS_TRITON_SPEC:-"tokenspeed-triton==3.8.10.post20260920"}

LADDER_ROOT=${LADDER_ROOT:-/work/nvme/$PROJ/$USER/Speculative_Ladder}
SRCDIR=${SRCDIR:-$LADDER_ROOT/vllm-build}
NEWVENV=${NEWVENV:-$LADDER_ROOT/vllm_env_gh200}
LOGFILE=${LOGFILE:-$LADDER_ROOT/logs/build-$(date +%Y%m%d-%H%M%S).log}

if [ ! -d "/work/nvme/$PROJ" ] && [ -z "${LADDER_ROOT_OK:-}" ]; then
  echo "ABORT: /work/nvme/$PROJ does not exist. Check the code with: accounts"
  echo "       (or set LADDER_ROOT yourself and LADDER_ROOT_OK=1)."
  exit 1
fi

mkdir -p "$(dirname "$LOGFILE")" || exit 1
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================="
echo " vLLM build for DeltaAI (aarch64, sm_90)"
echo "============================================================="
echo "commit    : $COMMIT"
echo "archs     : $ARCHS"
echo "python    : $PYVER"
echo "max jobs  : $MAX_JOBS"
echo "root      : $LADDER_ROOT"
echo "source    : $SRCDIR"
echo "new venv  : $NEWVENV"
echo "log       : $LOGFILE"
echo "started   : $(date)"
echo "host      : $(hostname) ($(uname -m))"
echo

if [ -e "$NEWVENV" ]; then
  echo "ABORT: $NEWVENV already exists."
  echo "       Remove it or set NEWVENV to a different path, then re-run."
  exit 1
fi

# ---------------------------------------------------------- 0. toolchain
# Module names on DeltaAI were not known when this was written. Try the
# likely ones, then insist on the tools themselves rather than on a name.
MODS_LOADED=""
try_module () {
  if module load "$1" 2>/dev/null; then MODS_LOADED="$MODS_LOADED $1"; return 0; fi
  return 1
}
if ! command -v nvcc >/dev/null 2>&1; then
  for m in ${CUDA_MODULE:-} cudatoolkit cuda nvhpc; do
    [ -n "$m" ] && try_module "$m" && command -v nvcc >/dev/null 2>&1 && break
  done
fi

echo "--- toolchain ---"
MISSING=0
for t in gcc g++ git nvcc; do
  if command -v "$t" >/dev/null 2>&1; then
    printf '%-6s %s\n' "$t" "$(command -v "$t")"
  else
    printf '%-6s MISSING\n' "$t"; MISSING=1
  fi
done
gcc --version 2>/dev/null | head -1
nvcc --version 2>/dev/null | tail -2
if [ "$MISSING" = "1" ]; then
  echo
  echo "ABORT: a build tool is missing. For nvcc, find the CUDA module with"
  echo "         module spider cuda"
  echo "       and re-run with CUDA_MODULE=<name>. The site also keeps a vLLM"
  echo "       recipe under /sw/user/python/ that shows which modules it uses."
  exit 1
fi
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
echo "CUDA_HOME=$CUDA_HOME"

# vLLM's build requirements include setuptools-rust, so a Rust toolchain is
# needed. Install one under LADDER_ROOT if the node has none.
if ! command -v cargo >/dev/null 2>&1; then
  echo "no cargo on PATH; installing rustup under $LADDER_ROOT/rust"
  export RUSTUP_HOME=$LADDER_ROOT/rust/rustup CARGO_HOME=$LADDER_ROOT/rust/cargo
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal || { echo "ABORT: rustup failed"; exit 1; }
  export PATH=$CARGO_HOME/bin:$PATH
fi
echo "cargo  $(command -v cargo)  $(cargo --version 2>/dev/null)"
echo

# ------------------------------------------------------------- 1. source
echo "=== [1/6] fetching vLLM source at $COMMIT ==="
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
# The Delta build used Python 3.12. DeltaAI's cray-python is 3.11, so prefer
# uv (which fetches a standalone 3.12) and fall back to whatever python3 is
# loaded, saying so. The extensions are abi3, so 3.11 would work; 3.12 keeps
# one fewer difference between the two builds.
echo "=== [2/6] creating $NEWVENV ==="
if command -v uv >/dev/null 2>&1; then
  uv venv --seed --python "$PYVER" "$NEWVENV" || exit 1
  echo "venv python from uv"
else
  for m in ${PYTHON_MODULE:-} cray-python; do
    [ -n "$m" ] && try_module "$m" && break
  done
  python3 -m venv "$NEWVENV" || exit 1
  echo "NOTE: no uv; venv built on $(python3 --version) from the loaded module."
  echo "      The Delta build is Python 3.12. Record this difference."
fi
# Job scripts read this file and load the same modules before activating.
echo "$MODS_LOADED" | tr ' ' '\n' | sed '/^$/d' > "$NEWVENV/ladder_modules"
# shellcheck disable=SC1091
source "$NEWVENV/bin/activate"
python3 --version
python3 -m pip install --upgrade pip wheel || exit 1
echo

# -------------------------------------------------------------- 3. torch
echo "=== [3/6] torch, and a wheel check on every runtime dependency ==="
python3 -m pip install "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cu130 \
  || { echo "ABORT: could not install $TORCH_SPEC for aarch64"; exit 1; }
python3 -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda)" || exit 1

# One line per requirement, each tried on its own, so a missing aarch64 wheel
# is named now rather than three hours into the compile.
WHEELTMP=$(mktemp -d)
NOWHEEL=""
if [ -f requirements/cuda.txt ]; then
  grep -v '^\s*#' requirements/cuda.txt | grep -v '^\s*-' | sed 's/#.*//' | sed '/^\s*$/d' \
  | while IFS= read -r req; do
      req=$(echo "$req" | sed 's/[[:space:]]*$//')
      if python3 -m pip install --quiet --dry-run --no-deps --ignore-installed \
           --only-binary :all: \
           --extra-index-url https://flashinfer.ai/whl/ \
           --extra-index-url https://download.pytorch.org/whl/cu130 \
           "$req" >/dev/null 2>&1; then
        printf '  ok       %s\n' "$req"
      else
        printf '  NO WHEEL %s\n' "$req"
        echo "$req" >> "$WHEELTMP/missing.txt"
      fi
    done
fi
if [ -s "$WHEELTMP/missing.txt" ]; then
  NOWHEEL=$(tr '\n' ';' < "$WHEELTMP/missing.txt")
  echo
  echo "Some dependencies have no aarch64 wheel (or need a source build):"
  sed 's/^/    /' "$WHEELTMP/missing.txt"
  echo "The build continues. If the install in step 5 fails on one of these,"
  echo "that is the cause. Video and audio packages are not used by this"
  echo "project; flashinfer, numba and the triton pair are."
  if [ "${STRICT_WHEELS:-0}" = "1" ]; then
    echo "ABORT: STRICT_WHEELS=1"; rm -rf "$WHEELTMP"; exit 1
  fi
fi
rm -rf "$WHEELTMP"
echo

# -------------------------------------------------------------- 4. build
echo "=== [4/6] building vLLM, this is the long part ==="
echo "started at $(date). The Delta build took one to three hours at MAX_JOBS=8."
export TORCH_CUDA_ARCH_LIST="$ARCHS"
export MAX_JOBS="$MAX_JOBS"
export NVCC_THREADS=${NVCC_THREADS:-2}
export VLLM_TARGET_DEVICE=cuda
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export CC=${CC:-$(command -v gcc)} CXX=${CXX:-$(command -v g++)}
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS  CC=$CC"

python3 -m pip install -r requirements/build/cuda.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  || { echo "ABORT: build requirements failed to install"; exit 1; }

python3 -m pip install -e . --no-build-isolation \
  --extra-index-url https://flashinfer.ai/whl/ \
  --extra-index-url https://download.pytorch.org/whl/cu130
BUILD_RC=$?
echo "build exit code: $BUILD_RC"
echo "finished at $(date)"
if [ $BUILD_RC -ne 0 ]; then
  echo
  echo "BUILD FAILED. Nothing outside $LADDER_ROOT was touched."
  [ -n "$NOWHEEL" ] && echo "Dependencies without a wheel: $NOWHEEL"
  echo "Full log: $LOGFILE"
  exit $BUILD_RC
fi
echo

# -------------------------------------------------------------- 5. pins
# The Delta venvs carry these two exact versions (env/gpu-support.md). pip may
# have resolved something newer; put them back and say what changed.
echo "=== [5/6] matching the triton pins of the Delta build ==="
python3 -m pip install "$TRITON_SPEC" "$TS_TRITON_SPEC" \
  || echo "WARNING: could not pin triton to the Delta versions; recorded below"
python3 -m pip check || echo "NOTE: pip check reports conflicts above; read them"
echo

# ------------------------------------------------------------- 6. verify
echo "=== [6/6] verifying ==="
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
if [ -z "$VD" ]; then
  echo "ABORT: cannot import vllm from the new venv."
  exit 1
fi
python3 - <<'PY'
import importlib, platform
import torch, vllm
print("vllm    ", vllm.__version__)
print("torch   ", torch.__version__, "cuda", torch.version.cuda)
print("python  ", platform.python_version(), platform.machine())
for name in ("triton", "numba", "flashinfer"):
    try:
        m = importlib.import_module(name)
        print("%-8s" % name, getattr(m, "__version__", "?"))
    except Exception as e:
        print("%-8s IMPORT FAILED: %r" % (name, e))
PY
python3 -m pip list 2>/dev/null | grep -i -E "^(triton|tokenspeed|torch|vllm|numba|flashinfer)" || true

CUOBJ=$(command -v cuobjdump || echo "$CUDA_HOME/bin/cuobjdump")
FAIL=0
if [ -x "$CUOBJ" ]; then
  for so in _C_stable_libtorch _moe_C_stable_libtorch _C _moe_C; do
    f="$VD/$so.abi3.so"
    [ -f "$f" ] || continue
    n=$("$CUOBJ" --list-elf "$f" 2>/dev/null | grep -c "sm_90")
    printf '  %-28s sm_90 cubins: %s\n' "$so" "$n"
    [ "$n" -eq 0 ] && FAIL=1
  done
else
  echo "  no cuobjdump found; cubin count skipped. The GPU smoke job is the real test."
fi

# Freeze what was built, for env/deltaai.md.
python3 -m pip freeze > "$NEWVENV/ladder_freeze.txt" 2>/dev/null
{
  echo "built    $(date)"
  echo "host     $(hostname) $(uname -m)"
  echo "commit   $(git -C "$SRCDIR" rev-parse HEAD)"
  echo "archs    $ARCHS"
  echo "gcc      $(gcc --version | head -1)"
  echo "nvcc     $(nvcc --version | tail -1)"
  echo "modules  $MODS_LOADED"
} > "$NEWVENV/ladder_build_info.txt"

echo
echo "============================================================="
if [ $FAIL -ne 0 ]; then
  echo "A vLLM extension carries NO sm_90 cubins. Do not submit jobs yet."
else
  echo "Build finished. A count is not a proof: the only proof that the"
  echo "kernels run is the smoke job on a GH200."
fi
echo
echo "next, still on the login node:"
echo "  source deltaai_env.sh          # PROJ=$PROJ must still be set"
echo "  bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7"
echo
echo "Copy $NEWVENV/ladder_build_info.txt and ladder_freeze.txt into the"
echo "repository under env/ so the build is on record."
echo "log: $LOGFILE"
echo "============================================================="
exit $FAIL
