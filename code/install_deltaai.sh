#!/bin/bash
# Install vLLM 0.33.0 on DeltaAI (aarch64, GH200) from the published wheel,
# into a project-owned virtualenv. Minutes, not hours, and no compile.
#
#   ssh <netid>@dtai-login.delta.ncsa.illinois.edu
#   accounts                                  # your project code
#   PROJ=<code> bash install_deltaai.sh
#
# Why 0.33.0 and not the project's old pin (bf13ecc2f): on DeltaAI the plan
# is to run every arm on Model Runner V2, where n-gram and DFlash share one
# runner and one scheduling mode. 0.33.0 is the first tagged release after
# the change that stops V2 inheriting V1's async fallbacks (PR #62521), it
# still ships both runners, and PyPI carries an aarch64 wheel for it.
#
# Every number from this venv is a new baseline. It is a different vLLM,
# torch, runner, CPU and GPU from every filed A40 and A100 row.
#
# Not verified, because nobody has run this yet:
#   - which CUDA build of torch the PyPI vLLM wheel was compiled against.
#     This installs torch 2.14.0+cu130, the pin in vLLM's requirements, from
#     PyTorch's own index, because PyPI's aarch64 torch may not be a CUDA
#     build. If the smoke job fails at import or at the first kernel, that
#     mismatch is the first suspect, and the fallback is a source build:
#       COMMIT=v0.33.0 TORCH_SPEC=torch==2.14.0 TRITON_SPEC= TS_TRITON_SPEC= \
#         NEWVENV=<path> PROJ=<code> bash build_deltaai.sh
#   - that the wheel's kernels cover sm_90. Official wheels are built for
#     Hopper, and the smoke job is the proof.

set -uo pipefail

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ABORT: this host is $(uname -m); install_deltaai.sh is for DeltaAI."
  exit 1
fi
if [ -z "${PROJ:-}" ]; then
  echo "ABORT: set PROJ to your DeltaAI project code (run: accounts)."
  exit 1
fi

VLLM_SPEC=${VLLM_SPEC:-"vllm==0.33.0"}
TORCH_SPEC=${TORCH_SPEC:-"torch==2.14.0+cu130"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
PYVER=${PYVER:-3.12}
LADDER_ROOT=${LADDER_ROOT:-/work/nvme/$PROJ/$USER/Speculative_Ladder}
NEWVENV=${NEWVENV:-$LADDER_ROOT/vllm033_gh200}
LOGFILE=${LOGFILE:-$LADDER_ROOT/logs/install-$(date +%Y%m%d-%H%M%S).log}

if [ ! -d "/work/nvme/$PROJ" ] && [ -z "${LADDER_ROOT_OK:-}" ]; then
  echo "ABORT: /work/nvme/$PROJ does not exist. Check the code with: accounts"
  exit 1
fi
if [ -e "$NEWVENV" ]; then
  echo "ABORT: $NEWVENV already exists. Remove it or set NEWVENV."
  exit 1
fi
mkdir -p "$(dirname "$LOGFILE")" || exit 1
exec > >(tee -a "$LOGFILE") 2>&1

echo "vllm   : $VLLM_SPEC"
echo "torch  : $TORCH_SPEC from $TORCH_INDEX"
echo "venv   : $NEWVENV"
echo "host   : $(hostname) ($(uname -m))   $(date)"
echo

MODS_LOADED=""
if command -v uv >/dev/null 2>&1; then
  uv venv --seed --python "$PYVER" "$NEWVENV" || exit 1
else
  for m in ${PYTHON_MODULE:-} cray-python; do
    if [ -n "$m" ] && module load "$m" 2>/dev/null; then MODS_LOADED="$m"; break; fi
  done
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || { echo "ABORT: vLLM 0.33.0 needs Python 3.11 or newer; got $(python3 --version)"; exit 1; }
  python3 -m venv "$NEWVENV" || exit 1
  echo "NOTE: no uv; venv built on $(python3 --version) from module '$MODS_LOADED'"
fi
echo "$MODS_LOADED" | sed '/^$/d' > "$NEWVENV/ladder_modules"
# shellcheck disable=SC1091
source "$NEWVENV/bin/activate"
python3 --version
python3 -m pip install --upgrade pip || exit 1

# torch first, from PyTorch's index, so the CUDA build is the one installed
# and vLLM's own torch==2.14.0 requirement is already satisfied.
python3 -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX" \
  || { echo "ABORT: could not install $TORCH_SPEC"; exit 1; }
python3 -m pip install "$VLLM_SPEC" --extra-index-url "$TORCH_INDEX" \
  || { echo "ABORT: could not install $VLLM_SPEC. Nothing outside $LADDER_ROOT was touched."; exit 1; }
python3 -m pip install "huggingface_hub[cli]" >/dev/null 2>&1 || true

echo
echo "--- verify ---"
python3 - <<'PY' || { echo "ABORT: the installed stack does not import"; exit 1; }
import importlib, platform
import torch, vllm
print("vllm    ", vllm.__version__)
print("torch   ", torch.__version__, "cuda build:", torch.version.cuda)
print("python  ", platform.python_version(), platform.machine())
assert torch.version.cuda, "torch is a CPU-only build; the +cu130 wheel did not survive the vllm install"
for name in ("triton", "numba", "flashinfer"):
    try:
        m = importlib.import_module(name)
        print("%-8s" % name, getattr(m, "__version__", "?"))
    except Exception as e:
        print("%-8s IMPORT FAILED: %r" % (name, e))
PY
python3 -m pip check || echo "NOTE: pip check reports conflicts above; read them"
python3 -m pip freeze > "$NEWVENV/ladder_freeze.txt"
{
  echo "installed $(date)"
  echo "host      $(hostname) $(uname -m)"
  echo "vllm      $VLLM_SPEC (PyPI wheel)"
  echo "torch     $TORCH_SPEC"
  echo "modules   $MODS_LOADED"
} > "$NEWVENV/ladder_build_info.txt"

echo
echo "Installed. A login node has no GPU, so nothing above proves a kernel"
echo "runs. Next:"
echo "  PROJ=$PROJ ACCOUNT=<account> source deltaai_env.sh"
echo "  bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7"
echo "log: $LOGFILE"
