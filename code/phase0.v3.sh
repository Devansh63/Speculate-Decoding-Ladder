#!/bin/bash
#SBATCH --job-name=phase0
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-%j.out

# Phase 0, one script for every GPU and every team member.
#
# Supersedes phase0_a40.sh and phase0_a100.sh, which are kept only as the
# record of what produced jobs 22598608 and 22685764. Use this one from now on.
#
# Two things make it portable. The GPU label is read from the device at run
# time rather than inferred from the partition, so results cannot be filed
# under the wrong hardware by a mistyped --partition. And every path is an
# overridable variable defaulting to the shared copy, so a teammate needs no
# setup of their own beyond a working vLLM.
#
# ---------------------------------------------------------------- how to run
#
# A40, the default partition:
#   sbatch --export=ALL,BUCKET=2K,LIMIT=50                 phase0.sh
#   sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC="1 4 12"   phase0.sh
#   sbatch --time=03:00:00 --export=ALL,BUCKET=32K,LIMIT=25,CONC="1 2 4" phase0.sh
#
# A100, override the partition:
#   sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2K,LIMIT=50 phase0.sh
#
# ------------------------------------------------------ the v3 knobs, by use
#
# V3=/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts/run_benchmark.v3.py
#
# C10 control AND the v3 no-op proof, one job (~15 min). Two identical off
# runs into one directory; the v3 job id in the file name keeps both, and
# check_lossless.v3.py pairs them as SAME-CONFIG:
#   sbatch --export=ALL,BUCKET=2K,LIMIT=50,CONC=1,ARMS="off:0",REPEAT=2,\
#     LADDER_RUN=$V3,LADDER_OUT=$SHARED/results/a40_c10 phase0.sh
#
# Prefill/decode split. MAXTOK=1 is prefill plus one step; subtract from the
# 256-token run for the marginal per-token cost. NOTE the intercept is
# arm-dependent -- a 1-token speculative run still pays one draft-and-verify
# cycle -- so only the DIFFERENCE is comparable across arms:
#   sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC=1,MAXTOK=1 phase0.sh
#
# n-gram k sweep, on a clean node (the existing sweep ran on gpub073, which is
# on the exclusion list):
#   sbatch --exclude=gpub073 --export=ALL,BUCKET=32K,LIMIT=25,CONC=1,\
#     ARMS="off:0 ngram:1 ngram:2 ngram:5 ngram:7" phase0.sh
#
# Coverage knob and temperature, v3 harness only:
#   sbatch --export=ALL,ARMS="off:0 ngram:5",PLMAX=7,LADDER_RUN=$V3 phase0.sh
#   sbatch --export=ALL,TEMP=0.7,LADDER_RUN=$V3 phase0.sh
#
# Do NOT point this at gpuH200x8. The shared vLLM build has no sm_90 kernels
# and no PTX, so the job would reserve a node and die at kernel launch. The
# guard below catches it, but the reservation is still wasted.
#
# ------------------------------------------------- the concurrency grid, why
#
# Qwen3-8B stores about 144 KiB of KV per token, and the measured cache on an
# A40 is 160,912 tokens. One 32K sequence is therefore about a fifth of the
# cache. Reachable concurrency:
#
#   bucket     A40 48 GB    A100 40 GB    grid to use
#   2K         ~70          ~52           1 8 32
#   8K         ~19          ~14           1 4 12
#   32K        ~4.9         ~3.6          1 2 4
#
# Asking for more than that does not fail. vLLM accepts --max-num-seqs 32 and
# then schedules fewer sequences than asked, so the row gets labelled with a
# concurrency it never ran at. That is worse than a crash. Check every log:
#   grep -i "maximum concurrency" <logfile>

set -uo pipefail

BUCKET=${BUCKET:-2K}
LIMIT=${LIMIT:-50}
SEED=${SEED:-1234}
CONC=${CONC:-"1 8 32"}

# ---------------------------------------------------------------- v3 knobs
# Every default here reproduces the pre-v3 script exactly.
MAXTOK=${MAXTOK:-256}                      # was hardcoded 256
ARMS=${ARMS:-"off:0 ngram:5 dflash:7"}     # rung:k, in order; off first
REPEAT=${REPEAT:-1}                        # runs of each arm, for controls

# Optional, v3 harness only. Unset means "do not pass the flag at all", so
# the v2 harness never sees an argument it does not know.
XARGS=""
[ -n "${TEMP:-}" ]  && XARGS="$XARGS --temperature $TEMP"
[ -n "${PLMIN:-}" ] && XARGS="$XARGS --prompt-lookup-min $PLMIN"
[ -n "${PLMAX:-}" ] && XARGS="$XARGS --prompt-lookup-max $PLMAX"
[ -n "${LOOP:-}" ]  && XARGS="$XARGS --loop-timing"

# ------------------------------------------------------------------ paths
# Shared by default. A teammate overrides nothing unless they want to.
SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
HF=${LADDER_HF:-/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub}
DATA=${LADDER_DATA:-$SHARED/ladder/data/longbench_v2}
RUN=${LADDER_RUN:-$SHARED/ladder/scripts/run_benchmark.py}

# Prefer a personal venv if one exists, fall back to the shared copy. This way
# the script works whether or not a teammate has built their own vLLM.
VENV=${LADDER_VENV:-}
if [ -z "$VENV" ]; then
  if [ -x "$HOME/vllm_env/bin/python3" ]; then
    VENV=$HOME/vllm_env
  else
    VENV=$SHARED/vllm_env
  fi
fi

module load cray-python/3.12.12
if [ ! -x "$VENV/bin/python3" ]; then
  echo "ABORT: no usable virtualenv at $VENV"
  echo "       set LADDER_VENV, or see notes/teammate-quickstart.md"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "venv: $VENV"

# Both must-do rungs on the stable runner. run_benchmark.py refuses to start
# without this, which is deliberate: without it n-gram silently falls back to
# the stable runner while DFlash stays on V2, and the comparison is void.
export VLLM_USE_V2_MODEL_RUNNER=0

# Compile cache is keyed by hardware, so the first run on a new GPU pays about
# four minutes and later ones about one. Shared, so nobody pays it twice.
export VLLM_CACHE_ROOT=${LADDER_CACHE:-$SHARED/.vllm_cache}

# A v3-only request against the v2 harness must fail HERE, not per arm.
# run() ends in "|| echo FAILED", so an argparse rejection would be swallowed
# and the job would hold a GPU for its full walltime producing nothing.
HARNESS_IS_V3=0
if grep -q 'HARNESS_VERSION = "v3' "$RUN" 2>/dev/null; then HARNESS_IS_V3=1; fi
echo "harness: $RUN (v3=$HARNESS_IS_V3)"

if [ "$HARNESS_IS_V3" != "1" ]; then
  if [ -n "$XARGS" ]; then
    echo "ABORT: TEMP/PLMIN/PLMAX/LOOP need the v3 harness, but $RUN is not v3."
    echo "       Point LADDER_RUN at run_benchmark.v3.py, or unset them."
    exit 1
  fi
  if [ "$REPEAT" -gt 1 ]; then
    echo "ABORT: REPEAT=$REPEAT needs the v3 harness. v2 writes"
    echo "       perreq_<run_id>.json with no job id, so run 2 would silently"
    echo "       overwrite run 1 and the repeat would buy nothing."
    echo "       Point LADDER_RUN at run_benchmark.v3.py."
    exit 1
  fi
fi

MODEL=${LADDER_MODEL:-$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
DRAFT=${LADDER_DRAFT:-$HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}

# ------------------------------------------------------ identify the hardware
# Read the GPU from the device, not from the partition name. A mislabelled
# result is harder to notice than a failed job and poisons the cross-hardware
# table, which is the whole point of the study.
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$GPUNAME" in
  *A40*)  GPUTAG=a40 ;;
  *A100*) GPUTAG=a100 ;;
  *H200*) GPUTAG=h200 ;;
  *H100*) GPUTAG=h100 ;;
  *)      GPUTAG=$(echo "$GPUNAME" | tr -c 'A-Za-z0-9' '_' | tr 'A-Z' 'a-z') ;;
esac

OUT=${LADDER_OUT:-$SHARED/results/$GPUTAG}
mkdir -p "$OUT" "$SHARED/logs" "$VLLM_CACHE_ROOT"

echo "gpu='$GPUNAME' cap=$CAP tag=$GPUTAG"
echo "bucket=$BUCKET limit=$LIMIT seed=$SEED conc='$CONC' job=${SLURM_JOB_ID:-none}"
echo "out=$OUT"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
date

# The shared vLLM build carries cubins for sm_80 and sm_86 only, verified with
# cuobjdump, and embeds no PTX, so there is no JIT fallback. Abort in seconds
# rather than four minutes into a model load.
case "$CAP" in
  80|86) echo "compute capability $CAP is covered by this build" ;;
  *) echo "ABORT: compute capability $CAP has no cubin in this vLLM build."
     echo "       See env/gpu-support.md. H200 needs a separate venv."
     exit 1 ;;
esac

# The no-speculation arm runs first in every concurrency group. Without it
# there is no speedup number, and speedup crossing 1.0x is what the milestone
# reports. Acceptance is never compared across different k without truncating
# to a common draft length first.
run () {  # $1 rung, $2 k, $3 extra args, $4 concurrency, $5 repeat index
  echo "### rung=$1 k=$2 conc=$4 rep=$5/$REPEAT maxtok=$MAXTOK " \
       "bucket=$BUCKET gpu=$GPUTAG  $(date +%H:%M:%S)"
  python3 "$RUN" \
    --dataset "$DATA/longbench_v2_${BUCKET}.jsonl" \
    --model "$MODEL" \
    --rung "$1" --k "$2" --bucket "$BUCKET" \
    --limit "$LIMIT" --max-tokens "$MAXTOK" --seed "$SEED" \
    --max-num-seqs "$4" \
    --outdir "$OUT" $3 $XARGS \
    || echo "FAILED rung=$1 k=$2 conc=$4 rep=$5 bucket=$BUCKET gpu=$GPUTAG"
}

# ARMS is "rung:k" pairs in order. The default is the old three arms with the
# old k values, so an existing submission is unchanged. --draft is added for
# dflash automatically rather than being spelled out per call.
for C in $CONC; do
  for ARM in $ARMS; do
    ARM_RUNG=${ARM%%:*}
    ARM_K=${ARM##*:}
    ARM_EXTRA=""
    [ "$ARM_RUNG" = "dflash" ] && ARM_EXTRA="--draft $DRAFT"
    REP=1
    while [ "$REP" -le "$REPEAT" ]; do
      run "$ARM_RUNG" "$ARM_K" "$ARM_EXTRA" "$C" "$REP"
      REP=$((REP + 1))
    done
  done
done

date
echo "done. results appended to $OUT/results.jsonl"
echo
echo "next:"
echo "  python3 $SHARED/ladder/scripts/summarize.py $OUT"
echo "  bash $SHARED/ladder/scripts/push_results.sh \"$GPUTAG $BUCKET sweep, job ${SLURM_JOB_ID:-}\""
