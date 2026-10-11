#!/bin/bash
#SBATCH --job-name=c10control
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=00:45:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/c10-%j.out

# The same-config control C10 is missing.
#
# C10 as written compares the `off` arm against the `off` arm at a DIFFERENT
# concurrency (check_lossless.py: "off c=1 vs off c=8") while the treatment
# compares a speculative arm against the baseline at MATCHING concurrency. The
# control varies two things, the treatment one, so the strong claim cannot be
# read off them. This job produces the missing control: two `off` runs at
# identical flags, differing only in process and output directory.
#
# --------------------------------------------------------------- how to run
#
#   sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch c10_control.sh
#
# Then, on the login node, once the job is done:
#
#   python3 $SHARED/ladder/scripts/c10_control.py \
#       --dirs /projects/bikd/$USER/Speculative_Ladder/results/c10_control/r1 \
#              /projects/bikd/$USER/Speculative_Ladder/results/c10_control/r2 \
#       --c $SHARED/results/a40/perreq_off_2K_k0_n50_s1234_c1.json \
#       --d $SHARED/results/a40/perreq_off_2K_k0_n50_s1234_c8.json
#
# The --c/--d pair is optional and reproduces the confounded control (12.7%
# as first recorded) so the two rates and the difference between them arrive
# with intervals on one common resample of prompts.
#
# --------------------------------------------------------------- design notes
#
# ONE JOB, TWO ARMS, TWO DIRECTORIES. Both runs land on the same node in the
# same hour, so node identity and soak are held constant and the only variation
# left is engine-level. Two separate jobs would add node variance to a
# measurement whose entire purpose is to bound engine variance. Cost of the
# choice: if the first arm dies the second still runs, and the log says so.
#
# SEPARATE OUTPUT DIRECTORIES ARE THE POINT. Per-request filenames encode
# rung/bucket/k/n/seed/concurrency — not the run — so a second run into the
# same directory is an overwrite. That is exactly how the existing paired
# realizations lost the token ids of their first run (see paper/claims.md,
# item 1 of the 11 Oct external review). r1/ and r2/ keep both.
#
# `off` ONLY, TWICE. Nothing speculative runs here, so the arm count in the log
# must be exactly 2. If a future edit reintroduces a pre-loop `run ngram 5`
# — the pattern kprobe_32k.sh carries — the log will show it, which is why the
# arm counter is printed at the end.
#
# COST. Floor from the nearest job that actually ran: job 22598608's `off` arm
# at 2K n=50 concurrency 1 took 393 s of generation; per-arm engine init was
# 172 s in job 22757104's log. Two arms is therefore about 19 minutes of wall
# clock, ~0.32 GPU-hours billed on the A40 partition. --time is 45 minutes so a
# slow node cannot eat the allocation.

set -uo pipefail

BUCKET=${BUCKET:-2K}
LIMIT=${LIMIT:-50}
SEED=${SEED:-1234}
CONC=1   # fixed on purpose: the control is a same-concurrency comparison

SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
HF=${LADDER_HF:-/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub}
DATA=${LADDER_DATA:-$SHARED/ladder/data/longbench_v2}
RUN=${LADDER_RUN:-$SHARED/ladder/scripts/run_benchmark.py}

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

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CACHE_ROOT=${LADDER_CACHE:-$SHARED/.vllm_cache}

MODEL=${LADDER_MODEL:-$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}

GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$GPUNAME" in
  *A40*)  GPUTAG=a40 ;;
  *A100*) GPUTAG=a100 ;;
  *H200*) GPUTAG=h200 ;;
  *H100*) GPUTAG=h100 ;;
  *)      GPUTAG=$(echo "$GPUNAME" | tr -c 'A-Za-z0-9' '_' | tr 'A-Z' 'a-z') ;;
esac

# One tree per user, never the shared results/ -- it is deliberately not
# group-writable and two jobs appending to one results.jsonl can interleave.
BASE=${LADDER_OUT:-/projects/bikd/$USER/Speculative_Ladder/results/c10_control}
if ! mkdir -p "$BASE/r1" "$BASE/r2" 2>/dev/null; then
  echo "ABORT: cannot create $BASE — you need write access to /projects/bikd/$USER"
  exit 1
fi
mkdir -p "$SHARED/logs" "$VLLM_CACHE_ROOT"

echo "gpu='$GPUNAME' cap=$CAP tag=$GPUTAG"
echo "bucket=$BUCKET limit=$LIMIT seed=$SEED conc=$CONC job=${SLURM_JOB_ID:-none}"
echo "out=$BASE/{r1,r2}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
date

case "$CAP" in
  80|86) echo "compute capability $CAP is covered by this build" ;;
  *) echo "ABORT: compute capability $CAP has no cubin in this vLLM build."
     echo "       See env/gpu-support.md. H200 needs a separate venv."
     exit 1 ;;
esac

ARMS=0
control_run () {  # $1 = subdirectory, $2 = label
  echo "### control arm=$2 dir=$1 bucket=$BUCKET limit=$LIMIT seed=$SEED conc=$CONC  $(date +%H:%M:%S)"
  ARMS=$((ARMS + 1))
  python3 "$RUN" \
    --dataset "$DATA/longbench_v2_${BUCKET}.jsonl" \
    --model "$MODEL" \
    --rung off --k 0 --bucket "$BUCKET" \
    --limit "$LIMIT" --max-tokens 256 --seed "$SEED" \
    --max-num-seqs "$CONC" \
    --outdir "$BASE/$1" \
    || echo "FAILED control arm=$2 dir=$1"
}

control_run r1 "1of2"
control_run r2 "2of2"

date
echo "### arms_run=$ARMS  (must be 2; anything else means the script changed)"
echo "### perreq files:"
ls -l "$BASE/r1"/perreq_*.json "$BASE/r2"/perreq_*.json 2>&1
echo
echo "done. records at $BASE/r1/results.jsonl and $BASE/r2/results.jsonl"
echo
echo "next, on a login node:"
echo "  python3 $SHARED/ladder/scripts/c10_control.py \\"
echo "      --dirs $BASE/r1 $BASE/r2 \\"
echo "      --c $SHARED/results/a40/perreq_off_2K_k0_n50_s1234_c1.json \\"
echo "      --d $SHARED/results/a40/perreq_off_2K_k0_n50_s1234_c8.json"
echo
echo "then bring both records and both perreq files back:"
echo "  cd $SHARED/repo && git pull && cp -r $BASE results/c10_control && \\"
echo "    git add results/c10_control && git commit -m 'results: C10 same-config control' && git push"
