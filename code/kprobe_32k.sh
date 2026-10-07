#!/bin/bash
#SBATCH --job-name=kprobe32k
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/kprobe32k-%j.out

# Is DFlash broken at 32K, or just running at the wrong draft length?
#
# At 32K, concurrency 1, DFlash k=7 scored 0.83x: worse than no speculation.
# But its acceptance by draft position is
#
#     0.296, 0.087, 0.038, 0.017, 0.010, 0.007, 0.004
#
# dead after position two, with a 6.6% acceptance rate on drafted tokens. Fit
# the marginal cost from R = 1 + k*c: the measured R = 1.766 at k=7 gives
# c = 0.109 per draft position. Position 2 returns 0.087 and costs 0.109, so it
# is already a losing trade, and every position after it is worse.
#
# Prediction: the optimal k at 32K is 1, giving
#
#     tau = 1.296    R ~= 1.109    speedup ~= 1.17x
#
# which would beat n-gram's 1.08x and reverse the reading that the trained head
# fails at long context. The failure would instead be a configuration chosen
# for short context and never revisited.
#
# That is the more useful result. "Use the cheap rung at long context" is a
# ranking a reader has to accept. "Shrink the draft length as context grows,
# and the trained head still wins" is something an operator can act on.
#
# This is not a grid. Context and load push the optimum the same way (at 2K,
# k=15 already lost to k=7 under saturation), so one axis at a time is enough
# to establish the direction. Concurrency 1 only, which is where the inversion
# was measured and where the prediction is sharpest.
#
#   sbatch kprobe_32k.sh
#
# About an hour. Six arms at 32K concurrency 1 are slow because decoding is
# serial with 32K of attention per step.

set -uo pipefail

BUCKET=${BUCKET:-32K}
LIMIT=${LIMIT:-25}
SEED=${SEED:-1234}
CONC=${CONC:-1}
KLIST=${KLIST:-"1 2 4 7"}

module load cray-python/3.12.12

SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
HF=${LADDER_HF:-/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub}
DATA=${LADDER_DATA:-$SHARED/ladder/data/longbench_v2}
RUN=${LADDER_RUN:-$SHARED/ladder/scripts/run_benchmark.py}

VENV=${LADDER_VENV:-}
if [ -z "$VENV" ]; then
  if [ -x "$HOME/vllm_env/bin/python3" ]; then VENV=$HOME/vllm_env; else VENV=$SHARED/vllm_env; fi
fi
if [ ! -x "$VENV/bin/python3" ]; then
  echo "ABORT: no usable virtualenv at $VENV"; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "venv: $VENV"

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CACHE_ROOT=${LADDER_CACHE:-$SHARED/.vllm_cache}

MODEL=${LADDER_MODEL:-$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
DRAFT=${LADDER_DRAFT:-$HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}

GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$GPUNAME" in
  *A40*)  GPUTAG=a40 ;;
  *A100*) GPUTAG=a100 ;;
  *H200*) GPUTAG=h200 ;;
  *)      GPUTAG=$(echo "$GPUNAME" | tr -c 'A-Za-z0-9' '_' | tr 'A-Z' 'a-z') ;;
esac
case "$CAP" in
  80|86) echo "compute capability $CAP is covered by this build" ;;
  *) echo "ABORT: compute capability $CAP has no cubin in this vLLM build"; exit 1 ;;
esac

OUT=${LADDER_OUT:-$SHARED/results/$GPUTAG}
mkdir -p "$OUT" "$SHARED/logs" "$VLLM_CACHE_ROOT"

echo "gpu='$GPUNAME' tag=$GPUTAG bucket=$BUCKET limit=$LIMIT conc=$CONC klist='$KLIST'"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
date

run () {  # $1 rung, $2 k, $3 extra args
  echo "### rung=$1 k=$2 conc=$CONC bucket=$BUCKET gpu=$GPUTAG  $(date +%H:%M:%S)"
  python3 "$RUN" \
    --dataset "$DATA/longbench_v2_${BUCKET}.jsonl" \
    --model "$MODEL" \
    --rung "$1" --k "$2" --bucket "$BUCKET" \
    --limit "$LIMIT" --max-tokens 256 --seed "$SEED" \
    --max-num-seqs "$CONC" \
    --outdir "$OUT" $3 \
    || echo "FAILED rung=$1 k=$2 conc=$CONC bucket=$BUCKET"
}

# The baseline and n-gram are re-run deliberately, not for completeness.
#
# The k arms need a baseline measured in the same engine session; borrowing
# job 22713173's would mean comparing against a different process on a
# possibly different node.
#
# And since per-request timing came back 0/25 on this vLLM build, the paired
# bootstrap is impossible and repeated runs are the only route to confidence
# intervals. This gives a second independent realization of the headline 32K
# concurrency-1 row, for about eighteen minutes of GPU time.
run off   0  ""
run ngram 5  ""

# k=7 replicates the measured 0.83x. If it does not land close, run-to-run
# variance is large enough that the whole 32K cell needs more repeats before
# anything is claimed from it, and that is worth knowing before the k result
# is believed.
for K in $KLIST; do
  run dflash "$K" "--draft $DRAFT"
done

date
echo "done. results appended to $OUT/results.jsonl"
echo
echo "next:"
echo "  python3 $SHARED/ladder/scripts/summarize.py $OUT"
echo "  bash $SHARED/ladder/scripts/push_results.sh \"32K k-probe, job ${SLURM_JOB_ID:-}\""
echo
echo "what to look for:"
echo "  dflash k=1 should beat k=7 and ideally exceed 1.00x"
echo "  dflash k=7 should land near 0.83x, replicating job 22713173"
echo "  off and ngram k=5 give the first repeat estimate of run-to-run variance"
