#!/bin/bash
#SBATCH --job-name=phase0
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-%j.out

# Phase 0: one context bucket, three rungs, across the concurrency axis,
# on the stable model runner.
#
# Submit one job per bucket:
#   sbatch --export=ALL,BUCKET=2K,LIMIT=100 phase0_a40.sh
#   sbatch --export=ALL,BUCKET=8K,LIMIT=100 phase0_a40.sh
#   sbatch --export=ALL,BUCKET=32K,LIMIT=50 --time=02:00:00 phase0_a40.sh

set -uo pipefail

BUCKET=${BUCKET:-2K}
LIMIT=${LIMIT:-100}
SEED=${SEED:-1234}

module load cray-python/3.12.12
source "$HOME/vllm_env/bin/activate"

# Both must-do rungs on the stable runner. run_benchmark.py refuses to start
# without this, which is deliberate: without it n-gram silently falls back to
# the stable runner while DFlash stays on V2, and the comparison is void.
export VLLM_USE_V2_MODEL_RUNNER=0

ROOT=/projects/bikd/$USER/Speculative_Ladder

# Shared compile cache, so the four minute first start becomes about one
# minute on later runs, for both of us.
export VLLM_CACHE_ROOT=$ROOT/.vllm_cache

DATA=$ROOT/ladder/data/longbench_v2
HF=/work/nvme/bikd/$USER/Speculative_Ladder/hf/hub
MODEL=$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DRAFT=$HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
OUT=$ROOT/results/a40
RUN=$ROOT/ladder/scripts/run_benchmark.py

mkdir -p "$OUT" "$ROOT/logs" "$VLLM_CACHE_ROOT"

echo "bucket=$BUCKET limit=$LIMIT seed=$SEED job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

# The no-speculation arm runs first in every concurrency group. Without it
# there is no speedup number, and speedup crossing 1.0x is what the milestone
# reports. Acceptance is never compared across different k without truncating
# to a common length first.
run () {  # $1 rung, $2 k, $3 extra args, $4 concurrency
  echo "### rung=$1 k=$2 conc=$4 bucket=$BUCKET  $(date +%H:%M:%S)"
  python3 "$RUN" \
    --dataset "$DATA/longbench_v2_${BUCKET}.jsonl" \
    --model "$MODEL" \
    --rung "$1" --k "$2" --bucket "$BUCKET" \
    --limit "$LIMIT" --max-tokens 256 --seed "$SEED" \
    --max-num-seqs "$4" \
    --outdir "$OUT" $3 \
    || echo "FAILED rung=$1 k=$2 conc=$4 bucket=$BUCKET"
}

for C in 1 8 32; do
  run off    0  ""                 $C
  run ngram  5  ""                 $C
  run dflash 7  "--draft $DRAFT"   $C
done

date
echo "done. results appended to $OUT/results.jsonl"
