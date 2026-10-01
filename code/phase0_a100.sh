#!/bin/bash
#SBATCH --job-name=phase0a100
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/phase0a100-%j.out

# Phase 0 on A100, identical harness to phase0_a40.sh.
#
# Same binary, same model, same prompts, same seed, same concurrency grid.
# The only thing that changes is the GPU, which is the point: tau (mean
# accepted length) should be essentially unchanged because acceptance is a
# property of the drafter and the text, while R (verification cost ratio)
# should move, because A100 sits at a different ridge point than A40.
# If tau moves materially between the two, something in the setup is wrong
# and the cross-hardware claim is void.
#
# The vLLM build in ~/vllm_env carries cubins for sm_80 and sm_86 only
# (verified with cuobjdump, no PTX embedded). sm_80 is A100, so this runs.
# H200 is sm_90 and will fail at kernel launch on this build. Do not point
# this script at gpuH200x8.
#
# Submit one job per bucket:
#   sbatch --export=ALL,BUCKET=2K,LIMIT=50 phase0_a100.sh
#   sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC="1 4 12" phase0_a100.sh
#   sbatch --export=ALL,BUCKET=32K,LIMIT=25,CONC="1 2 4" phase0_a100.sh
#
# Note on the concurrency grid. A100 on Delta is the 40 GB part, smaller than
# the A40's 48 GB, so the KV budget is tighter here than on A40. Qwen3-8B
# stores roughly 144 KiB of KV per token, so a 32K sequence costs about
# 4.6 GB on its own. Asking for 32 concurrent sequences at 32K is not
# physically possible on either GPU, and vLLM will quietly schedule fewer
# rather than fail, which would make the concurrency label a lie. Hence the
# per-bucket grids above. After every run, confirm the achieved number with
#   grep -i "maximum concurrency" <logfile>
# which vLLM prints at startup.

set -uo pipefail

BUCKET=${BUCKET:-2K}
LIMIT=${LIMIT:-50}
SEED=${SEED:-1234}
CONC=${CONC:-"1 8 32"}

module load cray-python/3.12.12
source "$HOME/vllm_env/bin/activate"

# Both must-do rungs on the stable runner. run_benchmark.py refuses to start
# without this, which is deliberate: without it n-gram silently falls back to
# the stable runner while DFlash stays on V2, and the comparison is void.
export VLLM_USE_V2_MODEL_RUNNER=0

ROOT=/projects/bikd/$USER/Speculative_Ladder

# Compile cache is keyed by hardware, so the first A100 run pays the full
# compile cost (about four minutes) and later ones are about one minute.
# Shared root, so Sahil's A100 runs reuse it.
export VLLM_CACHE_ROOT=$ROOT/.vllm_cache

DATA=$ROOT/ladder/data/longbench_v2
HF=/work/nvme/bikd/$USER/Speculative_Ladder/hf/hub
MODEL=$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DRAFT=$HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
OUT=$ROOT/results/a100
RUN=$ROOT/ladder/scripts/run_benchmark.py

mkdir -p "$OUT" "$ROOT/logs" "$VLLM_CACHE_ROOT"

echo "bucket=$BUCKET limit=$LIMIT seed=$SEED conc='$CONC' job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
date

# Fail loudly rather than silently producing numbers on the wrong GPU. If a
# future job lands on anything that is not sm_80 or sm_86, this build has no
# kernels for it and no PTX to fall back on.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$CAP" in
  80|86) echo "compute capability $CAP is covered by this build" ;;
  *) echo "ABORT: compute capability $CAP has no cubin in this vLLM build"; exit 1 ;;
esac

# The no-speculation arm runs first in every concurrency group. Without it
# there is no speedup number, and speedup crossing 1.0x is what the milestone
# reports. Acceptance is never compared across different k without truncating
# to a common draft length first.
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

for C in $CONC; do
  run off    0  ""                 $C
  run ngram  5  ""                 $C
  run dflash 7  "--draft $DRAFT"   $C
done

date
echo "done. results appended to $OUT/results.jsonl"
