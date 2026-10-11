#!/bin/bash
# Submit the two A100 cells that test the frozen prediction.
#
# The prediction is recorded in paper/claims.md under "Frozen prediction,
# 11 Oct 2026", committed before any A100 cell at full n was run. It says the
# A100 grid follows from the A40 grid by a single machine factor of 1.054
# applied to R, with tau carried over unchanged.
#
# RUN THIS ON A LOGIN NODE. It submits two jobs and exits. It is not itself a
# batch script, and it does not reimplement the harness: it calls the
# already-verified phase0.sh twice.
#
#   bash submit_a100_predict.sh
#
# Costs roughly 1.5 GPU-hours. A100 bills at about twice the A40 rate.
#
# ------------------------------------------------------------------ why these
#
# Three things here are easy to get wrong by hand, and each has already cost
# this project a job or a day.
#
#   LADDER_VENV. phase0.sh resolves its venv as LADDER_VENV, then
#   $HOME/vllm_env, then the original shared venv. Both fallbacks are
#   single-architecture builds with no sm_80 kernels, so an A100 job that
#   omits this variable reserves a node and dies at kernel launch. That is
#   job 22685764: seventeen minutes billed, nine arms lost, nothing written.
#
#   LADDER_OUT. results/ in the shared tree is deliberately not
#   group-writable, and two jobs appending to one results.jsonl can interleave
#   partial lines. Each user writes to their own tree.
#
#   LIMIT per bucket. The prediction was derived from A40 cells at n=50 (8K)
#   and n=25 (32K). Running a different prompt count does not test it.

set -uo pipefail

SHARED=/projects/bikd/dagraw2/Speculative_Ladder
VENV=$SHARED/vllm_env_multiarch
PHASE0=$SHARED/ladder/scripts/phase0.sh
OUT=/projects/bikd/$USER/Speculative_Ladder/results/a100

if [ ! -x "$VENV/bin/python3" ]; then
  echo "ABORT: no usable virtualenv at $VENV"
  echo "       Without the multi-architecture build an A100 job cannot run."
  exit 1
fi

if [ ! -f "$PHASE0" ]; then
  echo "ABORT: cannot find $PHASE0"
  exit 1
fi

if ! mkdir -p "$OUT" 2>/dev/null; then
  echo "ABORT: cannot create $OUT"
  echo "       You need write access to /projects/bikd/$USER. Ask Devansh."
  exit 1
fi

echo "user : $USER"
echo "venv : $VENV"
echo "out  : $OUT"
echo

JIDS=""
for CELL in "8K:50" "32K:25"; do
  B=${CELL%%:*}
  N=${CELL##*:}
  JID=$(sbatch --parsable \
    --job-name=a100pred \
    --partition=gpuA100x4 \
    --time=02:00:00 \
    --export=ALL,LADDER_VENV=$VENV,LADDER_OUT=$OUT,BUCKET=$B,LIMIT=$N,CONC=1 \
    "$PHASE0")
  if [ -z "$JID" ]; then
    echo "FAILED to submit $B — nothing was queued for it"
    continue
  fi
  echo "submitted  bucket=$B  n=$N  conc=1  ->  job $JID"
  JIDS="$JIDS,$JID"
done
JIDS=${JIDS#,}

echo
echo "watch:"
echo "  squeue -u \$USER"
echo "  sacct -j $JIDS --format=JobID,Partition,NodeList%12,State,Elapsed,ExitCode -X"
echo
echo "logs land in the shared directory, readable by both of you:"
echo "  $SHARED/logs/phase0-<jobid>.out"
echo
echo "read the results:"
echo "  module load cray-python/3.12.12"
echo "  python3 $SHARED/ladder/scripts/summarize.py $OUT"
echo
echo "------------------------------------------------------------------"
echo "What was predicted, before these jobs ran:"
echo
echo "   cell              predicted A100 R   predicted speedup"
echo "   8K  DFlash k=7    1.627              1.20x"
echo "   8K  n-gram k=5    1.181              1.09x"
echo "   32K DFlash k=7    1.861              0.78x"
echo "   32K n-gram k=5    1.360              1.02x"
echo
echo "Passes if every cell is within 5% of predicted AND the 32K ordering"
echo "holds (DFlash below 1.0, n-gram above). Fails if the median absolute"
echo "relative error across the four exceeds 10%, or 32K does not invert."
echo
echo "Note before reading 32K: the A100-SXM4 has 40 GB against the A40's 48,"
echo "so the 32K concurrency ceiling is lower and that cell may sit in a"
echo "different memory regime. A large 32K miss may be a capacity artifact"
echo "rather than a failure of the cost model. Judge the prediction on 8K."
echo "------------------------------------------------------------------"
