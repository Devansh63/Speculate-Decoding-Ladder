#!/bin/bash
# One piece of the router data job set (dtai3) on one GH200: a few cells of
# one workload file, back to back. Submit with submit_dtai3.sh, never with a
# bare sbatch: the account, partition, memory, walltime and log path come
# from there, and so do the piece's settings:
#
#   PIECE   a tag, letters and digits, unique per piece (directory suffix)
#   WL      the workload: a directory under $LADDER_ROUTER (default
#           $LADDER_ROOT/data/router), built by build_router_workloads.py
#   B       the bucket: nat 2K 8K 16K 32K mix (the file <B>.jsonl)
#   CELLS   cells to run, in order, from:
#             t       batch 1, timing: off, n-gram k 5 11, DFlash k 7 15
#             a       batch 1, per-request acceptance (the router's labels):
#                     the same four speculative arms as t, no off arm (vLLM
#                     refuses per-request acceptance without a drafter). All
#                     four, not k 11 and 15 alone: DFlash accepts differently at
#                     different k (dtai2 note), a k=5 run's steps are not a k=11
#                     run's steps cut short, and on the GH200 n-gram k=11 and
#                     DFlash k=15 are the arms whose greedy output changed
#             l       batch 1, the off arm alone with the target's top-5
#                     logprobs at every output position (added 28 Nov): the
#                     confidence feature for a step-level router, and the
#                     greedy reference the a cell's steps are aligned to
#             tT aT lT the same with thinking on, 2048 tokens, 25 prompts
#             c<N>    concurrency N: off, n-gram k 5, DFlash k 4 7 15
#             t1 t2 a1 a2 tT1 tT2 aT1 aT2 c<N>n c<N>d   halves of the cells
#                     above (added 28 Nov), which submit_dtai3.sh uses when a
#                     whole cell would not fit a 30-minute limit; t0 tn (and
#                     tT0 tTn) split t1 once more: off alone, n-gram k 5 11
#   LIM1    prompts in a batch-1 cell (default 50, 25 at 32K, never more
#           than the file holds less two for the warm-up)
#   LIML    prompts in a load cell (default 128 at 2K and 8K, 64 at 16K and
#           mix, 48 at 32K, capped the same way)
#
# Every cell: harness v5d, chat format (the target's own template, thinking
# off), natural stopping at the end of the turn, at most 1024 new tokens,
# max model length 40960. Results: $LADDER_ROOT/results/gh200r_<WL>_<B>_<cell>_p<PIECE>,
# one directory per cell and piece, so no two jobs ever append to one file.
#
# Written 25 Nov 2026. Tested offline only (test_dtai3.sh, stub engine).

set -uo pipefail

for v in PIECE WL B CELLS; do
  [ -n "${!v:-}" ] || { echo "ABORT: $v is not set. Submit with submit_dtai3.sh."; exit 1; }
done
[[ "$PIECE" =~ ^[A-Za-z0-9]+$ ]] || { echo "ABORT: PIECE='$PIECE': letters and digits only."; exit 1; }
for v in LADDER_ROOT LADDER_VENV LADDER_HF LADDER_CACHE LADDER_CAPS LADDER_MRV2; do
  [ -n "${!v:-}" ] || { echo "ABORT: $v is not set. PROJ=<code> ACCOUNT=<name> source deltaai_env.sh"; exit 1; }
done
[ "$LADDER_MRV2" = "1" ] || { echo "ABORT: LADDER_MRV2=$LADDER_MRV2; these jobs are for Model Runner V2."; exit 1; }
[ "$LADDER_CAPS" = "90" ] || { echo "ABORT: LADDER_CAPS=$LADDER_CAPS; on DeltaAI it is 90."; exit 1; }
if [ "$(uname -m)" != "aarch64" ] && [ "${LADDER_TEST_HOST:-0}" != "1" ]; then
  echo "ABORT: this node is $(uname -m); the dtai3 jobs run on DeltaAI's GH200 nodes."; exit 1
fi
ROUTER=${LADDER_ROUTER:-$LADDER_ROOT/data/router}
RES=${LADDER_RESULTS:-$LADDER_ROOT/results}
for v in LADDER_ROOT LADDER_VENV LADDER_HF LADDER_CACHE ROUTER RES; do
  case "${!v}" in /projects/*) echo "ABORT: $v=${!v} is under /projects; a ghx4 job may not see it."; exit 1 ;; esac
done
[ -f "$LADDER_VENV/ladder_modules" ] || { echo "ABORT: no $LADDER_VENV/ladder_modules."; exit 1; }

HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CELLSH=${LADDER_CELL:-$HERE/router_cell.sh}
[ -f "$CELLSH" ] || { echo "ABORT: no $CELLSH. Submit from the repository's code/ directory."; exit 1; }
F=$ROUTER/$WL/$B.jsonl
[ -s "$F" ] || { echo "ABORT: no workload $F (build_router_workloads.py)."; exit 1; }
ROWS=$(grep -c . "$F")
JOB=${SLURM_JOB_ID:-nojob}

# Prompts per cell, capped at what the file holds less the two warm-up rows.
cap () { local w=$1; [ "$w" -gt $((ROWS - 2)) ] && w=$((ROWS - 2)); echo "$w"; }
case "$B" in 32K) D1=25 ;; *) D1=50 ;; esac
case "$B" in 2K|8K) DL=128 ;; 32K) DL=48 ;; *) DL=64 ;; esac
LIM1=$(cap "${LIM1:-$D1}")
LIML=$(cap "${LIML:-$DL}")
# Workloads whose rows share a document (several instructions over one text)
# run with prefix caching off, so no prompt's prefill is a cache hit of the last.
case "$WL" in ragquote|synth) PCW=off ;; *) PCW="" ;; esac

echo "dtai3 piece $PIECE   workload $WL   bucket $B   cells '$CELLS'   job $JOB   host $(hostname)   $(date)"
echo "file $F: $ROWS rows; batch-1 cells measure $LIM1, load cells $LIML"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

GPULOG=$LADDER_ROOT/logs/dtai3-$PIECE-$JOB-gpu.csv
mkdir -p "$LADDER_ROOT/logs"
GPUPID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv -l 5 > "$GPULOG" 2>&1 &
  GPUPID=$!
fi
trap '[ -n "$GPUPID" ] && kill "$GPUPID" 2>/dev/null' EXIT

STATUS=(); FAILED=0
cell () {  # $1 cell name, then KEY=VALUE pairs for router_cell.sh
  local name=$1; shift
  echo "==================== cell $name  $(date +%H:%M:%S)"
  (
    unset CONC LIMIT PC PERREQ MAXTOK MAXLEN PFORMAT THINK PSTYLE SKIP_OFF K_NGRAM_GPU K_DFLASH LADDER_OUT LOGPROBS
    export BUCKET="$B" LADDER_DATA="$ROUTER/$WL" LADDER_OUT="$RES/gh200r_${WL}_${B}_${name}_p$PIECE"
    [ -n "$PCW" ] && export PC="$PCW"
    for kv in "$@"; do export "$kv"; done
    bash "$CELLSH"
  )
  local rc=$?
  STATUS+=("cell $name exit=$rc  -> gh200r_${WL}_${B}_${name}_p$PIECE")
  [ "$rc" -eq 0 ] || FAILED=$((FAILED + 1))
}

for c in $CELLS; do
  case "$c" in
    t)  cell t  CONC=1 LIMIT="$LIM1" "K_NGRAM_GPU=5 11" "K_DFLASH=7 15" ;;
    a)  cell a  CONC=1 LIMIT="$LIM1" SKIP_OFF=1 PERREQ=detailed "K_NGRAM_GPU=5 11" "K_DFLASH=7 15" ;;
    tT) cell tT CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 "K_NGRAM_GPU=5 11" "K_DFLASH=7 15" ;;
    aT) cell aT CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 SKIP_OFF=1 PERREQ=detailed "K_NGRAM_GPU=5 11" "K_DFLASH=7 15" ;;
    l)  cell l  CONC=1 LIMIT="$LIM1" LOGPROBS=5 K_NGRAM_GPU= K_DFLASH= ;;
    lT) cell lT CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 LOGPROBS=5 K_NGRAM_GPU= K_DFLASH= ;;
    # Halves (28 Nov), so a piece fits a 30-minute limit: the same arms as the
    # cell above, in two pieces. 1 = off (t only) and n-gram k 5 11; 2 = DFlash
    # k 7 15. Their rows together are exactly the whole cell's.
    t1)  cell t1  CONC=1 LIMIT="$LIM1" "K_NGRAM_GPU=5 11" K_DFLASH= ;;
    t2)  cell t2  CONC=1 LIMIT="$LIM1" SKIP_OFF=1 K_NGRAM_GPU= "K_DFLASH=7 15" ;;
    a1)  cell a1  CONC=1 LIMIT="$LIM1" SKIP_OFF=1 PERREQ=detailed "K_NGRAM_GPU=5 11" K_DFLASH= ;;
    a2)  cell a2  CONC=1 LIMIT="$LIM1" SKIP_OFF=1 PERREQ=detailed K_NGRAM_GPU= "K_DFLASH=7 15" ;;
    t0)  cell t0  CONC=1 LIMIT="$LIM1" K_NGRAM_GPU= K_DFLASH= ;;
    tn)  cell tn  CONC=1 LIMIT="$LIM1" SKIP_OFF=1 "K_NGRAM_GPU=5 11" K_DFLASH= ;;
    tT1) cell tT1 CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 "K_NGRAM_GPU=5 11" K_DFLASH= ;;
    tT0) cell tT0 CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 K_NGRAM_GPU= K_DFLASH= ;;
    tTn) cell tTn CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 SKIP_OFF=1 "K_NGRAM_GPU=5 11" K_DFLASH= ;;
    tT2) cell tT2 CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 SKIP_OFF=1 K_NGRAM_GPU= "K_DFLASH=7 15" ;;
    aT1) cell aT1 CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 SKIP_OFF=1 PERREQ=detailed "K_NGRAM_GPU=5 11" K_DFLASH= ;;
    aT2) cell aT2 CONC=1 LIMIT="$(cap 25)" THINK=on MAXTOK=2048 SKIP_OFF=1 PERREQ=detailed K_NGRAM_GPU= "K_DFLASH=7 15" ;;
    c[0-9]*)
      # c<N> all five arms; c<N>n off and n-gram k 5; c<N>d DFlash k 4 7 15 (28 Nov halves)
      n=${c#c}; half=""
      case "$n" in *n) half=n; n=${n%n} ;; *d) half=d; n=${n%d} ;; esac
      case "$n" in *[!0-9]*|"") echo "unknown cell $c"; STATUS+=("cell $c unknown"); FAILED=$((FAILED + 1)); continue ;; esac
      case "$half" in
        "") cell "$c" CONC="$n" LIMIT="$LIML" K_NGRAM_GPU=5 "K_DFLASH=4 7 15" ;;
        n)  cell "$c" CONC="$n" LIMIT="$LIML" K_NGRAM_GPU=5 K_DFLASH= ;;
        d)  cell "$c" CONC="$n" LIMIT="$LIML" SKIP_OFF=1 K_NGRAM_GPU= "K_DFLASH=4 7 15" ;;
      esac ;;
    *) echo "unknown cell $c"; STATUS+=("cell $c unknown"); FAILED=$((FAILED + 1)) ;;
  esac
done
echo
echo "==================== dtai3 piece $PIECE summary  $(date)"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo "gpu log: $GPULOG"
[ "$FAILED" -eq 0 ] || { echo "INCOMPLETE: $FAILED cell(s) failed. Read the cell logs above."; exit 1; }
echo "done."
