#!/bin/bash
# Plan, cost and (only with GO=1) submit the k sweep. Login node, no GPU.
#
#   cd /projects/bikd/$USER/Speculative_Ladder/repo/code
#
#   bash submit_ksweep.sh                       # the three all-domain jobs, printed only
#   bash submit_ksweep.sh GO=1                  # ... and submitted
#   bash submit_ksweep.sh BUCKETS=32K           # one bucket
#   bash submit_ksweep.sh DOMAINS=each          # one job per domain per bucket
#   bash submit_ksweep.sh DOMAINS="code_repository_understanding long_dialogue_history_understanding"
#   bash submit_ksweep.sh PART=gpuA100x4        # the A100 axis
#   bash submit_ksweep.sh CONCS_2K="1 8" CONCS_8K="1 4" CONCS_32K="1 2"   # add the load axis
#   bash submit_ksweep.sh RUNGS="ngram dflash" K_DFLASH="7 11 15" K_NGRAM="5 7"
#
# Arguments are KEY=VALUE and may also come from the environment. Everything
# ksweep.sh understands (PC, PERREQ, LOOP, MAXTOK, PLMIN, PLMAX, TEMP) passes
# straight through.
#
# One job per (bucket, domain, concurrency), so a failure at 32K does not cost
# 2K and 8K, and every ratio stays inside one job.
#
# Nothing is submitted without GO=1. RULES section 7: state the cost before
# running anything, say whose allocation, and call the figure a floor. The
# figure printed here is built from the filed A40 concurrency-1 off arms (jobs
# 22598608, 22685911, 22713173: 7.9, 9.3 and 17.2 seconds per prompt at 2K, 8K
# and 32K) plus an ASSUMED 3 minutes of model load per arm. Per-arm
# extrapolation has run about 2x low three times on this project, so read the
# total as a floor and budget double.

set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "${kv%%=*}"="${kv#*=}" ;;
    *) echo "arguments are KEY=VALUE, got '$kv'" >&2; exit 1 ;;
  esac
done

SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
MINE=/projects/bikd/$USER/Speculative_Ladder
HERE=$(cd "$(dirname "$0")" && pwd)

GO=${GO:-0}
PART=${PART:-gpuA40x4}
BUCKETS=${BUCKETS:-"2K 8K 32K"}
DOMAINS=${DOMAINS:-all}
CONCS_2K=${CONCS_2K:-1}
CONCS_8K=${CONCS_8K:-1}
CONCS_32K=${CONCS_32K:-1}
RUNGS=${RUNGS:-"ngram ngram_gpu dflash"}
K_NGRAM=${K_NGRAM:-"1 2 5 7 11"}
K_NGRAM_GPU=${K_NGRAM_GPU:-$K_NGRAM}
K_DFLASH=${K_DFLASH:-"1 2 4 7 11 15"}
LOAD_MIN=${LOAD_MIN:-3}
export RUNGS K_NGRAM K_NGRAM_GPU K_DFLASH
export LADDER_VENV=${LADDER_VENV:-$SHARED/vllm_env_multiarch}

case "$PART" in
  gpuA40*)  GPUTAG=a40;  RATE=0.5; EXCLUDE=${EXCLUDE:-gpub073} ;;
  gpuA100*) GPUTAG=a100; RATE=1.0; EXCLUDE=${EXCLUDE:-} ;;
  *) echo "PART=$PART: this build has kernels for A40 and A100 only (env/gpu-support.md)" >&2; exit 1 ;;
esac

for f in ksweep.sh run_benchmark.v4.py; do
  [ -f "$HERE/$f" ] || { echo "missing $HERE/$f" >&2; exit 1; }
done

# ------------------------------------------------------------ arms per job
count () { echo $#; }
has_rung () { case " $RUNGS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
ARMS=1
# shellcheck disable=SC2086
if has_rung ngram;     then ARMS=$((ARMS + 1 + $(count $K_NGRAM))); fi
# shellcheck disable=SC2086
if has_rung ngram_gpu; then ARMS=$((ARMS + $(count $K_NGRAM_GPU))); fi
# shellcheck disable=SC2086
if has_rung dflash;    then ARMS=$((ARMS + $(count $K_DFLASH))); fi

# ------------------------------------------------------------- domain list
DOMROOT=${DOMROOT:-$MINE/data/domains}
if [ "$DOMAINS" = "each" ]; then
  if [ ! -d "$DOMROOT" ]; then
    echo "no $DOMROOT. Run: python3 make_domains.py --check, then without --check" >&2
    exit 1
  fi
  DOMAINS=$( (cd "$DOMROOT" && ls -d */ 2>/dev/null || true) | tr -d '/' | tr '\n' ' ')
  [ -n "${DOMAINS// /}" ] || { echo "no domain directories under $DOMROOT" >&2; exit 1; }
fi

sec_per_prompt () { case "$1" in 2K) echo 7.9 ;; 8K) echo 9.3 ;; 32K) echo 17.2 ;; esac; }

echo "partition $PART ($GPUTAG), assumed charge factor $RATE per GPU hour"
echo "allocation: $USER under bikd-delta-gpu"
echo "arms per job: $ARMS  (rungs: $RUNGS)"
echo "  k ngram: $K_NGRAM | k ngram_gpu: $K_NGRAM_GPU | k dflash: $K_DFLASH"
echo

TOTAL_MIN=0
NJOBS=0
PLAN=$(mktemp)
trap 'rm -f "$PLAN"' EXIT

for B in $BUCKETS; do
  case "$B" in
    2K) CONCS=$CONCS_2K ;; 8K) CONCS=$CONCS_8K ;; 32K) CONCS=$CONCS_32K ;;
    *) echo "bucket '$B' is not 2K, 8K or 32K (case sensitive)" >&2; exit 1 ;;
  esac
  for D in $DOMAINS; do
    if [ "$D" = "all" ]; then
      FILE=$SHARED/ladder/data/longbench_v2/longbench_v2_${B}.jsonl
      case "$B" in 32K) N=${LIMIT:-25} ;; *) N=${LIMIT:-50} ;; esac
      OUT=$MINE/results/${GPUTAG}_ksweep
    else
      FILE=$DOMROOT/$D/longbench_v2_${B}.jsonl
      if [ ! -s "$FILE" ]; then
        echo "skip  $B $D: no file (make_domains.py skipped it or was not run)"
        continue
      fi
      N=${LIMIT:-$(grep -c . "$FILE")}
      OUT=$MINE/results/${GPUTAG}_ksweep_d_$D
    fi
    for C in $CONCS; do
      MIN=$(awk -v a="$ARMS" -v n="$N" -v s="$(sec_per_prompt "$B")" -v l="$LOAD_MIN" \
            'BEGIN { printf "%d", a * (l + n * s / 60) + 0.5 }')
      HRS=$(awk -v m="$MIN" 'BEGIN { h = int(m * 1.6 / 60) + 1; if (h < 1) h = 1; print h }')
      TOTAL_MIN=$((TOTAL_MIN + MIN))
      NJOBS=$((NJOBS + 1))
      printf "job   %-3s %-40s c=%-2s n=%-3s ~%3d min, walltime %d h  -> %s\n" \
             "$B" "$D" "$C" "$N" "$MIN" "$HRS" "$OUT"
      echo "$B|$D|$C|$N|$HRS|$OUT" >> "$PLAN"
    done
  done
done

echo
awk -v m="$TOTAL_MIN" -v r="$RATE" -v j="$NJOBS" 'BEGIN {
  printf "%d jobs, floor %.1f GPU hours wall, %.1f charged at factor %.1f. Budget double: %.1f charged.\n",
         j, m / 60, m / 60 * r, r, m / 60 * r * 2 }'
echo "Check what is left before submitting:  accounts"

if [ "$GO" != "1" ]; then
  echo
  echo "Nothing submitted. Re-run with GO=1 to submit, and log every job id in LOG.md."
  exit 0
fi

echo
cd "$HERE"
while IFS='|' read -r B D C N HRS OUT; do
  mkdir -p "$OUT"
  EX=()
  [ -n "$EXCLUDE" ] && EX=(--exclude="$EXCLUDE")
  # Variables go through the environment, not the --export list, so values
  # holding spaces (the k lists) survive intact.
  JID=$(BUCKET=$B DOMAIN=$D CONC=$C LIMIT=$N LADDER_OUT=$OUT \
        sbatch --parsable --partition="$PART" --time="$(printf '%02d' "$HRS"):00:00" \
               ${EX[@]+"${EX[@]}"} --export=ALL ksweep.sh)
  echo "submitted $JID  $B $D c=$C n=$N -> $OUT"
done < "$PLAN"

echo
echo "after they finish:"
echo "  sacct -j <jobid> --format=JobID,Partition,NodeList%12,State,Elapsed,ExitCode -X"
echo "  python3 ksweep_table.py $MINE/results/${GPUTAG}_ksweep*"
echo "  bash push_ksweep.sh \"k sweep, jobs ...\""
