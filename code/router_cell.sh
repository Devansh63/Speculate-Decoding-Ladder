#!/bin/bash
# One cell of the router data job set (dtai3): every arm of one workload file,
# one after another, each in its own process, into one results directory.
# A variant of ksweep.sh for harness v5d (RULES section 6: ksweep.sh is left as
# it is, so the dtai2 rows stay reproducible from the script that made them).
#
# Called by dtai3_job.sh inside its own allocation, never with sbatch.
#
# Differences from ksweep.sh:
#   - the workload file is $LADDER_DATA/<BUCKET>.jsonl, and BUCKET may be
#     nat, 2K, 8K, 16K, 32K or mix (the router files of build_router_workloads.py)
#   - harness v5d, chat format by default (PFORMAT=chat), natural stopping,
#     MAXTOK default 1024, MAXLEN (max model length) default 40960
#   - SKIP_OFF=1 drops the off arm: vLLM 0.33.0 refuses per-request
#     acceptance without a drafter, so an acceptance cell's off arm can only
#     fail (the dtai2 acc cells exited 1 for that reason alone)
#   - Delta and stable-runner branches removed: DeltaAI, Model Runner V2 only
#
# Knobs (environment):
#   BUCKET LIMIT          mandatory
#   CONC                  concurrency (max_num_seqs), default 1
#   PFORMAT THINK         chat|raw, off|on          default chat, off
#   MAXTOK MAXLEN         default 1024, 40960
#   PSTYLE                context|qa                default context
#   PERREQ                none|summary|detailed     default none
#   PC                    on|off (prefix caching)   default: engine default (on)
#   SKIP_OFF              1 to skip the off arm
#   K_NGRAM_GPU K_DFLASH  k lists; "" drops the rung
#   LADDER_OUT LADDER_DATA LADDER_VENV LADDER_HF LADDER_CACHE   mandatory

set -uo pipefail

BUCKET=${BUCKET:-}
CONC=${CONC:-1}
SEED=1234
MAXTOK=${MAXTOK:-1024}
MAXLEN=${MAXLEN:-40960}
PFORMAT=${PFORMAT:-chat}
THINK=${THINK:-off}
PSTYLE=${PSTYLE:-context}
K_NGRAM_GPU=${K_NGRAM_GPU-5 11}
K_DFLASH=${K_DFLASH-7 15}

case "$BUCKET" in
  nat|2K|8K|16K|32K|mix) ;;
  *) echo "ABORT: BUCKET='$BUCKET'. One of nat 2K 8K 16K 32K mix (case sensitive)."; exit 1 ;;
esac
case "$CONC" in *[!0-9]*|""|0) echo "ABORT: CONC='$CONC' must be a positive integer."; exit 1 ;; esac
case "${LIMIT:-}" in *[!0-9]*|""|0) echo "ABORT: LIMIT='${LIMIT:-}' must be a positive integer."; exit 1 ;; esac
case "$PFORMAT" in chat|raw) ;; *) echo "ABORT: PFORMAT='$PFORMAT'"; exit 1 ;; esac
case "$THINK" in off|on) ;; *) echo "ABORT: THINK='$THINK'"; exit 1 ;; esac
for K in $K_DFLASH; do
  [ "$K" -le 15 ] || { echo "ABORT: dflash k=$K. The b16 head drafts at most 15 tokens."; exit 1; }
done
for v in LADDER_OUT LADDER_DATA LADDER_VENV LADDER_HF LADDER_CACHE; do
  [ -n "${!v:-}" ] || { echo "ABORT: $v is not set."; exit 1; }
done

HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
RUN=${LADDER_RUN:-$HERE/run_benchmark.v5d.py}
grep -q 'HARNESS_VERSION = "v5d' "$RUN" 2>/dev/null || { echo "ABORT: $RUN is not harness v5d."; exit 1; }
DATASET=$LADDER_DATA/$BUCKET.jsonl
[ -s "$DATASET" ] || { echo "ABORT: no workload at $DATASET (run build_router_workloads.py)."; exit 1; }
HAVE=$(grep -c . "$DATASET")
if [ $((LIMIT + 2)) -gt "$HAVE" ]; then
  echo "ABORT: LIMIT=$LIMIT needs $((LIMIT + 2)) rows (two for the warm-up); $DATASET holds $HAVE."
  exit 1
fi

XARGS="--prompt-format $PFORMAT --prompt-style $PSTYLE --max-model-len $MAXLEN"
[ "$PFORMAT" = "chat" ] && XARGS="$XARGS --thinking $THINK"
[ -n "${PERREQ:-}" ] && [ "$PERREQ" != "none" ] && XARGS="$XARGS --per-request-spec $PERREQ"
case "${PC:-}" in
  "") ;;
  on) XARGS="$XARGS --prefix-caching" ;;
  off) XARGS="$XARGS --no-prefix-caching" ;;
  *) echo "ABORT: PC='$PC' must be on or off."; exit 1 ;;
esac

if [ -f "$LADDER_VENV/ladder_modules" ]; then
  while IFS= read -r M; do [ -n "$M" ] && module load "$M"; done < "$LADDER_VENV/ladder_modules"
else
  echo "ABORT: no $LADDER_VENV/ladder_modules (use the venv install_deltaai.sh built)."; exit 1
fi
[ -x "$LADDER_VENV/bin/python3" ] || { echo "ABORT: no usable virtualenv at $LADDER_VENV"; exit 1; }
source "$LADDER_VENV/bin/activate"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_CACHE_ROOT=$LADDER_CACHE
MODEL=${LADDER_MODEL:-$LADDER_HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
DRAFT=${LADDER_DRAFT:-$LADDER_HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
[ "$CAP" = "${LADDER_CAPS:-90}" ] || { echo "ABORT: compute capability $CAP, expected ${LADDER_CAPS:-90} (GH200)."; exit 1; }

mkdir -p "$LADDER_OUT" "$VLLM_CACHE_ROOT"
echo "harness: $RUN"
echo "bucket=$BUCKET conc=$CONC limit=$LIMIT maxtok=$MAXTOK maxlen=$MAXLEN format=$PFORMAT thinking=$THINK"
echo "k_ngram_gpu='$K_NGRAM_GPU' k_dflash='$K_DFLASH' skip_off=${SKIP_OFF:-0} extra='$XARGS'"
echo "dataset=$DATASET ($HAVE rows)"
echo "out=$LADDER_OUT"
date

ROWS_BEFORE=0
[ -f "$LADDER_OUT/results.jsonl" ] && ROWS_BEFORE=$(grep -c . "$LADDER_OUT/results.jsonl")
ARMS_RUN=0; ARMS_FAILED=0
run () {  # $1 rung, $2 k, $3 extra args
  ARMS_RUN=$((ARMS_RUN + 1))
  echo "### arm $ARMS_RUN rung=$1 k=$2 conc=$CONC bucket=$BUCKET  $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  python3 "$RUN" --dataset "$DATASET" --model "$MODEL" --rung "$1" --k "$2" --bucket "$BUCKET" \
    --limit "$LIMIT" --max-tokens "$MAXTOK" --seed "$SEED" --max-num-seqs "$CONC" \
    --outdir "$LADDER_OUT" $3 $XARGS
  if [ $? -ne 0 ]; then
    ARMS_FAILED=$((ARMS_FAILED + 1)); echo "FAILED rung=$1 k=$2 conc=$CONC bucket=$BUCKET"
  fi
}
[ "${SKIP_OFF:-0}" = "1" ] || run off 0 ""
for K in $K_NGRAM_GPU; do run ngram_gpu "$K" ""; done
for K in $K_DFLASH; do run dflash "$K" "--draft $DRAFT"; done
date
ROWS_AFTER=0
[ -f "$LADDER_OUT/results.jsonl" ] && ROWS_AFTER=$(grep -c . "$LADDER_OUT/results.jsonl")
GREW=$((ROWS_AFTER - ROWS_BEFORE))
echo "arms run: $ARMS_RUN   failed: $ARMS_FAILED   rows added: $GREW"
if [ "$ARMS_RUN" -eq 0 ] || [ "$ARMS_FAILED" -ne 0 ] || [ "$GREW" -ne "$ARMS_RUN" ]; then
  echo "INCOMPLETE: expected $ARMS_RUN new rows, found $GREW. Read the log before using this cell."
  exit 1
fi
echo "done."
