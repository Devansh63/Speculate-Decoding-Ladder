#!/bin/bash
#SBATCH --job-name=ksweep
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-ksweep-%j.out

# The k sweep: every rung at every draft length, one bucket, one domain, one
# concurrency per job. A variant of phase0.v3.sh, which is untouched (RULES
# section 6). The log name keeps the phase0 prefix so push_results.sh's
# existing glob picks it up.
#
# The question it answers is the TA's: does adaptive k help at all? It does
# not if the best k is the same in every cell. The filed A40 sweeps cannot
# say: DFlash has k = 1, 2, 7 at 2K and 8K and 1, 2, 4, 7 at 32K, the best is
# the largest k tested in every bucket, and 11 and 15 (the head's native
# block) were never run at concurrency 1. n-gram has k = 5 only in the repo.
#
# Do not call sbatch on this file by hand for more than one cell. Use
#   bash submit_ksweep.sh
# on the login node. It prints every job and a cost floor and submits nothing
# until GO=1.
#
# ------------------------------------------------------------ what one job is
#
# Arms, in this order, all in ONE job so every ratio is within-job:
#
#   off                       baseline, async scheduling at the engine default
#   off, async off            the matched baseline for CPU n-gram (see below)
#   ngram      k in K_NGRAM       CPU prompt lookup; the engine runs it sync
#   ngram_gpu  k in K_NGRAM_GPU   GPU prompt lookup; allowed to run async
#   dflash     k in K_DFLASH      the trained head
#
# Why two baselines. At the pin, async scheduling defaults ON for off and
# dflash and OFF with a warning for CPU ngram (vllm/config/vllm.py 1456 to
# 1505 at bf13ecc2f). Every filed n-gram speedup is therefore a ratio across
# scheduling modes. Here each arm is compared only with the off arm that
# resolved to the same state; ksweep_table.py refuses any other ratio.
#
# ------------------------------------------------------------------- knobs
#
#   BUCKET        2K | 8K | 32K                              (case sensitive)
#   CONC          one concurrency per job                    default 1
#   DOMAIN        all, or a slug written by make_domains.py  default all
#   RUNGS         subset of "ngram ngram_gpu dflash"         default all three
#   K_NGRAM       default "1 2 5 7 11"
#   K_NGRAM_GPU   default: same as K_NGRAM
#   K_DFLASH      default "1 2 4 7 11 15"
#   LIMIT         default 50 / 50 / 25 for DOMAIN=all (RULES section 6);
#                 for a domain, the line count of its file
#   PC            on | off   prefix caching for EVERY arm    default: engine default
#   PERREQ        none | summary | detailed  per-prompt acceptance   default none
#   LOOP          1 = one prompt at a time with per-prompt latency (CONC=1 only;
#                 a separate population, never pooled with batched rows)
#   MAXTOK, SEED, TEMP, PLMIN, PLMAX   as in phase0.v3.sh
#
# Mandatory, as on every job (RULES section 4): LADDER_VENV and LADDER_OUT.
# LADDER_OUT is used as given; submit_ksweep.sh sets one directory per domain.
#
# n-gram GPU asserts that the proposal length equals num_speculative_tokens.
# That is satisfied here because k is static per arm. If an ngram_gpu arm
# fails, the job continues and the failure is counted at the end.

set -uo pipefail

BUCKET=${BUCKET:-2K}
CONC=${CONC:-1}
DOMAIN=${DOMAIN:-all}
SEED=${SEED:-1234}
MAXTOK=${MAXTOK:-256}
# LADDER_MRV2=1: vLLM 0.33.0 on Model Runner V2 with harness v5 (DeltaAI).
# There `ngram` and `ngram_gpu` are the same GPU speculator and every arm runs
# async, so the default drops CPU n-gram and with it the sync off baseline.
MRV2=${LADDER_MRV2:-0}
if [ "$MRV2" = "1" ]; then
  RUNGS=${RUNGS:-"ngram_gpu dflash"}
else
  RUNGS=${RUNGS:-"ngram ngram_gpu dflash"}
fi
K_NGRAM=${K_NGRAM:-"1 2 5 7 11"}
K_NGRAM_GPU=${K_NGRAM_GPU:-$K_NGRAM}
K_DFLASH=${K_DFLASH:-"1 2 4 7 11 15"}

case "$BUCKET" in
  2K|8K|32K) ;;
  *) echo "ABORT: BUCKET='$BUCKET'. It is case sensitive: 2K, 8K or 32K."; exit 1 ;;
esac
case "$CONC" in
  *[!0-9]*|"") echo "ABORT: CONC='$CONC' must be one integer per job."; exit 1 ;;
esac
for K in $K_DFLASH; do
  if [ "$K" -gt 15 ]; then
    echo "ABORT: dflash k=$K. The b16 head drafts at most 15 tokens."; exit 1
  fi
done

if [ -z "${LADDER_VENV:-}" ] || [ -z "${LADDER_OUT:-}" ]; then
  echo "ABORT: LADDER_VENV and LADDER_OUT are mandatory (RULES section 4)."
  exit 1
fi

SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
HF=${LADDER_HF:-/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub}
VENV=$LADDER_VENV
OUT=$LADDER_OUT

# The harness comes from the directory the job was submitted from (your clone
# of the repository), because the shared scripts directory is not writable by
# teammates and does not hold v4.
if [ "$MRV2" = "1" ]; then HV=v5; else HV=v4; fi
RUN=${LADDER_RUN:-${SLURM_SUBMIT_DIR:-$(pwd)}/run_benchmark.$HV.py}
if ! grep -q "HARNESS_VERSION = \"$HV" "$RUN" 2>/dev/null; then
  echo "ABORT: $RUN is not harness $HV (LADDER_MRV2=$MRV2). Submit from the"
  echo "       repository's code/ directory or set LADDER_RUN."
  exit 1
fi

if [ "$DOMAIN" = "all" ]; then
  DATA=${LADDER_DATA:-$SHARED/ladder/data/longbench_v2}
  DOMARG=""
else
  DATA=${LADDER_DATA:-${LADDER_DOMROOT:-/projects/bikd/$USER/Speculative_Ladder/data/domains}/$DOMAIN}
  DOMARG="--domain $DOMAIN"
fi
DATASET=$DATA/longbench_v2_${BUCKET}.jsonl
if [ ! -s "$DATASET" ]; then
  echo "ABORT: no workload at $DATASET"
  [ "$DOMAIN" != "all" ] && echo "       run make_domains.py on the login node first"
  exit 1
fi

if [ -z "${LIMIT:-}" ]; then
  if [ "$DOMAIN" = "all" ]; then
    case "$BUCKET" in 32K) LIMIT=25 ;; *) LIMIT=50 ;; esac
  else
    LIMIT=$(grep -c . "$DATASET")
  fi
fi
HAVE=$(grep -c . "$DATASET")
if [ "$LIMIT" -gt "$HAVE" ]; then
  echo "ABORT: LIMIT=$LIMIT but $DATASET holds $HAVE prompts. The harness"
  echo "       would run $HAVE and label the row n$LIMIT."
  exit 1
fi

XARGS="$DOMARG"
[ -n "${TEMP:-}" ]  && XARGS="$XARGS --temperature $TEMP"
[ -n "${PLMIN:-}" ] && XARGS="$XARGS --prompt-lookup-min $PLMIN"
[ -n "${PLMAX:-}" ] && XARGS="$XARGS --prompt-lookup-max $PLMAX"
[ -n "${PERREQ:-}" ] && [ "$PERREQ" != "none" ] && XARGS="$XARGS --per-request-spec $PERREQ"
case "${PC:-}" in
  "")  ;;
  on)  XARGS="$XARGS --prefix-caching" ;;
  off) XARGS="$XARGS --no-prefix-caching" ;;
  *)   echo "ABORT: PC='$PC' must be on or off."; exit 1 ;;
esac
if [ -n "${LOOP:-}" ]; then
  if [ "$CONC" != "1" ]; then
    echo "ABORT: LOOP needs CONC=1."; exit 1
  fi
  XARGS="$XARGS --loop-timing"
fi

# A venv built by build_deltaai.sh records the modules it was built under.
# Otherwise this is Delta, where the venvs sit on cray-python 3.12.12.
if [ -f "$VENV/ladder_modules" ]; then
  while IFS= read -r M; do [ -n "$M" ] && module load "$M"; done < "$VENV/ladder_modules"
else
  module load cray-python/3.12.12
fi
if [ ! -x "$VENV/bin/python3" ]; then
  echo "ABORT: no usable virtualenv at $VENV"; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export VLLM_USE_V2_MODEL_RUNNER=$MRV2
export VLLM_CACHE_ROOT=${LADDER_CACHE:-$SHARED/.vllm_cache}

MODEL=${LADDER_MODEL:-$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
DRAFT=${LADDER_DRAFT:-$HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}

GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$GPUNAME" in
  *A40*)  GPUTAG=a40 ;;
  *A100*) GPUTAG=a100 ;;
  *GH200*) GPUTAG=gh200 ;;
  *H200*) GPUTAG=h200 ;;
  *H100*) GPUTAG=h100 ;;
  *)      GPUTAG=$(echo "$GPUNAME" | tr -c 'A-Za-z0-9' '_' | tr 'A-Z' 'a-z') ;;
esac
# Which compute capabilities the venv in use was compiled for. The Delta
# multiarch venv is sm_80 and sm_86 as far as anything has been run; the
# DeltaAI venv is sm_90 only. deltaai_env.sh sets LADDER_CAPS=90.
CAPS_OK=${LADDER_CAPS:-"80 86"}
case " $CAPS_OK " in
  *" $CAP "*) echo "compute capability $CAP is covered by this build ($CAPS_OK)" ;;
  *) echo "ABORT: compute capability $CAP is not in LADDER_CAPS='$CAPS_OK'."
     echo "       See env/gpu-support.md and env/deltaai.md."
     exit 1 ;;
esac

mkdir -p "$OUT" "$VLLM_CACHE_ROOT"

echo "venv: $VENV"
echo "harness: $RUN (model runner V2: $MRV2)"
echo "gpu='$GPUNAME' cap=$CAP tag=$GPUTAG host=$(hostname)"
echo "bucket=$BUCKET domain=$DOMAIN conc=$CONC limit=$LIMIT maxtok=$MAXTOK seed=$SEED job=${SLURM_JOB_ID:-none}"
echo "rungs='$RUNGS' k_ngram='$K_NGRAM' k_ngram_gpu='$K_NGRAM_GPU' k_dflash='$K_DFLASH'"
echo "extra='$XARGS'"
echo "dataset=$DATASET"
echo "out=$OUT"
date

ROWS_BEFORE=0
[ -f "$OUT/results.jsonl" ] && ROWS_BEFORE=$(grep -c . "$OUT/results.jsonl")
ARMS_RUN=0
ARMS_FAILED=0

run () {  # $1 rung, $2 k, $3 extra args
  ARMS_RUN=$((ARMS_RUN + 1))
  echo "### arm $ARMS_RUN rung=$1 k=$2 conc=$CONC bucket=$BUCKET domain=$DOMAIN" \
       "gpu=$GPUTAG extra='$3'  $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  python3 "$RUN" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rung "$1" --k "$2" --bucket "$BUCKET" \
    --limit "$LIMIT" --max-tokens "$MAXTOK" --seed "$SEED" \
    --max-num-seqs "$CONC" \
    --outdir "$OUT" $3 $XARGS
  if [ $? -ne 0 ]; then
    ARMS_FAILED=$((ARMS_FAILED + 1))
    echo "FAILED rung=$1 k=$2 conc=$CONC bucket=$BUCKET domain=$DOMAIN gpu=$GPUTAG"
  fi
}

has_rung () { case " $RUNGS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Baselines first. Without the off arm there is no speedup; without the
# sync off arm there is no fair one for CPU n-gram.
run off 0 ""
if has_rung ngram && [ "$MRV2" != "1" ]; then
  run off 0 "--async-scheduling off"
fi

if has_rung ngram; then
  for K in $K_NGRAM; do run ngram "$K" ""; done
fi
if has_rung ngram_gpu; then
  for K in $K_NGRAM_GPU; do run ngram_gpu "$K" ""; done
fi
if has_rung dflash; then
  for K in $K_DFLASH; do run dflash "$K" "--draft $DRAFT"; done
fi

date
ROWS_AFTER=0
[ -f "$OUT/results.jsonl" ] && ROWS_AFTER=$(grep -c . "$OUT/results.jsonl")
GREW=$((ROWS_AFTER - ROWS_BEFORE))
echo
echo "arms run: $ARMS_RUN   failed: $ARMS_FAILED   rows added to results.jsonl: $GREW"
echo "next, on the login node:"
echo "  python3 ksweep_table.py $OUT"

# COMPLETED must mean the measurements exist (RULES section 6). A job that
# lost an arm exits non-zero so sacct shows FAILED instead.
if [ "$ARMS_FAILED" -ne 0 ] || [ "$GREW" -ne "$ARMS_RUN" ]; then
  echo "INCOMPLETE: expected $ARMS_RUN new rows, found $GREW. Read the log before using this cell."
  exit 1
fi
echo "done."
