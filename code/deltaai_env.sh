#!/bin/bash
# DeltaAI site settings for the Ladder job scripts. SOURCE it, do not run it:
#
#   PROJ=<code> ACCOUNT=<name> source deltaai_env.sh
#
# PROJ is your four-letter DeltaAI project code and ACCOUNT the Slurm account;
# both are printed by `accounts`. Neither is guessed here.
#
# It exports the LADDER_* variables ksweep.sh and submit_ksweep.sh read, then
# checks that the venv, the two model snapshots and the three workload files
# are where the jobs will look, and prints the exact command for anything
# missing. It changes nothing on disk except creating empty directories under
# LADDER_ROOT.
#
# Nothing here points at the Delta venvs: those are x86_64 and cannot run on
# DeltaAI. Models and workloads are architecture-neutral, so the Delta copies
# are reused if this account can read them (/work/nvme is shared between the
# two systems; whether /projects is has not been confirmed).

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "source this file:  PROJ=<code> ACCOUNT=<name> source deltaai_env.sh"
  exit 1
fi
if [ -z "${PROJ:-}" ] || [ -z "${ACCOUNT:-}" ]; then
  echo "deltaai_env: set PROJ and ACCOUNT first (run: accounts)"
  return 1
fi

export PROJ ACCOUNT
export LADDER_ROOT=${LADDER_ROOT:-/work/nvme/$PROJ/$USER/Speculative_Ladder}
export LADDER_VENV=${LADDER_VENV:-$LADDER_ROOT/vllm_env_gh200}
export LADDER_CACHE=${LADDER_CACHE:-$LADDER_ROOT/.vllm_cache}
export LADDER_CAPS=${LADDER_CAPS:-90}
mkdir -p "$LADDER_ROOT"/{logs,results,data,hf/hub} "$LADDER_CACHE"

_QWEN=models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
_DFLASH=models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
_DELTA_HF=/work/nvme/bikd/dagraw2/Speculative_Ladder/hf/hub
_DELTA_DATA=/projects/bikd/dagraw2/Speculative_Ladder/ladder/data/longbench_v2

# Models: the Delta cache if readable, else our own.
if [ -z "${LADDER_HF:-}" ]; then
  if [ -r "$_DELTA_HF/$_QWEN/config.json" ] && [ -r "$_DELTA_HF/$_DFLASH/config.json" ]; then
    export LADDER_HF=$_DELTA_HF
  else
    export LADDER_HF=$LADDER_ROOT/hf/hub
  fi
fi
# Workloads: our copy if present, else the Delta files if readable.
if [ -z "${LADDER_DATA:-}" ]; then
  if [ -s "$LADDER_ROOT/data/longbench_v2/longbench_v2_2K.jsonl" ]; then
    export LADDER_DATA=$LADDER_ROOT/data/longbench_v2
  elif [ -r "$_DELTA_DATA/longbench_v2_2K.jsonl" ]; then
    export LADDER_DATA=$_DELTA_DATA
  else
    export LADDER_DATA=$LADDER_ROOT/data/longbench_v2
  fi
fi

echo "LADDER_ROOT  $LADDER_ROOT"
echo "LADDER_VENV  $LADDER_VENV"
echo "LADDER_HF    $LADDER_HF"
echo "LADDER_DATA  $LADDER_DATA"
echo "LADDER_CAPS  $LADDER_CAPS    account $ACCOUNT"
echo

_OK=1
if [ ! -x "$LADDER_VENV/bin/python3" ]; then
  _OK=0
  echo "MISSING venv. Build it (login node, tmux, no GPU hours):"
  echo "    PROJ=$PROJ bash build_deltaai.sh"
fi
if [ ! -r "$LADDER_HF/$_QWEN/config.json" ]; then
  _OK=0
  echo "MISSING Qwen3-8B at the pinned revision. With the venv active:"
  echo "    hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --cache-dir $LADDER_HF"
fi
if [ ! -r "$LADDER_HF/$_DFLASH/config.json" ]; then
  _OK=0
  echo "MISSING the DFlash head at the pinned revision. With the venv active:"
  echo "    hf download z-lab/Qwen3-8B-DFlash-b16 --revision 9b41424b7109f9c5413454f481b09a82b85333f4 --cache-dir $LADDER_HF"
fi
for _b in 2K 8K 32K; do
  if [ ! -s "$LADDER_DATA/longbench_v2_${_b}.jsonl" ]; then
    _OK=0
    echo "MISSING workload $LADDER_DATA/longbench_v2_${_b}.jsonl"
  fi
done
if [ ! -s "$LADDER_DATA/longbench_v2_2K.jsonl" ]; then
  echo "  Copy the three files, do NOT rebuild them: a rebuild is not guaranteed"
  echo "  to be byte-identical to what the filed cells read. On DELTA, where"
  echo "  /work/nvme is the same filesystem:"
  echo "    mkdir -p $LADDER_ROOT/data && cp -r $_DELTA_DATA $LADDER_ROOT/data/"
fi
if [ "$_OK" = "1" ]; then
  echo "all present. Smoke job (prints the plan; add GO=1 to submit):"
  echo "    bash submit_ksweep.sh PART=ghx4 BUCKETS=2K LIMIT=5 K_NGRAM=5 K_DFLASH=7"
fi
unset _OK _b _QWEN _DFLASH _DELTA_HF _DELTA_DATA
