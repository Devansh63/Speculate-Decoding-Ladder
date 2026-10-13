#!/usr/bin/env python3
"""
Block 1, step 3 -- parameterise phase0.sh.

phase0.sh currently hardcodes --max-tokens 256 (line 143) and the three arms
with fixed k (lines 150-152). Three queued experiments need those to move:

  * the prefill/decode split needs MAXTOK=1 and MAXTOK=512 on the same cell;
  * the C10 control and the v3 no-op proof need the SAME arm run twice into
    one directory;
  * the n-gram k question needs ARMS="ngram:1 ngram:2 ngram:5 ngram:7".

Every default below reproduces the current script exactly: MAXTOK=256,
ARMS="off:0 ngram:5 dflash:7" in that order, REPEAT=1, no extra flags. A job
submitted the old way produces the same runs in the same order.

Two guards, because both of these would otherwise waste a reservation:

  * REPEAT>1 against the v2 harness is refused. v2 names per-request files
    perreq_<run_id>.json with no job id, so the second run silently overwrites
    the first and the repeat buys nothing. v3 adds _j<jobid>; the check looks
    for "v3" in the harness path and greps the file for the version string.
  * Any v3-only flag (TEMP, PLMIN, PLMAX, LOOP) against the v2 harness is
    refused up front. argparse would reject the unknown flag per arm, and the
    "|| echo FAILED" in run() would swallow it as a failed arm -- so the job
    would burn its full walltime producing nothing, which is exactly how
    22713396 spent 28 minutes.

Usage
-----
  python3 b1_03_patch_phase0.py /path/to/ladder/scripts/phase0.sh
  python3 b1_03_patch_phase0.py ... --write

Then, for the next job (the C10 control and the v3 no-op proof in one):

  sbatch --export=ALL,BUCKET=2K,LIMIT=50,CONC=1,ARMS="off:0",REPEAT=2,\\
LADDER_RUN=/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts/run_benchmark.v3.py,\\
LADDER_OUT=/projects/bikd/dagraw2/Speculative_Ladder/results/a40_c10 phase0.sh
"""

import argparse
import difflib
import os
import sys

EDITS = [

    ("1. new knobs, with guards",
     '''BUCKET=${BUCKET:-2K}
LIMIT=${LIMIT:-50}
SEED=${SEED:-1234}
CONC=${CONC:-"1 8 32"}''',
     '''BUCKET=${BUCKET:-2K}
LIMIT=${LIMIT:-50}
SEED=${SEED:-1234}
CONC=${CONC:-"1 8 32"}

# ---------------------------------------------------------------- v3 knobs
# Every default here reproduces the pre-v3 script exactly.
MAXTOK=${MAXTOK:-256}                      # was hardcoded 256
ARMS=${ARMS:-"off:0 ngram:5 dflash:7"}     # rung:k, in order; off first
REPEAT=${REPEAT:-1}                        # runs of each arm, for controls

# Optional, v3 harness only. Unset means "do not pass the flag at all", so
# the v2 harness never sees an argument it does not know.
XARGS=""
[ -n "${TEMP:-}" ]  && XARGS="$XARGS --temperature $TEMP"
[ -n "${PLMIN:-}" ] && XARGS="$XARGS --prompt-lookup-min $PLMIN"
[ -n "${PLMAX:-}" ] && XARGS="$XARGS --prompt-lookup-max $PLMAX"
[ -n "${LOOP:-}" ]  && XARGS="$XARGS --loop-timing"''' ),

    ("2. refuse v3-only requests against a v2 harness",
     '''MODEL=${LADDER_MODEL:-$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}''',
     '''# A v3-only request against the v2 harness must fail HERE, not per arm.
# run() ends in "|| echo FAILED", so an argparse rejection would be swallowed
# and the job would hold a GPU for its full walltime producing nothing.
HARNESS_IS_V3=0
if grep -q 'HARNESS_VERSION = "v3' "$RUN" 2>/dev/null; then HARNESS_IS_V3=1; fi
echo "harness: $RUN (v3=$HARNESS_IS_V3)"

if [ "$HARNESS_IS_V3" != "1" ]; then
  if [ -n "$XARGS" ]; then
    echo "ABORT: TEMP/PLMIN/PLMAX/LOOP need the v3 harness, but $RUN is not v3."
    echo "       Point LADDER_RUN at run_benchmark.v3.py, or unset them."
    exit 1
  fi
  if [ "$REPEAT" -gt 1 ]; then
    echo "ABORT: REPEAT=$REPEAT needs the v3 harness. v2 writes"
    echo "       perreq_<run_id>.json with no job id, so run 2 would silently"
    echo "       overwrite run 1 and the repeat would buy nothing."
    echo "       Point LADDER_RUN at run_benchmark.v3.py."
    exit 1
  fi
fi

MODEL=${LADDER_MODEL:-$HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}'''),

    ("3. run() takes max-tokens and the extra flags",
     '''run () {  # $1 rung, $2 k, $3 extra args, $4 concurrency
  echo "### rung=$1 k=$2 conc=$4 bucket=$BUCKET gpu=$GPUTAG  $(date +%H:%M:%S)"
  python3 "$RUN" \\
    --dataset "$DATA/longbench_v2_${BUCKET}.jsonl" \\
    --model "$MODEL" \\
    --rung "$1" --k "$2" --bucket "$BUCKET" \\
    --limit "$LIMIT" --max-tokens 256 --seed "$SEED" \\
    --max-num-seqs "$4" \\
    --outdir "$OUT" $3 \\
    || echo "FAILED rung=$1 k=$2 conc=$4 bucket=$BUCKET gpu=$GPUTAG"
}''',
     '''run () {  # $1 rung, $2 k, $3 extra args, $4 concurrency, $5 repeat index
  echo "### rung=$1 k=$2 conc=$4 rep=$5/$REPEAT maxtok=$MAXTOK " \\
       "bucket=$BUCKET gpu=$GPUTAG  $(date +%H:%M:%S)"
  python3 "$RUN" \\
    --dataset "$DATA/longbench_v2_${BUCKET}.jsonl" \\
    --model "$MODEL" \\
    --rung "$1" --k "$2" --bucket "$BUCKET" \\
    --limit "$LIMIT" --max-tokens "$MAXTOK" --seed "$SEED" \\
    --max-num-seqs "$4" \\
    --outdir "$OUT" $3 $XARGS \\
    || echo "FAILED rung=$1 k=$2 conc=$4 rep=$5 bucket=$BUCKET gpu=$GPUTAG"
}'''),

    ("4. drive the loop from ARMS and REPEAT",
     '''for C in $CONC; do
  run off    0  ""                 $C
  run ngram  5  ""                 $C
  run dflash 7  "--draft $DRAFT"   $C
done''',
     '''# ARMS is "rung:k" pairs in order. The default is the old three arms with the
# old k values, so an existing submission is unchanged. --draft is added for
# dflash automatically rather than being spelled out per call.
for C in $CONC; do
  for ARM in $ARMS; do
    ARM_RUNG=${ARM%%:*}
    ARM_K=${ARM##*:}
    ARM_EXTRA=""
    [ "$ARM_RUNG" = "dflash" ] && ARM_EXTRA="--draft $DRAFT"
    REP=1
    while [ "$REP" -le "$REPEAT" ]; do
      run "$ARM_RUNG" "$ARM_K" "$ARM_EXTRA" "$C" "$REP"
      REP=$((REP + 1))
    done
  done
done'''),

    ("5. usage examples for the queued experiments",
     '''# A100, override the partition:
#   sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2K,LIMIT=50 phase0.sh''',
     '''# A100, override the partition:
#   sbatch --partition=gpuA100x4 --export=ALL,BUCKET=2K,LIMIT=50 phase0.sh
#
# ------------------------------------------------------ the v3 knobs, by use
#
# V3=/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts/run_benchmark.v3.py
#
# C10 control AND the v3 no-op proof, one job (~15 min). Two identical off
# runs into one directory; the v3 job id in the file name keeps both, and
# check_lossless.v3.py pairs them as SAME-CONFIG:
#   sbatch --export=ALL,BUCKET=2K,LIMIT=50,CONC=1,ARMS="off:0",REPEAT=2,\\
#     LADDER_RUN=$V3,LADDER_OUT=$SHARED/results/a40_c10 phase0.sh
#
# Prefill/decode split. MAXTOK=1 is prefill plus one step; subtract from the
# 256-token run for the marginal per-token cost. NOTE the intercept is
# arm-dependent -- a 1-token speculative run still pays one draft-and-verify
# cycle -- so only the DIFFERENCE is comparable across arms:
#   sbatch --export=ALL,BUCKET=8K,LIMIT=50,CONC=1,MAXTOK=1 phase0.sh
#
# n-gram k sweep, on a clean node (the existing sweep ran on gpub073, which is
# on the exclusion list):
#   sbatch --exclude=gpub073 --export=ALL,BUCKET=32K,LIMIT=25,CONC=1,\\
#     ARMS="off:0 ngram:1 ngram:2 ngram:5 ngram:7" phase0.sh
#
# Coverage knob and temperature, v3 harness only:
#   sbatch --export=ALL,ARMS="off:0 ngram:5",PLMAX=7,LADDER_RUN=$V3 phase0.sh
#   sbatch --export=ALL,TEMP=0.7,LADDER_RUN=$V3 phase0.sh'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to phase0.sh")
    ap.add_argument("--write", action="store_true",
                    help="write phase0.v3.sh beside the original")
    a = ap.parse_args()

    src = open(a.path).read()
    if "ARMS=${ARMS:-" in src:
        sys.exit(f"{a.path} already looks patched (ARMS is present) -- stopping "
                 f"rather than double-applying.")

    text, problems = src, []
    for desc, anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            problems.append(f"  {desc}: anchor found {n} times, expected 1")
            continue
        text = text.replace(anchor, repl, 1)

    print(f"patching {a.path}  ({len(src.splitlines())} lines)")
    if problems:
        print("\nANCHOR FAILURES -- nothing written:")
        print("\n".join(problems))
        sys.exit(2)
    print(f"all {len(EDITS)} anchors matched exactly once")

    # Write BEFORE printing the diff. If stdout is piped into head, the pipe
    # closes mid-diff and anything after it never runs -- which silently
    # skipped the write three times before this was reordered.
    if a.write:
        dest = os.path.join(os.path.dirname(os.path.abspath(a.path)),
                            "phase0.v3.sh")
        open(dest, "w").write(text)
        os.chmod(dest, 0o755)
        print(f"wrote {dest}")
        print(f"Check it parses before submitting anything:  bash -n {dest}")
        print("Defaults are unchanged, so an old-style submission behaves "
              "identically.")
    else:
        print("(--write to produce phase0.v3.sh; nothing written)")

    print(f"{len(src.splitlines())} -> {len(text.splitlines())} lines; "
          f"diff follows")
    sys.stdout.writelines(difflib.unified_diff(
        src.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile="phase0.sh", tofile="phase0.v3.sh", n=3))


if __name__ == "__main__":
    main()
