#!/usr/bin/env bash
# Block 0 runner. Run ONE stage at a time:
#
#   bash run_block0.sh check        # verify the scripts arrived intact
#   bash run_block0.sh analyse      # steps 2, 3, 4 -- READ ONLY, writes no repo files
#   bash run_block0.sh push-dry     # step 1, dry run: shows what it would copy
#   bash run_block0.sh push         # step 1 for real: copies, commits, pushes a100 data
#   bash run_block0.sh attribution  # step 5 audit (read only)
#   bash run_block0.sh attribution-write   # step 5, writes + commits the exception note
#
# Deliberately NOT a single "do everything" target. Steps 2, 3 and 4 produce
# output that a person has to read and act on; step 1 changes the repo. Chaining
# them would hide the decisions.
#
# Outputs land in $OUTDIR (default: ./block0-out) so they can be pasted back.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/projects/bikd/dagraw2/Speculative_Ladder/repo}
A40=${A40:-/projects/bikd/dagraw2/Speculative_Ladder/results/a40}
A100=${A100:-/projects/bikd/ssashi/Speculative_Ladder/results/a100}
LOGS=${LOGS:-/projects/bikd/dagraw2/Speculative_Ladder/logs}
OUTDIR=${OUTDIR:-$HERE/block0-out}
HARNESS=${HARNESS:-v2}

mkdir -p "$OUTDIR"
hr() { printf '\n%s\n%s\n' "────────────────────────────────────────────────────────" "  $*"; }

stage_check() {
  hr "verify the scripts arrived intact"
  if [ ! -f "$HERE/MANIFEST.md5" ]; then
    echo "  no MANIFEST.md5 next to this script -- cannot verify"; return 1
  fi
  cd "$HERE"
  if command -v md5sum >/dev/null; then
    md5sum -c MANIFEST.md5
  else
    # macOS / some HPC images ship md5 instead of md5sum
    while read -r want name; do
      got=$(md5 -q "$name" 2>/dev/null || openssl md5 -r "$name" | cut -d' ' -f1)
      [ "$got" = "$want" ] && echo "$name: OK" || echo "$name: FAILED (got $got)"
    done < MANIFEST.md5
  fi
  echo
  echo "  Any FAILED line means the file was corrupted in transit."
  echo "  Do not run it -- re-fetch instead."
  hr "python and paths"
  python3 -VV
  for P in "$REPO" "$A40" "$A100" "$LOGS"; do
    if [ -r "$P" ]; then echo "  OK      $P"; else echo "  MISSING $P"; fi
  done
}

stage_analyse() {
  hr "step 2 -- decomposition and the frozen-prediction scoreboard"
  echo "  (read only: reads results.jsonl, writes only to $OUTDIR)"

  python3 "$HERE/decompose.py" "$A40" --compare "$A100" \
      --fit-k --harness "$HARNESS" --exclude-host gpub073 \
      > "$OUTDIR/02a-decompose.txt" 2>&1
  echo "  -> $OUTDIR/02a-decompose.txt"

  python3 "$HERE/b0_02_score_prediction.py" "$A100" --harness "$HARNESS" \
      > "$OUTDIR/02b-prediction-entry.md" 2> "$OUTDIR/02b-prediction-entry.err"
  echo "  -> $OUTDIR/02b-prediction-entry.md   (the claims.md entry -- READ IT)"
  [ -s "$OUTDIR/02b-prediction-entry.err" ] && \
    echo "  -> $OUTDIR/02b-prediction-entry.err  (warnings, check these)"

  hr "step 3 -- the 32K n-gram tau gap, via token_ids"
  python3 "$HERE/b0_03_tau_diff.py" "$A40" "$A100" \
      --pattern 'perreq_ngram_32K_k5_*_c1.json' \
      > "$OUTDIR/03-tau-diff.txt" 2>&1
  echo "  -> $OUTDIR/03-tau-diff.txt (exit $?)"
  echo "     If it says tree B has no per-request files, that IS the result:"
  echo "     the tau gap stays open and 1.390 must not appear in a table."

  hr "step 4 -- KV pool and the concurrency ceiling, from the engine logs"
  # shellcheck disable=SC2086
  python3 "$HERE/b0_04_kv_extract.py" $LOGS/phase0-*.out \
      > "$OUTDIR/04-kv.txt" 2>&1
  echo "  -> $OUTDIR/04-kv.txt"
  echo "     If it reports NOT SEPARABLE, the 2.4 GiB figure comes out of C19"
  echo "     entirely -- keep the direction and the maxconc column only."

  hr "done -- nothing was written to the repo"
  echo "  Paste these three back for the write-ups:"
  ls -1 "$OUTDIR"
  echo
  echo "  claims.md is append-only, so do NOT append 02b until the wording is settled."
}

stage_push_dry() {
  hr "step 1 (DRY) -- what would be copied into the repo"
  DRYRUN=1 bash "$HERE/b0_01_push_a100.sh"
}

stage_push() {
  hr "step 1 (REAL) -- copy, commit and push the A100 data"
  echo "  Did you read the dry run above? Ctrl-C now if not. 5 seconds."
  sleep 5
  DRYRUN=0 bash "$HERE/b0_01_push_a100.sh"
}

stage_attribution()       { hr "step 5 -- audit only";        bash "$HERE/b0_05_attribution_audit.sh" | tee "$OUTDIR/05-attribution.txt"; }
stage_attribution_write() { hr "step 5 -- audit and write";   WRITE=1 bash "$HERE/b0_05_attribution_audit.sh" | tee "$OUTDIR/05-attribution.txt"; }

case "${1:-}" in
  check)              stage_check ;;
  analyse|analyze)    stage_analyse ;;
  push-dry)           stage_push_dry ;;
  push)               stage_push ;;
  attribution)        stage_attribution ;;
  attribution-write)  stage_attribution_write ;;
  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1 ;;
esac
