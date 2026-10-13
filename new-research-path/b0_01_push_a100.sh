#!/usr/bin/env bash
# Block 0, step 1 -- push results/a100 + the logs that produced it.
#
# Why first: three separate review disagreements in the last two days traced to
# this data living only in Sahil's tree. Nothing downstream is checkable by a
# second reader until this lands.
#
# Safe by default: DRYRUN=1 prints what it would do and touches nothing.
#   bash b0_01_push_a100.sh              # inspect only
#   DRYRUN=0 bash b0_01_push_a100.sh     # copy, commit, push
#
# Reads Sahil's tree read-only. Never writes outside $REPO.

set -euo pipefail

REPO=${REPO:-/projects/bikd/dagraw2/Speculative_Ladder/repo}
SRC=${SRC:-/projects/bikd/ssashi/Speculative_Ladder/results/a100}
MYLOGS=${MYLOGS:-/projects/bikd/dagraw2/Speculative_Ladder/logs}
SAHILLOGS=${SAHILLOGS:-/projects/bikd/ssashi/ladder/logs}
DEST=$REPO/results/a100
LOGDEST=$REPO/results/logs/a100
DRYRUN=${DRYRUN:-1}

say() { printf '\n== %s\n' "$*"; }
run() { if [ "$DRYRUN" = "1" ]; then echo "  [dry] $*"; else eval "$@"; fi; }

say "0. preconditions"
[ -d "$REPO/.git" ] || { echo "ERROR: $REPO is not a git repo"; exit 1; }
[ -r "$SRC/results.jsonl" ] || { echo "ERROR: cannot read $SRC/results.jsonl"; exit 1; }
git -C "$REPO" rev-parse --abbrev-ref HEAD
git -C "$REPO" status --porcelain | head -20
echo "  (if the working tree is dirty above, stop and deal with that first)"

say "1. what is in the source tree"
ls -l "$SRC"
echo "--- total size ---"
du -sh "$SRC"
echo "--- per-request files present? ---"
PERREQ=$(find "$SRC" -maxdepth 1 -name 'perreq_*.json' | wc -l)
echo "  perreq_*.json count: $PERREQ"
if [ "$PERREQ" -eq 0 ]; then
  echo "  NOTE: no per-request files. The token_ids diff in step 3 will have"
  echo "        nothing to compare on the A100 side -- expected, not an error."
fi

say "2. which jobs produced these rows"
JOBS=$(python3 - "$SRC/results.jsonl" <<'PY'
import json,sys
js=[]
for l in open(sys.argv[1]):
    l=l.strip()
    if not l: continue
    j=(json.loads(l).get("env") or {}).get("slurm_job")
    if j and j not in js: js.append(j)
print(" ".join(js))
PY
)
echo "  jobs: $JOBS"
echo "  rows: $(wc -l < "$SRC/results.jsonl")"

say "3. logs to include (gzipped; they are ~100 KB each raw)"
FOUND=""
for J in $JOBS 22713396; do
  for D in "$MYLOGS" "$SAHILLOGS"; do
    F="$D/phase0-$J.out"
    if [ -r "$F" ]; then
      printf '  %-12s %s (%s)\n' "$J" "$F" "$(du -h "$F" | cut -f1)"
      FOUND="$FOUND $F"
      break
    fi
  done
done
[ -n "$FOUND" ] || echo "  WARNING: no logs found for any job"

say "4. copy"
run "mkdir -p '$DEST' '$LOGDEST'"
run "cp -v '$SRC/results.jsonl' '$DEST/results.jsonl'"
if [ "$PERREQ" -gt 0 ]; then
  run "cp -v $SRC/perreq_*.json '$DEST/'"
fi
for F in $FOUND; do
  B=$(basename "$F")
  run "gzip -c '$F' > '$LOGDEST/$B.gz'"
done

say "5. a README so a second reader knows what this tree is"
if [ "$DRYRUN" = "1" ]; then
  echo "  [dry] would write $DEST/README.md"
else
  cat > "$DEST/README.md" <<EOF
# results/a100

A100-SXM4-40GB rows, produced on Sahil's allocation and copied here on
$(date +%Y-%m-%d) so that every figure quoting an A100 number is checkable.

Source of truth at the time of copy: \`$SRC\`.
Jobs: $JOBS
Rows: $(wc -l < "$SRC/results.jsonl")
Per-request files: $PERREQ

Raw engine logs for these jobs are under \`results/logs/a100/\` (gzipped).
Job 22713396 is included deliberately: it is \`COMPLETED\` with exit code 0:0,
ran 28:42, and produced **zero rows** -- it predates the multi-architecture
rebuild and died on \`cudaErrorNoKernelImageForDevice\`. It is the clearest
example of this project's standing convention that COMPLETED does not mean
measured.

Known gaps, so nobody rediscovers them:
* \`per_request_timing\` is 0 on every row -- this vLLM build never populates
  engine request metrics. There is no per-request latency anywhere.
* \`prefix_caching\` reads \`null\` on every row, meaning the flag was not
  passed. The engine default was ON: the logs report prefix cache hit rates of
  4-8%. Cite the logs, not the rows, for caching state.
* Per-request filenames carry no job id, so where a cell was run twice the
  second run overwrote the first. Cross-node replication for those cells is
  aggregate-only.
EOF
  echo "  wrote $DEST/README.md"
fi

say "6. stage and show the diff"
run "git -C '$REPO' add results/a100 results/logs/a100"
if [ "$DRYRUN" = "1" ]; then
  echo "  [dry] would show: git -C $REPO status --short"
else
  git -C "$REPO" status --short
  echo "--- bytes being added ---"
  git -C "$REPO" diff --cached --stat | tail -5
fi

say "7. commit (authored per RULES section 1) and push"
MSG="results: add A100 grid and engine logs (jobs$(echo " $JOBS" | tr -s ' '))"
run "git -C '$REPO' -c user.name='Devansh Agrawal' -c user.email='dagraw2@illinois.edu' commit -m '$MSG'"
run "git -C '$REPO' push origin HEAD"

say "done"
echo "DRYRUN=$DRYRUN  (set DRYRUN=0 to actually copy, commit and push)"
