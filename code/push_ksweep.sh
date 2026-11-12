#!/bin/bash
# Sync the k-sweep outputs from your Delta results tree into the git clone and
# push. A variant of push_results.sh, which copies only results/a40 and
# results/a100 and is left untouched (RULES section 6).
#
# Run on the login node, never in a job.
#
#   bash push_ksweep.sh "A40 k sweep, jobs 2301xxxx 2301yyyy"
#
# Copies every results/<gpu>_ksweep* directory (results.jsonl and the
# per-request files), regenerates the k-sweep table beside the raw rows, and
# compresses the phase0-ksweep logs. Commits under whatever identity the clone
# is configured with; no attribution lines are added.

set -euo pipefail

ROOT=/projects/bikd/$USER/Speculative_Ladder
REPO=${REPO:-$ROOT/repo}
MSG=${1:-"results: k sweep outputs from Delta"}

if [ ! -d "$REPO/.git" ]; then
  echo "No clone at $REPO. See notes/github-from-delta.md for setup." >&2
  exit 1
fi

module load cray-python/3.12.12 >/dev/null 2>&1 || true

cd "$REPO"
git pull --rebase --autostash
mkdir -p results/logs

FOUND=0
for src in "$ROOT"/results/*_ksweep*; do
  [ -f "$src/results.jsonl" ] || continue
  FOUND=1
  name=$(basename "$src")
  mkdir -p "results/$name"
  cp "$src/results.jsonl" "results/$name/results.jsonl"
  cp "$src"/perreq_*.json "results/$name/" 2>/dev/null || true
done
if [ "$FOUND" = "0" ]; then
  echo "no results.jsonl under $ROOT/results/*_ksweep*" >&2
  exit 1
fi

# One table over every k-sweep directory, so the domain comparison is in it.
python3 code/ksweep_table.py results/*_ksweep* > results/ksweep_table.txt 2>&1 \
  || echo "ksweep_table.py failed; raw rows still committed"

for f in /projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-ksweep-*.out; do
  [ -e "$f" ] || continue
  # The log directory is shared; take only this user's jobs.
  [ "$(stat -c %U "$f" 2>/dev/null)" = "$USER" ] || continue
  gzip -c "$f" > "results/logs/$(basename "$f").gz"
done

git add -A results
if git diff --cached --quiet; then
  echo "nothing new to commit"
  exit 0
fi
git commit -m "$MSG"
git push
echo "pushed: $MSG"
