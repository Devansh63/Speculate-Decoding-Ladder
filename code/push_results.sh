#!/bin/bash
# Sync measurement outputs from the live Delta run tree into the git clone and
# push, so every result is in the repo before it can be lost or confused with
# a later run.
#
# Run on the login node, never in a job.
#
#   bash push_results.sh "A40 2K concurrency sweep, job 22598608"
#
# First time setup, once per account, see notes/github-from-delta.md.

set -euo pipefail

ROOT=/projects/bikd/$USER/Speculative_Ladder
REPO=${REPO:-$ROOT/repo}
MSG=${1:-"Sync measurement results from Delta"}

if [ ! -d "$REPO/.git" ]; then
  echo "No clone at $REPO. See notes/github-from-delta.md for setup." >&2
  exit 1
fi

module load cray-python/3.12.12 >/dev/null 2>&1 || true

cd "$REPO"
git pull --rebase --autostash

mkdir -p results/a40 results/a100 results/logs notes

for g in a40 a100; do
  src=$ROOT/results/$g
  [ -d "$src" ] || continue
  [ -f "$src/results.jsonl" ] && cp "$src/results.jsonl" "results/$g/results.jsonl"
  # Per request token id hashes. These are what prove the speculative arms
  # produced byte identical output to the no speculation arm, which is the
  # losslessness claim, so they are part of the result, not a by-product.
  cp "$src"/perreq_*.json "results/$g/" 2>/dev/null || true
  # Regenerate the human readable table so the repo always carries one that
  # matches the raw rows beside it.
  if [ -f "results/$g/results.jsonl" ]; then
    python3 code/summarize.py "results/$g" > "results/$g/summary.txt" 2>/dev/null \
      || echo "summarize failed for $g, raw rows still committed"
  fi
done

# Slurm logs are the provenance for every row: which node, which GPU, what
# vLLM printed at startup, whether concurrency was actually achieved. They are
# mostly progress bar noise, so compress rather than skip.
for f in "$ROOT"/logs/phase0*.out; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  gzip -c "$f" > "results/logs/$b.gz"
done

git add -A results notes
if git diff --cached --quiet; then
  echo "nothing new to commit"
  exit 0
fi
git commit -m "$MSG"
git push
echo "pushed: $MSG"
