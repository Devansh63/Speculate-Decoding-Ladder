#!/bin/bash
# Copy the DeltaAI GH200 results (the smoke job's gh200_ksweep and every
# gh200v2_* directory, pieces included) into the git clone under results/gh200/
# and push. Run on a DeltaAI login node, never in a job, after
# `source deltaai_env.sh`.
#
#   bash push_dtai2.sh "GH200: smoke 3612174, c1-2K k0 pre 3612400, profile 3614889 3614890"
#
# What is copied, per results directory: results.jsonl, the per-request
# files (perreq_*.json) and any other small .json or .csv beside them. Also
# the Slurm logs of this user's dtai2 and phase0-ksweep jobs, gzipped, and
# their GPU clock logs. Not copied: the nsys reports (.nsys-rep, several MB
# each); they stay in $LADDER_ROOT/results/gh200v2_nsys/nsys/. Set NSYS=1 to
# commit them too.
#
# Commits under whatever identity the clone is configured with
# (git config user.name / user.email); it refuses to run without one. Pushing
# needs a GitHub token stored for this clone: notes/github-from-delta.md.

set -euo pipefail

: "${LADDER_ROOT:?source deltaai_env.sh first}"
case "$LADDER_ROOT" in /projects/*) echo "LADDER_ROOT is under /projects; run deltaai_env.sh on DeltaAI" >&2; exit 1 ;; esac
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
MSG=${1:-"results: DeltaAI GH200 outputs"}
NSYS=${NSYS:-0}

[ -d "$REPO/.git" ] || { echo "no git clone at $REPO" >&2; exit 1; }
cd "$REPO"
if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "This clone has no commit identity. Set it once:" >&2
  echo "  git -C $REPO config user.name  \"<your name>\"" >&2
  echo "  git -C $REPO config user.email \"<the email on your GitHub account>\"" >&2
  exit 1
fi

git pull --rebase --autostash
DST=results/gh200
mkdir -p "$DST/logs"

n=0
for src in "$LADDER_ROOT"/results/gh200_ksweep* "$LADDER_ROOT"/results/gh200v2_*; do
  [ -d "$src" ] || continue
  name=$(basename "$src")
  mkdir -p "$DST/$name"
  # Small text files only: results.jsonl, per-request json, metadata json, csv.
  find "$src" -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.json' -o -name '*.csv' -o -name '*.txt' \) \
    -exec cp -p {} "$DST/$name/" \;
  if [ -d "$src/nsys" ]; then
    mkdir -p "$DST/$name/nsys"
    find "$src/nsys" -maxdepth 1 -type f ! -name '*.nsys-rep' ! -name '*.sqlite' -size -5M -exec cp -p {} "$DST/$name/nsys/" \;
    if [ "$NSYS" = "1" ]; then
      find "$src/nsys" -maxdepth 1 -type f -name '*.nsys-rep' -exec cp -p {} "$DST/$name/nsys/" \;
    else
      ls "$src/nsys" | grep '\.nsys-rep$' > "$DST/$name/nsys/REPORTS_ON_DELTAAI.txt" || true
    fi
  fi
  n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "no gh200 results under $LADDER_ROOT/results" >&2; exit 1; }

for f in "$LADDER_ROOT"/logs/dtai2-*.out "$LADDER_ROOT"/logs/phase0-ksweep-*.out; do
  [ -e "$f" ] || continue
  [ "$(stat -c %U "$f" 2>/dev/null)" = "${USER:-$(id -un)}" ] || continue
  gzip -c "$f" > "$DST/logs/$(basename "$f").gz"
done
for f in "$LADDER_ROOT"/logs/dtai2-*-gpu.csv; do
  [ -e "$f" ] && gzip -c "$f" > "$DST/logs/$(basename "$f").gz"
done

git add "$DST"
if git diff --cached --quiet; then echo "nothing new to commit"; exit 0; fi
echo "committing $(git diff --cached --name-only | wc -l) files from $n results directories:"
git diff --cached --stat | tail -1
git commit -q -m "$MSG"
git push
echo "pushed: $MSG"
