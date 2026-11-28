#!/bin/bash
# Copy the dtai3 (router data) results and this user's dtai3 logs into the
# repository and push them. A copy of push_dtai2.sh for the gh200r_* directories
# (25 Nov 2026); push_dtai2.sh is unchanged.
#
#   cd $LADDER_ROOT/repo/code && bash push_dtai3.sh "results: dtai3 tier 0 pieces <job ids>"
#
# Results land in results/gh200r/<directory>/ (results.jsonl plain, perreq_*.json
# gzipped as perreq_*.json.gz), logs in results/gh200r/logs/ (gzipped).
# Run as yourself: each teammate pushes their own pieces; directory names carry
# the piece tag, so two teammates' pieces never collide.
set -euo pipefail
: "${LADDER_ROOT:?source deltaai_env.sh first}"
case "$LADDER_ROOT" in /projects/*) echo "LADDER_ROOT is under /projects; run deltaai_env.sh on DeltaAI" >&2; exit 1 ;; esac
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
MSG=${1:-"results: DeltaAI GH200 dtai3 router data"}
[ -d "$REPO/.git" ] || { echo "no git clone at $REPO" >&2; exit 1; }
cd "$REPO"
if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "This clone has no commit identity. Set it once:" >&2
  echo "  git -C $REPO config user.name  \"<your name>\"" >&2
  echo "  git -C $REPO config user.email \"<the email on your GitHub account>\"" >&2
  exit 1
fi
git pull --rebase --autostash
DST=results/gh200r
mkdir -p "$DST/logs"
n=0
for src in "$LADDER_ROOT"/results/gh200r_*; do
  [ -d "$src" ] || continue
  name=$(basename "$src")
  mkdir -p "$DST/$name"
  # results.jsonl as it is; the per-request files gzipped (token ids at up to
  # 1,024 tokens, and the l cells' logprobs, would add hundreds of MB of JSON).
  find "$src" -maxdepth 1 -type f -name '*.jsonl' -exec cp -p {} "$DST/$name/" \;
  for f in "$src"/*.json; do
    [ -f "$f" ] || continue
    g="$DST/$name/$(basename "$f").gz"
    [ -f "$g" ] && [ "$g" -nt "$f" ] && continue
    gzip -n -c "$f" > "$g"
  done
  n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "no gh200r_* results under $LADDER_ROOT/results" >&2; exit 1; }
for f in "$LADDER_ROOT"/logs/dtai3-*.out "$LADDER_ROOT"/logs/dtai3-*-gpu.csv; do
  [ -e "$f" ] || continue
  [ "$(stat -c %U "$f" 2>/dev/null)" = "${USER:-$(id -un)}" ] || continue
  gzip -c "$f" > "$DST/logs/$(basename "$f").gz"
done
# The workload manifest, so every result can be traced to the exact files it ran on.
ROUTER=${LADDER_ROUTER:-$LADDER_ROOT/data/router}
[ -f "$ROUTER/manifest.json" ] && cp -p "$ROUTER/manifest.json" "$DST/workload_manifest_${USER:-$(id -un)}.json"
git add "$DST"
if git diff --cached --quiet; then echo "nothing new to commit"; exit 0; fi
echo "committing $(git diff --cached --name-only | wc -l) files from $n results directories:"
git diff --cached --stat | tail -1
git commit -q -m "$MSG"
git push
echo "pushed: $MSG"
