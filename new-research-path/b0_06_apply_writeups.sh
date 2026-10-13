#!/usr/bin/env bash
# Block 0, step 6 -- put the Block 0 findings into the repo and commit them.
#
# Splits block0-writeups.md on its section markers and routes each one:
#
#   §A  frozen prediction scored      -> append to paper/claims.md
#   §B  C20, machine factor by rung   -> append to paper/claims.md
#   §C  C19 correction                -> new notes/2026-10-13-c19-correction.md
#                                        + a pointer line in the original note
#   §D  three new threats             -> append to paper/threats.md
#
# claims.md and threats.md are append-only. This script only ever appends to
# them, never rewrites. The original memory-tax note is not edited either --
# it gets one pointer line, so a reader of the old claim finds the correction.
#
#   bash b0_06_apply_writeups.sh            # dry run: prints what would be written
#   DRYRUN=0 bash b0_06_apply_writeups.sh   # write and commit

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/projects/bikd/dagraw2/Speculative_Ladder/repo}
SRC=${SRC:-$HERE/block0-writeups.md}
OLDNOTE=notes/2026-10-11-memory-tax-and-remaining-axes.md
NEWNOTE=notes/2026-10-13-c19-correction.md
DRYRUN=${DRYRUN:-1}
STAMP=$(date +%Y-%m-%d)

[ -r "$SRC" ] || { echo "ERROR: cannot read $SRC"; exit 1; }
[ -d "$REPO/.git" ] || { echo "ERROR: $REPO is not a git repo"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 - "$SRC" "$TMP" <<'PY'
import re, sys, os
src, out = sys.argv[1], sys.argv[2]
text = open(src).read()
# Sections start at "## §X" and run to the next "## §" (or EOF).
parts = re.split(r'(?m)^## +§([A-D])\b[^\n]*\n', text)
# parts = [preamble, 'A', bodyA, 'B', bodyB, ...]
found = {}
for i in range(1, len(parts) - 1, 2):
    body = parts[i + 1]
    body = re.sub(r'(?m)^---\s*$', '', body).strip() + "\n"
    found[parts[i]] = body
for k in "ABCD":
    if k not in found:
        sys.exit(f"section {k} not found in {src}")
    open(os.path.join(out, k + ".md"), "w").write(found[k])
    print(f"  §{k}: {len(found[k].splitlines())} lines, "
          f"{len(found[k])} bytes")
PY

cd "$REPO"

echo
echo "== preflight =="
git rev-parse --abbrev-ref HEAD
DIRTY=$(git status --porcelain | grep -v '^?? new-research-path/block0-out/' || true)
if [ -n "$DIRTY" ]; then
  echo "  working tree has other changes:"; echo "$DIRTY"
  echo "  (they will be left alone; only the files below are staged)"
fi
for F in paper/claims.md paper/threats.md "$OLDNOTE"; do
  if [ -f "$F" ]; then echo "  OK      $F ($(wc -l < "$F") lines)"
  else echo "  MISSING $F -- will be created"; fi
done

write() {  # $1 target, $2 source fragment, $3 description
  if [ "$DRYRUN" = "1" ]; then
    echo "  [dry] append $(wc -l < "$2") lines to $1   ($3)"
  else
    { printf '\n'; cat "$2"; } >> "$1"
    echo "  appended $3 -> $1"
  fi
}

echo
echo "== routing =="
write paper/claims.md "$TMP/A.md" "frozen prediction scored"
write paper/claims.md "$TMP/B.md" "C20 machine factor by rung"
write paper/threats.md "$TMP/D.md" "threats 9-11"

if [ "$DRYRUN" = "1" ]; then
  echo "  [dry] create $NEWNOTE from section C"
  echo "  [dry] add a pointer line to $OLDNOTE"
else
  mkdir -p notes
  { echo "# C19 corrected: the drafter's weights are charged to the KV pool"
    echo
    echo "Written $STAMP. Supersedes the magnitude paragraph of"
    echo "\`$OLDNOTE\`, which is left in place unedited so the original"
    echo "reasoning stays readable."
    echo
    cat "$TMP/C.md"; } > "$NEWNOTE"
  echo "  wrote $NEWNOTE"
  if [ -f "$OLDNOTE" ] && ! grep -q "$NEWNOTE" "$OLDNOTE"; then
    printf '\n> **Corrected %s.** The magnitude in this note was re-derived by pairing within job; see `%s`. The direction and the ceiling consequence stand; the pooled-across-jobs comparison that appeared to refute them was invalid.\n' \
      "$STAMP" "$NEWNOTE" >> "$OLDNOTE"
    echo "  added pointer to $OLDNOTE"
  fi
fi

echo
echo "== evidence files =="
EV=new-research-path/block0-out
if [ -d "$EV" ]; then
  ls -1 "$EV"
  if [ "$DRYRUN" = "1" ]; then
    echo "  [dry] git add -f $EV  (it is gitignored; forced so the evidence is in the repo)"
  else
    git add -f "$EV"
  fi
else
  echo "  $EV not present -- run 'bash run_block0.sh analyse' first"
fi

echo
echo "== commit =="
if [ "$DRYRUN" = "1" ]; then
  echo "  [dry] would stage: paper/claims.md paper/threats.md $NEWNOTE $OLDNOTE $EV"
  echo "  [dry] commit as Devansh Agrawal <dagraw2@illinois.edu>"
  echo
  echo "DRYRUN=1. Read block0-writeups.md, then re-run with DRYRUN=0."
else
  git add paper/claims.md paper/threats.md "$NEWNOTE" "$OLDNOTE" 2>/dev/null || true
  git -c user.name="Devansh Agrawal" -c user.email="dagraw2@illinois.edu" commit -m \
"claims: score the frozen prediction, register C20, correct C19

- frozen prediction: 2 of 4 cells within 5%, median 6.18%, all errors
  optimistic; pass/fail splits by rung, not bucket. Ordering criterion is
  ambiguous as worded and the ambiguity is recorded, not resolved.
- C20: the machine factor is rung-independent at 2K (1.046 vs 1.056) and
  diverges with context (1.070 vs 1.296 at 32K). R_draft measured from the
  k-sweeps: DFlash 0.299/0.445/0.673, n-gram 0.071/-/0.139. The fixed
  drafting term dominates c*k by 4-7x and grows twice as fast.
- C19 corrected: 2.4-2.5 GiB paired within job across 17 pairs; the pooled
  comparison that refuted it mixed a 48 GB card with a 40 GB one. New: the
  cost rises to 4.01 GiB at c=32, so admissible concurrency cannot be
  projected from c=1.
- threats 9-11: tau does not transfer across machines (21 of 25 outputs
  differ); sequence-level agreement is a length artefact; n-gram may have
  been run at a suboptimal k throughout."
  echo
  git log -1 --format='%h | %an <%ae> | %s'
  echo
  echo "Push with:  git -C $REPO push origin HEAD"
fi
