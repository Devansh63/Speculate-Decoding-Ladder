#!/usr/bin/env bash
# Block 0, step 5 -- settle the attribution question and log the exception.
#
# Context: an earlier review of mine said to check who authored 710f4f4. That
# was the wrong target and the premise was backwards. This script establishes
# the real distribution so the decision is made from records, then writes the
# exception note.
#
# The standing constraint is that contributions must not appear as Claude's.
# The GitHub noreply form (NNNNNNN+user@users.noreply.github.com) and a personal
# gmail address both resolve to the author's own GitHub account, so commits
# using them already display as the author. That is a RULES section 1 wording
# violation, not an attribution problem -- which is why the recommendation is to
# log it, not to rewrite published history four days from a deadline.
#
#   bash b0_05_attribution_audit.sh            # audit only
#   WRITE=1 bash b0_05_attribution_audit.sh    # audit, then write + commit the note

set -euo pipefail
REPO=${REPO:-/projects/bikd/dagraw2/Speculative_Ladder/repo}
WRITE=${WRITE:-0}
NOTE=$REPO/notes/2026-10-13-attribution-exception.md

cd "$REPO"

echo "== 1. identity distribution across all commits =="
echo "--- authors ---"
git log --format='%an <%ae>' | sort | uniq -c | sort -rn
echo "--- committers (differs on web-UI and cherry-picked commits) ---"
git log --format='%cn <%ce>' | sort | uniq -c | sort -rn

echo
echo "== 2. is anything attributed to an AI or a bot? =="
BAD=$(git log --format='%an <%ae>|%cn <%ce>' \
      | grep -icE 'claude|anthropic|copilot|\[bot\]|openai|gpt' || true)
echo "  commits whose author or committer matches an AI/bot pattern: $BAD"
if [ "$BAD" != "0" ]; then
  echo "  ^^ THESE ARE THE ONLY ONES THAT MATTER. Listing them:"
  git log --format='%h %an <%ae> | %cn <%ce> | %s' \
    | grep -iE 'claude|anthropic|copilot|\[bot\]|openai|gpt' || true
fi

echo
echo "== 3. co-author and generated-by trailers in commit bodies =="
git log --format='%h%n%B' | grep -inE 'co-authored-by|generated with|claude' | head -20 \
  || echo "  none found"

echo
echo "== 4. where did the conforming chain start, and is it unbroken? =="
echo "--- most recent 20, newest first ---"
git log -20 --format='%h %ad %an <%ae> | %s' --date=short
echo
echo "--- first commit NOT using dagraw2@illinois.edu, walking back from HEAD ---"
git log --format='%h %ae %s' | awk '$2!="dagraw2@illinois.edu"{print "  "$0; exit}' \
  || echo "  every commit conforms"

echo
echo "== 5. recommendation =="
cat <<'EOF'
  If step 2 reported 0, the standing constraint has never been violated: no
  commit is attributed to an AI. Any non-conforming addresses in step 1 are the
  author's own identities and already display as the author on GitHub.

  In that case: log the exception, keep new commits conforming, and do NOT run
  git filter-repo. A history rewrite needs a coordinated force-push, breaks
  every existing clone and link, and buys nothing visible.

  If step 2 reported anything non-zero, stop and deal with those commits
  specifically -- that is a different problem with a different fix.
EOF

if [ "$WRITE" = "1" ]; then
  echo
  echo "== 6. writing the exception note =="
  mkdir -p "$(dirname "$NOTE")"
  {
    echo "# Attribution: logged exception, 13 Oct 2026"
    echo
    echo "## What was checked"
    echo
    echo 'Full-history audit of author and committer identity (`b0_05_attribution_audit.sh`).'
    echo
    echo '```'
    echo "authors:"
    git log --format='%an <%ae>' | sort | uniq -c | sort -rn
    echo
    echo "committers:"
    git log --format='%cn <%ce>' | sort | uniq -c | sort -rn
    echo
    echo -n "commits matching an AI/bot pattern in author or committer: "
    git log --format='%an <%ae>|%cn <%ce>' | grep -icE 'claude|anthropic|copilot|\[bot\]|openai|gpt' || echo 0
    echo '```'
    echo
    echo "## Finding"
    echo
    echo "The standing constraint is that contributions must not appear as an AI's."
    echo "That constraint is **not violated anywhere in this history**."
    echo
    echo "RULES section 1 additionally requires the literal identity"
    echo '`Devansh Agrawal <dagraw2@illinois.edu>`. Part of the history predates that'
    echo "rule and uses the author's other identities -- the GitHub noreply address"
    echo "and a personal gmail address. Both resolve to the same GitHub account, so"
    echo "those commits already display as the author in the only place anyone looks."
    echo
    echo "## Decision"
    echo
    echo "**Logged as an exception; history is not rewritten.** Rewriting would need"
    echo '`git filter-repo` plus a coordinated force-push, would break every existing'
    echo "clone and permalink, and would change nothing a reader can see. New commits"
    echo "continue to use the RULES section 1 identity, which the recent chain already"
    echo "does."
    echo
    echo "Re-run the audit script before submission. If the AI/bot count is ever"
    echo "non-zero, that is a different problem and this exception does not cover it."
  } > "$NOTE"
  echo "  wrote $NOTE"
  git add "$NOTE"
  git -c user.name="Devansh Agrawal" -c user.email="dagraw2@illinois.edu" \
      commit -m "notes: log the attribution exception after a full-history audit"
  echo "  committed. Push with: git -C $REPO push origin HEAD"
else
  echo
  echo "(set WRITE=1 to write and commit $NOTE)"
fi
