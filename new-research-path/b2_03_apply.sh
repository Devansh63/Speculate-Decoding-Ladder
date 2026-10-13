#!/usr/bin/env bash
# Block 2 -- capture every result from 13 Oct, put the raw data in the repo,
# route the write-ups, and commit.
#
# Results first, by design: every claim below cites data that this script
# commits in the same pass, so a second reader can check it in one clone.
#
#   bash b2_03_apply.sh            # dry run: prints what it would do
#   DRYRUN=0 bash b2_03_apply.sh   # capture, write, commit

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${R:-/projects/bikd/dagraw2/Speculative_Ladder/repo}
S=${S:-/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts}
LOGS=${LOGS:-/projects/bikd/dagraw2/Speculative_Ladder/logs}
A40=${A40:-/projects/bikd/dagraw2/Speculative_Ladder/results/a40}
C10=${C10:-/projects/bikd/dagraw2/Speculative_Ladder/results/a40_c10}
VENV=${VENV:-/projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch}
SRC=${SRC:-$HERE/block2-writeups.md}
OUT=$R/new-research-path/block2-out
DRYRUN=${DRYRUN:-1}
STAMP=$(date +%Y-%m-%d)

say() { printf '\n== %s\n' "$*"; }
run() { if [ "$DRYRUN" = "1" ]; then echo "  [dry] $*"; else eval "$@"; fi; }

[ -r "$SRC" ] || { echo "ERROR: cannot read $SRC"; exit 1; }
[ -d "$R/.git" ] || { echo "ERROR: $R is not a git repo"; exit 1; }

say "0. split the write-ups"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 - "$SRC" "$TMP" <<'PY'
import os, re, sys
src, out = sys.argv[1], sys.argv[2]
parts = re.split(r'(?m)^## +§([E-H])\b[^\n]*\n', open(src).read())
found = {}
for i in range(1, len(parts) - 1, 2):
    body = re.sub(r'(?m)^---\s*$', '', parts[i + 1]).strip() + "\n"
    found[parts[i]] = body
for k in "EFGH":
    if k not in found:
        sys.exit(f"section {k} missing from {src}")
    open(os.path.join(out, k + ".md"), "w").write(found[k])
    print(f"  §{k}: {len(found[k].splitlines())} lines")
PY

say "1. capture the results (read only; this is the evidence the claims cite)"
run "mkdir -p '$OUT'"

# -- the C10 control, the day's headline
if [ -r "$C10/results.jsonl" ]; then
  run "python3 '$S/check_lossless.v3.py' '$C10' > '$OUT/c10-control.txt' 2>&1"
  run "python3 '$HERE/decompose.py' '$C10' --harness v3 > '$OUT/c10-throughput.txt' 2>&1"
  echo "  C10 control: $(wc -l < "$C10/results.jsonl") rows, \
$(find "$C10" -name 'perreq_*.json' | wc -l) per-request file(s)"
else
  echo "  WARNING: $C10 missing -- the C21 claim cites data that will not be committed"
fi

# -- achieved concurrency across every log
run "python3 '$HERE/b2_02_achieved_concurrency.py' $LOGS/phase0-*.out \
     > '$OUT/achieved-concurrency.txt' 2>&1"

# -- accuracy needs the venv for the tokenizer
if [ -x "$VENV/bin/python3" ]; then
  run "'$VENV/bin/python3' '$HERE/b2_01_score_accuracy.py' '$A40' --by-domain \
       > '$OUT/accuracy-a40.txt' 2>&1"
else
  echo "  WARNING: no venv at $VENV -- accuracy not captured"
fi

# -- the decomposition and R_draft fits, for the record
run "python3 '$HERE/decompose.py' '$A40' --fit-k --harness v2 \
     > '$OUT/decompose-a40-fitk.txt' 2>&1"

say "2. the raw C10 data into the repo"
run "mkdir -p '$R/results/a40_c10' '$R/results/logs/a40_c10'"
if [ -r "$C10/results.jsonl" ]; then
  run "cp -v '$C10/results.jsonl' '$R/results/a40_c10/'"
  run "cp -v $C10/perreq_*.json '$R/results/a40_c10/'"
fi
for J in 22875382 22876140; do
  [ -r "$LOGS/phase0-$J.out" ] && \
    run "gzip -c '$LOGS/phase0-$J.out' > '$R/results/logs/a40_c10/phase0-$J.out.gz'"
done

if [ "$DRYRUN" = "1" ]; then
  echo "  [dry] would write $R/results/a40_c10/README.md"
else
  cat > "$R/results/a40_c10/README.md" <<EOF
# results/a40_c10 -- the non-determinism floor

Three realizations of ONE cell: A40, 2K bucket, rung \`off\`, n=50,
concurrency 1, seed 1234, prefix caching on (read back off the engine).
Harness \`v3-2026-10-13\`.

* job 22875382, gpub015 -- two runs (REPEAT=2)
* job 22876140, gpub036 -- two runs (REPEAT=2)

Job 22875382 produced only ONE per-request file: both its runs shared a job id
and the first version of the v3 file-name fix still overwrote run 1. The fix
now picks the first free name (\`_j<job>\`, then \`_r2\`, \`_r3\`), which is why
22876140 kept both. Three surviving realizations is what the control needed.

Result: 150/150 sequences bit-identical across all three pairs, per-token
hazard 0.000%. Zero events in 12,800 token positions bounds the true hazard at
about 0.023% per token (rule of three, 95%). See C21 in \`paper/claims.md\`.

Throughput, same cell, four runs: 32.5698 and 32.5696 on gpub015, 32.5703 and
32.5644 on gpub036 -- a 0.018% total spread across two nodes.

**Still missing:** no speculative arm has been run twice at one configuration,
so only the BASELINE is shown reproducible. See threat 12.
EOF
  echo "  wrote $R/results/a40_c10/README.md"
fi

say "3. route the write-ups"
w() {  # $1 target, $2 fragment, $3 label
  if [ "$DRYRUN" = "1" ]; then
    echo "  [dry] append $(wc -l < "$2") lines to $1   ($3)"
  else
    { printf '\n'; cat "$2"; } >> "$1"
    echo "  appended $3 -> $1"
  fi
}
w "$R/paper/claims.md"  "$TMP/E.md" "C21 engine determinism, C10 refuted"
w "$R/paper/claims.md"  "$TMP/F.md" "C22 tokens change, answers do not"
w "$R/paper/claims.md"  "$TMP/G.md" "concurrency threat retired"
w "$R/paper/threats.md" "$TMP/H.md" "threats 12-16"

say "4. tools and regenerated candidates"
run "cp -v '$HERE/b2_01_score_accuracy.py' '$HERE/b2_02_achieved_concurrency.py' \
     '$HERE/decompose.py' '$R/code/'"
# run_benchmark.v3 / check_lossless.v3 were REGENERATED after the overwrite fix
for F in run_benchmark.v3.py check_lossless.v3.py phase0.v3.sh; do
  [ -r "$S/$F" ] && run "cp -v '$S/$F' '$R/code/'"
done
run "cp -v '$SRC' '$R/new-research-path/'"

say "5. commit"
if [ "$DRYRUN" = "1" ]; then
  echo "  [dry] git add results/ paper/ code/ new-research-path/"
  echo "  [dry] commit as Devansh Agrawal <dagraw2@illinois.edu>"
  echo
  echo "DRYRUN=1. Read block2-writeups.md, then re-run with DRYRUN=0."
else
  cd "$R"
  git add results/a40_c10 results/logs/a40_c10 paper/claims.md paper/threats.md \
          code new-research-path 2>/dev/null
  git add -f new-research-path/block2-out 2>/dev/null
  git -c user.name="Devansh Agrawal" -c user.email="dagraw2@illinois.edu" commit -m \
"claims: C21 engine determinism refutes C10's strong form; C22 answers invariant

Results committed alongside the claims that cite them.

C21 -- jobs 22875382 (gpub015) and 22876140 (gpub036), 2K off c=1, three
realizations across two nodes: 150/150 sequences bit-identical, per-token
hazard 0.000%, true hazard bounded at ~0.023%. At matched concurrency the
engine does not disagree with itself, so C10's control was measuring
CONCURRENCY, not non-determinism, and its strong form is refuted. The
speculative hazards in results/a40 (n-gram 0.556%/tok, DFlash 0.738-0.781%)
need no noise subtraction and sit 24-34x above the bound -- so speculative
decoding on this build is not bitwise lossless.

C22 -- accuracy scored from stored token ids against the LongBench-v2 answer
key. At 8K nine arms score identically (34.7%, n=49); every difference
elsewhere is one prompt; and the baseline differs from ITSELF by one prompt
across concurrency. Four of five domain-bucket cells perfectly invariant.
Speculation changes tokens and does not change answers.

Concurrency threat retired -- 54 arms checked against the engine's own
'Running: N reqs'; every arm with an explicit concurrency reached it. But
sustained means run below peak (c=32 -> 23.5) and job 22569403 ran unlabelled
at up to 77 concurrent, so it must be excluded from any mixed-concurrency
analysis.

Threats 12-16: only the baseline is shown reproducible (no speculative arm has
been run twice at one config); determinism is scoped to 2K/c=1/A40/caching-on;
check_lossless's verdict block is stale; per-cell costs were estimated from
A100 and are ~1.5x low on A40; accuracy differs by bucket in a direction
difficulty cannot explain."
  echo
  git log -1 --format='%h | %an <%ae> | %s'
  echo
  echo "Push with:  git -C $R push origin HEAD"
fi
