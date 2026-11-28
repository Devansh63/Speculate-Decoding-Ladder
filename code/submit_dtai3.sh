#!/bin/bash
# Submit the router data job set (dtai3), one Slurm job per piece. Prints the
# plan and submits nothing unless GO=1.
#
#   cd $LADDER_ROOT/repo/code
#   PROJ=<code> ACCOUNT=<name> source deltaai_env.sh
#   bash submit_dtai3.sh TIERS=S                 # the plan for the smoke piece
#   bash submit_dtai3.sh TIERS=S GO=1            # run it first (about 15 minutes)
#   Pieces too long for a 30-minute limit are split by the planner below
#   (tags <piece>s1, <piece>s2, ...); ONLY and SKIP take the table's tags.
#   bash submit_dtai3.sh TIERS=0 GO=1            # then tier 0, and so on
#
# Knobs (KEY=VALUE arguments or environment):
#   TIERS   which tiers, default "0"; S is the smoke piece
#   ONLY    only these pieces (by tag), e.g. ONLY="qa2 qa8"
#   SKIP    skip these pieces
#   GO      1 to submit
#   PART    partition, default ghx4
#
# Tiers (notes/2026-11-25-router-data-jobs.md has the reasons and costs):
#   S  smoke: 4 prompts of QA 2K through both cells, to see chat mode work
#      on DeltaAI before any real hour is spent
#   0  chat-format core: QA at 2K, 8K, 16K, 32K at batch 1 and under load;
#      code, summ, fewshot, mdqa at 2K, 8K, 16K, 32K at batch 1. Answers
#      whether the crossover and the switch points survive the chat format,
#      and fills 16K, where the switch point is
#   1  copy-varying request types from real text: RAG with four instructions
#      over one document, code edits, multi-turn WildChat, SWE-agent traces,
#      the padding test (real padding and repeated-sentence padding at 8K),
#      and Spec-Bench's six categories (translation and math are nowhere else)
#   2  AI-written prompts (needs gen_router_ai.py and DEEPSEEK_API_KEY first):
#      instructions over real documents across copy levels, DeepSeek-written
#      code edits, LLM padding for the padding test, twins of real chats
#   3  extras: the four tasks under load, mixed 2K and 32K batches, thinking on
#
# Each piece writes its own directories (gh200r_<workload>_<bucket>_<cell>_p<piece>),
# so pieces can run in any order and at the same time.

set -euo pipefail
for kv in "$@"; do
  case "$kv" in *=*) export "${kv%%=*}"="${kv#*=}" ;; *) echo "arguments are KEY=VALUE, got '$kv'" >&2; exit 1 ;; esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
TIERS=${TIERS:-0}
GO=${GO:-0}
PART=${PART:-ghx4}

# tier|piece|workload|bucket|cells|lim1 (optional)
PIECES='
S|smoke|qa|2K|t a l|4
0|qa2|qa|2K|t a l|
0|qa8|qa|8K|t a l|
0|qa16t|qa|16K|t|
0|qa16a|qa|16K|a l|
0|qa32t|qa|32K|t|
0|qa32a|qa|32K|a l|
0|qa2La|qa|2K|c4 c8|
0|qa2Lb|qa|2K|c16 c32|
0|qa8La|qa|8K|c4|
0|qa8Lb|qa|8K|c8 c16|
0|qa16La|qa|16K|c2|
0|qa16Lb|qa|16K|c4|
0|qa16Lc|qa|16K|c8|
0|qa32La|qa|32K|c2|
0|qa32Lb|qa|32K|c4|
0|qa32Lc|qa|32K|c8|
0|code2|code|2K|t a l|
0|code8|code|8K|t a l|
0|code16t|code|16K|t|
0|code16a|code|16K|a l|
0|code32t|code|32K|t|
0|code32a|code|32K|a l|
0|summ2t|summ|2K|t|
0|summ2a|summ|2K|a l|
0|summ8t|summ|8K|t|
0|summ8a|summ|8K|a l|
0|summ16t|summ|16K|t|
0|summ16a|summ|16K|a l|
0|summ32t|summ|32K|t|
0|summ32a|summ|32K|a l|
0|fewshot2|fewshot|2K|t a l|
0|fewshot8|fewshot|8K|t a l|
0|fewshot16t|fewshot|16K|t|
0|fewshot16a|fewshot|16K|a l|
0|fewshot32t|fewshot|32K|t|
0|fewshot32a|fewshot|32K|a l|
0|mdqa2|mdqa|2K|t a l|
0|mdqa8|mdqa|8K|t a l|
0|mdqa16t|mdqa|16K|t|
0|mdqa16a|mdqa|16K|a l|
0|mdqa32t|mdqa|32K|t|
0|mdqa32a|mdqa|32K|a l|
1|rq2|ragquote|2K|t a l|62
1|rq8|ragquote|8K|t a l|62
1|rq16t|ragquote|16K|t|62
1|rq16a|ragquote|16K|a l|62
1|rq32t|ragquote|32K|t|46
1|rq32a|ragquote|32K|a l|46
1|ce0t|codeedit|nat|t|
1|ce0a|codeedit|nat|a l|
1|ce2t|codeedit|2K|t|
1|ce2a|codeedit|2K|a l|
1|ce8t|codeedit|8K|t|
1|ce8a|codeedit|8K|a l|
1|ce16t|codeedit|16K|t|
1|ce16a|codeedit|16K|a l|
1|chat2|chat|2K|t a l|
1|chat8t|chat|8K|t|
1|chat8a|chat|8K|a l|
1|ag8|agent|8K|t a l|
1|ag16t|agent|16K|t|
1|ag16a|agent|16K|a l|
1|ag32t|agent|32K|t|
1|ag32a|agent|32K|a l|
1|eqmdR|eq_mdqa_realpad|8K|t a l|
1|eqmdX|eq_mdqa_repeatpad|8K|t a l|
1|eqfsR|eq_fewshot_realpad|8K|t a l|
1|eqfsX|eq_fewshot_repeatpad|8K|t a l|
1|sbmtt|sb_mt|nat|t|78
1|sbmta|sb_mt|nat|a l|78
1|sbtr|sb_trans|nat|t a l|78
1|sbsu|sb_summ|nat|t a l|78
1|sbqa|sb_qa|nat|t a l|78
1|sbma|sb_math|nat|t a l|78
1|sbrg|sb_rag|nat|t a l|78
2|syn2|synth|2K|t a l|58
2|syn8t|synth|8K|t|58
2|syn8a|synth|8K|a l|58
2|syn16t|synth|16K|t|58
2|syn16a|synth|16K|a l|58
2|syn32t|synth|32K|t|30
2|syn32a|synth|32K|a l|30
2|cea2|codeedit_ai|2K|t a l|
2|cea8|codeedit_ai|8K|t a l|
2|cea16t|codeedit_ai|16K|t|
2|cea16a|codeedit_ai|16K|a l|
2|eqmdL|eq_mdqa_llmpad|8K|t a l|
2|eqfsL|eq_fewshot_llmpad|8K|t a l|
2|twR|twins_real|nat|t a l|
2|twA|twins_ai|nat|t a l|
3|code2L|code|2K|c16 c32|
3|summ2L|summ|2K|c16 c32|
3|fewshot2L|fewshot|2K|c16 c32|
3|mdqa2L|mdqa|2K|c16 c32|
3|code8L|code|8K|c16|
3|summ8L|summ|8K|c16|
3|fewshot8L|fewshot|8K|c16|
3|mdqa8L|mdqa|8K|c16|
3|mixL|mix|mix|c8 c16|
3|qa2Tt|qa|2K|tT|
3|qa2Ta|qa|2K|aT lT|
3|qa8Tt|qa|8K|tT|
3|qa8Ta|qa|8K|aT lT|
'

miss=0
for v in LADDER_ROOT LADDER_VENV LADDER_HF LADDER_CACHE ACCOUNT; do
  [ -n "${!v:-}" ] || { echo "missing $v" >&2; miss=1; }
done
[ "$miss" = "0" ] || { echo "Run first:  PROJ=<code> ACCOUNT=<name> source deltaai_env.sh" >&2; exit 1; }
if [ "$(uname -m)" != "aarch64" ] && [ "${LADDER_TEST_HOST:-0}" != "1" ]; then
  echo "This is a $(uname -m) machine. Submit from a DeltaAI login node." >&2; exit 1
fi
[ "${LADDER_CAPS:-}" = "90" ] || { echo "LADDER_CAPS=${LADDER_CAPS:-unset}; source deltaai_env.sh" >&2; exit 1; }
[ "${LADDER_MRV2:-0}" = "1" ] || { echo "LADDER_MRV2 must be 1 (Model Runner V2)." >&2; exit 1; }
[ -f "$LADDER_VENV/ladder_modules" ] || { echo "no $LADDER_VENV/ladder_modules" >&2; exit 1; }
for f in dtai3_job.sh router_cell.sh run_benchmark.v5d.py; do
  [ -f "$HERE/$f" ] || { echo "missing $HERE/$f (git pull?)" >&2; exit 1; }
done
grep -q 'HARNESS_VERSION = "v5d' "$HERE/run_benchmark.v5d.py" || { echo "run_benchmark.v5d.py is not harness v5d" >&2; exit 1; }
ROUTER=${LADDER_ROUTER:-$LADDER_ROOT/data/router}
case "$ROUTER" in /projects/*) echo "LADDER_ROUTER is under /projects" >&2; exit 1 ;; esac
mkdir -p "$LADDER_ROOT/logs" "$LADDER_ROOT/results"

# The plan (28 Nov): every piece's time from measured GH200 numbers, and any
# piece that would not fit a 30-minute limit split until it does. Short limits
# are what start on ghx4: ghx4 had 5,195 jobs waiting on 28 Nov, and on 18 Nov
# 30-minute pieces backfilled while 1-hour pieces waited. Splitting costs almost
# nothing: every arm already loads the model in its own process.
#
# Measured, and where each number comes from:
#   prefill and decode per token, per arm and context length, batch 1: the
#     dtai2 per-request timings (results/gh200*/perreq_*.json, 400 to 1,300
#     requests per arm; e.g. 2K off 0.056 s and 5.85 ms per token, DFlash k 15
#     2.65 ms; 32K off 1.45 s and 7.11 ms)
#   decode step under load: dtai2's load cells, prefill taken out
#   model load per arm: the smoke piece (job 3689701): 45 to 55 s once the
#     compile cache is warm, 80 to 130 s cold
# Sized for the worst case so nothing times out: every answer at the token cap
# (1,024; 2,048 thinking), n-gram no faster than the off arm (chat prompts copy
# less than the raw ones did), DFlash 30% slower than measured, loads cold.
# A piece gets 20 minutes if that worst case plus 1.5 fits in 17, 30 if it fits
# in 27; otherwise its cells are split into separate pieces and, where needed,
# into halves (t1 t2, a1 a2, c<N>n c<N>d; dtai3_job.sh), then packed back
# together as far as 27 minutes allows. Printed GPU time is the expected case
# (measured speeds, outputs at a per-workload guess).
plan () {
  PIECES="$PIECES" TIERS="$TIERS" ONLY="${ONLY:-}" SKIP="${SKIP:-}" ROUTER="$ROUTER" \
  "$LADDER_VENV/bin/python3" - <<'EOF'
import os, re
PRE = {"nat": 0.03, "2K": 0.056, "8K": 0.247, "16K": 0.65, "32K": 1.45, "mix": 0.75}
DEC = {  # ms per token, batch 1, measured (dtai2)
    "off":  {"nat": 5.85, "2K": 5.85, "8K": 6.12, "16K": 6.6, "32K": 7.11, "mix": 6.6},
    "ng5":  {"nat": 4.87, "2K": 4.87, "8K": 4.73, "16K": 5.1, "32K": 5.46, "mix": 5.1},
    "ng11": {"nat": 4.75, "2K": 4.75, "8K": 4.58, "16K": 5.0, "32K": 5.40, "mix": 5.0},
    "df4":  {"nat": 3.18, "2K": 3.18, "8K": 4.16, "16K": 5.2, "32K": 6.26, "mix": 5.2},
    "df7":  {"nat": 2.88, "2K": 2.88, "8K": 3.84, "16K": 4.9, "32K": 6.04, "mix": 4.9},
    "df15": {"nat": 2.65, "2K": 2.65, "8K": 3.31, "16K": 4.6, "32K": 5.85, "mix": 4.6},
}
STEP = {"2K": {4: 6.15, 8: 6.57, 16: 7.4, 32: 9.8}, "8K": {4: 6.9, 8: 8.2, 16: 11.3},
        "16K": {2: 7.5, 4: 8.5, 8: 12.0}, "32K": {2: 8.3, 4: 10.4, 8: 16.2}, "mix": {8: 11.0, 16: 16.0}}
OUT = {"qa": 600, "code": 300, "summ": 600, "fewshot": 80, "mdqa": 60, "ragquote": 400, "codeedit": 700,
       "codeedit_ai": 600, "chat": 550, "agent": 350, "synth": 500, "twins_real": 550, "twins_ai": 550,
       "mix": 600, "sb_mt": 700, "sb_trans": 80, "sb_summ": 250, "sb_qa": 150, "sb_math": 350, "sb_rag": 300}
B1 = {"t": ["off", "ng5", "ng11", "df7", "df15"], "t1": ["off", "ng5", "ng11"], "t2": ["df7", "df15"],
      "a": ["ng5", "ng11", "df7", "df15"], "a1": ["ng5", "ng11"], "a2": ["df7", "df15"], "l": ["off"],
      "t0": ["off"], "tn": ["ng5", "ng11"]}
HALF = {"t": ["t1", "t2"], "a": ["a1", "a2"], "tT": ["tT1", "tT2"], "aT": ["aT1", "aT2"],
        "t1": ["t0", "tn"], "tT1": ["tT0", "tTn"]}
FIT20, FIT30, JOB = 17.0, 27.0, 1.5

def out_tok(wl, think, worst):
    cap = 2048 if think else 1024
    if worst:
        return cap
    o = OUT.get(wl, 60 if "mdqa" in wl else 80 if "fewshot" in wl else 500)
    return min(cap, o * (2.5 if think else 1))

def cell_min(wl, b, cell, n1, nl, nT, worst):
    think = "T" in cell
    o = out_tok(wl, think, worst)
    load = 1.5 if worst else 1.0
    m = re.fullmatch(r"c(\d+)([nd]?)", cell)
    if m:
        c, half = int(m.group(1)), m.group(2)
        arms = {"": ["off", "ng5", "df4", "df7", "df15"], "n": ["off", "ng5"], "d": ["df4", "df7", "df15"]}[half]
        step = STEP.get(b, {}).get(c) or max(STEP.get(b, {"x": 12.0}).values())
        per_arm = nl * PRE[b] + -(-nl // c) * o * step / 1000
        if not worst:
            per_arm *= 0.8
        return len(arms) * ((load + 0.25) + per_arm / 60)
    base = cell.replace("T", "")
    arms = B1[base]
    n = nT if think else n1
    tot = 0.0
    for a in arms:
        d = DEC[a][b]
        if worst:
            d = DEC["off"][b] if a.startswith("ng") else d * 1.3
        per = n * (PRE[b] + o * d / 1000) * (1.05 if base.startswith("a") else 1.1 if base == "l" else 1.0)
        tot += load + per / 60
    return tot

def wall(m):
    need = m + JOB
    return "00:20:00" if need <= FIT20 else "00:30:00" if need <= FIT30 else "01:00:00"

tiers = os.environ["TIERS"].split()
only, skip = os.environ["ONLY"].split(), os.environ["SKIP"].split()
router = os.environ["ROUTER"]
for line in os.environ["PIECES"].strip().splitlines():
    tier, piece, wl, b, cells, lim1 = line.split("|")
    if tier not in tiers or (only and piece not in only) or piece in skip:
        continue
    f = os.path.join(router, wl, b + ".jsonl")
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        print("|".join(["HELD", tier, piece, wl, b, cells, "-", "-", "-", "-", f"HELD: no {f}"])); continue
    rows = sum(1 for x in open(f) if x.strip())
    if rows < 7:
        print("|".join(["HELD", tier, piece, wl, b, cells, "-", str(rows), "-", "-", f"HELD: only {rows} rows"])); continue
    d1 = 25 if b == "32K" else 50
    n1 = min(int(lim1) if lim1 else d1, rows - 2)
    nl = min({"2K": 128, "8K": 128, "32K": 48}.get(b, 64), rows - 2)
    nT = min(25, rows - 2)
    W = lambda c: cell_min(wl, b, c, n1, nl, nT, True)
    def expand(c):  # split a cell until each part fits 27 minutes, or cannot split further
        if W(c) + JOB <= FIT30:
            return [c]
        m = re.fullmatch(r"c(\d+)", c)
        parts = [f"c{m.group(1)}n", f"c{m.group(1)}d"] if m else HALF.get(c)
        return [x for p in parts for x in expand(p)] if parts else [c]
    units = [u for c in cells.split() for u in expand(c)]
    groups, cur = [], []
    for u in units:  # pack back together, in order, as far as 27 minutes allows
        if cur and sum(map(W, cur)) + W(u) + JOB > FIT30:
            groups.append(cur); cur = []
        cur.append(u)
    groups.append(cur)
    for i, g in enumerate(groups):
        tag = piece if len(groups) == 1 else f"{piece}s{i + 1}"
        worst = sum(map(W, g))
        exp = sum(cell_min(wl, b, c, n1, nl, nT, False) for c in g)
        print("|".join(["SUB", tier, tag, wl, b, " ".join(g), lim1, str(rows),
                        f"{worst:.0f}", f"{exp:.1f}", wall(worst)]))
EOF
}

PLAN=$(plan) || { echo "the planner failed (above); nothing submitted" >&2; exit 1; }
# A rerun must not queue a piece twice (28 Nov: sbatch waited on a busy
# controller mid-tier; an interrupted run is simply run again). Skip a piece
# whose job name is already queued or running, or whose every cell already
# has a results file.
QUEUED=$(squeue -u "${USER:-$(id -un)}" -h -o "%j" 2>/dev/null | sort -u || true)
done_already () {  # $1 wl $2 b $3 cells $4 piece
  local c
  for c in $3; do [ -s "$LADDER_ROOT/results/gh200r_$1_$2_${c}_p$4/results.jsonl" ] || return 1; done
  return 0
}
printf '%-4s %-13s %-22s %-5s %-16s %5s %6s %6s  %s\n' tier piece workload bucket cells rows worst expect status
total=0; nwall=0; n=0; nsub=0; seen=" "
SUBMITTED=()
while IFS='|' read -r kind tier piece wl b cells lim1 rows worst exp wall; do
  [ -z "$kind" ] && continue
  if [ "$kind" = "HELD" ]; then
    printf '%-4s %-13s %-22s %-5s %-16s %5s %6s %6s  %s\n' "$tier" "$piece" "$wl" "$b" "$cells" "$rows" - - "$wall"
    continue
  fi
  case "$seen" in *" $piece "*) echo "planner made the tag $piece twice; stopping" >&2; exit 1 ;; esac
  seen="$seen$piece "
  name="dtai3-${wl}_${b}-p$piece"
  if grep -qxF "$name" <<< "$QUEUED"; then
    printf '%-4s %-13s %-22s %-5s %-16s %5s %6s %6s  %s\n' "$tier" "$piece" "$wl" "$b" "$cells" "$rows" "$worst" "$exp" "already queued, skipped"
    continue
  fi
  if done_already "$wl" "$b" "$cells" "$piece"; then
    printf '%-4s %-13s %-22s %-5s %-16s %5s %6s %6s  %s\n' "$tier" "$piece" "$wl" "$b" "$cells" "$rows" "$worst" "$exp" "results exist, skipped"
    continue
  fi
  printf '%-4s %-13s %-22s %-5s %-16s %5s %6s %6s  %s\n' "$tier" "$piece" "$wl" "$b" "$cells" "$rows" "$worst" "$exp" "ok, walltime $wall"
  total=$(awk -v t="$total" -v e="$exp" 'BEGIN { print t + e }'); n=$((n + 1))
  [ "$wall" = "01:00:00" ] && nwall=$((nwall + 1))
  if [ "$GO" = "1" ]; then
    # env -u: a DeepSeek key exported in this shell for gen_router_ai.py must
    # not travel into the job's environment (--export=ALL would carry it).
    if ! jid=$(cd "$HERE" && PIECE="$piece" WL="$wl" B="$b" CELLS="$cells" LIM1="$lim1" \
          env -u DEEPSEEK_API_KEY sbatch --parsable --partition="$PART" --account="$ACCOUNT" \
            --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g \
            --time="$wall" --job-name="$name" \
            --output="$LADDER_ROOT/logs/dtai3-$piece-%j.out" --export=ALL dtai3_job.sh); then
      echo "sbatch refused $piece; stopping. Submitted before it:" >&2
      for s in "${SUBMITTED[@]}"; do echo "  $s" >&2; done
      exit 1
    fi
    SUBMITTED+=("$jid $piece"); nsub=$((nsub + 1))
    sleep "${PACE:-0.5}"   # do not flood a busy controller
  fi
done <<< "$PLAN"
echo
echo "$n pieces, about $(awk -v t="$total" 'BEGIN { printf "%d h %d min", t / 60, t % 60 }') of GPU time expected (each piece asks for 1 GPU); $nwall need a 1-hour limit."
if [ "$GO" = "1" ]; then
  echo "submitted $nsub:"; for s in "${SUBMITTED[@]}"; do echo "  $s"; done
  echo "logs: $LADDER_ROOT/logs/dtai3-<piece>-<job>.out   results: $LADDER_ROOT/results/gh200r_*"
else
  echo "plan only. Add GO=1 to submit."
fi
