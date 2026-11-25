#!/bin/bash
# Submit the router data job set (dtai3), one Slurm job per piece. Prints the
# plan and submits nothing unless GO=1.
#
#   cd $LADDER_ROOT/repo/code
#   PROJ=<code> ACCOUNT=<name> source deltaai_env.sh
#   bash submit_dtai3.sh TIERS=S                 # the plan for the smoke piece
#   bash submit_dtai3.sh TIERS=S GO=1            # run it first (about 15 minutes)
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
#      and the padding test (real padding and repeated-sentence padding at 8K)
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
S|smoke|qa|2K|t a|4
0|qa2|qa|2K|t a|
0|qa8|qa|8K|t a|
0|qa16t|qa|16K|t|
0|qa16a|qa|16K|a|
0|qa32t|qa|32K|t|
0|qa32a|qa|32K|a|
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
0|code2|code|2K|t a|
0|code8|code|8K|t a|
0|code16t|code|16K|t|
0|code16a|code|16K|a|
0|code32t|code|32K|t|
0|code32a|code|32K|a|
0|summ2|summ|2K|t a|
0|summ8|summ|8K|t a|
0|summ16t|summ|16K|t|
0|summ16a|summ|16K|a|
0|summ32t|summ|32K|t|
0|summ32a|summ|32K|a|
0|fewshot2|fewshot|2K|t a|
0|fewshot8|fewshot|8K|t a|
0|fewshot16t|fewshot|16K|t|
0|fewshot16a|fewshot|16K|a|
0|fewshot32t|fewshot|32K|t|
0|fewshot32a|fewshot|32K|a|
0|mdqa2|mdqa|2K|t a|
0|mdqa8|mdqa|8K|t a|
0|mdqa16t|mdqa|16K|t|
0|mdqa16a|mdqa|16K|a|
0|mdqa32t|mdqa|32K|t|
0|mdqa32a|mdqa|32K|a|
1|rq2|ragquote|2K|t a|62
1|rq8|ragquote|8K|t a|62
1|rq16t|ragquote|16K|t|62
1|rq16a|ragquote|16K|a|62
1|rq32t|ragquote|32K|t|46
1|rq32a|ragquote|32K|a|46
1|ce0|codeedit|nat|t a|
1|ce2|codeedit|2K|t a|
1|ce8|codeedit|8K|t a|
1|ce16t|codeedit|16K|t|
1|ce16a|codeedit|16K|a|
1|chat2|chat|2K|t a|
1|chat8|chat|8K|t a|
1|ag8|agent|8K|t a|
1|ag16t|agent|16K|t|
1|ag16a|agent|16K|a|
1|ag32t|agent|32K|t|
1|ag32a|agent|32K|a|
1|eqmdR|eq_mdqa_realpad|8K|t a|
1|eqmdX|eq_mdqa_repeatpad|8K|t a|
1|eqfsR|eq_fewshot_realpad|8K|t a|
1|eqfsX|eq_fewshot_repeatpad|8K|t a|
2|syn2|synth|2K|t a|58
2|syn8|synth|8K|t a|58
2|syn16t|synth|16K|t|58
2|syn16a|synth|16K|a|58
2|syn32t|synth|32K|t|30
2|syn32a|synth|32K|a|30
2|cea2|codeedit_ai|2K|t a|
2|cea8|codeedit_ai|8K|t a|
2|cea16t|codeedit_ai|16K|t|
2|cea16a|codeedit_ai|16K|a|
2|eqmdL|eq_mdqa_llmpad|8K|t a|
2|eqfsL|eq_fewshot_llmpad|8K|t a|
2|twR|twins_real|nat|t a|
2|twA|twins_ai|nat|t a|
3|code2L|code|2K|c16 c32|
3|summ2L|summ|2K|c16 c32|
3|fewshot2L|fewshot|2K|c16 c32|
3|mdqa2L|mdqa|2K|c16 c32|
3|code8L|code|8K|c16|
3|summ8L|summ|8K|c16|
3|fewshot8L|fewshot|8K|c16|
3|mdqa8L|mdqa|8K|c16|
3|mixL|mix|mix|c8 c16|
3|qa2T|qa|2K|tT aT|
3|qa8Tt|qa|8K|tT|
3|qa8Ta|qa|8K|aT|
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

# A rough time model, only for the plan and the walltime (minutes). Measured
# GH200 batch-1 numbers at 256 tokens put model loading under a minute per arm
# (task pieces: 5 arms of 50 prompts in 7 to 11 minutes); 1.5 is the guess here.
# Expected output lengths in chat format are guesses per workload, to be
# replaced by the smoke piece's and tier 0's measured ones.
est () {  # $1 workload $2 bucket $3 cells $4 lim1 $5 rows -> minutes
  python3 - "$@" <<'EOF'
import sys
wl, b, cells, lim1, rows = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4], int(sys.argv[5])
P = {"nat": 0.05, "2K": 0.15, "8K": 0.6, "16K": 1.4, "32K": 3.5, "mix": 1.8}[b]
tok = {"nat": 0.0075, "2K": 0.0075, "8K": 0.008, "16K": 0.009, "32K": 0.011, "mix": 0.009}[b]
out = {"qa": 450, "code": 150, "summ": 700, "fewshot": 60, "mdqa": 30, "ragquote": 300, "codeedit": 600,
       "codeedit_ai": 450, "chat": 500, "agent": 250, "synth": 400, "twins_real": 500, "twins_ai": 500,
       "mix": 450}.get(wl, 30 if "mdqa" in wl else 60 if "fewshot" in wl else 400)
d1 = 25 if b == "32K" else 50
n1 = min(int(lim1) if lim1 else d1, rows - 2)
dl = {"2K": 128, "8K": 128, "32K": 48}.get(b, 64)
nl = min(dl, rows - 2)
load = 1.5
m = 0.0
for c in cells:
    if c in ("t", "tT"):
        o = 1800 if c == "tT" else out
        n = min(25, rows - 2) if c == "tT" else n1
        per = P + o * tok
        m += 5 * load + n * per * (1 + 4 / 1.6) / 60
    elif c in ("a", "aT"):
        o = 1800 if c == "aT" else out
        n = min(25, rows - 2) if c == "aT" else n1
        m += 2 * load + n * (P + o * tok) * (2 / 1.6) / 60
    elif c.startswith("c"):
        k = int(c[1:])
        arm = nl * P + nl * out * tok / (0.8 * k)
        m += 5 * load + arm * (1 + 4 / 1.3) / 60
print(int(m + 0.999))
EOF
}

printf '%-4s %-11s %-22s %-5s %-16s %5s %6s  %s\n' tier piece workload bucket cells rows est_min status
total=0; n=0; nsub=0
SUBMITTED=()
while IFS='|' read -r tier piece wl b cells lim1; do
  [ -z "$tier" ] && continue
  case " $TIERS " in *" $tier "*) ;; *) continue ;; esac
  if [ -n "${ONLY:-}" ]; then case " $ONLY " in *" $piece "*) ;; *) continue ;; esac; fi
  case " ${SKIP:-} " in *" $piece "*) continue ;; esac
  f=$ROUTER/$wl/$b.jsonl
  if [ ! -s "$f" ]; then
    printf '%-4s %-11s %-22s %-5s %-16s %5s %6s  %s\n' "$tier" "$piece" "$wl" "$b" "$cells" - - "HELD: no $f"
    continue
  fi
  rows=$(grep -c . "$f")
  m=$(est "$wl" "$b" "$cells" "$lim1" "$rows")
  # Walltime: the estimate times 1.3 plus 8 minutes, rounded up to 30, 60 or 120
  # minutes. Short limits start sooner on ghx4 (18 Nov: two 30-minute pieces
  # backfilled while 1-hour ones waited); a limit can be lowered later with
  # scontrol update JobId=<id> TimeLimit=..., never raised.
  need=$(( m * 13 / 10 + 8 ))
  if [ "$need" -le 30 ]; then wall="00:30:00"; elif [ "$need" -le 60 ]; then wall="01:00:00"; else wall="02:00:00"; fi
  status="ok, walltime $wall"
  [ "$rows" -lt 7 ] && status="HELD: only $rows rows"
  printf '%-4s %-11s %-22s %-5s %-16s %5s %6s  %s\n' "$tier" "$piece" "$wl" "$b" "$cells" "$rows" "$m" "$status"
  case "$status" in HELD*) continue ;; esac
  total=$((total + m)); n=$((n + 1))
  if [ "$GO" = "1" ]; then
    jid=$(cd "$HERE" && PIECE="$piece" WL="$wl" B="$b" CELLS="$cells" LIM1="$lim1" \
          sbatch --parsable --partition="$PART" --account="$ACCOUNT" \
            --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g \
            --time="$wall" --job-name="dtai3-${wl}_${b}-p$piece" \
            --output="$LADDER_ROOT/logs/dtai3-$piece-%j.out" --export=ALL dtai3_job.sh)
    SUBMITTED+=("$jid $piece"); nsub=$((nsub + 1))
  fi
done <<< "$PIECES"
echo
echo "$n pieces, about $((total / 60)) h $((total % 60)) min of GPU time by the rough model (each piece asks for 1 GPU)."
if [ "$GO" = "1" ]; then
  echo "submitted $nsub:"; for s in "${SUBMITTED[@]}"; do echo "  $s"; done
  echo "logs: $LADDER_ROOT/logs/dtai3-<piece>-<job>.out   results: $LADDER_ROOT/results/gh200r_*"
  echo "to move waiting 1-hour pieces through ghx4-interactive one at a time: bash dtai3_feeder.sh"
else
  echo "plan only. Add GO=1 to submit."
fi
