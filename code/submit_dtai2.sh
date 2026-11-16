#!/bin/bash
# Plan and (only with GO=1) submit the improved DeltaAI job set, dtai2. Login
# node, no GPU. Each bundle is one GH200 job (dtai2_bundle.sh); read that
# file for what every cell measures and why.
#
# PROPOSED, NOT YET RUN. Written 16 Nov 2026. Rationale, evidence and a
# verification checklist: notes/2026-11-16-dtai2-proposed-jobs.md. Offline
# tests: bash test_dtai2.sh.
#
#   ssh <netid>@dtai-login.delta.ncsa.illinois.edu     # DeltaAI, not Delta
#   cd /work/nvme/<proj>/$USER/Speculative_Ladder/repo/code && git pull
#   PROJ=<code> ACCOUNT=<name> source deltaai_env.sh
#
#   bash submit_dtai2.sh                                      # plan only
#   bash submit_dtai2.sh SMOKE=<dir or results.jsonl>         # plan + smoke check
#   bash submit_dtai2.sh SMOKE=<...> BUNDLES="c1-2K profile" GO=1
#
# The domain bundles need two login-node steps first (minutes, no GPU). The
# document counts are twice make_domains.py's defaults, for router training:
#   "$LADDER_VENV/bin/python3" make_domains.py --src "$LADDER_DATA" --dst "$LADDER_ROOT/data" --n 50 --n32 25 --check
#   "$LADDER_VENV/bin/python3" make_domains.py --src "$LADDER_DATA" --dst "$LADDER_ROOT/data" --n 50 --n32 25
#   "$LADDER_VENV/bin/python3" merge_domains.py
#
# KEY=VALUE arguments (also read from the environment):
#   BUNDLES   default, in this order:
#               c1-2K profile c1-8K c1-32K conc-2K conc-8K conc-32K dom-2K dom-8K dom-32K
#             The first two are the quickest and catch a problem before the
#             long ones spend hours on it. Submit them, read their logs, then
#             submit the rest.
#   SMOKE     the smoke job's results directory or results.jsonl. GO=1 refuses
#             to submit unless it passes (or FORCE=1).
#   FORCE     1 = submit without a passing smoke check. Do not.
#   CELLS     passed to every bundle: run only these cells
#   NSYS      0 = skip the nsys cells in the profile bundle
#   LOOPB     1 = add the second per-prompt timing pass to the c1 jobs. Off by
#             default: decided 16 Nov to run everything once first. To add it
#             later on its own: BUNDLES="c1-2K c1-8K c1-32K" CELLS=loopB LOOPB=1
#   LOAD_MIN  minutes of engine start per arm. Default auto: measured from
#             SMOKE_LOG if given (the slowest arm's wall time less its timed
#             run and warm-up), else 3. Engine start is not scaled by
#             GH200_FACTOR, and at GH200 speed it is a large share of the total.
#   SMOKE_LOG the smoke job's Slurm log (logs/phase0-ksweep-<job>.out)
#   PART      default ghx4. ghx4-interactive charges 2x and allows one job.
#
#   GH200_FACTOR  GH200 time as a fraction of A40 time, for the estimate.
#             Default auto: measured from any GH200 rows already on disk (the
#             smoke job's, or earlier dtai2 jobs'); 1.0 if there are none.
#   BUDGET_H  GPU-hours available, for the printout only. Default 200.
#
# These jobs run on DeltaAI's GH200 nodes only. The cost estimate is the one
# place A40 numbers appear: no GH200 timing exists yet, so it starts from the
# filed Delta A40 timings by bucket and concurrency (results/a40 and
# results/a40_c10, slowest arm of each cell) and scales them by GH200_FACTOR.
# Until a GH200 row exists that is an A40-speed ceiling, and the GH200 should
# be faster. Slurm charges for time used, not for the walltime requested.

set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "${kv%%=*}"="${kv#*=}" ;;
    *) echo "arguments are KEY=VALUE, got '$kv'" >&2; exit 1 ;;
  esac
done

HERE=$(cd "$(dirname "$0")" && pwd)
GO=${GO:-0}
BUDGET_H=${BUDGET_H:-200}
FORCE=${FORCE:-0}
PART=${PART:-ghx4}
BUNDLES=${BUNDLES:-"c1-2K profile c1-8K c1-32K conc-2K conc-8K conc-32K dom-2K dom-8K dom-32K"}
LOAD_MIN=${LOAD_MIN:-auto}

# ------------------------------------------------------------ environment
miss=0
for v in LADDER_ROOT LADDER_VENV LADDER_HF LADDER_DATA LADDER_CACHE ACCOUNT; do
  [ -n "${!v:-}" ] || { echo "missing $v" >&2; miss=1; }
done
if [ "$miss" = "1" ]; then
  echo "Run first:  PROJ=<code> ACCOUNT=<name> source deltaai_env.sh" >&2
  exit 1
fi
# DeltaAI, not Delta: its login nodes are aarch64 like its GH200 nodes.
if [ "$(uname -m)" != "aarch64" ] && [ "${LADDER_TEST_HOST:-0}" != "1" ]; then
  echo "This is a $(uname -m) machine. Submit from a DeltaAI login node (dtai-login.delta.ncsa.illinois.edu), not Delta." >&2
  exit 1
fi
[ "${LADDER_CAPS:-}" = "90" ] || { echo "LADDER_CAPS=${LADDER_CAPS:-unset}; on DeltaAI it is 90. source deltaai_env.sh" >&2; exit 1; }
for v in LADDER_ROOT LADDER_VENV LADDER_HF LADDER_DATA LADDER_CACHE LADDER_DOMDATA LADDER_RESULTS; do
  case "${!v:-}" in
    /projects/*) echo "$v=${!v} is under /projects, which a ghx4 job may not see. Use the copy under /work/nvme." >&2; exit 1 ;;
  esac
done
[ -f "$LADDER_VENV/ladder_modules" ] || { echo "no $LADDER_VENV/ladder_modules: use the venv install_deltaai.sh built" >&2; exit 1; }
if [ "${LADDER_MRV2:-0}" != "1" ]; then
  echo "LADDER_MRV2=${LADDER_MRV2:-unset}: these bundles are for vLLM 0.33.0 on Model Runner V2." >&2
  exit 1
fi
for f in dtai2_bundle.sh ksweep.sh run_benchmark.v5c.py nsys_v5.py; do
  [ -f "$HERE/$f" ] || { echo "missing $HERE/$f (git pull?)" >&2; exit 1; }
done
grep -q 'HARNESS_VERSION = "v5c' "$HERE/run_benchmark.v5c.py" || { echo "run_benchmark.v5c.py is not harness v5c" >&2; exit 1; }
[ -x "$LADDER_VENV/bin/python3" ] || { echo "no venv at $LADDER_VENV (point LADDER_VENV at the venv the smoke job used)" >&2; exit 1; }
PY=$LADDER_VENV/bin/python3   # the login node's own python3 may be old

# Prompts needed per bucket: the c=1 cells read 50 / 50 / 25, the
# concurrency cells 128 / 128 / 48 (checked only if a conc bundle is asked
# for), each plus two that harness v5c warms up on and does not measure.
need () {
  case " $BUNDLES " in
    *" conc-$1 "*) case "$1" in 2K|8K) echo 130 ;; 32K) echo 50 ;; esac ;;
    *)             case "$1" in 2K|8K) echo 52 ;;  32K) echo 27 ;; esac ;;
  esac
}
for b in 2K 8K 32K; do
  f=$LADDER_DATA/longbench_v2_${b}.jsonl
  [ -s "$f" ] || { echo "no workload $f" >&2; exit 1; }
  have=$(grep -c . "$f")
  [ "$have" -ge "$(need $b)" ] || { echo "$f holds $have prompts; the bundles asked for need $(need $b)" >&2; exit 1; }
done
mkdir -p "$LADDER_ROOT/logs" "$LADDER_ROOT/results"
DOMDATA=${LADDER_DOMDATA:-$LADDER_ROOT/data/domains_balanced}

echo "venv     $LADDER_VENV"
echo "account  $ACCOUNT   partition $PART   user $USER"
echo "results  $LADDER_ROOT/results/gh200v2_*"
echo

# ------------------------------------------------------------ smoke check
SMOKE_OK=0
if [ -n "${SMOKE:-}" ]; then
  SF=$SMOKE
  [ -d "$SF" ] && SF=$SF/results.jsonl
  if [ ! -s "$SF" ]; then
    echo "SMOKE: no rows at $SF"
  else
    if "$PY" - "$SF" <<'EOF'
import json, sys
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
v5 = [r for r in rows if str(r.get("harness", "")).startswith("v5")]
if not v5:
    print("SMOKE: no harness-v5 rows"); sys.exit(1)
bad = [r.get("run_id") for r in v5 if (r.get("resolved") or {}).get("use_v2_model_runner") is not True]
asyncs = {str((r.get("resolved") or {}).get("async_scheduling")) for r in v5}
rungs = sorted({r.get("rung") for r in v5})
timed = sum(int(r.get("per_request_engine_timing", r.get("per_request_timing")) or 0) for r in v5)
print(f"SMOKE: {len(v5)} v5 rows, rungs {rungs}, async states {sorted(asyncs)}")
print(f"SMOKE: requests with engine timing: {timed}"
      + ("  (per-request prefill and decode time ARE recorded)" if timed else
         "  (none. Harness v5 reads field names vLLM 0.33.0 does not have; run the"
         " smoke job on v5c, LADDER_RUN=$PWD/run_benchmark.v5c.py, to see them)"))
ok = True
vers = sorted({str((r.get("env") or {}).get("vllm")) for r in v5})
if any(not v.startswith("0.33.0") for v in vers):
    print(f"SMOKE: FAIL, vLLM versions {vers}; the dtai2 set is written for 0.33.0"); ok = False
gpus = sorted({str((r.get("env") or {}).get("gpu")) for r in v5})
if any("GH200" not in g and "H100" not in g for g in gpus):
    print(f"SMOKE: WARNING, GPU names {gpus}: expected a GH200 (H100)")
for r in v5:
    if r.get("rung") != "off" and not (r.get("spec") or {}).get("drafts"):
        print(f"SMOKE: FAIL, {r.get('run_id')} counted no drafts: the drafter did not run"); ok = False
if bad:
    print(f"SMOKE: FAIL, not on Model Runner V2: {bad}"); ok = False
if len(asyncs) != 1:
    print("SMOKE: FAIL, arms resolved to different async-scheduling states"); ok = False
for need in ("off", "dflash"):
    if need not in rungs:
        print(f"SMOKE: FAIL, no {need} row"); ok = False
if not ({"ngram", "ngram_gpu"} & set(rungs)):
    print("SMOKE: FAIL, no n-gram row"); ok = False
sys.exit(0 if ok else 1)
EOF
    then SMOKE_OK=1; echo "SMOKE: pass"; else echo "SMOKE: did not pass"; fi
  fi
  echo
fi

case " ${CELLS:-} " in
  *" loopB "*) [ "${LOOPB:-0}" = "1" ] || { echo "CELLS asks for loopB: add LOOPB=1" >&2; exit 1; } ;;
esac

# ------------------------------------------------------------ GH200 timing
# A40 off-arm seconds per prompt at c=1, 256 tokens: 7.86 / 9.25 / 17.2.
FACTOR=${GH200_FACTOR:-auto}
FACTOR_SRC="set by GH200_FACTOR"
if [ "$FACTOR" = "auto" ]; then
  CAL=$("$PY" - "${SMOKE:-}" "$LADDER_ROOT/results" <<'EOF' 2>/dev/null || true
import glob, json, os, sys
smoke, root = sys.argv[1], sys.argv[2]
files = glob.glob(os.path.join(root, "gh200*", "results.jsonl"))
if smoke:
    files.append(os.path.join(smoke, "results.jsonl") if os.path.isdir(smoke) else smoke)
a40 = {"2K": 7.86, "8K": 9.25, "32K": 17.2}
ratios, used = [], set()
for f in files:
    try:
        for line in open(f):
            r = json.loads(line)
            if (r.get("rung") == "off" and r.get("max_num_seqs") == 1 and r.get("max_tokens") == 256
                    and not r.get("loop_timing") and r.get("bucket") in a40 and r.get("limit")
                    and any(g in str((r.get("env") or {}).get("gpu", "")) for g in ("GH200", "H100"))):
                ratios.append((r["elapsed_s"] / r["limit"]) / a40[r["bucket"]])
                used.add(os.path.basename(os.path.dirname(f)))
    except Exception:
        pass
if ratios:
    print("%.3f %d %s" % (max(ratios), len(ratios), ",".join(sorted(used))))
EOF
)
  if [ -n "$CAL" ]; then
    read -r FACTOR n_rows src <<< "$CAL"
    FACTOR_SRC="measured: slowest of $n_rows GH200 off rows ($src)"
  else
    FACTOR=1.0
    FACTOR_SRC="no GH200 rows on disk yet: A40-speed ceiling"
  fi
fi

# ------------------------------------------------------------ engine start
# Per-arm overhead from the smoke job's log: ksweep.sh stamps each arm's start
# ("### arm N ... HH:MM:SS") and prints `date` after the last one. Overhead =
# the arm's wall time, less its timed run, less its two warm-up prompts
# (priced at the timed run's seconds per prompt). The slowest arm is used.
LOAD_SRC="set by LOAD_MIN"
if [ "$LOAD_MIN" = "auto" ]; then
  LOAD_MIN=3
  LOAD_SRC="assumed; pass SMOKE_LOG=<the smoke job's log> to measure it"
  if [ -n "${SMOKE_LOG:-}" ] && [ -n "${SMOKE:-}" ]; then
    LM=$("$PY" - "$SMOKE_LOG" "$SMOKE" <<'EOF' 2>/dev/null || true
import json, math, os, re, sys
log, smoke = sys.argv[1], sys.argv[2]
sf = os.path.join(smoke, "results.jsonl") if os.path.isdir(smoke) else smoke
hms = lambda s: sum(int(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
starts, end = [], None
for line in open(log, errors="replace"):
    m = re.match(r"### arm \d+ .*\s(\d\d:\d\d:\d\d)\s*$", line)
    if m:
        starts.append(hms(m.group(1))); end = None; continue
    m = re.match(r"[A-Z][a-z]{2} [A-Z][a-z]{2} +\d+ (\d\d:\d\d:\d\d) ", line)
    if m and starts:
        end = hms(m.group(1))
rows = [json.loads(l) for l in open(sf) if l.strip()]
if starts and end is not None and len(rows) >= len(starts):
    rows = rows[-len(starts):]
    edges = starts + [end]
    over = []
    for i, r in enumerate(rows):
        wall = (edges[i + 1] - edges[i]) % 86400
        per = r["elapsed_s"] / max(1, r.get("limit") or 1)
        over.append(wall - r["elapsed_s"] - 2 * per)
    print("%d %d" % (max(1, math.ceil(max(over) / 60.0)), len(over)))
EOF
)
    if [ -n "$LM" ]; then
      read -r LOAD_MIN n_arms <<< "$LM"
      LOAD_SRC="measured: slowest of $n_arms smoke arms ($SMOKE_LOG)"
    else
      LOAD_SRC="assumed; could not read arm times from $SMOKE_LOG"
    fi
  fi
fi

# ------------------------------------------------------------ plan
# Seconds per prompt, slowest arm, filed A40 rows, by bucket and concurrency,
# scaled by FACTOR below.
# c=2 at 2K and 8K and c=16 at 8K are interpolated.
spp () {  # bucket conc kind -> seconds per prompt
  case "$3" in
    pre)  case "$1" in 2K) echo 0.4 ;; 8K) echo 1.5 ;; 32K) echo 8.5 ;; esac; return ;;
    t64)  echo 12 ; return ;;
    t512) echo 38 ; return ;;
    nsys) case "$1" in 2K) echo 5 ;; *) echo 20 ;; esac; return ;;
  esac
  case "$1:$2" in
    2K:1) echo 7.9 ;;  2K:2) echo 4.5 ;;  2K:4) echo 2.6 ;;  2K:8) echo 1.5 ;;
    2K:16) echo 1.1 ;; 2K:32) echo 0.95 ;;
    8K:1) echo 9.6 ;;  8K:2) echo 6.0 ;;  8K:4) echo 3.8 ;;  8K:8) echo 3.0 ;; 8K:16) echo 2.6 ;;
    32K:1) echo 22.7 ;; 32K:2) echo 14.8 ;; 32K:4) echo 13.4 ;; 32K:8) echo 12.5 ;;
    *) echo "no timing for $1 at c=$2" >&2; echo 30 ;;
  esac
}
cellmin () {  # arms bucket conc prompts kind [factor] -> minutes
  local load=$LOAD_MIN f=${6:-$FACTOR}
  [ "$5" = "nsys" ] && load=$((load + 2))
  # n + 2: every arm also runs two warm-up prompts before its timed run.
  awk -v a="$1" -v s="$(spp "$2" "$3" "$5")" -v f="$f" -v n="$4" -v l="$load" \
      'BEGIN { printf "%d", a * (l + (n + 2) * s * f / 60) + 0.5 }'
}
# One line per cell: name arms bucket conc prompts kind
cells_of () {
  local b=${1#*-}
  case "$1" in
    c1-*)
      local n=50; [ "$b" = "32K" ] && n=25
      echo "core 12 $b 1 $n dec"; echo "pre 4 $b 1 $n pre"; echo "loopA 12 $b 1 $n dec"
      echo "acc 7 $b 1 $n dec";   echo "k0 2 $b 1 $n dec"
      [ "$b" = "32K" ] && echo "nopc 4 $b 1 $n dec"
      [ "${LOOPB:-0}" = "1" ] && echo "loopB 12 $b 1 $n dec"
      echo "rep 4 $b 1 $n dec" ;;
    conc-2K)  echo "pre1 3 2K 1 128 pre"; for c in 1 4 8 16 32; do echo "c$c 7 2K $c 128 dec"; done
              echo "preTop 3 2K 32 128 pre" ;;
    conc-8K)  echo "pre1 3 8K 1 128 pre"; for c in 1 4 8 16;    do echo "c$c 7 8K $c 128 dec"; done
              echo "preTop 3 8K 16 128 pre" ;;
    conc-32K) echo "pre1 3 32K 1 48 pre"; for c in 1 2 4 8;     do echo "c$c 7 32K $c 48 dec"; done
              echo "preTop 3 32K 8 48 pre" ;;
    dom-*)
      # Without the file, an upper bound: six domains at --n 50 / --n32 25.
      local n=300; [ "$b" = "32K" ] && n=150
      local f=$DOMDATA/longbench_v2_$b.jsonl
      [ -s "$f" ] && n=$(( $(grep -c . "$f") - 2 ))   # the last two are the warm-up
      echo "domloop 4 $b 1 $n dec"; echo "domacc 7 $b 1 $n dec" ;;
    profile)
      echo "t64 3 32K 1 10 t64"; echo "t512 3 32K 1 10 t512"
      echo "nsys2K 2 2K 1 3 nsys"; echo "nsys32K 3 32K 1 3 nsys" ;;
    *) echo "unknown bundle '$1'" >&2; return 1 ;;
  esac
}

TOTAL=0
PLAN=()
DOM_MISSING=()
printf "%-9s %-72s %8s %8s\n" bundle "cells (arms)" "estimate" walltime
for bn in $BUNDLES; do
  case "$bn" in
    c1-2K|c1-8K|c1-32K|conc-2K|conc-8K|conc-32K|dom-2K|dom-8K|dom-32K|profile) ;;
    *) echo "unknown bundle '$bn'. One of c1-2K c1-8K c1-32K conc-2K conc-8K conc-32K dom-2K dom-8K dom-32K profile." >&2; exit 1 ;;
  esac
  lines=$(cells_of "$bn") || exit 1
  m=0; mc=0; desc=""
  while read -r name arms b c n kind; do
    [ -n "${CELLS:-}" ] && case " $CELLS " in *" $name "*) ;; *) continue ;; esac
    m=$((m + $(cellmin "$arms" "$b" "$c" "$n" "$kind")))
    mc=$((mc + $(cellmin "$arms" "$b" "$c" "$n" "$kind" 1.0)))
    desc="$desc $name($arms)"
  done <<< "$lines"
  # Walltime from the A40-speed ceiling whatever the factor: a factor measured
  # on a few smoke prompts must not be able to make a job time out. A longer
  # request costs nothing extra; Slurm charges for time used.
  h=$(awk -v m="$mc" 'BEGIN { h = int(m * 1.6 / 60) + 1; if (h > 47) h = 47; print h }')
  # The profiler's slowdown with CUDA-graph node tracing has not been measured.
  [ "$bn" = "profile" ] && [ "${NSYS:-1}" = "1" ] && [ "$h" -lt 4 ] && h=4
  note=""
  case "$bn" in
    dom-*) [ -s "$DOMDATA/longbench_v2_${bn#*-}.jsonl" ] || { note="  NO DOMAIN FILE"; DOM_MISSING+=("$bn"); } ;;
  esac
  TOTAL=$((TOTAL + m))
  PLAN+=("$bn|$h")
  printf "%-9s %-72s %5d min %5d h%s\n" "$bn" "${desc# }" "$m" "$h" "$note"
done
echo
awk -v m="$TOTAL" -v n="${#PLAN[@]}" -v f="$FACTOR" -v b="$BUDGET_H" 'BEGIN {
  printf "%d DeltaAI jobs, estimate %.1f GPU-hours (one GH200 each, charge factor 1.0), %.0f%% of %s.\n", n, m / 60, 100 * m / 60 / b, b
  if (f == 1.0) printf "No GH200 timing on disk yet, so this is the A40-speed ceiling; GH200 should be faster.\n" }'
echo "GH200/A40 time factor: $FACTOR ($FACTOR_SRC). Override with GH200_FACTOR=<x>."
echo "Engine start per arm: $LOAD_MIN min ($LOAD_SRC). Not scaled by the factor."
echo "Check the balance first:  accounts"
if [ "${#DOM_MISSING[@]}" -gt 0 ]; then
  echo
  echo "Domain files missing for: ${DOM_MISSING[*]}. Build them first (login node, minutes):"
  echo "  \"\$LADDER_VENV/bin/python3\" make_domains.py --src \"\$LADDER_DATA\" --dst \"\$LADDER_ROOT/data\" --n 50 --n32 25 --check"
  echo "  \"\$LADDER_VENV/bin/python3\" make_domains.py --src \"\$LADDER_DATA\" --dst \"\$LADDER_ROOT/data\" --n 50 --n32 25"
  echo "  \"\$LADDER_VENV/bin/python3\" merge_domains.py"
fi
case " $BUNDLES " in
  *" profile "*)
    if [ "${NSYS:-1}" = "1" ]; then
      found=$( (module -t avail 2>&1 || true) | grep -i -E "nsight|cudatoolkit|nvhpc|^cuda" | head -5 | tr '\n' ' ' || true)
      echo
      echo "nsys on this login node: $(command -v nsys || echo 'not on PATH')   modules that may carry it: ${found:-none listed}"
      echo "If neither shows it, run 'module spider nsight' and pass NSYS_MODULES=<name>, or the profile bundle skips its nsys cells."
    fi ;;
esac

if [ "$GO" != "1" ]; then
  echo
  echo "Nothing submitted. Add GO=1 (and SMOKE=<dir>) to submit, then log every job id in LOG.md."
  exit 0
fi
if [ "$SMOKE_OK" != "1" ] && [ "$FORCE" != "1" ]; then
  echo
  echo "REFUSED: no passing smoke check. Pass SMOKE=<the smoke job's results dir> after reading its log."
  exit 1
fi

echo
cd "$HERE"
for p in "${PLAN[@]}"; do
  bn=${p%%|*}; h=${p#*|}
  case " ${DOM_MISSING[*]:-} " in
    *" $bn "*) echo "NOT SUBMITTED: $bn (no domain file)"; continue ;;
  esac
  JID=$(BUNDLE=$bn sbatch --parsable --partition="$PART" --account="$ACCOUNT" \
          --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g \
          --time="$(printf '%02d' "$h"):00:00" --job-name="dtai2-$bn" \
          --output="$LADDER_ROOT/logs/dtai2-$bn-%j.out" --export=ALL dtai2_bundle.sh)
  echo "submitted $JID  bundle $bn  walltime ${h}h  log $LADDER_ROOT/logs/dtai2-$bn-$JID.out"
done
echo
echo "watch:   squeue -u $USER      (PD with QOSMaxJobsPerUserLimit = waiting its turn)"
echo "after:   sacct -j <id> --format=JobID,State,Elapsed,NodeList%12,ExitCode -X"
echo "         \"\$LADDER_VENV/bin/python3\" ksweep_table.py $LADDER_ROOT/results/gh200v2_ksweep_<bucket>"
