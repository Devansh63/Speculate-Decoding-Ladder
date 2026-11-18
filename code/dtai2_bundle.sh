#!/bin/bash
# The improved DeltaAI job set ("dtai2"). One job = one bundle of k-sweep
# cells on one GH200, run back to back. Submit with submit_dtai2.sh, never
# with a bare sbatch: the account, partition, memory and log path come from
# there.
#
# PROPOSED, NOT YET RUN. The full rationale, the evidence behind every cell
# and a verification checklist are in notes/2026-11-16-dtai2-proposed-jobs.md.
# Offline tests: bash test_dtai2.sh (stubs only, no GPU, no network).
#
# It grew out of a smaller first draft of four bundles that was never pushed;
# this set supersedes it. Results go to results/gh200v2_*, which nothing else
# writes, so these rows never mix with the gh200_* rows of submit_ksweep.sh
# (Sahil's smoke job and sweeps).
#
# Written 16 Nov 2026 for vLLM 0.33.0 on Model Runner V2. Tested only against
# stub scripts, not on DeltaAI.
#
# Revised 16 Nov 2026 after the outside review (the note's section 10):
#   - every cell runs on harness v5c (per-request engine timing under vLLM
#     0.33.0's field names; warm-up on two prompts that are not measured;
#     stricter k-schedule checks). The nsys cells run v5c under the profiler.
#   - every results directory carries its bucket (gh200v2_ksweep_2K, ...).
#     ksweep.sh counts the rows of results.jsonl before and after a cell, so
#     two jobs appending to one file made both report INCOMPLETE, and RULES
#     section 3 warns that two writers can interleave lines.
#   - SEED TEMP PLMIN PLMAX and LADDER_WARMUP are cleared for every cell.
#   - the domain cells measure the file's prompts less the last two, which
#     are the warm-up.
#   The directory names in the bundle list below are prefixes: read each as
#   <name>_<bucket>.
#
# DeltaAI only: GH200 nodes (aarch64 Grace CPU, H100 GPU, sm_90), partition
# ghx4. Delta's A40 and A100 nodes cannot run it: the venv is aarch64, the
# vLLM is 0.33.0 on Model Runner V2, and ksweep.sh refuses any GPU whose
# compute capability is not in LADDER_CAPS=90. ksweep.sh still carries
# Delta's #SBATCH lines, but they are comments here: bundles run it with
# bash inside their own allocation, never with sbatch. The only A40 numbers
# in this job set are in submit_dtai2.sh's cost estimate, used until GH200
# timings exist.
#
# ------------------------------------------------------------------ knobs
#
#   BUNDLE   c1-2K c1-8K c1-32K | conc-2K conc-8K conc-32K |
#            dom-2K dom-8K dom-32K | task-2K task-8K | profile (mandatory)
#   CELLS    run only these cells of the bundle, e.g. "core rep"   default all
#   NSYS     0 skips the nsys cells of the profile bundle          default 1
#   LOOPB    1 adds the second per-prompt timing pass to c1 jobs   default 0
#   PIECE, ONLY_NGRAM_GPU, ONLY_DFLASH   run part of a bundle as its own job,
#            into directories suffixed _p<PIECE> (see "pieces" below)
#   LADDER_DOMDATA  the merged domain files (merge_domains.py)
#   LADDER_TASKDATA the task-type files (build_task_workloads.py)
#                   default $LADDER_ROOT/data/tasks
#                   default $LADDER_ROOT/data/domains_balanced
#
# Mandatory environment, exported by deltaai_env.sh and passed through by
# sbatch --export=ALL: LADDER_ROOT LADDER_VENV LADDER_MRV2=1 LADDER_CAPS
# LADDER_HF LADDER_DATA.
#
# ------------------------------------------------------------------ arm sets
#
# ksweep.sh always runs off first. Rungs on V2 are ngram_gpu and dflash.
#   FULL     ngram_gpu k = 1 2 5 7 11; dflash k = 1 2 4 7 11 15     12 arms
#   REDUCED  ngram_gpu k = 5; dflash k = 7 15                        4 arms
#   CONCSET  ngram_gpu k = 2 5; dflash k = 2 4 7 15                  7 arms
#   PRESET   ngram_gpu k = 5; dflash k = 7                           3 arms
#
# Revised 16 Nov 2026 (same day) for router training and the oracle:
#   - per-prompt timing over the FULL k list (loopA; a second pass, loopB,
#     is defined but off unless LOOPB=1);
#   - acceptance passes widened from REDUCED to CONCSET;
#   - the domain files built with twice the documents (see submit_dtai2.sh);
#   - prefill-only cells on each concurrency job's own prompt set.
#
# ------------------------------------------------------------------ bundles
#
# c1-<B>, concurrency 1, the bucket's usual 50 / 50 / 25 prompts:
#   core  FULL                                   -> results/gh200v2_ksweep
#   pre   REDUCED at max_tokens 1                -> results/gh200v2_prefill
#         With one output token there is no decode step, so each arm's time
#         is its prefill (DFlash's included). Removes the prefill dilution
#         found in the 15 Nov check of C23: per-step costs then come out of
#         the k sweep by subtraction, not by a fit.
#   loopA FULL, per-prompt latency, first pass   -> results/gh200v2_loopA
#         Every setting the router could pick, timed on every prompt: the
#         per-prompt oracle over the router's whole choice set.
#   acc   CONCSET, per-request acceptance        -> results/gh200v2_acc
#         Per-prompt acceptance at the DFlash k a router might choose. It
#         cannot be inferred from k=7: DFlash accepts more at early positions
#         when it drafts more (job 22569403, k=15 against k=7).
#   k0    off + DFlash with the k schedule 1-1024:0
#                                                -> results/gh200v2_k0
#         DFlash keeps its drafter's context K/V current every step and never
#         drafts. k0 minus off is the cost of carrying an idle DFlash drafter:
#         its context build over the prompt at prefill (the same quantity as
#         pre(dflash) minus pre(off); subtract it), the target's auxiliary
#         hidden states every step, and the per-step context write, which here
#         runs eagerly for one row per step and in a k=7 arm runs inside the
#         draft graph for 1+k rows. An upper-bound diagnostic for upkeep, not
#         the upkeep term of the cost model.
#   nopc  (32K only) REDUCED, prefix caching off -> results/gh200v2_nopc
#   loopB OFF BY DEFAULT (LOOPB=1). FULL, per-prompt latency, second pass
#                                                -> results/gh200v2_loopB
#         The same as loopA. Pick each prompt's winner on one pass and score it
#         with the other, so the oracle is not inflated by taking the minimum
#         of noisy timings; A against B is also the per-prompt timing noise.
#         Decided 16 Nov to run everything once first and add this only if
#         the first results call for it: rep against core differs by more
#         than about 0.5% on some arm, the GPU clock log is not flat, or many
#         prompts (a fifth or more) have their top two settings within about
#         1%. Then run it on its own, on any node, any day:
#           bash submit_dtai2.sh BUNDLES="c1-2K c1-8K c1-32K" CELLS=loopB LOOPB=1 ...
#   rep   REDUCED, settings identical to core    -> results/gh200v2_rep
#         Last in the job on purpose: rep against core is a repeat across the
#         whole job (no speculative arm has ever been run twice at one
#         setting). acc against rep on their shared arms (off, ngram_gpu 5,
#         dflash 7 and 15) is the cost of per-request logging.
#
# conc-<B>, one prompt set per bucket at every concurrency, CONCSET arms:
#   2K  128 prompts at c = 1 4 8 16 32           -> results/gh200v2_conc
#   8K  128 prompts at c = 1 4 8 16
#   32K  48 prompts at c = 1 2 4 8   (c=16 at 32K does not fit in KV memory)
#   About 4 to 8 rounds of requests at the top concurrency, so the steady
#   state is not swamped by the ramp and the drain. 50 prompts at c=16, the
#   first draft's design, is about three rounds.
#   pre1, preTop   PRESET at max_tokens 1 on the same prompts, at c=1 and at
#                  the job's top concurrency -> results/gh200v2_conc_prefill
#   Splits every batched time into prompt reading and generation, which is
#   the per-step cost curve against batch size that the cost model needs.
#
# dom-<B>, the domain-balanced file from merge_domains.py (built from
# make_domains.py --n 50 --n32 25), every prompt:
#   domloop  REDUCED, per-prompt latency         -> results/gh200v2_dom_loop
#   domacc   CONCSET, per-request acceptance     -> results/gh200v2_dom_acc
#   Every per-request record carries its prompt's domain; split by it later.
#   domloop stays at four settings: it answers the domain question. The
#   router's oracle over every setting comes from loopA and loopB.
#
# task-<B> (added 18 Nov), task-type workloads from build_task_workloads.py,
# 50 prompts each at exactly <B> tokens, concurrency 1, every request run to
# 256 tokens (IGNORE_EOS=1, so arms that change the text still generate the
# same number of tokens), the whole prompt in the file (PSTYLE=context):
#   code     code completion (LongBench lcc, repobench-p)   -> gh200v2_tcode_<B>
#   summ     report summarization (gov_report)              -> gh200v2_tsumm_<B>
#   fewshot  few-shot dialogue summarization (samsum)       -> gh200v2_tfewshot_<B>
#   mdqa     multi-document QA (hotpotqa)                   -> gh200v2_tmdqa_<B>
#   Each with TASKSET: off, GPU n-gram k 5 and 11, DFlash k 7 and 15 (the best
#   k of each drafter at 2K and 8K in the core cells, and a middle one).
#   32K is not offered: none of these tasks has 25 documents that long
#   without padding.
#
# profile, all at concurrency 1:
#   t64, t512  32K, 10 prompts, max_tokens 64 and 512, off + ngram_gpu 5 +
#              dflash 7          -> results/gh200v2_outlen_t64 / _t512
#   nsys2K     3 prompts, max_tokens 64, off + dflash 7, under Nsight Systems
#   nsys32K    the same plus ngram_gpu 5 (its lookup also grows with context)
#                                -> results/gh200v2_nsys
#
# Every job also writes, next to its log:
#   logs/dtai2-<bundle>-<job>-gpu.csv   clocks, power, temperature every 5 s.
#     GH200 shares one power budget between the Grace CPU and the GPU, so a
#     clock drop mid-job would show up here and nowhere else.
#   results/gh200v2_meta/<bundle>_j<job>.json   the DFlash checkpoint's
#     architectures and causal flags, and which of the four V2 DFlash
#     speculator classes vLLM 0.33.0's dispatch rule picks for it.
#
# run_id carries _c, _loop, _nopc, _prs and _ks, but NOT max_tokens, which is
# why the prefill and output-length cells have their own directories. loopA
# and loopB share run_ids by design, so they have one directory each.

set -uo pipefail

BUNDLE=${BUNDLE:-}
case "$BUNDLE" in
  c1-2K|c1-8K|c1-32K|conc-2K|conc-8K|conc-32K|dom-2K|dom-8K|dom-32K|task-2K|task-8K|profile) ;;
  *) echo "ABORT: BUNDLE='$BUNDLE'. One of c1-2K c1-8K c1-32K conc-2K conc-8K conc-32K dom-2K dom-8K dom-32K task-2K task-8K profile."
     exit 1 ;;
esac

# ------------------------------------------------------------ pieces
# Added 16 Nov 2026 evening, after 13 hours in which nothing asking for 2 h or
# more started on ghx4 (2747 jobs waiting) while both 1 h jobs started within
# 20 minutes. A piece is part of a bundle sized to fit a 1 h walltime:
#   PIECE=<tag>              letters and digits. Every results directory of
#                            this job gets the suffix _p<tag>, so pieces that
#                            run at the same time never append to one file.
#                            Give every piece of a bundle its own tag.
#   ONLY_NGRAM_GPU="<ks>"    keep only these GPU n-gram k values of each cell
#   ONLY_DFLASH="<ks>"       keep only these DFlash k values of each cell
#                            ("none" drops the rung; unset keeps the cell's list)
# Each piece still runs its own off arm first (ksweep.sh always does), so every
# speedup in a piece divides by an off arm timed on the same GPU in the same
# job. A cell the filters leave with no speculative arm is skipped, not run as
# off alone. Splitting a bundle by CELLS across parallel jobs also needs a
# PIECE per job: the conc cells share gh200v2_conc_<b> and the two nsys cells
# share gh200v2_nsys.
PIECE=${PIECE:-}
if [ -n "$PIECE" ] && ! [[ "$PIECE" =~ ^[A-Za-z0-9]+$ ]]; then
  echo "ABORT: PIECE='$PIECE'. Letters and digits only (it goes into directory names)."
  exit 1
fi
if [ -z "$PIECE" ] && { [ -n "${ONLY_NGRAM_GPU+x}" ] || [ -n "${ONLY_DFLASH+x}" ]; }; then
  echo "ABORT: ONLY_NGRAM_GPU or ONLY_DFLASH without PIECE. A piece needs a tag so its rows land in their own directories."
  exit 1
fi
keep_k () {  # $1 the cell's k list, $2 the k values to keep -> the kept ones, in the cell's order
  local out="" k
  for k in $1; do case " $2 " in *" $k "*) out="$out $k" ;; esac; done
  echo "${out# }"
}

for v in LADDER_ROOT LADDER_VENV LADDER_MRV2 LADDER_CAPS LADDER_HF LADDER_DATA LADDER_CACHE; do
  if [ -z "${!v:-}" ]; then
    echo "ABORT: $v is not set. On the login node: PROJ=<code> ACCOUNT=<name> source deltaai_env.sh,"
    echo "       then submit with submit_dtai2.sh."
    exit 1
  fi
done
if [ "$LADDER_MRV2" != "1" ]; then
  echo "ABORT: LADDER_MRV2=$LADDER_MRV2. These bundles are for vLLM 0.33.0 on Model Runner V2."
  exit 1
fi
# DeltaAI, not Delta. LADDER_TEST_HOST=1 is for testing on a laptop only.
if [ "$(uname -m)" != "aarch64" ] && [ "${LADDER_TEST_HOST:-0}" != "1" ]; then
  echo "ABORT: this node is $(uname -m). The dtai2 jobs run on DeltaAI's GH200 nodes (aarch64), not on Delta."
  exit 1
fi
if [ "$LADDER_CAPS" != "90" ]; then
  echo "ABORT: LADDER_CAPS=$LADDER_CAPS. On DeltaAI it is 90 (GH200); deltaai_env.sh sets it."
  exit 1
fi
# Delta's /projects may not be mounted on DeltaAI's compute nodes. Nothing a
# job reads or writes may live there.
for v in LADDER_ROOT LADDER_VENV LADDER_HF LADDER_DATA LADDER_CACHE LADDER_DOMDATA LADDER_TASKDATA LADDER_RESULTS; do
  case "${!v:-}" in
    /projects/*) echo "ABORT: $v=${!v} is under /projects. Use the copies under /work/nvme."; exit 1 ;;
  esac
done
# Without this file ksweep.sh falls back to Delta's module cray-python/3.12.12.
if [ ! -f "$LADDER_VENV/ladder_modules" ]; then
  echo "ABORT: no $LADDER_VENV/ladder_modules. Use the venv install_deltaai.sh built (it writes that file)."
  exit 1
fi

HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
KS=${LADDER_KSWEEP:-$HERE/ksweep.sh}
V5C=$HERE/run_benchmark.v5c.py
RES=${LADDER_RESULTS:-$LADDER_ROOT/results}
DOMDATA=${LADDER_DOMDATA:-$LADDER_ROOT/data/domains_balanced}
TASKDATA=${LADDER_TASKDATA:-$LADDER_ROOT/data/tasks}
NSYS=${NSYS:-1}
JOB=${SLURM_JOB_ID:-nojob}
[ -f "$KS" ] || { echo "ABORT: no $KS. Submit from the repository's code/ directory."; exit 1; }
grep -q 'HARNESS_VERSION = "v5c' "$V5C" 2>/dev/null || { echo "ABORT: no harness v5c at $V5C (git pull?)."; exit 1; }
# nsys_v5.py profiles this harness instead of its default, v5.
export LADDER_NSYS_HARNESS=$V5C

echo "dtai2 bundle $BUNDLE   job $JOB   host $(hostname)   $(date)"
echo "ksweep: $KS"
echo "results root: $RES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo

# ------------------------------------------------------------ GPU clock log
GPULOG=$LADDER_ROOT/logs/dtai2-$BUNDLE-$JOB-gpu.csv
mkdir -p "$LADDER_ROOT/logs"
GPUQ="timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active"
nvidia-smi --query-gpu="$GPUQ" --format=csv >/dev/null 2>&1 || GPUQ="timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used"
GPUPID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu="$GPUQ" --format=csv -l 5 > "$GPULOG" 2>&1 &
  GPUPID=$!
  echo "gpu log: $GPULOG (every 5 s)"
fi
trap '[ -n "$GPUPID" ] && kill "$GPUPID" 2>/dev/null' EXIT

# ------------------------------------------------------------ DFlash metadata
DRAFT=${LADDER_DRAFT:-$LADDER_HF/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}
mkdir -p "$RES/gh200v2_meta"
META=$RES/gh200v2_meta/${BUNDLE}_j$JOB.json
if [ -x "$LADDER_VENV/bin/python3" ]; then
  # In a subshell with the venv's modules, as ksweep.sh loads them, so the
  # vLLM import sees the same libraries the arms will.
  (
  if [ -f "$LADDER_VENV/ladder_modules" ] && command -v module >/dev/null 2>&1; then
    while IFS= read -r M; do [ -n "$M" ] && module load "$M" >/dev/null 2>&1; done < "$LADDER_VENV/ladder_modules"
  fi
  timeout 300 "$LADDER_VENV/bin/python3" - "$DRAFT" "$META" "$BUNDLE" "$JOB" <<'EOF' || echo "WARNING: DFlash metadata not recorded"
import json, os, socket, sys
draft, meta, bundle, job = sys.argv[1:5]
out = {"bundle": bundle, "job": job, "host": socket.gethostname(), "draft": draft}
try:
    c = json.load(open(os.path.join(draft, "config.json")))
    dc = c.get("dflash_config") if isinstance(c.get("dflash_config"), dict) else {}
    lt = c.get("layer_types")
    out["config"] = {k: c.get(k) for k in ("architectures", "model_type", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size",
                     "block_size", "is_causal")}
    out["config"]["dflash_config_causal"] = dc.get("causal")
    out["config"]["layer_types"] = sorted(set(lt)) if isinstance(lt, list) else lt
except Exception as e:
    out["config_error"] = repr(e)
try:
    import vllm
    from vllm.config.speculative import XPRESS_ARCHITECTURES as xp
    out["vllm"] = vllm.__version__
    a = set((out.get("config") or {}).get("architectures") or [])
    out["speculator_class"] = ("XPressSpeculator" if a & set(xp) else
                               "LiLiCorrSpeculator" if "LiLiCorrDraftModel" in a else
                               "DFlash2Speculator" if a & {"DFlash2DraftModel", "DFlash2BeamDraftModel"} else
                               "DFlashSpeculator")
except Exception as e:
    out["vllm_error"] = repr(e)
json.dump(out, open(meta, "w"), indent=1)
print("dflash metadata:", json.dumps(out))
EOF
  )
fi
echo

FULL=("K_NGRAM=1 2 5 7 11" "K_NGRAM_GPU=1 2 5 7 11" "K_DFLASH=1 2 4 7 11 15")
REDUCED=("K_NGRAM=5" "K_NGRAM_GPU=5" "K_DFLASH=7 15")
CONCSET=("K_NGRAM=2 5" "K_NGRAM_GPU=2 5" "K_DFLASH=2 4 7 15")
PRESET=("K_NGRAM=5" "K_NGRAM_GPU=5" "K_DFLASH=7")
TASKSET=("K_NGRAM=5 11" "K_NGRAM_GPU=5 11" "K_DFLASH=7 15")

STATUS=()
FAILED=0
RAN=0

want_cell () {  # $1 cell name; true if CELLS is unset or lists it
  [ -z "${CELLS:-}" ] && return 0
  case " $CELLS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

cell () {  # $1 name, $2 bucket, then KEY=VALUE pairs for ksweep.sh
  local name=$1 bucket=$2; shift 2
  want_cell "$name" || { echo "skip cell $name (not in CELLS='$CELLS')"; return 0; }
  RAN=$((RAN + 1))
  echo "==================== cell $name  bucket=$bucket  $(date +%H:%M:%S)"
  # A subshell, with every per-cell knob cleared first, so nothing set for
  # one cell (PC=off, LOOP, MAXTOK=1, a k schedule, a k list) can leak into
  # the next. LADDER_DATA is only ever overridden inside the subshell.
  (
    unset CONC LIMIT PC PERREQ LOOP MAXTOK K_NGRAM K_NGRAM_GPU K_DFLASH RUNGS \
          LADDER_RUN LADDER_OUT LADDER_KSCHED LADDER_WARMUP SEED TEMP PLMIN PLMAX \
          PSTYLE IGNORE_EOS
    export BUCKET="$bucket" DOMAIN=all LADDER_RUN="$V5C"
    for kv in "$@"; do export "$kv"; done
    if [ -n "$PIECE" ]; then
      # RUNGS unset means ksweep.sh's Model Runner V2 default.
      left=""
      for r in ${RUNGS:-ngram_gpu dflash}; do
        case "$r" in
          ngram_gpu) [ -n "${ONLY_NGRAM_GPU+x}" ] && export K_NGRAM_GPU="$(keep_k "${K_NGRAM_GPU:-}" "$ONLY_NGRAM_GPU")"
                     [ -n "${K_NGRAM_GPU:-}" ] && left="$left ngram_gpu" ;;
          dflash)    [ -n "${ONLY_DFLASH+x}" ] && export K_DFLASH="$(keep_k "${K_DFLASH:-}" "$ONLY_DFLASH")"
                     [ -n "${K_DFLASH:-}" ] && left="$left dflash" ;;
          *)         left="$left $r" ;;
        esac
      done
      if [ -z "$left" ]; then
        echo "skip cell $name: piece $PIECE leaves it no speculative arm"
        exit 97
      fi
      export RUNGS="${left# }" LADDER_OUT="${LADDER_OUT}_p$PIECE"
    fi
    mkdir -p "$LADDER_OUT"
    echo "cell $name: CONC=${CONC:-1} LIMIT=${LIMIT:-default} PC=${PC:-default} LOOP=${LOOP:-} PERREQ=${PERREQ:-none} MAXTOK=${MAXTOK:-256} RUNGS=${RUNGS:-default} K_NGRAM_GPU='${K_NGRAM_GPU:-}' K_DFLASH='${K_DFLASH:-}' KSCHED=${LADDER_KSCHED:-none} RUN=$(basename "$LADDER_RUN") DATA=$LADDER_DATA OUT=$LADDER_OUT"
    bash "$KS"
  )
  local rc=$?
  if [ "$rc" -eq 97 ] && [ -n "$PIECE" ]; then
    RAN=$((RAN - 1)); STATUS+=("cell $name skipped: nothing of it in piece $PIECE"); return 0
  fi
  STATUS+=("cell $name bucket=$bucket exit=$rc")
  [ "$rc" -eq 0 ] || FAILED=$((FAILED + 1))
}

fail_cell () {  # $1 name, $2 reason
  want_cell "$1" || return 0
  echo "SKIP cell $1: $2"
  STATUS+=("cell $1 not run: $2")
  FAILED=$((FAILED + 1))
}

B=${BUNDLE#*-}
case "$BUNDLE" in
  c1-*)
    cell core "$B" CONC=1 "${FULL[@]}"                    LADDER_OUT="$RES/gh200v2_ksweep_$B"
    cell pre   "$B" CONC=1 MAXTOK=1 "${REDUCED[@]}"         LADDER_OUT="$RES/gh200v2_prefill_$B"
    cell loopA "$B" CONC=1 LOOP=1 "${FULL[@]}"             LADDER_OUT="$RES/gh200v2_loopA_$B"
    cell acc   "$B" CONC=1 PERREQ=detailed "${CONCSET[@]}" LADDER_OUT="$RES/gh200v2_acc_$B"
    cell k0    "$B" CONC=1 RUNGS=dflash K_DFLASH=7 LADDER_KSCHED=1-1024:0 \
               LADDER_OUT="$RES/gh200v2_k0_$B"
    [ "$B" = "32K" ] && cell nopc "$B" CONC=1 PC=off "${REDUCED[@]}" LADDER_OUT="$RES/gh200v2_nopc_$B"
    if [ "${LOOPB:-0}" = "1" ]; then
      cell loopB "$B" CONC=1 LOOP=1 "${FULL[@]}"           LADDER_OUT="$RES/gh200v2_loopB_$B"
    elif [ -n "${CELLS:-}" ]; then
      fail_cell loopB "CELLS asks for loopB but LOOPB is not 1"
    fi
    cell rep   "$B" CONC=1 "${REDUCED[@]}"                 LADDER_OUT="$RES/gh200v2_rep_$B"
    ;;
  conc-*)
    case "$B" in
      2K)  N=128; CS="1 4 8 16 32" ;;
      8K)  N=128; CS="1 4 8 16" ;;
      32K) N=48;  CS="1 2 4 8" ;;
    esac
    TOP=${CS##* }
    cell pre1 "$B" CONC=1 LIMIT="$N" MAXTOK=1 "${PRESET[@]}" LADDER_OUT="$RES/gh200v2_conc_prefill_$B"
    for C in $CS; do
      cell "c$C" "$B" CONC="$C" LIMIT="$N" "${CONCSET[@]}" LADDER_OUT="$RES/gh200v2_conc_$B"
    done
    cell preTop "$B" CONC="$TOP" LIMIT="$N" MAXTOK=1 "${PRESET[@]}" LADDER_OUT="$RES/gh200v2_conc_prefill_$B"
    ;;
  dom-*)
    F=$DOMDATA/longbench_v2_$B.jsonl
    if [ -s "$F" ]; then
      HAVE=$(grep -c . "$F")
      N=$((HAVE - 2))   # the last two prompts of the file are the warm-up
      echo "domain file $F: $HAVE prompts, $N measured, the last two for the warm-up"
      cell domloop "$B" CONC=1 LOOP=1 LIMIT="$N" "${REDUCED[@]}" LADDER_DATA="$DOMDATA" \
           LADDER_OUT="$RES/gh200v2_dom_loop_$B"
      cell domacc  "$B" CONC=1 PERREQ=detailed LIMIT="$N" "${CONCSET[@]}" LADDER_DATA="$DOMDATA" \
           LADDER_OUT="$RES/gh200v2_dom_acc_$B"
    else
      fail_cell domloop "no $F (run make_domains.py and merge_domains.py first)"
      fail_cell domacc  "no $F"
    fi
    ;;
  task-*)
    for T in code summ fewshot mdqa; do
      F=$TASKDATA/$T/longbench_v2_$B.jsonl
      if [ -s "$F" ]; then
        N=$(( $(grep -c . "$F") - 2 )); [ "$N" -gt 50 ] && N=50
        cell "$T" "$B" CONC=1 LIMIT="$N" "${TASKSET[@]}" PSTYLE=context IGNORE_EOS=1 \
             LADDER_DATA="$TASKDATA/$T" LADDER_OUT="$RES/gh200v2_t${T}_$B"
      else
        fail_cell "$T" "no $F (run build_task_workloads.py first)"
      fi
    done
    ;;
  profile)
    cell t64  32K CONC=1 LIMIT=10 MAXTOK=64  K_NGRAM=5 K_NGRAM_GPU=5 K_DFLASH=7 \
              LADDER_OUT="$RES/gh200v2_outlen_t64"
    cell t512 32K CONC=1 LIMIT=10 MAXTOK=512 K_NGRAM=5 K_NGRAM_GPU=5 K_DFLASH=7 \
              LADDER_OUT="$RES/gh200v2_outlen_t512"
    if [ "$NSYS" = "1" ]; then
      if ! command -v nsys >/dev/null 2>&1; then
        # Try the usual module names; which one DeltaAI uses is not verified.
        for m in ${NSYS_MODULES:-cudatoolkit nvhpc cuda}; do
          module load "$m" >/dev/null 2>&1 && command -v nsys >/dev/null 2>&1 && { echo "nsys from module $m"; break; }
        done
      fi
      if command -v nsys >/dev/null 2>&1; then
        echo "nsys: $(command -v nsys)  $(nsys --version 2>/dev/null | head -1)"
        cell nsys2K  2K  CONC=1 LIMIT=3 MAXTOK=64 RUNGS=dflash K_DFLASH=7 \
             LADDER_RUN="$HERE/nsys_v5.py" LADDER_OUT="$RES/gh200v2_nsys"
        cell nsys32K 32K CONC=1 LIMIT=3 MAXTOK=64 RUNGS="ngram_gpu dflash" \
             K_NGRAM=5 K_NGRAM_GPU=5 K_DFLASH=7 \
             LADDER_RUN="$HERE/nsys_v5.py" LADDER_OUT="$RES/gh200v2_nsys"
      else
        echo "No nsys on PATH after trying modules '${NSYS_MODULES:-cudatoolkit nvhpc cuda}'."
        echo "Find it on a login node (module spider nsight) and resubmit with CELLS='nsys2K nsys32K' NSYS_MODULES=<name>."
        fail_cell nsys2K "nsys not found"
        fail_cell nsys32K "nsys not found"
      fi
    fi
    ;;
esac

echo
echo "==================== dtai2 bundle $BUNDLE summary  $(date)"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo "cells run: $RAN   cells failed, skipped or incomplete: $FAILED"
echo "gpu log: $GPULOG"
echo "next, on the login node:"
echo "  \"\$LADDER_VENV/bin/python3\" ksweep_table.py $RES/gh200v2_ksweep_<bucket>"
# COMPLETED must mean the measurements exist (RULES section 6).
[ "$FAILED" -eq 0 ] || { echo "INCOMPLETE: read the cell logs above before using any of this bundle."; exit 1; }
echo "done."
