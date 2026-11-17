#!/bin/bash
# Offline tests for the proposed dtai2 job set. No GPU, no network, no Slurm,
# no vLLM: everything the jobs touch is replaced by a stub in a temporary
# directory, which is deleted at the end (KEEP=1 keeps it).
#
#   cd code && bash test_dtai2.sh
#
# What is real and what is a stub:
#   real   dtai2_bundle.sh, submit_dtai2.sh, run_benchmark.v5.py,
#          run_benchmark.v5b.py, run_benchmark.v5c.py, ksweep.sh,
#          make_domains.py, merge_domains.py
#   stub   vllm and torch (a fake LLM that returns 8 tokens per prompt and
#          echoes its config back), the venv, nvidia-smi (answers "GH200",
#          compute capability 9.0), sbatch, module, nsys, the workloads
#
# So these tests show that the scripts pass the right settings to the right
# places, refuse what they should refuse, and write rows where they should.
# They say nothing about how vLLM 0.33.0 behaves on a GH200. That is the smoke
# job's and the first real job's to show (see the note's verification list).
#
# Groups:
#   T1 syntax                       T6 guards (host, /projects, venv, names)
#   T2 harness v5b == v5            T7 domain files (make_domains + merge)
#   T3 k schedule option            T8 submitter (plan, smoke gate, factor)
#   T4 every bundle's cell list     T9 end to end through the real ksweep.sh
#   T5 settings never leak between cells
#
#   T2c harness v5c: engine timing, warm-up outside the measured set, the
#       all-zero schedule check
#
# Written 16 Nov 2026 with the dtai2 job set; revised the same day after the
# outside review (harness v5c, per-bucket results directories, a wider leak
# test, fixed walltimes). Needs GNU timeout on PATH, as DeltaAI has: on macOS
# the DFlash metadata check fails without it. Exit 0 only if every check passes.

set -u
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d "${TMPDIR:-/tmp}/dtai2test.XXXXXX")
PASS=0; FAIL=0
ok  () { PASS=$((PASS + 1)); echo "  ok    $1"; }
bad () { FAIL=$((FAIL + 1)); echo "  FAIL  $1"; [ -n "${2:-}" ] && echo "        $2"; }
check () { if eval "$2"; then ok "$1"; else bad "$1" "${3:-}"; fi; }
cleanup () { [ "${KEEP:-0}" = "1" ] && echo "kept $W" || rm -rf "$W"; }
trap cleanup EXIT

# ---------------------------------------------------------------- stubs
mkdir -p "$W"/{bin,stub/vllm/config,stub/torch,venv/bin,data,root,smoke,hf}
QWEN=$W/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DFL=$W/hf/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
mkdir -p "$QWEN" "$DFL"
printf '{"architectures":["DFlashDraftModel"],"num_hidden_layers":5,"num_key_value_heads":8,"head_dim":128,"layer_types":["full_attention"]}\n' > "$DFL/config.json"

cat > "$W/stub/vllm/__init__.py" <<'EOF'
__version__ = "0.33.0-stub"
import os, types
class SamplingParams:
    def __init__(self, temperature=0.0, max_tokens=256, seed=0):
        self.max_tokens = max_tokens
class _Spec:
    def __init__(self, d):
        self.method = d.get("method")
        self.num_speculative_tokens = d.get("num_speculative_tokens")
        drop = os.environ.get("STUB_DROP_SCHED") == "1"
        self.num_speculative_tokens_per_batch_size = None if drop else d.get("num_speculative_tokens_per_batch_size")
class LLM:
    def __init__(self, **kw):
        self.kw = kw; sc = kw.get("speculative_config")
        NS = types.SimpleNamespace
        self.llm_engine = NS(vllm_config=NS(
            use_v2_model_runner=True,
            scheduler_config=NS(max_num_seqs=kw.get("max_num_seqs"), async_scheduling=True),
            cache_config=NS(enable_prefix_caching=kw.get("enable_prefix_caching", True)),
            speculative_config=_Spec(sc) if sc else None))
        self.n = 0
    def generate(self, prompts, sp, use_tqdm=True):
        NS = types.SimpleNamespace; out = []
        m = None
        if os.environ.get("STUB_METRICS") == "1":   # vLLM 0.33.0's RequestStateStats
            m = NS(arrival_time=1.7e9, queued_ts=10.0, scheduled_ts=10.5, first_token_ts=12.0,
                   last_token_ts=15.0, first_token_latency=2.0)
        for p in prompts:
            toks = list(range(min(sp.max_tokens, 8)))
            out.append(NS(prompt_token_ids=list(range(len(p) // 10)), metrics=m,
                          outputs=[NS(token_ids=toks, finish_reason="length", spec_decode_metrics=None)]))
        self.n += len(prompts); return out
    def get_metrics(self):
        sc = self.kw.get("speculative_config")
        if not sc: return []
        M = types.SimpleNamespace
        ks = sc.get("num_speculative_tokens_per_batch_size")
        if ks and all(e[2] == 0 for e in ks) and os.environ.get("STUB_K0_DRAFTS") != "1":
            return [M(name="vllm:spec_decode_num_drafts", value=0),
                    M(name="vllm:spec_decode_num_draft_tokens", value=0),
                    M(name="vllm:spec_decode_num_accepted_tokens", value=0)]
        return [M(name="vllm:spec_decode_num_drafts", value=self.n * 3),
                M(name="vllm:spec_decode_num_draft_tokens", value=self.n * 9),
                M(name="vllm:spec_decode_num_accepted_tokens", value=self.n * 4)]
EOF
: > "$W/stub/vllm/config/__init__.py"
printf 'XPRESS_ARCHITECTURES = frozenset({"Qwen3XPressModel", "DFlashQwen3XPressModel"})\n' > "$W/stub/vllm/config/speculative.py"
printf '__version__ = "2.14.0-stub"\nclass cuda:\n    @staticmethod\n    def is_available(): return False\n' > "$W/stub/torch/__init__.py"

ln -s "$(command -v python3)" "$W/venv/bin/python3"
: > "$W/venv/bin/activate"
: > "$W/venv/ladder_modules"

cat > "$W/bin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in *clocks_event_reasons*) exit 6 ;; esac
case "$*" in *"-l 5"*) echo "timestamp, clocks.sm"; while true; do echo "t, 1980 MHz"; sleep 1; done ;; esac
case "$*" in *compute_cap*) echo "9.0"; exit 0 ;; esac
case "$*" in *"--query-gpu=name --format"*) echo "NVIDIA GH200 480GB"; exit 0 ;; esac
echo "NVIDIA GH200 480GB, 97871 MiB, 570.00"
EOF
printf '#!/bin/bash\necho "BUNDLE=$BUNDLE $* PIECE=${PIECE:-} ONLY_NGRAM_GPU=${ONLY_NGRAM_GPU-unset} ONLY_DFLASH=${ONLY_DFLASH-unset}" >> "%s/sbatch.log"; echo $((RANDOM + 500000))\n' "$W" > "$W/bin/sbatch"
printf '#!/bin/bash\n[ "$1" = load ] && [ "$2" = none_such ] && exit 1\nexit 0\n' > "$W/bin/module"
chmod +x "$W"/bin/*

# A stub ksweep.sh that records the settings each cell hands it.
cat > "$W/ks.sh" <<'EOF'
#!/bin/bash
echo "$(basename "$LADDER_OUT")|$BUCKET|${CONC:-}|${LIMIT:-}|${MAXTOK:-}|${PERREQ:-}|${LOOP:-}|${PC:-}|${K_NGRAM_GPU:-}|${K_DFLASH:-}|${RUNGS:-}|${LADDER_KSCHED:-}|$(basename "${LADDER_RUN:-v5}")|$(basename "$LADDER_DATA")|${SEED:-}${TEMP:-}${PLMIN:-}${PLMAX:-}${LADDER_WARMUP:-}" >> "$CELLLOG"
exit 0
EOF
chmod +x "$W/ks.sh"

python3 - "$W/data" <<'EOF'
import json, os, random, sys
random.seed(1)
doms = (["Single-Document QA"] * 60 + ["Multi-Document QA"] * 40 + ["Code Repository Understanding"] * 20
        + ["Long In-context Learning"] * 25 + ["Long-dialogue History Understanding"] * 12
        + ["Long Structured Data Understanding"] * 8)
random.shuffle(doms)
rows = [{"_id": "d%03d" % i, "domain": d, "context": "word " * 80, "question": "q?",
         "choice_A": "a", "choice_B": "b", "choice_C": "c", "choice_D": "d"} for i, d in enumerate(doms)]
extra = [{"_id": "x%03d" % i, "domain": "Single-Document QA", "context": "word " * 80, "question": "q?",
          "choice_A": "a", "choice_B": "b", "choice_C": "c", "choice_D": "d"} for i in range(200)]
for b in ("2K", "8K"):
    with open(os.path.join(sys.argv[1], "longbench_v2_%s.jsonl" % b), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows + extra)
with open(os.path.join(sys.argv[1], "longbench_v2_32K.jsonl"), "w") as f:
    f.writelines(json.dumps(r) + "\n" for i, r in enumerate(rows + extra) if i % 3 == 0)
EOF

export PATH="$W/bin:$PATH" PYTHONPATH="$W/stub" LADDER_TEST_HOST=1 USER=tester ACCOUNT=acct
export LADDER_ROOT="$W/root" LADDER_VENV="$W/venv" LADDER_HF="$W/hf" LADDER_DATA="$W/data" \
       LADDER_CACHE="$W/cache" LADDER_MRV2=1 LADDER_CAPS=90 SLURM_JOB_ID=4242
unset LADDER_DOMDATA LADDER_KSCHED CELLS LOOPB GH200_FACTOR SMOKE GO PC LOOP MAXTOK LIMIT K_DFLASH 2>/dev/null

# ---------------------------------------------------------------- T1
echo "T1 syntax"
for f in dtai2_bundle.sh submit_dtai2.sh ksweep.sh dtai2_pieces.sh; do check "bash -n $f" "bash -n '$CODE/$f'"; done
for f in run_benchmark.v5b.py run_benchmark.v5c.py merge_domains.py nsys_v5.py; do
  check "compiles: $f" "python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' '$CODE/$f'"
done
check "v5b passes ksweep.sh's harness check" "grep -q 'HARNESS_VERSION = \"v5' '$CODE/run_benchmark.v5b.py'"
check "v5c passes ksweep.sh's harness check" "grep -q 'HARNESS_VERSION = \"v5c' '$CODE/run_benchmark.v5c.py'"
EMD=$(printf '\342\200\224')   # U+2014, spelled in octal so this file holds none
check "no em dashes in the new files" "! grep -q '$EMD' '$CODE/dtai2_bundle.sh' '$CODE/submit_dtai2.sh' '$CODE/run_benchmark.v5b.py' '$CODE/run_benchmark.v5c.py' '$CODE/merge_domains.py' '$CODE/../notes/2026-11-16-dtai2-proposed-jobs.md'"

# ---------------------------------------------------------------- T2
echo "T2 harness v5b reproduces v5 when no schedule is asked for"
H () {  # harness outdir args...
  local h=$1 o=$2; shift 2
  VLLM_USE_V2_MODEL_RUNNER=1 python3 "$CODE/$h" --dataset "$W/data/longbench_v2_2K.jsonl" --model "$QWEN" \
    --bucket 2K --limit 4 --max-num-seqs 1 --outdir "$o" "$@" > "$o.log" 2>&1
}
for h in v5 v5b; do
  mkdir -p "$W/h$h"
  H run_benchmark.$h.py "$W/h$h" --rung off
  H run_benchmark.$h.py "$W/h$h" --rung ngram_gpu --k 5
  H run_benchmark.$h.py "$W/h$h" --rung dflash --k 7 --draft "$DFL"
done
python3 - "$W/hv5/results.jsonl" "$W/hv5b/results.jsonl" > "$W/t2.txt" 2>&1 <<'EOF'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1])]; b = [json.loads(l) for l in open(sys.argv[2])]
skip = {"harness", "elapsed_s", "throughput_tok_s", "env", "perreq"}
add = {"k_schedule_requested", "draft_config"}
assert len(a) == len(b) == 3, (len(a), len(b))
for x, y in zip(a, b):
    dx = {k: v for k, v in x.items() if k not in skip}
    dy = {k: v for k, v in y.items() if k not in skip | add}
    ry = dict(dy["resolved"]); ry.pop("spec_k_schedule", None); dy["resolved"] = ry
    assert dx == dy, {k: (dx.get(k), dy.get(k)) for k in set(dx) | set(dy) if dx.get(k) != dy.get(k)}
    assert y["k_schedule_requested"] is None
assert b[2]["draft_config"]["speculator_class"] == "DFlashSpeculator", b[2]["draft_config"]
assert b[2]["draft_config"]["layer_types"] == ["full_attention"]
print("same")
EOF
check "v5b rows equal v5 rows apart from the added fields" "grep -q '^same' '$W/t2.txt'" "$(tail -3 "$W/t2.txt")"

# ---------------------------------------------------------------- T2c
echo "T2c harness v5c: timing fields, warm-up, and v5's rows otherwise"
mkdir -p "$W/hv5c" "$W/hv5cm" "$W/hv5cf"
H run_benchmark.v5c.py "$W/hv5c" --rung off
H run_benchmark.v5c.py "$W/hv5c" --rung ngram_gpu --k 5
H run_benchmark.v5c.py "$W/hv5c" --rung dflash --k 7 --draft "$DFL"
STUB_METRICS=1 H run_benchmark.v5c.py "$W/hv5cm" --rung off
STUB_METRICS=1 H run_benchmark.v5.py  "$W/hv5cm" --rung off
H run_benchmark.v5c.py "$W/hv5cf" --rung off --warmup first2
LADDER_WARMUP=first2 H run_benchmark.v5c.py "$W/hv5cf" --rung ngram_gpu --k 5
head -5 "$W/data/longbench_v2_2K.jsonl" > "$W/short.jsonl"
VLLM_USE_V2_MODEL_RUNNER=1 python3 "$CODE/run_benchmark.v5c.py" --dataset "$W/short.jsonl" --model "$QWEN" \
  --bucket 2K --limit 4 --max-num-seqs 1 --outdir "$W/hshort" --rung off > "$W/hshort.log" 2>&1
check "v5c refuses a file with no prompts to spare for the warm-up" "grep -q 'needs 6 prompts' '$W/hshort.log' && [ ! -e '$W/hshort/results.jsonl' ]"
python3 - "$W" > "$W/t2c.txt" 2>&1 <<'EOF'
import glob, json, os, sys
W = sys.argv[1]
R = lambda d: [json.loads(l) for l in open(os.path.join(W, d, "results.jsonl"))]
P = lambda d, r: json.load(open(os.path.join(W, d, r["perreq"])))
a, c = R("hv5"), R("hv5c")
skip = {"harness", "elapsed_s", "throughput_tok_s", "env", "perreq"}
add = {"k_schedule_requested", "draft_config", "warmup", "per_request_engine_timing"}
assert len(a) == len(c) == 3
for x, y in zip(a, c):
    dx = {k: v for k, v in x.items() if k not in skip}
    dy = {k: v for k, v in y.items() if k not in skip | add}
    ry = dict(dy["resolved"]); ry.pop("spec_k_schedule", None); dy["resolved"] = ry
    assert dx == dy, {k: (dx.get(k), dy.get(k)) for k in set(dx) | set(dy) if dx.get(k) != dy.get(k)}
    ids = [r["id"] for r in P("hv5c", y)]
    w = y["warmup"]
    assert w["mode"] == "outside" and len(w["ids"]) == 2 and not w["measured_too"], w
    assert len(ids) == 4 and not set(ids) & set(w["ids"]), (ids, w)
    assert ids == [r["id"] for r in P("hv5", x)], "measured prompts differ from v5's"
m = R("hv5cm")
assert m[0]["harness"].startswith("v5c") and m[0]["per_request_engine_timing"] == 4, m[0]
t = P("hv5cm", m[0])[0]["timing"]
assert abs(t["prefill_s"] - 1.5) < 1e-9 and abs(t["decode_s"] - 3.0) < 1e-9 and abs(t["queue_s"] - 0.5) < 1e-9, t
assert abs(t["engine_s"] - 5.0) < 1e-9 and abs(t["ttft_s"] - 2.0) < 1e-9, t
assert m[1]["harness"].startswith("v5-") and m[1]["per_request_timing"] == 0, "v5 was expected to miss the 0.33.0 fields"
f = R("hv5cf")
for y in f:
    ids = [r["id"] for r in P("hv5cf", y)]
    assert y["warmup"]["mode"] == "first2" and y["warmup"]["measured_too"] == sorted(ids[:2]), y["warmup"]
print("v5c ok")
EOF
check "v5c: v5's rows and prompts, warm-up outside them, engine timing read, first2 recorded" "grep -q '^v5c ok' '$W/t2c.txt'" "$(tail -3 "$W/t2c.txt")"

# ---------------------------------------------------------------- T3
echo "T3 the k schedule option"
mkdir -p "$W/ks"
K () { VLLM_USE_V2_MODEL_RUNNER=1 python3 "$CODE/run_benchmark.v5c.py" --dataset "$W/data/longbench_v2_2K.jsonl" \
         --model "$QWEN" --bucket 2K --limit 4 --max-num-seqs 1 --outdir "$W/ks" "$@" > "$W/ks.out" 2>&1; }
LADDER_KSCHED=1-1024:0 K --rung dflash --k 7 --draft "$DFL"; check "schedule from LADDER_KSCHED runs" "[ $? -eq 0 ]"
LADDER_KSCHED=1-1024:0 K --rung off;                         check "schedule ignored for off" "grep -q 'ignored for rung off' '$W/ks.out'"
STUB_DROP_SCHED=1 LADDER_KSCHED=1-1024:0 K --rung dflash --k 7 --draft "$DFL"
check "engine dropping the schedule aborts" "grep -q 'the engine dropped it' '$W/ks.out'"
STUB_K0_DRAFTS=1 LADDER_KSCHED=1-1024:0 K --rung dflash --k 7 --draft "$DFL"
check "an all-zero schedule that still drafted aborts" "grep -q 'zero everywhere but the engine counted' '$W/ks.out'"
K --rung ngram_gpu --k 5 --k-schedule 1-4:7;        check "k above the arm's k refused" "grep -q 'outside 0..5' '$W/ks.out'"
K --rung ngram_gpu --k 5 --k-schedule 2-4:3;        check "schedule not starting at 1 refused" "grep -q 'must start at batch size 1' '$W/ks.out'"
K --rung ngram_gpu --k 5 --k-schedule 1-4:3,4-8:2;  check "overlapping ranges refused" "grep -q 'overlap' '$W/ks.out'"
K --rung ngram_gpu --k 5 --k-schedule 1:3;          check "malformed entry refused" "grep -q 'is not start-end:k' '$W/ks.out'"
LADDER_KSCHED=1-1024:0 K --rung ngram_gpu --k 5 --k-schedule 1-1:5,2-8:2,9-64:1
python3 - "$W/ks/results.jsonl" > "$W/t3.txt" 2>&1 <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
ids = [x["run_id"] for x in r]
assert ids == ["dflash_2K_k7_n4_s1234_c1_ks1to1024k0", "off_2K_k0_n4_s1234_c1",
               "ngram_gpu_2K_k5_n4_s1234_c1_ks1to1k5-2to8k2-9to64k1"], ids
assert r[0]["resolved"]["spec_k_schedule"] == [[1, 1024, 0]]
assert r[0]["spec"]["drafts"] == 0 and r[0]["harness"].startswith("v5c")
assert r[2]["k_schedule_requested"] == [[1, 1, 5], [2, 8, 2], [9, 64, 1]]
print("rows ok")
EOF
check "rows: tagged run_ids, read-back schedule, no row from refused runs, flag beats env" "grep -q 'rows ok' '$W/t3.txt'" "$(tail -3 "$W/t3.txt")"

# ---------------------------------------------------------------- T4 + T5
echo "T4 every bundle hands ksweep.sh the intended cells; T5 nothing leaks between cells"
FULLK="1 2 5 7 11|1 2 4 7 11 15"; RED="5|7 15"; CON="2 5|2 4 7 15"; PRE="5|7"; V5C=run_benchmark.v5c.py
expect_c1 () {
  local b=$1
  printf '%s\n' \
    "gh200v2_ksweep_$b|$b|1||||||$FULLK|||$V5C|data|" \
    "gh200v2_prefill_$b|$b|1||1||||$RED|||$V5C|data|" \
    "gh200v2_loopA_$b|$b|1||||1||$FULLK|||$V5C|data|" \
    "gh200v2_acc_$b|$b|1|||detailed|||$CON|||$V5C|data|" \
    "gh200v2_k0_$b|$b|1|||||||7|dflash|1-1024:0|$V5C|data|"
  [ "$b" = 32K ] && echo "gh200v2_nopc_$b|$b|1|||||off|$RED|||$V5C|data|"
  echo "gh200v2_rep_$b|$b|1||||||$RED|||$V5C|data|"
}
expect_conc () {
  local b=$1 n=$2 cs=$3 top=${3##* }
  echo "gh200v2_conc_prefill_$b|$b|1|$n|1||||$PRE|||$V5C|data|"
  for c in $cs; do echo "gh200v2_conc_$b|$b|$c|$n|||||$CON|||$V5C|data|"; done
  echo "gh200v2_conc_prefill_$b|$b|$top|$n|1||||$PRE|||$V5C|data|"
}
# junk in the parent environment that must never reach any cell
export PC=on LOOP=1 MAXTOK=999 K_DFLASH=99 LADDER_KSCHED=9-9:9 LIMIT=7 PERREQ=summary RUNGS=ngram \
       SEED=9 TEMP=0.9 PLMIN=1 PLMAX=9 LADDER_WARMUP=first2 CONC=77 K_NGRAM=98 K_NGRAM_GPU=99 \
       LADDER_RUN=/nonexistent/harness.py LADDER_OUT=/nonexistent/out
run_bundle () {  # bundle [extra env...] -> cell log in $W/cells.$1
  local b=$1; shift
  export CELLLOG="$W/cells.$b"; : > "$CELLLOG"
  env "$@" BUNDLE="$b" LADDER_KSWEEP="$W/ks.sh" bash "$CODE/dtai2_bundle.sh" > "$W/out.$b" 2>&1
  echo $?
}
for b in 2K 8K 32K; do
  rc=$(run_bundle "c1-$b")
  check "c1-$b exits 0" "[ $rc -eq 0 ]"
  check "c1-$b cells" "diff <(expect_c1 $b) '$W/cells.c1-$b' >/dev/null" "$(diff <(expect_c1 $b) "$W/cells.c1-$b" | head -4)"
done
for spec in "2K 128 1 4 8 16 32" "8K 128 1 4 8 16" "32K 48 1 2 4 8"; do
  set -- $spec; b=$1; n=$2; shift 2
  rc=$(run_bundle "conc-$b")
  check "conc-$b exits 0" "[ $rc -eq 0 ]"
  check "conc-$b cells" "diff <(expect_conc $b $n '$*') '$W/cells.conc-$b' >/dev/null" "$(diff <(expect_conc $b $n "$*") "$W/cells.conc-$b" | head -4)"
done
rc=$(run_bundle c1-8K LOOPB=1 CELLS=loopB)
check "LOOPB=1 CELLS=loopB runs only the second per-prompt pass" "[ $rc -eq 0 ] && [ \"\$(cat '$W/cells.c1-8K')\" = 'gh200v2_loopB_8K|8K|1||||1||$FULLK|||$V5C|data|' ]"
rc=$(run_bundle c1-8K CELLS=loopB)
check "CELLS=loopB without LOOPB=1 fails loudly" "[ $rc -ne 0 ] && grep -q 'LOOPB is not 1' '$W/out.c1-8K'"
rc=$(run_bundle profile NSYS_MODULES=none_such)
check "profile without nsys: output-length cells run, job marked incomplete" "[ $rc -ne 0 ] && [ \$(grep -c . '$W/cells.profile') -eq 2 ] && grep -q 'nsys not found' '$W/out.profile'"
printf '#!/bin/bash\necho fake nsys\n' > "$W/bin/nsys"; chmod +x "$W/bin/nsys"
rc=$(run_bundle profile)
printf '%s\n' "gh200v2_outlen_t64|32K|1|10|64||||5|7|||$V5C|data|" "gh200v2_outlen_t512|32K|1|10|512||||5|7|||$V5C|data|" \
  "gh200v2_nsys|2K|1|3|64|||||7|dflash||nsys_v5.py|data|" "gh200v2_nsys|32K|1|3|64||||5|7|ngram_gpu dflash||nsys_v5.py|data|" > "$W/exp.profile"
check "profile with nsys: four cells" "[ $rc -eq 0 ] && diff '$W/exp.profile' '$W/cells.profile' >/dev/null" "$(diff "$W/exp.profile" "$W/cells.profile" | head -4)"
rm -f "$W/bin/nsys"
rc=$(run_bundle dom-8K)
check "dom-8K without its domain file fails loudly" "[ $rc -ne 0 ] && grep -q 'run make_domains.py' '$W/out.dom-8K'"
check "DFlash metadata written with the predicted class" "grep -q DFlashSpeculator '$W/root/results/gh200v2_meta/c1-2K_j4242.json'"
check "GPU clock log written" "[ -s '$W/root/logs/dtai2-c1-2K-4242-gpu.csv' ]"
unset PC LOOP MAXTOK K_DFLASH LADDER_KSCHED LIMIT PERREQ RUNGS SEED TEMP PLMIN PLMAX LADDER_WARMUP \
      CONC K_NGRAM K_NGRAM_GPU LADDER_RUN LADDER_OUT

# ---------------------------------------------------------------- T6
echo "T6 guards"
rc=$(run_bundle c1-2K LADDER_TEST_HOST=0); check "bundle refuses a non-aarch64 node" "[ $rc -ne 0 ] || [ \"\$(uname -m)\" = aarch64 ]"
rc=$(run_bundle c1-2K LADDER_CACHE=/projects/x); check "bundle refuses a /projects path" "[ $rc -ne 0 ] && grep -q 'under /projects' '$W/out.c1-2K'"
rc=$(run_bundle c1-2K LADDER_RESULTS=/projects/x); check "bundle refuses results under /projects" "[ $rc -ne 0 ] && grep -q 'under /projects' '$W/out.c1-2K'"
rc=$(run_bundle c1-2K LADDER_CAPS=86); check "bundle refuses LADDER_CAPS other than 90" "[ $rc -ne 0 ]"
rc=$(run_bundle c1-4K); check "bundle refuses an unknown bundle" "[ $rc -ne 0 ]"
mv "$W/venv/ladder_modules" "$W/lm"; rc=$(run_bundle c1-2K); mv "$W/lm" "$W/venv/ladder_modules"
check "bundle refuses a venv without ladder_modules" "[ $rc -ne 0 ] && grep -q ladder_modules '$W/out.c1-2K'"
S () { (cd "$CODE" && env "$@" bash submit_dtai2.sh > "$W/sub.out" 2>&1); echo $?; }
rc=$(S LADDER_TEST_HOST=0);           check "submitter refuses a non-aarch64 host" "[ $rc -ne 0 ] || [ \"\$(uname -m)\" = aarch64 ]"
rc=$(S LADDER_DATA=/projects/x);      check "submitter refuses /projects data" "[ $rc -ne 0 ] && grep -q 'under /projects' '$W/sub.out'"
rc=$(S BUNDLES=c1-4K);                check "submitter refuses an unknown bundle" "[ $rc -ne 0 ] && grep -q 'unknown bundle' '$W/sub.out'"
rc=$(S BUNDLES=c1-8K CELLS=loopB);    check "submitter refuses CELLS=loopB without LOOPB=1" "[ $rc -ne 0 ]"

# ---------------------------------------------------------------- T7
echo "T7 domain files"
(cd "$CODE" && python3 make_domains.py --src "$W/data" --dst "$W/root/data" --n 50 --n32 25 > "$W/md.out" 2>&1)
check "make_domains.py (unchanged) writes per-domain files" "[ -d '$W/root/data/domains/single_document_qa' ]" "$(tail -3 "$W/md.out")"
(cd "$CODE" && python3 merge_domains.py > "$W/mg.out" 2>&1)
python3 - "$W/root/data/domains_balanced" > "$W/t7.txt" 2>&1 <<'EOF'
import json, os, sys
f = lambda b: [json.loads(l) for l in open(os.path.join(sys.argv[1], "longbench_v2_%s.jsonl" % b))]
a, b, c = f("2K"), f("8K"), f("32K")
assert [r["_id"] for r in a] == [r["_id"] for r in b], "2K and 8K differ"
assert {r["_id"] for r in c} <= {r["_id"] for r in a}, "32K not a subset"
doms = [r["domain"] for r in a]
k = len(set(doms)); assert len(set(doms[:k])) == k, "not round-robin at the start"
m = json.load(open(os.path.join(sys.argv[1], "manifest.json")))
assert m["n"]["2K"] == len(a) and m["input_sha256"]
print("merged ok", len(a), len(c))
EOF
check "merged: 2K and 8K same order, 32K a subset, round-robin, manifest" "grep -q 'merged ok' '$W/t7.txt'" "$(tail -3 "$W/t7.txt")"
(cd "$CODE" && python3 merge_domains.py > "$W/mg2.out" 2>&1)
check "rerun leaves matching files alone" "[ \$(grep -c 'left alone' '$W/mg2.out') -eq 3 ]"
echo '{"_id":"zz","domain":"X"}' >> "$W/root/data/domains_balanced/longbench_v2_2K.jsonl"
(cd "$CODE" && python3 merge_domains.py > "$W/mg3.out" 2>&1)
check "refuses to overwrite a changed file" "grep -q 'refusing to overwrite' '$W/mg3.out'"
python3 -c "import sys; p = sys.argv[1]; l = open(p).readlines(); open(p, 'w').writelines(l[:-1])" "$W/root/data/domains_balanced/longbench_v2_2K.jsonl"
rc=$(run_bundle dom-8K)
N8=$(( $(grep -c . "$W/root/data/domains_balanced/longbench_v2_8K.jsonl") - 2 ))
printf '%s\n' "gh200v2_dom_loop_8K|8K|1|$N8|||1||$RED|||$V5C|domains_balanced|" \
              "gh200v2_dom_acc_8K|8K|1|$N8||detailed|||$CON|||$V5C|domains_balanced|" > "$W/exp.dom"
check "dom-8K cells read the merged file less its two warm-up prompts" "[ $rc -eq 0 ] && diff '$W/exp.dom' '$W/cells.dom-8K' >/dev/null" "$(diff "$W/exp.dom" "$W/cells.dom-8K" | head -4)"

# ---------------------------------------------------------------- T8
echo "T8 submitter"
python3 - "$W/smoke" <<'EOF'
import json, os, sys
mk = lambda rung, el, v2=True, ver="0.33.0", drafts=40: {"harness": "v5-2026-11-15", "run_id": rung + "_2K", "rung": rung, "bucket": "2K",
     "limit": 5, "max_tokens": 256, "max_num_seqs": 1, "loop_timing": False, "elapsed_s": el,
     "spec": {"drafts": 0 if rung == "off" else drafts},
     "env": {"gpu": "NVIDIA GH200 480GB", "vllm": ver}, "resolved": {"use_v2_model_runner": v2, "async_scheduling": True}}
open(os.path.join(sys.argv[1], "results.jsonl"), "w").writelines(json.dumps(r) + "\n" for r in
     [mk("off", 12.0), mk("ngram_gpu", 10.0), mk("dflash", 8.0)])
open(os.path.join(sys.argv[1], "bad.jsonl"), "w").writelines(json.dumps(r) + "\n" for r in
     [mk("off", 12.0), mk("ngram_gpu", 10.0), mk("dflash", 8.0, v2=False)])
open(os.path.join(sys.argv[1], "oldver.jsonl"), "w").writelines(json.dumps(r) + "\n" for r in
     [mk("off", 12.0, ver="0.32.1"), mk("ngram_gpu", 10.0, ver="0.32.1"), mk("dflash", 8.0, ver="0.32.1")])
open(os.path.join(sys.argv[1], "nodrafts.jsonl"), "w").writelines(json.dumps(r) + "\n" for r in
     [mk("off", 12.0), mk("ngram_gpu", 10.0), mk("dflash", 8.0, drafts=0)])
# ksweep.sh's log shape: each arm stamped at its start, `date` after the last.
# Arm walls 100, 200 and 400 s; timed runs 12, 10 and 8 s on 5 prompts.
# Slowest overhead: 400 - 8 - 2 * 8 / 5 = 388.8 s, so 7 minutes.
open(os.path.join(sys.argv[1], "smoke.out"), "w").write(
    "Mon Nov 16 23:55:00 CST 2026\n"
    "### arm 1 rung=off k=0 conc=1 bucket=2K domain=all gpu=gh200 extra=''  23:55:00\n"
    "### arm 2 rung=ngram_gpu k=5 conc=1 bucket=2K domain=all gpu=gh200 extra=''  23:56:40\n"
    "### arm 3 rung=dflash k=7 conc=1 bucket=2K domain=all gpu=gh200 extra='--draft x'  00:00:00\n"
    "Tue Nov 17 00:06:40 CST 2026\n")
EOF
rm -rf "$W/root/results"
rc=$(S); check "plan only exits 0 and submits nothing" "[ $rc -eq 0 ] && grep -q 'Nothing submitted' '$W/sub.out' && [ ! -s '$W/sbatch.log' ]"
check "plan lists ten bundles" "[ \$(grep -cE '^(c1|conc|dom|profile)' '$W/sub.out') -eq 10 ]"
check "no GH200 rows: factor 1.0" "grep -q 'factor: 1.0 (no GH200 rows' '$W/sub.out'"
check "no smoke log: engine start assumed at 3 min" "grep -q 'Engine start per arm: 3 min (assumed' '$W/sub.out'"
WT1=$(awk '$1=="c1-32K" {print $(NF-1)}' "$W/sub.out")
# Hours requested at the A40-speed ceiling with 3 min of engine start: 1.6 times
# the estimate, plus one, and never under 4 for the profile bundle. The domain
# bundles are left out: their size depends on the domain files.
WT_EXPECT="c1-2K=11 profile=4 c1-8K=12 c1-32K=16 conc-2K=10 conc-8K=11 conc-32K=14 "
WTALL=$(awk '$1 ~ /^(c1|conc|profile)/ {printf "%s=%s ", $1, $(NF-1)}' "$W/sub.out")
check "walltimes at the ceiling are the expected hours" "[ '$WTALL' = '$WT_EXPECT' ]" "got: $WTALL"
rc=$(S GO=1); check "GO without a smoke check is refused" "[ $rc -ne 0 ] && grep -q REFUSED '$W/sub.out'"
rc=$(S GO=1 SMOKE="$W/smoke/bad.jsonl"); check "a smoke row off Model Runner V2 fails the check" "[ $rc -ne 0 ] && grep -q 'not on Model Runner V2' '$W/sub.out'"
rc=$(S GO=1 SMOKE="$W/smoke/oldver.jsonl"); check "a smoke row from another vLLM version fails the check" "[ $rc -ne 0 ] && grep -q 'written for 0.33.0' '$W/sub.out'"
rc=$(S GO=1 SMOKE="$W/smoke/nodrafts.jsonl"); check "a speculative smoke row with no drafts fails the check" "[ $rc -ne 0 ] && grep -q 'counted no drafts' '$W/sub.out'"
rc=$(S SMOKE="$W/smoke" SMOKE_LOG="$W/smoke/smoke.out"); check "engine start measured from the smoke log (7 min, across midnight)" "grep -q 'Engine start per arm: 7 min (measured: slowest of 3 smoke arms' '$W/sub.out'"
rc=$(S SMOKE="$W/smoke"); check "GH200 factor measured from the smoke row" "grep -q 'factor: 0.305 (measured' '$W/sub.out'"
WT2=$(awk '$1=="c1-32K" {print $(NF-1)}' "$W/sub.out")
check "walltime stays at the A40-speed ceiling ($WT1 h) with the factor applied" "[ '$WT1' = '$WT2' ] && [ -n '$WT1' ]"
: > "$W/sbatch.log"; rm -rf "$W/root/data/domains_balanced"
rc=$(S GO=1 SMOKE="$W/smoke")
check "GO submits seven, holds back the three domain bundles" "[ \$(grep -c . '$W/sbatch.log') -eq 7 ] && [ \$(grep -c 'NOT SUBMITTED' '$W/sub.out') -eq 3 ]"
check "sbatch gets ghx4, one GPU, 16 cores, 64 GB, the bundle" "grep -q -- '--partition=ghx4 --account=acct --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g' '$W/sbatch.log' && grep -q '^BUNDLE=c1-2K' '$W/sbatch.log'"
head -30 "$W/data/longbench_v2_32K.jsonl" > "$W/d32" && cp "$W/data/longbench_v2_32K.jsonl" "$W/d32.full" && mv "$W/d32" "$W/data/longbench_v2_32K.jsonl"
rc=$(S BUNDLES=conc-32K); check "too few 32K prompts for conc-32K refused" "[ $rc -ne 0 ] && grep -q 'need 50' '$W/sub.out'"
rc=$(S BUNDLES=c1-32K);   check "the same file is enough for c1-32K" "[ $rc -eq 0 ]"
mv "$W/d32.full" "$W/data/longbench_v2_32K.jsonl"

# ---------------------------------------------------------------- T9
echo "T9 end to end through the real ksweep.sh and harness (stub engine)"
rm -rf "$W/root/results"
(export SLURM_SUBMIT_DIR="$CODE"; CELLS="pre k0 loopA" BUNDLE=c1-2K bash "$CODE/dtai2_bundle.sh" > "$W/e2e.out" 2>&1)
check "c1-2K pre, k0, loopA complete" "[ $? -eq 0 ] && grep -q '^done' '$W/e2e.out'" "$(grep -E 'ABORT|FAILED' "$W/e2e.out" | head -3)"
python3 - "$W/root/results" > "$W/t9.txt" 2>&1 <<'EOF'
import json, os, sys
R = lambda d: [json.loads(l) for l in open(os.path.join(sys.argv[1], d, "results.jsonl"))]
p, k, l = R("gh200v2_prefill_2K"), R("gh200v2_k0_2K"), R("gh200v2_loopA_2K")
assert len(p) == 4 and all(r["max_tokens"] == 1 for r in p), [r["run_id"] for r in p]
assert [r["run_id"] for r in k] == ["off_2K_k0_n50_s1234_c1", "dflash_2K_k7_n50_s1234_c1_ks1to1024k0"]
assert k[1]["resolved"]["spec_k_schedule"] == [[1, 1024, 0]] and k[1]["spec"]["drafts"] == 0
assert len(l) == 12 and all(r["run_id"].endswith("_loop") for r in l)
assert all(r["harness"].startswith("v5c") and r["warmup"]["mode"] == "outside"
           and not r["warmup"]["measured_too"] for r in p + k + l)
print("e2e ok")
EOF
check "rows: 4 prefill at max_tokens 1, k0 with the schedule and no drafts, 12 loop rows, all on v5c" "grep -q 'e2e ok' '$W/t9.txt'" "$(tail -3 "$W/t9.txt")"

# ---------------------------------------------------------------- T10
echo "T10 pieces: part of a bundle as its own job, and the submitter's arm counts"
stub_arms () {  # cell log -> arms per cell, as ksweep.sh would run them (off + each k of each rung run)
  awk -F'|' '{ r = $11; a = 1
    if (r == "" || r ~ /(^| )ngram_gpu( |$)/) a += split($9, x, " ")
    if (r == "" || r ~ /(^| )dflash( |$)/)    a += split($10, y, " ")
    printf "%d ", a }' "$1"
}
plan_arms () {  # submitter output, bundle -> arms per cell from its plan line
  awk -v b="$2" '$1 == b' "$1" | grep -o '([0-9]*)' | tr -d '()' | tr '\n' ' '
}
printf '#!/bin/bash\necho fake nsys\n' > "$W/bin/nsys"; chmod +x "$W/bin/nsys"
for b in c1-2K c1-8K c1-32K conc-2K conc-8K conc-32K profile; do
  rc=$(run_bundle "$b"); S BUNDLES="$b" >/dev/null
  check "$b: submitter's arm counts equal what the bundle runs" "[ '$(stub_arms "$W/cells.$b")' = '$(plan_arms "$W/sub.out" "$b")' ]" "bundle: $(stub_arms "$W/cells.$b")  plan: $(plan_arms "$W/sub.out" "$b")"
done
for pc in "a|none|" "b|1 2|none" "c||7 11 15"; do
  IFS='|' read -r tag ng df <<< "$pc"
  args=(PIECE="$tag"); [ -n "$ng" ] && args+=(ONLY_NGRAM_GPU="$ng"); [ -n "$df" ] && args+=(ONLY_DFLASH="$df")
  rc=$(run_bundle c1-32K "${args[@]}"); S BUNDLES=c1-32K "${args[@]}" >/dev/null
  check "piece $tag of c1-32K: submitter's arm counts equal what the bundle runs" "[ '$(stub_arms "$W/cells.c1-32K")' = '$(plan_arms "$W/sub.out" c1-32K)' ]" "bundle: $(stub_arms "$W/cells.c1-32K")  plan: $(plan_arms "$W/sub.out" c1-32K)"
done
rm -f "$W/bin/nsys"
rc=$(run_bundle c1-2K PIECE=a ONLY_DFLASH=none CELLS="core rep")
printf '%s\n' "gh200v2_ksweep_2K_pa|2K|1||||||1 2 5 7 11||ngram_gpu||$V5C|data|" \
  "gh200v2_rep_2K_pa|2K|1||||||5||ngram_gpu||$V5C|data|" > "$W/exp.pa"
check "piece a: n-gram arms only, into _pa directories" "[ $rc -eq 0 ] && diff '$W/exp.pa' '$W/cells.c1-2K' >/dev/null" "$(diff "$W/exp.pa" "$W/cells.c1-2K" | head -4)"
rc=$(run_bundle c1-2K PIECE=b ONLY_NGRAM_GPU=none ONLY_DFLASH="1 2 4" CELLS="core pre k0")
check "piece b: core keeps dflash 1 2 4; pre and k0, left with nothing, are skipped and the job passes" \
  "[ $rc -eq 0 ] && [ \"\$(cat '$W/cells.c1-2K')\" = 'gh200v2_ksweep_2K_pb|2K|1|||||||1 2 4|dflash||$V5C|data|' ] && [ \$(grep -c 'skipped: nothing of it in piece b' '$W/out.c1-2K') -eq 2 ] && grep -q 'cells run: 1 ' '$W/out.c1-2K'"
rc=$(run_bundle conc-2K PIECE=x CELLS="c1")
check "a CELLS split with a PIECE writes its conc cell to its own directory" "[ $rc -eq 0 ] && [ \"\$(cat '$W/cells.conc-2K')\" = 'gh200v2_conc_2K_px|2K|1|128|||||$CON|ngram_gpu dflash||$V5C|data|' ]"
rc=$(run_bundle c1-2K ONLY_DFLASH=7);            check "bundle refuses ONLY_DFLASH without PIECE" "[ $rc -ne 0 ] && grep -q 'without PIECE' '$W/out.c1-2K' && [ ! -s '$W/cells.c1-2K' ]"
rc=$(run_bundle c1-2K PIECE=a_b);                check "bundle refuses a PIECE that is not letters and digits" "[ $rc -ne 0 ] && [ ! -s '$W/cells.c1-2K' ]"
rc=$(S BUNDLES=c1-2K ONLY_DFLASH=7);             check "submitter refuses ONLY_DFLASH without PIECE" "[ $rc -ne 0 ] && grep -q 'needs PIECE' '$W/sub.out'"
rc=$(S BUNDLES=c1-2K WALL_H=0);                  check "submitter refuses WALL_H=0" "[ $rc -ne 0 ]"
rc=$(S BUNDLES=c1-2K CELLS=k0 PIECE=z ONLY_DFLASH="1 2"); check "submitter refuses a piece that leaves nothing to run" "[ $rc -ne 0 ] && grep -q 'nothing to run' '$W/sub.out'"
: > "$W/sbatch.log"
rc=$(S GO=1 SMOKE="$W/smoke" BUNDLES=c1-2K CELLS=core PIECE=a ONLY_DFLASH=none WALL_H=1)
check "GO with a piece: one job, 1 h, named dtai2-c1-2K-pa, its own log" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/sbatch.log') -eq 1 ] && grep -q -- '--time=01:00:00 --job-name=dtai2-c1-2K-pa --output=.*/dtai2-c1-2K-pa-%j.out' '$W/sbatch.log' && grep -q '^BUNDLE=c1-2K' '$W/sbatch.log'"
rc=$(S BUNDLES=c1-32K WALL_H=1);                 check "WALL_H below the estimate warns" "[ $rc -eq 0 ] && grep -q 'WARNING: c1-32K: the GH200 estimate' '$W/sub.out'"

P () { (cd "$CODE" && env "$@" bash dtai2_pieces.sh SMOKE="$W/smoke" > "$W/pieces.out" 2>&1); echo $?; }
rc=$(P GH200_FACTOR=0.199 LOAD_MIN=3)
check "pieces plan: 46 pieces, none over 75% of its hour at the 2K factor, nothing submitted" "[ $rc -eq 0 ] && grep -q '^46 pieces planned' '$W/pieces.out' && ! grep -q WARNING '$W/pieces.out'" "$(grep -E 'WARNING|STOPPED' "$W/pieces.out" | head -3)"
: > "$W/sbatch.log"
rc=$(P GO=1)
check "pieces GO: 46 jobs, all 1 h, 46 distinct job names" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/sbatch.log') -eq 46 ] && [ \$(grep -c -- '--time=01:00:00' '$W/sbatch.log') -eq 46 ] && [ \$(grep -o -- '--job-name=[^ ]*' '$W/sbatch.log' | sort -u | grep -c .) -eq 46 ]"
: > "$W/sbatch.log"
rc=$(P GO=1 PIECES_FOR=conc-8K)
check "PIECES_FOR limits the list to one bundle" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/sbatch.log') -eq 6 ] && ! grep -qv '^BUNDLE=conc-8K' '$W/sbatch.log'"
: > "$W/sbatch.log"
rc=$(P GO=1 LADDER_DATA=/projects/x)
check "pieces stop at the first refusal and submit nothing after it" "[ $rc -ne 0 ] && grep -q 'STOPPED at piece 1' '$W/pieces.out' && [ ! -s '$W/sbatch.log' ]"

TRIAL="c1-32K:prek0 c1-2K:coreD2 conc-32K:c2"
: > "$W/sbatch.log"; rc=$(P GO=1 ONLY_PIECES="$TRIAL")
check "ONLY_PIECES submits exactly the three named pieces" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/sbatch.log') -eq 3 ] && grep -q 'job-name=dtai2-c1-32K-pprek0 ' '$W/sbatch.log' && grep -q 'job-name=dtai2-c1-2K-pcoreD2 ' '$W/sbatch.log' && grep -q 'job-name=dtai2-conc-32K-pc2 ' '$W/sbatch.log'"
check "the coreD2 piece passes both filters to the job" "grep 'pcoreD2 ' '$W/sbatch.log' | grep -q 'ONLY_NGRAM_GPU=none' && grep 'pcoreD2 ' '$W/sbatch.log' | grep -q 'ONLY_DFLASH=7 11 15'"
: > "$W/sbatch.log"; rc=$(P GO=1 SKIP_PIECES="$TRIAL")
check "SKIP_PIECES submits the other 43, none of the three" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/sbatch.log') -eq 43 ] && ! grep -q 'pprek0 .*BUNDLE=c1-32K\|BUNDLE=c1-32K.*PIECE=prek0' '$W/sbatch.log' && ! grep -q 'PIECE=coreD2.*BUNDLE=c1-2K\|BUNDLE=c1-2K.*PIECE=coreD2' '$W/sbatch.log'"
: > "$W/sbatch.log"; rc=$(P GO=1 ONLY_PIECES="c1-2K:coreX")
check "an unknown piece name is refused before anything is submitted" "[ $rc -ne 0 ] && grep -q \"unknown piece 'c1-2K:coreX'\" '$W/pieces.out' && [ ! -s '$W/sbatch.log' ]"

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]
