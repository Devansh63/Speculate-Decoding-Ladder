#!/bin/bash
# Offline tests for the proposed dtai2 job set. No GPU, no network, no Slurm,
# no vLLM: everything the jobs touch is replaced by a stub in a temporary
# directory, which is deleted at the end (KEEP=1 keeps it).
#
#   cd code && bash test_dtai2.sh
#
# What is real and what is a stub:
#   real   dtai2_bundle.sh, submit_dtai2.sh, run_benchmark.v5.py,
#          run_benchmark.v5b.py, ksweep.sh, make_domains.py, merge_domains.py
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
# Written 16 Nov 2026 with the dtai2 job set. Exit 0 only if every check passes.

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
        for p in prompts:
            toks = list(range(min(sp.max_tokens, 8)))
            out.append(NS(prompt_token_ids=list(range(len(p) // 10)), metrics=None,
                          outputs=[NS(token_ids=toks, finish_reason="length", spec_decode_metrics=None)]))
        self.n += len(prompts); return out
    def get_metrics(self):
        if not self.kw.get("speculative_config"): return []
        M = types.SimpleNamespace
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
printf '#!/bin/bash\necho "BUNDLE=$BUNDLE $*" >> "%s/sbatch.log"; echo $((RANDOM + 500000))\n' "$W" > "$W/bin/sbatch"
printf '#!/bin/bash\n[ "$1" = load ] && [ "$2" = none_such ] && exit 1\nexit 0\n' > "$W/bin/module"
chmod +x "$W"/bin/*

# A stub ksweep.sh that records the settings each cell hands it.
cat > "$W/ks.sh" <<'EOF'
#!/bin/bash
echo "$(basename "$LADDER_OUT")|$BUCKET|${CONC:-}|${LIMIT:-}|${MAXTOK:-}|${PERREQ:-}|${LOOP:-}|${PC:-}|${K_NGRAM_GPU:-}|${K_DFLASH:-}|${RUNGS:-}|${LADDER_KSCHED:-}|$(basename "${LADDER_RUN:-v5}")|$(basename "$LADDER_DATA")" >> "$CELLLOG"
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
for f in dtai2_bundle.sh submit_dtai2.sh ksweep.sh; do check "bash -n $f" "bash -n '$CODE/$f'"; done
for f in run_benchmark.v5b.py merge_domains.py nsys_v5.py; do
  check "compiles: $f" "python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' '$CODE/$f'"
done
check "v5b passes ksweep.sh's harness check" "grep -q 'HARNESS_VERSION = \"v5' '$CODE/run_benchmark.v5b.py'"
EMD=$(printf '\342\200\224')   # U+2014, spelled in octal so this file holds none
check "no em dashes in the new files" "! grep -q '$EMD' '$CODE/dtai2_bundle.sh' '$CODE/submit_dtai2.sh' '$CODE/run_benchmark.v5b.py' '$CODE/merge_domains.py' '$CODE/../notes/2026-11-16-dtai2-proposed-jobs.md'"

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

# ---------------------------------------------------------------- T3
echo "T3 the k schedule option"
mkdir -p "$W/ks"
K () { VLLM_USE_V2_MODEL_RUNNER=1 python3 "$CODE/run_benchmark.v5b.py" --dataset "$W/data/longbench_v2_2K.jsonl" \
         --model "$QWEN" --bucket 2K --limit 4 --max-num-seqs 1 --outdir "$W/ks" "$@" > "$W/ks.out" 2>&1; }
LADDER_KSCHED=1-1024:0 K --rung dflash --k 7 --draft "$DFL"; check "schedule from LADDER_KSCHED runs" "[ $? -eq 0 ]"
LADDER_KSCHED=1-1024:0 K --rung off;                         check "schedule ignored for off" "grep -q 'ignored for rung off' '$W/ks.out'"
STUB_DROP_SCHED=1 LADDER_KSCHED=1-1024:0 K --rung dflash --k 7 --draft "$DFL"
check "engine dropping the schedule aborts" "grep -q 'the engine dropped it' '$W/ks.out'"
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
assert r[2]["k_schedule_requested"] == [[1, 1, 5], [2, 8, 2], [9, 64, 1]]
print("rows ok")
EOF
check "rows: tagged run_ids, read-back schedule, no row from refused runs, flag beats env" "grep -q 'rows ok' '$W/t3.txt'" "$(tail -3 "$W/t3.txt")"

# ---------------------------------------------------------------- T4 + T5
echo "T4 every bundle hands ksweep.sh the intended cells; T5 nothing leaks between cells"
FULLK="1 2 5 7 11|1 2 4 7 11 15"; RED="5|7 15"; CON="2 5|2 4 7 15"; PRE="5|7"
expect_c1 () {
  local b=$1
  printf '%s\n' \
    "gh200v2_ksweep|$b|1||||||$FULLK|||v5|data" \
    "gh200v2_prefill|$b|1||1||||$RED|||v5|data" \
    "gh200v2_loopA|$b|1||||1||$FULLK|||v5|data" \
    "gh200v2_acc|$b|1|||detailed|||$CON|||v5|data" \
    "gh200v2_k0|$b|1|||||||7|dflash|1-1024:0|run_benchmark.v5b.py|data"
  [ "$b" = 32K ] && echo "gh200v2_nopc|$b|1|||||off|$RED|||v5|data"
  echo "gh200v2_rep|$b|1||||||$RED|||v5|data"
}
expect_conc () {
  local b=$1 n=$2 cs=$3 top=${3##* }
  echo "gh200v2_conc_prefill|$b|1|$n|1||||$PRE|||v5|data"
  for c in $cs; do echo "gh200v2_conc|$b|$c|$n|||||$CON|||v5|data"; done
  echo "gh200v2_conc_prefill|$b|$top|$n|1||||$PRE|||v5|data"
}
# junk in the parent environment that must never reach any cell
export PC=on LOOP=1 MAXTOK=999 K_DFLASH=99 LADDER_KSCHED=9-9:9 LIMIT=7 PERREQ=summary RUNGS=ngram
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
check "LOOPB=1 CELLS=loopB runs only the second per-prompt pass" "[ $rc -eq 0 ] && [ \"\$(cat '$W/cells.c1-8K')\" = 'gh200v2_loopB|8K|1||||1||$FULLK|||v5|data' ]"
rc=$(run_bundle c1-8K CELLS=loopB)
check "CELLS=loopB without LOOPB=1 fails loudly" "[ $rc -ne 0 ] && grep -q 'LOOPB is not 1' '$W/out.c1-8K'"
rc=$(run_bundle profile NSYS_MODULES=none_such)
check "profile without nsys: output-length cells run, job marked incomplete" "[ $rc -ne 0 ] && [ \$(grep -c . '$W/cells.profile') -eq 2 ] && grep -q 'nsys not found' '$W/out.profile'"
printf '#!/bin/bash\necho fake nsys\n' > "$W/bin/nsys"; chmod +x "$W/bin/nsys"
rc=$(run_bundle profile)
printf '%s\n' "gh200v2_outlen_t64|32K|1|10|64||||5|7|||v5|data" "gh200v2_outlen_t512|32K|1|10|512||||5|7|||v5|data" \
  "gh200v2_nsys|2K|1|3|64|||||7|dflash||nsys_v5.py|data" "gh200v2_nsys|32K|1|3|64||||5|7|ngram_gpu dflash||nsys_v5.py|data" > "$W/exp.profile"
check "profile with nsys: four cells" "[ $rc -eq 0 ] && diff '$W/exp.profile' '$W/cells.profile' >/dev/null" "$(diff "$W/exp.profile" "$W/cells.profile" | head -4)"
rm -f "$W/bin/nsys"
rc=$(run_bundle dom-8K)
check "dom-8K without its domain file fails loudly" "[ $rc -ne 0 ] && grep -q 'run make_domains.py' '$W/out.dom-8K'"
check "DFlash metadata written with the predicted class" "grep -q DFlashSpeculator '$W/root/results/gh200v2_meta/c1-2K_j4242.json'"
check "GPU clock log written" "[ -s '$W/root/logs/dtai2-c1-2K-4242-gpu.csv' ]"
unset PC LOOP MAXTOK K_DFLASH LADDER_KSCHED LIMIT PERREQ RUNGS

# ---------------------------------------------------------------- T6
echo "T6 guards"
rc=$(run_bundle c1-2K LADDER_TEST_HOST=0); check "bundle refuses a non-aarch64 node" "[ $rc -ne 0 ] || [ \"\$(uname -m)\" = aarch64 ]"
rc=$(run_bundle c1-2K LADDER_CACHE=/projects/x); check "bundle refuses a /projects path" "[ $rc -ne 0 ] && grep -q 'under /projects' '$W/out.c1-2K'"
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
sed -i '$d' "$W/root/data/domains_balanced/longbench_v2_2K.jsonl"
rc=$(run_bundle dom-8K)
N8=$(grep -c . "$W/root/data/domains_balanced/longbench_v2_8K.jsonl")
printf '%s\n' "gh200v2_dom_loop|8K|1|$N8|||1||$RED|||v5|domains_balanced" \
              "gh200v2_dom_acc|8K|1|$N8||detailed|||$CON|||v5|domains_balanced" > "$W/exp.dom"
check "dom-8K cells read the merged file at its full length" "[ $rc -eq 0 ] && diff '$W/exp.dom' '$W/cells.dom-8K' >/dev/null" "$(diff "$W/exp.dom" "$W/cells.dom-8K" | head -4)"

# ---------------------------------------------------------------- T8
echo "T8 submitter"
python3 - "$W/smoke" <<'EOF'
import json, os, sys
mk = lambda rung, el, v2=True: {"harness": "v5-2026-11-15", "run_id": rung + "_2K", "rung": rung, "bucket": "2K",
     "limit": 5, "max_tokens": 256, "max_num_seqs": 1, "loop_timing": False, "elapsed_s": el,
     "env": {"gpu": "NVIDIA GH200 480GB"}, "resolved": {"use_v2_model_runner": v2, "async_scheduling": True}}
open(os.path.join(sys.argv[1], "results.jsonl"), "w").writelines(json.dumps(r) + "\n" for r in
     [mk("off", 12.0), mk("ngram_gpu", 10.0), mk("dflash", 8.0)])
open(os.path.join(sys.argv[1], "bad.jsonl"), "w").writelines(json.dumps(r) + "\n" for r in
     [mk("off", 12.0), mk("ngram_gpu", 10.0), mk("dflash", 8.0, v2=False)])
EOF
rm -rf "$W/root/results"
rc=$(S); check "plan only exits 0 and submits nothing" "[ $rc -eq 0 ] && grep -q 'Nothing submitted' '$W/sub.out' && [ ! -s '$W/sbatch.log' ]"
check "plan lists ten bundles" "[ \$(grep -cE '^(c1|conc|dom|profile)' '$W/sub.out') -eq 10 ]"
check "no GH200 rows: factor 1.0" "grep -q 'factor: 1.0 (no GH200 rows' '$W/sub.out'"
WT1=$(awk '$1=="c1-32K" {print $(NF-1)}' "$W/sub.out")
rc=$(S GO=1); check "GO without a smoke check is refused" "[ $rc -ne 0 ] && grep -q REFUSED '$W/sub.out'"
rc=$(S GO=1 SMOKE="$W/smoke/bad.jsonl"); check "a smoke row off Model Runner V2 fails the check" "[ $rc -ne 0 ] && grep -q 'not on Model Runner V2' '$W/sub.out'"
rc=$(S SMOKE="$W/smoke"); check "GH200 factor measured from the smoke row" "grep -q 'factor: 0.305 (measured' '$W/sub.out'"
WT2=$(awk '$1=="c1-32K" {print $(NF-1)}' "$W/sub.out")
check "walltime stays at the A40-speed ceiling ($WT1 h) with the factor applied" "[ '$WT1' = '$WT2' ] && [ -n '$WT1' ]"
: > "$W/sbatch.log"; rm -rf "$W/root/data/domains_balanced"
rc=$(S GO=1 SMOKE="$W/smoke")
check "GO submits seven, holds back the three domain bundles" "[ \$(grep -c . '$W/sbatch.log') -eq 7 ] && [ \$(grep -c 'NOT SUBMITTED' '$W/sub.out') -eq 3 ]"
check "sbatch gets ghx4, one GPU, 16 cores, 64 GB, the bundle" "grep -q -- '--partition=ghx4 --account=acct --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g' '$W/sbatch.log' && grep -q '^BUNDLE=c1-2K' '$W/sbatch.log'"
head -30 "$W/data/longbench_v2_32K.jsonl" > "$W/d32" && cp "$W/data/longbench_v2_32K.jsonl" "$W/d32.full" && mv "$W/d32" "$W/data/longbench_v2_32K.jsonl"
rc=$(S BUNDLES=conc-32K); check "too few 32K prompts for conc-32K refused" "[ $rc -ne 0 ] && grep -q 'need 48' '$W/sub.out'"
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
p, k, l = R("gh200v2_prefill"), R("gh200v2_k0"), R("gh200v2_loopA")
assert len(p) == 4 and all(r["max_tokens"] == 1 for r in p), [r["run_id"] for r in p]
assert [r["run_id"] for r in k] == ["off_2K_k0_n50_s1234_c1", "dflash_2K_k7_n50_s1234_c1_ks1to1024k0"]
assert k[1]["harness"].startswith("v5b") and k[1]["resolved"]["spec_k_schedule"] == [[1, 1024, 0]]
assert len(l) == 12 and all(r["run_id"].endswith("_loop") for r in l)
assert all(r["harness"].startswith("v5-") for r in p + l)
print("e2e ok")
EOF
check "rows: 4 prefill at max_tokens 1, k0 on v5b with the schedule, 12 loop rows on v5" "grep -q 'e2e ok' '$W/t9.txt'" "$(tail -3 "$W/t9.txt")"

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]
