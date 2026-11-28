#!/bin/bash
# Offline tests for the dtai3 (router data) job set. No GPU, no Slurm, no vLLM:
# the engine, nvidia-smi, sbatch and module are stubs in a temporary directory
# (deleted at the end; KEEP=1 keeps it). The Qwen3-8B tokenizer is REAL when
# QWEN_TOK points at a directory holding its tokenizer.json and
# tokenizer_config.json (on DeltaAI: the pinned snapshot under $LADDER_HF);
# without it the chat-template tests are skipped, not passed.
#
#   cd code && QWEN_TOK=$LADDER_HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 bash test_dtai3.sh
#
# Groups:
#   T1 syntax of every new file
#   T2 harness v5d in raw mode writes what v5c writes (same run_id, same prompts)
#   T3 harness v5d in chat mode: template, token-id prompts, stop ids, thinking,
#      multi-turn rows, the too-long guard
#   T4 router_cell.sh: arm order, SKIP_OFF, guards, the row-count check
#   T5 dtai3_job.sh: what every cell type hands router_cell.sh
#   T6 submit_dtai3.sh: the plan, held pieces, what sbatch receives
#   T7 end to end: dtai3_job.sh -> router_cell.sh -> harness v5d (stub engine)
#
# The builders were tested on the real sources separately (the note's section
# "How the files were checked"); they need the network and minutes, so they are
# not run here. What these tests cannot show: how vLLM 0.33.0 behaves. The smoke
# piece (TIERS=S) is that test.

set -u
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d "${TMPDIR:-/tmp}/dtai3test.XXXXXX")
PASS=0; FAIL=0; SKIPPED=0
ok () { PASS=$((PASS + 1)); echo "  ok    $1"; }
bad () { FAIL=$((FAIL + 1)); echo "  FAIL  $1"; }
check () { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
skip () { SKIPPED=$((SKIPPED + 1)); echo "  skip  $1"; }
cleanup () { [ "${KEEP:-0}" = "1" ] && echo "kept $W" || rm -rf "$W"; }
trap cleanup EXIT

mkdir -p "$W"/{bin,stub/vllm/config,stub/torch,venv/bin,root/data/router,root/logs,root/results,hf,cache}
QWEN=$W/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DFL=$W/hf/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
mkdir -p "$QWEN" "$DFL"
printf '{"architectures":["DFlashDraftModel"],"num_hidden_layers":5}\n' > "$DFL/config.json"
HAVE_TOK=0
if [ -n "${QWEN_TOK:-}" ] && [ -f "$QWEN_TOK/tokenizer.json" ] && [ -f "$QWEN_TOK/tokenizer_config.json" ] \
   && python3 -c "import transformers" 2>/dev/null; then
  cp "$QWEN_TOK/tokenizer.json" "$QWEN_TOK/tokenizer_config.json" "$QWEN/"
  [ -f "$QWEN_TOK/config.json" ] && cp "$QWEN_TOK/config.json" "$QWEN/"
  HAVE_TOK=1
fi

cat > "$W/stub/vllm/__init__.py" <<'EOF'
__version__ = "0.33.0-stub"
import os, json, types
LOG = os.environ.get("STUB_LOG", "/dev/null")
class SamplingParams:
    def __init__(self, temperature=0.0, max_tokens=256, seed=0, ignore_eos=False, stop_token_ids=None,
                 logprobs=None):
        self.max_tokens, self.ignore_eos, self.stop_token_ids = max_tokens, ignore_eos, stop_token_ids
        self.logprobs = logprobs
        open(LOG, "a").write(json.dumps({"sp": {"max_tokens": max_tokens, "ignore_eos": ignore_eos,
                                                "stop": stop_token_ids, "logprobs": logprobs}}) + "\n")
class _Spec:
    def __init__(self, d):
        self.method = d.get("method"); self.num_speculative_tokens = d.get("num_speculative_tokens")
        self.num_speculative_tokens_per_batch_size = d.get("num_speculative_tokens_per_batch_size")
class LLM:
    def __init__(self, **kw):
        self.kw = kw; sc = kw.get("speculative_config"); NS = types.SimpleNamespace
        open(LOG, "a").write(json.dumps({"llm": {k: v for k, v in kw.items() if k != "speculative_config"},
                                         "spec": sc}) + "\n")
        self.llm_engine = NS(vllm_config=NS(use_v2_model_runner=True,
            scheduler_config=NS(max_num_seqs=kw.get("max_num_seqs"), async_scheduling=True),
            cache_config=NS(enable_prefix_caching=kw.get("enable_prefix_caching", True)),
            speculative_config=_Spec(sc) if sc else None))
        self.n = 0
    def generate(self, prompts, sp, use_tqdm=True):
        NS = types.SimpleNamespace; out = []
        kinds = sorted({type(p).__name__ for p in prompts})
        first = prompts[0]
        open(LOG, "a").write(json.dumps({"gen": len(prompts), "kinds": kinds,
            "head": (first.get("prompt_token_ids")[:3] if isinstance(first, dict) else first[:30])}) + "\n")
        m = NS(arrival_time=1.7e9, queued_ts=10.0, scheduled_ts=10.5, first_token_ts=12.0, last_token_ts=15.0,
               first_token_latency=2.0)
        prs = self.kw.get("per_request_spec_decode_metrics")
        for p in prompts:
            ids = p["prompt_token_ids"] if isinstance(p, dict) else list(range(max(1, len(p) // 4)))
            toks = list(range(min(sp.max_tokens, 8)))
            sm = NS(num_drafts=3, num_accepted_tokens=4, per_step_accepted=[1, 2, 1]) if prs else None
            # vLLM's shape: per position {token_id: Logprob(logprob, rank)}, top-N plus the chosen token
            lp = ([{t: NS(logprob=-0.1, rank=1), 1000 + t: NS(logprob=-2.5, rank=2)} for t in toks]
                  if getattr(sp, "logprobs", None) else None)
            out.append(NS(prompt_token_ids=ids, metrics=m,
                          outputs=[NS(token_ids=toks, finish_reason="stop", spec_decode_metrics=sm, logprobs=lp)]))
        self.n += len(prompts); return out
    def get_metrics(self):
        sc = self.kw.get("speculative_config")
        if not sc: return []
        M = types.SimpleNamespace
        return [M(name="vllm:spec_decode_num_drafts", value=self.n * 3),
                M(name="vllm:spec_decode_num_draft_tokens", value=self.n * 9),
                M(name="vllm:spec_decode_num_accepted_tokens", value=self.n * 4)]
EOF
: > "$W/stub/vllm/config/__init__.py"
printf 'XPRESS_ARCHITECTURES = frozenset()\n' > "$W/stub/vllm/config/speculative.py"
printf '__version__ = "2.14.0-stub"\nclass cuda:\n    @staticmethod\n    def is_available(): return False\n' > "$W/stub/torch/__init__.py"
ln -s "$(command -v python3)" "$W/venv/bin/python3"
: > "$W/venv/bin/activate"; : > "$W/venv/ladder_modules"
cat > "$W/bin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in *"-l 5"*) echo "timestamp, clocks.sm"; while true; do echo "t, 1980 MHz"; sleep 1; done ;; esac
case "$*" in *compute_cap*) echo "9.0"; exit 0 ;; esac
echo "NVIDIA GH200 480GB, 97871 MiB, 570.00"
EOF
printf '#!/bin/bash\necho "PIECE=$PIECE WL=$WL B=$B CELLS=$CELLS LIM1=${LIM1:-} $*" >> "%s/sbatch.log"; echo $((RANDOM + 500000))\n' "$W" > "$W/bin/sbatch"
printf '#!/bin/bash\nexit 0\n' > "$W/bin/module"
chmod +x "$W"/bin/*

# Small workload files in the builder's layout.
python3 - "$W/root/data/router" <<'EOF'
import json, os, sys
root = sys.argv[1]
def put(wl, b, rows):
    os.makedirs(os.path.join(root, wl), exist_ok=True)
    with open(os.path.join(root, wl, b + ".jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
ctx = lambda i, n=200: {"_id": f"p{i}", "context": f"Document {i}. " + "word " * n + "\nQuestion: what?"}
for wl, b, n in [("qa", "2K", 130), ("qa", "8K", 60), ("qa", "16K", 66), ("qa", "32K", 50), ("ragquote", "2K", 64),
                 ("code", "16K", 27), ("mix", "mix", 66), ("codeedit", "nat", 52)]:
    put(wl, b, [ctx(i) for i in range(n)])
put("chat", "2K", [{"_id": f"c{i}", "messages": [{"role": "user", "content": f"hi {i}"},
                    {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more please"}]}
                   for i in range(52)])
put("toolong", "2K", [ctx(i, 6000) for i in range(10)])
EOF

export PATH="$W/bin:$PATH" PYTHONPATH="$W/stub" LADDER_TEST_HOST=1 USER=tester ACCOUNT=acct
export LADDER_ROOT="$W/root" LADDER_VENV="$W/venv" LADDER_HF="$W/hf" LADDER_CACHE="$W/cache" \
       LADDER_CAPS=90 LADDER_MRV2=1 VLLM_USE_V2_MODEL_RUNNER=1 STUB_LOG="$W/stub.log"
H5C=$CODE/run_benchmark.v5c.py; H5D=$CODE/run_benchmark.v5d.py
D=$W/root/data/router

echo "T1 syntax"
for f in router_cell.sh dtai3_job.sh submit_dtai3.sh push_dtai3.sh dtai3_feeder.sh; do
  check "bash -n $f" "bash -n '$CODE/$f'"
done
for f in run_benchmark.v5d.py build_router_workloads.py gen_router_ai.py; do
  check "py_compile $f" "python3 -m py_compile '$CODE/$f'"
done

echo "T2 harness v5d raw mode == v5c"
for h in v5c v5d; do
  python3 "$CODE/run_benchmark.$h.py" --dataset "$D/qa/2K.jsonl" --model "$QWEN" --rung off --bucket 2K \
    --limit 3 --max-tokens 16 --max-num-seqs 1 --outdir "$W/t2_$h" --prompt-style context >/dev/null 2>&1
done
check "both wrote one row" "[ \$(grep -c . $W/t2_v5c/results.jsonl) = 1 ] && [ \$(grep -c . $W/t2_v5d/results.jsonl) = 1 ]"
check "same run_id" "python3 -c \"import json;a=json.loads(open('$W/t2_v5c/results.jsonl').readline());b=json.loads(open('$W/t2_v5d/results.jsonl').readline());exit(a['run_id']!=b['run_id'])\""
check "v5d raw row says prompt_format raw, no stop ids" "python3 -c \"import json;b=json.loads(open('$W/t2_v5d/results.jsonl').readline());exit(not(b['prompt_format']=='raw' and b['stop_token_ids'] is None and b['harness'].startswith('v5d')))\""
check "same per-request ids and prompt lengths" "python3 -c \"
import json,glob
a=json.load(open(glob.glob('$W/t2_v5c/perreq_*')[0]));b=json.load(open(glob.glob('$W/t2_v5d/perreq_*')[0]))
exit([(r['id'],r['prompt_tokens']) for r in a]!=[(r['id'],r['prompt_tokens']) for r in b])\""
python3 "$H5D" --dataset "$D/chat/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --max-tokens 16 \
  --max-num-seqs 1 --outdir "$W/t2_bad" > "$W/t2_bad.log" 2>&1
check "multi-turn rows refuse raw mode" "grep -q 'run with --prompt-format chat' $W/t2_bad.log"
python3 "$H5D" --dataset "$D/qa/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --thinking on \
  --max-num-seqs 1 --outdir "$W/t2_bad2" > "$W/t2_bad2.log" 2>&1
check "thinking on without chat is refused" "grep -q 'needs --prompt-format chat' $W/t2_bad2.log"

echo "T3 harness v5d chat mode"
if [ "$HAVE_TOK" = "1" ]; then
  : > "$W/stub.log"
  python3 "$H5D" --dataset "$D/qa/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --max-tokens 64 \
    --max-num-seqs 1 --outdir "$W/t3" --prompt-format chat --max-model-len 40960 > "$W/t3.log" 2>&1
  check "row written, run_id ends _chat" "python3 -c \"import json;r=json.loads(open('$W/t3/results.jsonl').readline());exit(not r['run_id'].endswith('_chat'))\""
  check "engine got token-id prompts starting <|im_start|>user" "grep -q '\"kinds\": \\[\"dict\"\\], \"head\": \\[151644, 872, 198\\]' $W/stub.log"
  check "stop ids <|im_end|> and <|endoftext|> reach SamplingParams" "grep -q '\"stop\": \\[151645, 151643\\]' $W/stub.log"
  check "no ignore_eos in chat mode" "! grep -q '\"ignore_eos\": true' $W/stub.log"
  check "row records format, thinking, template sha, finished_stop" "python3 -c \"
import json;r=json.loads(open('$W/t3/results.jsonl').readline())
exit(not(r['prompt_format']=='chat' and r['thinking']=='off' and len(r['chat_template_sha256'])==64 and r['finished_stop']==3))\""
  check "templated prompt longer than the raw one" "python3 -c \"
import json,glob;a=json.load(open(glob.glob('$W/t2_v5d/perreq_*')[0]));b=json.load(open(glob.glob('$W/t3/perreq_*')[0]))
exit(not all(y['prompt_tokens']>10 for y in b))\""
  python3 "$H5D" --dataset "$D/chat/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --max-tokens 64 \
    --max-num-seqs 1 --outdir "$W/t3m" --prompt-format chat --max-model-len 40960 > "$W/t3m.log" 2>&1
  check "multi-turn rows run in chat mode" "[ \$(grep -c . $W/t3m/results.jsonl 2>/dev/null) = 1 ]"
  python3 "$H5D" --dataset "$D/qa/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --max-tokens 64 \
    --max-num-seqs 1 --outdir "$W/t3t" --prompt-format chat --thinking on --max-model-len 40960 > "$W/t3t.log" 2>&1
  check "thinking on: run_id _chatT" "python3 -c \"import json;r=json.loads(open('$W/t3t/results.jsonl').readline());exit(not r['run_id'].endswith('_chatT'))\""
  python3 "$H5D" --dataset "$D/qa/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --max-tokens 64 \
    --max-num-seqs 1 --outdir "$W/t3lp" --prompt-format chat --logprobs 5 --max-model-len 40960 > "$W/t3lp.log" 2>&1
  check "logprobs on the off arm: run_id _chat_lp5, every request read" "python3 -c \"
import json,glob;r=json.loads(open('$W/t3lp/results.jsonl').readline())
q=json.load(open(glob.glob('$W/t3lp/perreq_*')[0]))[0]['logprobs']
exit(not(r['run_id'].endswith('_chat_lp5') and r['logprobs']==5 and r['logprobs_seen']==3
         and q['chosen'][0]==-0.1 and q['top_ids'][0]==[0,1000] and len(q['top_lps'])==len(q['chosen'])))\""
  python3 "$H5D" --dataset "$D/qa/2K.jsonl" --model "$QWEN" --rung ngram_gpu --k 5 --bucket 2K --limit 3 \
    --max-num-seqs 1 --outdir "$W/t3lpx" --prompt-format chat --logprobs 5 > "$W/t3lpx.log" 2>&1
  check "logprobs refused on a speculative arm" "grep -q 'off arm only' $W/t3lpx.log && [ ! -f $W/t3lpx/results.jsonl ]"
  python3 "$H5D" --dataset "$D/toolong/2K.jsonl" --model "$QWEN" --rung off --bucket 2K --limit 3 --max-tokens 64 \
    --max-num-seqs 1 --outdir "$W/t3l" --prompt-format chat --max-model-len 4096 > "$W/t3l.log" 2>&1
  check "too-long prompts abort before the engine loads" "grep -q 'prompts too long' $W/t3l.log && [ ! -f $W/t3l/results.jsonl ]"
else
  skip "chat template tests (set QWEN_TOK to a Qwen3-8B tokenizer directory)"
fi

echo "T4 router_cell.sh"
cell () { (cd "$CODE" && env "$@" bash router_cell.sh) ; }
OUT4=$W/root/results/t4
if [ "$HAVE_TOK" = "1" ]; then
  : > "$W/stub.log"
  cell BUCKET=2K LIMIT=3 LADDER_DATA="$D/qa" LADDER_OUT="$OUT4" > "$W/t4.log" 2>&1
  check "five arms in order off, ngram 5 11, dflash 7 15" "[ \"\$(grep -o 'rung=[a-z_]* k=[0-9]*' $W/t4.log | tr '\n' ' ')\" = 'rung=off k=0 rung=ngram_gpu k=5 rung=ngram_gpu k=11 rung=dflash k=7 rung=dflash k=15 ' ]"
  check "five rows, cell done" "[ \$(grep -c . $OUT4/results.jsonl) = 5 ] && grep -q '^done' $W/t4.log"
  check "max_model_len 40960 and chat format reach the engine" "grep -q '\"max_model_len\": 40960' $W/stub.log && grep -q '\"kinds\": \\[\"dict\"\\]' $W/stub.log"
  cell BUCKET=2K LIMIT=3 SKIP_OFF=1 PERREQ=detailed K_NGRAM_GPU=11 K_DFLASH=15 LADDER_DATA="$D/qa" LADDER_OUT="$W/root/results/t4a" > "$W/t4a.log" 2>&1
  check "SKIP_OFF: two arms, no off" "! grep -q 'rung=off' $W/t4a.log && [ \$(grep -c . $W/root/results/t4a/results.jsonl) = 2 ]"
  check "per-request acceptance recorded" "python3 -c \"
import json,glob;b=json.load(open(sorted(glob.glob('$W/root/results/t4a/perreq_*'))[0]))
exit(not all(r['spec_decode_metrics']['per_step_accepted']==[1,2,1] for r in b))\""
else
  skip "router_cell arm runs (need the tokenizer for chat mode)"
fi
if [ "$HAVE_TOK" = "1" ]; then
  cell BUCKET=2K LIMIT=3 LOGPROBS=5 K_NGRAM_GPU= K_DFLASH= LADDER_DATA="$D/qa" LADDER_OUT="$W/root/results/t4l" > "$W/t4l.log" 2>&1
  check "LOGPROBS: the off arm alone, one row" "[ \"\$(grep -o 'rung=[a-z_]* k=[0-9]*' $W/t4l.log | tr '\n' ' ')\" = 'rung=off k=0 ' ] && [ \$(grep -c . $W/root/results/t4l/results.jsonl) = 1 ]"
fi
cell BUCKET=2K LIMIT=3 LOGPROBS=5 LADDER_DATA="$D/qa" LADDER_OUT="$W/x" > "$W/t4e.log" 2>&1
check "LOGPROBS with drafter arms refused" "grep -q 'off-arm-only cell' $W/t4e.log"
cell BUCKET=4K LIMIT=3 LADDER_DATA="$D/qa" LADDER_OUT="$W/x" > "$W/t4b.log" 2>&1
check "unknown bucket refused" "grep -q \"BUCKET='4K'\" $W/t4b.log"
cell BUCKET=16K LIMIT=30 LADDER_DATA="$D/code" LADDER_OUT="$W/x" > "$W/t4c.log" 2>&1
check "LIMIT beyond the file refused" "grep -q 'needs 32 rows' $W/t4c.log"
cell BUCKET=2K LIMIT=3 K_DFLASH=16 LADDER_DATA="$D/qa" LADDER_OUT="$W/x" > "$W/t4d.log" 2>&1
check "DFlash k above 15 refused" "grep -q 'at most 15' $W/t4d.log"

echo "T5 dtai3_job.sh cell settings"
cat > "$W/cell_stub.sh" <<'EOF'
#!/bin/bash
echo "$(basename "$LADDER_OUT")|$BUCKET|${CONC:-}|${LIMIT:-}|${SKIP_OFF:-}|${PERREQ:-}|${THINK:-}|${MAXTOK:-}|${PC:-}|${K_NGRAM_GPU:-}|${K_DFLASH:-}|$(basename "$LADDER_DATA")|${LOGPROBS:-}" >> "$CELLLOG"
EOF
chmod +x "$W/cell_stub.sh"
job () { (cd "$CODE" && env LADDER_CELL="$W/cell_stub.sh" CELLLOG="$W/cells.log" "$@" bash dtai3_job.sh) > "$W/job.log" 2>&1; }
: > "$W/cells.log"; job PIECE=x1 WL=qa B=2K CELLS="t a c4 c32"
check "t: c1, 50 prompts, ngram 5 11, dflash 7 15" "grep -q '^gh200r_qa_2K_t_px1|2K|1|50||||||5 11|7 15|qa|$' $W/cells.log"
check "a: no off, detailed, all four arms (ngram 5 11, dflash 7 15)" "grep -q '^gh200r_qa_2K_a_px1|2K|1|50|1|detailed||||5 11|7 15|qa|$' $W/cells.log"
check "c4 and c32: 128 prompts, ngram 5, dflash 4 7 15" "grep -q '^gh200r_qa_2K_c4_px1|2K|4|128||||||5|4 7 15|qa|$' $W/cells.log && grep -q '^gh200r_qa_2K_c32_px1|2K|32|128|' $W/cells.log"
: > "$W/cells.log"; job PIECE=x2 WL=qa B=32K CELLS="t c8"
check "32K: 25 prompts at batch 1, 48 under load" "grep -q '^gh200r_qa_32K_t_px2|32K|1|25|' $W/cells.log && grep -q '^gh200r_qa_32K_c8_px2|32K|8|48|' $W/cells.log"
: > "$W/cells.log"; job PIECE=x3 WL=code B=16K CELLS="t"
check "a short file caps the prompt count (27 rows -> 25)" "grep -q '^gh200r_code_16K_t_px3|16K|1|25|' $W/cells.log"
: > "$W/cells.log"; job PIECE=x4 WL=ragquote B=2K CELLS="t" LIM1=62
check "ragquote: LIM1 62 and prefix caching off" "grep -q '^gh200r_ragquote_2K_t_px4|2K|1|62|||||off|5 11|7 15|ragquote|$' $W/cells.log"
: > "$W/cells.log"; job PIECE=x5 WL=qa B=2K CELLS="tT aT"
check "thinking cells: on, 2048 tokens, 25 prompts" "grep -q '^gh200r_qa_2K_tT_px5|2K|1|25|||on|2048|' $W/cells.log && grep -q '^gh200r_qa_2K_aT_px5|2K|1|25|1|detailed|on|2048|' $W/cells.log"
: > "$W/cells.log"; job PIECE=x11 WL=qa B=2K CELLS="l lT"
check "l: off alone, 50 prompts, logprobs 5; lT: thinking, 25, 2048" "grep -q '^gh200r_qa_2K_l_px11|2K|1|50||||||||qa|5$' $W/cells.log && grep -q '^gh200r_qa_2K_lT_px11|2K|1|25|||on|2048||||qa|5$' $W/cells.log"
: > "$W/cells.log"; job PIECE=x6 WL=mix B=mix CELLS="c8"
check "mix: 64 prompts under load" "grep -q '^gh200r_mix_mix_c8_px6|mix|8|64|' $W/cells.log"
job PIECE=x7 WL=nosuch B=2K CELLS="t"
check "missing workload refused" "grep -q 'no workload' $W/job.log"
job PIECE=x8 WL=qa B=2K CELLS="t zz"
check "unknown cell fails the piece" "grep -q 'unknown cell zz' $W/job.log && grep -q INCOMPLETE $W/job.log"
job PIECE=x-9 WL=qa B=2K CELLS="t"
check "bad piece tag refused" "grep -q 'letters and digits only' $W/job.log"
job PIECE=x10 WL=qa B=2K CELLS="t" LADDER_ROOT=/projects/bikd/x
check "/projects refused" "grep -q 'under /projects' $W/job.log"

echo "T6 submit_dtai3.sh"
(cd "$CODE" && bash submit_dtai3.sh TIERS="S 0") > "$W/plan.log" 2>&1
check "plan lists the smoke piece with all three batch-1 cells" "grep -q '^S    smoke .* t a l ' $W/plan.log"
check "pieces with no file are held, not submitted" "grep -q 'HELD: no .*/summ/2K.jsonl' $W/plan.log"
check "plan only without GO" "grep -q 'plan only' $W/plan.log && [ ! -f $W/sbatch.log ]"
(cd "$CODE" && bash submit_dtai3.sh TIERS=0 ONLY="qa2 qa32La" GO=1) > "$W/go.log" 2>&1
check "GO=1 submits exactly the pieces asked for" "[ \$(grep -c . $W/sbatch.log) = 2 ]"
check "sbatch gets the piece settings and a dtai3 job name" "grep -q 'PIECE=qa2 WL=qa B=2K CELLS=t a l LIM1= ' $W/sbatch.log && grep -q 'job-name=dtai3-qa_2K-pqa2' $W/sbatch.log"
check "one GPU, 16 cores, 64g, ghx4" "grep -q -- '--partition=ghx4 --account=acct --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g' $W/sbatch.log"
check "job name matches the feeder's pattern" "[[ 'dtai3-eq_mdqa_realpad_8K-peqmdR' =~ ^dtai3-(.+)-p[A-Za-z0-9]+\$ ]]"

echo "T7 end to end through the real router_cell.sh and harness v5d"
if [ "$HAVE_TOK" = "1" ]; then
  (cd "$CODE" && PIECE=e2e WL=qa B=2K CELLS="t a l" LIM1=3 bash dtai3_job.sh) > "$W/e2e.log" 2>&1
  check "piece done, all three cells exit 0" "grep -q '^done' $W/e2e.log && grep -q 'cell t exit=0' $W/e2e.log && grep -q 'cell a exit=0' $W/e2e.log && grep -q 'cell l exit=0' $W/e2e.log"
  check "l cell: one off row with logprobs for every request" "python3 -c \"
import json
l=[json.loads(x) for x in open('$W/root/results/gh200r_qa_2K_l_pe2e/results.jsonl')]
exit(not(len(l)==1 and l[0]['rung']=='off' and l[0]['logprobs_seen']==3))\""
  check "rows: 5 timing, 4 acceptance (one per arm), all v5d chat" "python3 -c \"
import json
t=[json.loads(l) for l in open('$W/root/results/gh200r_qa_2K_t_pe2e/results.jsonl')]
a=[json.loads(l) for l in open('$W/root/results/gh200r_qa_2K_a_pe2e/results.jsonl')]
exit(not(len(t)==5 and len(a)==4 and all(r['harness'].startswith('v5d') and r['prompt_format']=='chat' for r in t+a)))\""
  (cd "$CODE" && PIECE=e2e2 WL=chat B=2K CELLS="t" LIM1=3 bash dtai3_job.sh) > "$W/e2e2.log" 2>&1
  check "multi-turn workload end to end" "grep -q '^done' $W/e2e2.log"
else
  skip "end to end (needs the tokenizer)"
fi

echo
echo "passed $PASS, failed $FAIL, skipped $SKIPPED"
[ "$FAIL" -eq 0 ]
