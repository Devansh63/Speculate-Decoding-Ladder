# Router data jobs (dtai3), 25 Nov 2026 (revised 28 Nov)

**Status:** built and tested offline (stub engine, real Qwen3 tokenizer, real
source data); nothing has run on DeltaAI yet. The smoke piece is the first
real test. Revised 28 Nov after a third check and a look at what recent
drafter-routing work trains on (section 1b): a logprob cell (`l`), Spec-Bench's
six categories, a fixed `mix`, DeepSeek's thinking mode turned off, and
gzipped per-request files on push.

## 1. Why these jobs

The 24 Nov reports found four gaps, and each one decides what the router can
learn:

1. **Every prompt so far was raw text.** Harness v5c sends the prompt through
   `llm.generate` with no chat template. Qwen3-8B then never emits an
   end-of-text token (0 of 525 no-speculation outputs) and some tasks loop
   ("Answer: Jupiter" repeated). Served traffic is chat format, and DFlash was
   trained on chat-style data with target-regenerated responses. Every
   conclusion so far might shift in chat format.
2. **No data where the decision changes.** 16K was never measured, tasks never
   above 8K, load only on QA.
3. **Per-prompt routing was not learnable** from the 525 prompts: prompt
   statistics explained 3 to 4% of the per-prompt difference, because inside
   each task the instruction never varies. The signal that matters (how much
   the output copies the prompt) needs data whose instructions vary in how
   much copying they ask for.
4. **No test of padding.** LongBench v1 has 0 to 6 documents long enough for
   each task at 32K, so 32K task data has to be constructed. Whether a
   construction behaves like natural text has to be measured, not assumed.

## 1b. What the router needs, from recent work (28 Nov)

Our per-prompt router learned nothing (3 to 4% explained, 0% gain under
cross-validation). Recent drafter-routing work points to why, and to what
data a learnable router needs:

- **The best drafter changes inside one answer.** SpecRouter (NeurIPS 2026)
  chooses among retrieval, a small draft model and a tree drafter before every
  draft-verify block; retrieval wins while the model copies its prompt and
  loses once it writes new text. It predicts each mechanism's accepted length,
  prices it with a calibrated cost table, and beats the best single mechanism
  by 27 to 30% on two 8B targets. WhiFlash (EMNLP 2026 Findings, already
  threat 18's omission) routes per step between DFlash and EAGLE-3 on
  Qwen3-8B with an entropy or a small neural policy, and shows that
  task-level routing leaves most of the oracle's gain unused.
- **The labels can be full-information.** Not-a-Bandit (arXiv:2510.20064)
  scores every drafter on the trajectory the chosen one produced. Our runs are
  greedy and lossless, so every arm writes the same tokens (where it does not,
  the divergence point is known): n-gram's proposal at every position can be
  replayed offline from the token ids, and DFlash's acceptance is measured at
  each of its steps by the `a` cell (k 7 and 15 start steps at different
  positions).
- **The features are step-level.** What a step-level policy reads: the
  target's confidence at the current position (WhiFlash's entropy policy;
  SpecRouter reads the hidden state, which vLLM does not expose), the
  current n-gram match length, and each drafter's recent acceptance. Only the
  first needs the GPU, so the `l` cell records it: the target's top-5
  logprobs at every generated position, on the off arm alone.
- **The request mix.** Spec-Bench (translation, summarization, QA, math, RAG,
  MT-bench) is what this literature reports on, and translation and math were
  in no dtai3 workload. Agentic traces are where retrieval drafting wins most
  (SuffixDecoding, NeurIPS 2025, SWE-Bench); we have them (`agent`).

So the training example is a step, not a prompt: about 50 prompts of up to
1,024 tokens give tens of thousands of labelled steps per cell, and the
prompt-level counts below are enough. The cost side (time per step for each
drafter, k and load) comes from the `t` and `c` cells and the two-constant
cost model (notes/2026-11-18-cost-model.md).

## 2. What is new

| File | What it is |
|---|---|
| `code/run_benchmark.v5d.py` | Harness v5d: a variant of v5c. `--prompt-format chat` applies the target's own chat template, passes token ids, stops at `<|im_end|>`; `--thinking on/off`; multi-turn rows (`messages`); a too-long guard before the engine loads. Raw mode reproduces v5c (same run_id, same prompts). |
| `code/build_router_workloads.py` | Builds every workload below into `$LADDER_ROOT/data/router/<workload>/<bucket>.jsonl` plus `manifest.json`, and writes the requests for the AI step. |
| `code/gen_router_ai.py` | Asks DeepSeek for the AI-written inputs (standard library only; key from the environment only). |
| `code/router_cell.sh` | One cell: every arm of one workload file. A variant of `ksweep.sh` for v5d. |
| `code/dtai3_job.sh` | One Slurm job = one piece = a few cells of one workload file. |
| `code/submit_dtai3.sh` | The piece table, tiers, a time estimate, and `GO=1` to submit. |
| `code/dtai3_feeder.sh` | `dtai2_feeder.sh` for dtai3 job names. |
| `code/push_dtai3.sh` | Copies `gh200r_*` results and dtai3 logs into `results/gh200r/` and pushes. |
| `code/test_dtai3.sh` | Offline tests (59 checks with the tokenizer). |

Nothing existing was edited (RULES section 6). `test_dtai2.sh` still passes
(116 of 116).

## 3. The cells

Every cell: harness v5d, chat format, thinking off, natural stopping, at most
1,024 new tokens, max model length 40,960 (Qwen3-8B's native maximum; the raw
runs used 33,792, which a 32K prompt plus 1,024 tokens would exceed).

| Cell | Concurrency | Arms | What it gives the router |
|---|---|---|---|
| `t` | 1 | off; n-gram k 5, 11; DFlash k 7, 15 | per-prompt time for each drafter and k |
| `a` | 1 | n-gram k 5, 11; DFlash k 7, 15, per-request acceptance | the labels: accepted tokens at every step, per prompt, measured for every k the router chooses between |
| `c<N>` | N | off; n-gram k 5; DFlash k 4, 7, 15 | where load moves the switch point and the best k |
| `l` | 1 | off alone, `--logprobs 5` | the target's top-5 logprobs at every generated position: the step-level confidence feature, and the greedy reference sequence the `a` cell's steps are aligned to. Off arm only, so no timing or speculative row carries the extra work (added 28 Nov) |
| `tT`, `aT`, `lT` | 1 | as `t`, `a`, `l`, thinking on, 2,048 tokens, 25 prompts | Qwen3's reasoning mode, a different output distribution |

The `a` cell has no off arm: vLLM 0.33.0 refuses per-request acceptance
without a drafter, which is why every dtai2 `acc` cell exited 1.

The `a` cell measures all four arms rather than reading k 5 and 7 off the
k 11 and 15 steps. Reading them off is exact only for n-gram; for DFlash the
k 7 block is not the first 7 tokens of the k 15 block. And k 11 and 15 are
the two arms whose output left greedy in Devansh's environment (dtai2, 24
Nov), so labels taken only from them would carry that divergence. With
the pieces this forces to split, it raises the total from about 28 to about
34 GPU-hours.

**Compare arms per generated token.** With natural stopping, an arm that
changes the text can also change its length; use `decode_s / gen_tokens` per
request, not total elapsed time alone.

## 4. The workloads

All lengths are Qwen3-8B tokens of the user content; the template adds about
10 to 20. "Measured" is per cell; every file also holds two warm-up rows.

| Workload | Buckets (measured) | Source | Construction |
|---|---|---|---|
| `qa` | 2K, 8K (50 at batch 1, 128 under load), 16K (50; 64), 32K (25; 48) | LongBench v2 | natural, the raw runs' rule; the first rows are exactly the raw runs' prompts |
| `code` | 2K, 8K, 16K (50), 32K (25) | LongBench lcc, repobench-p; LongBench v2 code repositories at 32K | natural to 16K; `repo_prefix` at 32K (a real repository cut at a line boundary) |
| `summ` | 2K, 8K (50), 16K (25), 32K (25) | gov_report; LongBench v2 single documents at 32K | natural to 16K; `long_document` at 32K |
| `fewshot` | 2K, 8K (50), 16K (50), 32K (25) | SAMSum | natural to 8K; `many_shot_real` above (more real SAMSum examples) |
| `mdqa` | 2K, 8K, 16K (50), 32K (25) | HotpotQA | natural to 16K; `real_distractors` at 32K (passages of other real HotpotQA items, shuffled, the item's own passages kept) |
| `ragquote` | 2K, 8K, 16K (62), 32K (46) | LongBench v2 QA documents not used by `qa` | one document, four instructions: extract word for word, answer with quotes, explain in your own words, letter only. The instruction is the only variable |
| `codeedit` | nat (50, InstructCoder), 2K, 8K, 16K (50) | InstructCoder valid; LongBench lcc and repobench-p files | four edit requests over real code: rename everywhere (whole file back), comment one function, refactor one function, explain one function |
| `chat` | 2K, 8K (50) | WildChat-1M, English, not toxic, not redacted, 2+ user turns | natural length: the conversation up to its last user turn |
| `agent` | 8K, 16K (50), 32K (25) | SWE-agent trajectories (nebius) | natural length: a real trace cut before an assistant turn |
| `eq_*` | 8K (50 each) | the 2K mdqa and fewshot items | the padding test, section 5 |
| `mix` | mix (64) | QA 2K and 32K alternating, each document once | load cells with mixed lengths in one batch (28 Nov: 23 of 52 rows had paired a 2K prompt with the 32K prompt of the same document, a prefix-cache hit; now no document appears twice) |
| `sb_mt`, `sb_trans`, `sb_summ`, `sb_qa`, `sb_math`, `sb_rag` | nat (78 each) | Spec-Bench, pinned to commit fd2c1cd and the file's sha256 | natural; MT-bench first turns only (a second turn needs the model's own first answer) (added 28 Nov) |
| `synth` | 2K, 8K, 16K (58), 32K (30) | LongBench v2 documents, all six domains | DeepSeek-written instructions across six copy levels (section 6) |
| `codeedit_ai` | 2K, 8K, 16K (50) | the `codeedit` excerpts | DeepSeek-written edit requests |
| `twins_real`, `twins_ai` | nat (50 each) | WildChat single-turn requests; DeepSeek twins of them | the synthetic-prompt test (section 6) |

Not built: **64K.** Qwen3-8B's native context is 40,960 positions; 64K needs
YaRN rope scaling, which changes the model being measured. Deliberately left
out.

Not used: **SPEED-Bench's throughput splits.** Their low-entropy prompts are
padded to the bucket length with the sentence "Answer now please." repeated
on line after line, and many rows are placeholders reading "FULL BENCHMARK
DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH" (checked on the
16K and 2K parquet files, 25 Nov). A repeated sentence is exactly the padding
that hands n-gram free matches, so the `eq_*_repeatpad` cells reproduce it on
purpose, as the negative control.

## 5. Padding: the policy and the test

**Policy.** No prompt used for training is padded with invented or repeated
text. Length comes from real text of the task's own kind: real distractor
passages (how HotpotQA defines its distractor setting), real many-shot
examples, a real repository prefix, a real long document. Every row records
its `construction`, so no constructed prompt is ever pooled with a natural one
by accident.

**The test** (tier 1 and 2, 8K, mdqa and fewshot, 50 prompts each):

| Workload | What the 8K prompt is |
|---|---|
| `mdqa` / `fewshot` 8K (tier 0) | natural: a real document long enough on its own |
| `eq_*_realpad` | the 2K cell's exact text, padded to 8K with real passages or real examples |
| `eq_*_llmpad` | the same 2K text, padded with DeepSeek-written passages or dialogues |
| `eq_*_repeatpad` | the same 2K text, padded with "Answer now please." lines (SPEED-Bench's padding) |

A construction passes for a task if, against the natural 8K cell, every arm's
speedup is within 2%, the DFlash-to-n-gram tau ratio within 0.05, and
acceptance by draft position within the run-to-run spread. The 32K `fewshot`
(many-shot) and `mdqa` (distractors) cells are trustworthy only if
`realpad` passes. `repeatpad` is expected to fail, and how badly it fails is
the number that shows what padding does to n-gram.

## 6. The AI step (DeepSeek)

**What DeepSeek writes:** inputs only.

- `synth`: for each of 106 real documents, six requests, one per copy level
  (verbatim, edit, quote, summarize, brief, create); each document is used
  with two of them, rotating so every level appears equally.
- `codeedit_ai`: four edit requests per real code excerpt (whole file, one
  function, fix or feature, question).
- Filler for `eq_*_llmpad`: invented encyclopedia passages and invented chat
  dialogues. Used only in the padding test.
- `twins_ai`: for each of 50 real WildChat requests, a new request of the same
  kind and size.

**What it never writes:** the text the drafters are measured on. That is
always Qwen3-8B's own output in the GPU job. Acceptance depends on what the
target writes; it is why drafter training recipes regenerate every response
with the target, and why our labels are measurements, not an LLM's judgment.

**Cost:** about 2.7M tokens in and 0.5M out, about $1.40 at deepseek-flash
peak rates (half off-peak). `gen_router_ai.py --dry-run` prints it.

**The synthetic-prompt test:** `twins_real` against `twins_ai`. If the
distribution of per-prompt tau ratios differs, synthetic prompts need
weighting or validation before training on them; either way the router is
evaluated only on real prompts.

## 7. Tiers and cost

Estimates from the submitter's rough model (1.5 minutes to load each arm,
guessed output lengths per workload); the smoke piece and the first tier-0
pieces replace the guesses.

| Tier | Pieces | Est. GPU time | What it answers |
|---|---|---|---|
| S | 1 | 17 min | Does chat mode work on DeltaAI (template, stop ids, per-request acceptance, logprobs)? |
| 0 | 42 | about 15 h | Does the crossover survive the chat format? Where is the switch point at 16K? Do tasks flip the winner at 16K or 32K? How does load move it? |
| 1 | 33 | about 14 h | Does the instruction change the winner on one document? Code edits, real chats, agent traces, Spec-Bench's six categories; the padding test |
| 2 | 15 | about 7 h | AI-written prompts across copy levels; LLM padding; synthetic against real prompts |
| 3 | 13 | about 5 h | Tasks under load, mixed-length batches, thinking on |

Every piece fits a 30-minute or 1-hour walltime (short limits start sooner on
ghx4; LOG, 18 Nov): the heaviest pieces are split into a `t` piece and an
`a` piece. Total about 41 GPU-hours (104 pieces). The model overestimates:
against the dtai2 off arms at 256 tokens it predicts 33% too much time at 2K
and almost twice the time at 32K, so the real figure is likely lower.

**Charges.** Pieces that `dtai3_feeder.sh` moves to ghx4-interactive are
charged at twice the GPU-hour rate (qos_ghx4int, LOG 18 Nov), so 41 GPU-hours
can cost up to 82 charged hours if everything goes through the feeder. Check
the balance with `accounts` before submitting tiers 2 and 3.

**Suggested split:** Devansh runs S and 0; Sahil runs 1 on identical inputs:
either `LADDER_ROUTER=/work/nvme/bikd/dagraw2/Speculative_Ladder/data/router`
if he can read that directory, or his own build (the build is deterministic:
every random choice is seeded from the item id). Either way, compare the
`sha256` of each file in the two `manifest.json` files before running.
Whoever has hours left runs 2 and 3.

## 8. How to run it

On the DeltaAI login node, in tmux (the build downloads about 1.2 GB with
the default two WildChat and two SWE-agent shards, and takes 5 to 20 minutes):

```bash
ssh dagraw2@dtai-login.delta.ncsa.illinois.edu
tmux new -s build
cd /work/nvme/bikd/dagraw2/Speculative_Ladder/repo && git pull && cd code
export LADDER_VENV=/work/nvme/bikd/ssashi/Speculative_Ladder/vllm033_gh200   # the venv every dtai2 job used
PROJ=bikd ACCOUNT=bikd-dtai-gh source deltaai_env.sh
"$LADDER_VENV/bin/python3" -c "import transformers, pyarrow, huggingface_hub, jinja2" || { "$LADDER_VENV/bin/python3" -m venv "$LADDER_ROOT/buildenv" && "$LADDER_ROOT/buildenv/bin/pip" install -q transformers pyarrow huggingface_hub jinja2; }
BPY=$( "$LADDER_VENV/bin/python3" -c "import transformers, pyarrow, jinja2" 2>/dev/null && echo "$LADDER_VENV/bin/python3" || echo "$LADDER_ROOT/buildenv/bin/python3" )
"$BPY" build_router_workloads.py --check
"$BPY" build_router_workloads.py
PATH="$(dirname "$BPY"):$PATH" QWEN_TOK=$LADDER_HF/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 bash test_dtai3.sh
```

(`LADDER_VENV`: Devansh has no venv of his own on DeltaAI; every dtai2 job ran
on Sahil's, as every dtai2 log records, and the dtai3 jobs must use the same
binary for their ratios to sit beside dtai2's. `deltaai_env.sh` keeps a
`LADDER_VENV` that is already set. Sahil leaves the line out.

The build venv exists so the measurement venv is never changed by a
`pip install`. It needs `jinja2` as well: `transformers` applies chat
templates with it but does not install it (28 Nov, on DeltaAI). The `PATH`
prefix gives the test the python that has `transformers`.)

Then the smoke piece, and read its log before anything else:

```bash
bash submit_dtai3.sh TIERS=S GO=1
```

Smoke checklist (log `$LADDER_ROOT/logs/dtai3-smoke-<job>.out`):
- each arm prints `chat template sha256 ...` and `stop ids [151645, 151643]`;
- no `INCOMPLETE`; `results.jsonl` in `gh200r_qa_2K_t_psmoke` has 5 rows and in
  `gh200r_qa_2K_a_psmoke` 4 (one per arm);
- `finished_stop` close to 4 in each row (most answers end on their own) and
  `gen_tokens` well under 4 x 1024;
- in the `a` rows, `per_request_spec_seen` is 4;
- `gh200r_qa_2K_l_psmoke` has 1 row with `logprobs_seen` 4, and its
  `perreq_*.json` holds `logprobs.chosen` lists as long as `gen_tokens`, with
  no `error` key (if vLLM 0.33.0 on Model Runner V2 refuses logprobs, only
  the `l` cells fail; everything else is unaffected).

Then tier 0, with the feeder moving waiting pieces through ghx4-interactive:

```bash
bash submit_dtai3.sh TIERS=0 GO=1
tmux new -s feeder3 "bash dtai3_feeder.sh"
```

The AI step, on the login node (or a laptop, copying `data/router/_ai` there
and back if the login node cannot reach the API). The key goes in the shell
only; `read -rs` keeps it out of the history, and `submit_dtai3.sh` calls
`sbatch` with the key removed from its environment, so it never reaches a
GPU job:

```bash
read -rs DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY
"$BPY" gen_router_ai.py --dry-run
"$BPY" gen_router_ai.py --limit 1     # a one-request-per-file pilot: all ok?
"$BPY" gen_router_ai.py
"$BPY" build_router_workloads.py --only ai
bash submit_dtai3.sh TIERS=2 GO=1
```

Push results as they land:

```bash
bash push_dtai3.sh "results: dtai3 tier 0 pieces <job ids>"
```

## 9. How the files were checked (25 Nov)

- `build_router_workloads.py` run on the real sources (one WildChat shard and
  one SWE-agent shard; the defaults read two of each): every file in
  section 4 built with the counts shown. The first rows of `qa` at 2K, 8K, 32K
  and of the 2K and 8K task files are exactly the raw runs' measured and
  warm-up prompts (ids checked against `results/gh200`), so every chat cell
  has a raw twin already measured.
- `gen_router_ai.py` against a local mock of the API: 404 requests answered,
  a rerun asked nothing, and the AI workloads assembled.
- A second pass the same day found six problems, all fixed before any job
  ran: (1) `mix` held the same document twice under one id when it was in
  both QA buckets (ids now start `2K:` or `32K:`); (2) the cut to length
  dropped one of the item's own passages in 13 of 27 `mdqa` 32K rows, 17 of
  52 `eq_mdqa_realpad` and 10 of 52 `eq_mdqa_llmpad`, making those questions
  unanswerable (the builder now reshuffles until every own passage is kept);
  (3) 4 of the 32K `code` repositories were also 32K `qa` documents (now
  excluded); (4) the `a` cell's k 5 and 7 labels were approximations from
  the diverging arms (now measured, section 3); (5) the submitter's estimate
  ran the system python, and `sbatch --export=ALL` would have copied an
  exported `DEEPSEEK_API_KEY` into every job's environment (now the venv's
  python, and `env -u DEEPSEEK_API_KEY sbatch`); (6) four pieces needed a
  2-hour walltime once the `a` cell had four arms (now split). After the
  fixes, a rebuild and audit: no duplicate ids, every own passage kept, no
  document shared between the 32K `qa`, `code` and `summ` files, every
  named function present in its code edit, the longest templated prompt
  33,263 tokens (under 40,960 minus 2,048).
- `test_dtai3.sh`: 53 of 53 with the real tokenizer (36 pass and 3 skip
  without it). The chat path was checked with Qwen3-8B's real template:
  `<|im_start|>user ... <|im_start|>assistant` with the empty think block when
  thinking is off, and stop ids 151645 and 151643.

- A third pass on 28 Nov (after section 1b's reading) found and fixed:
  `mix` paired a 2K and a 32K prompt of the same document in 23 of its first
  52 rows (a partial prefix-cache hit on the 32K prefill); `gen_router_ai.py`
  did not turn off deepseek-flash's thinking mode, which DeepSeek's docs now
  make the default (reasoning would have used `max_tokens` before the JSON;
  it now sends `"thinking": {"type": "disabled"}`, and the mock API rejects a
  request without it); per-request files would have added hundreds of MB of
  JSON to the repository (`push_dtai3.sh` now gzips them). Added: the `l`
  cell (harness v5d `--logprobs`, off arm only), Spec-Bench. Five pieces that
  then needed 2 hours were split. Harness v5d had produced no row before this
  change, so it was changed in place (version `v5d-2026-11-28`) rather than
  copied. Checked: the rebuilt `mix` uses 66 distinct documents, Spec-Bench
  builds 80 rows per category, `test_dtai3.sh` 59 of 59, `test_dtai2.sh` 116
  of 116, a pilot of `gen_router_ai.py` against the mock (10 of 10).

## 10. Known limits

- Not observed on a GH200: token-id prompts and `stop_token_ids` through vLLM
  0.33.0's `LLM.generate`, and per-request acceptance on chat prompts. The
  smoke piece shows both.
- The time model is a guess, and chat output lengths are unknown until tier 0.
- `max_model_len` 40,960 against 33,792 in the raw runs changes the engine's
  memory plan; it is the same for every arm of a cell, but note it before
  putting a raw and a chat ratio side by side.
- WildChat and the SWE-agent traces are other people's conversations and
  runs. They are filtered (English, not flagged toxic, not redacted) but not
  read by hand; dataset licences were not checked beyond their cards.
- 25 prompts at 32K, as before.
- `agent` rows share SWE-agent's system prompt, so with prefix caching on
  (the default) every agent prefill reuses it. That is how an agent is served,
  and it is the same for every arm, so it is kept.
- The `l` cell's confidence comes from the off arm. Under lossless greedy
  decoding the speculative arms write the same tokens, so it applies to them
  up to the first divergence (n-gram k 11 and DFlash k 15 diverged from greedy
  on the GH200 in dtai2; the open question to Sahil is why).
- Thinking on with greedy decoding can loop until the 2,048-token cap (Qwen
  recommends sampling for thinking mode). The `tT` and `aT` cells are timing
  and acceptance measurements, not answer quality; check `finished_stop`
  before using them.
