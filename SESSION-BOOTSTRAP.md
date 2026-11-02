# SESSION-BOOTSTRAP

**Paste this whole file as the first message of a new assistant session, then
say what you want done.** It is deliberately a pointer document, not a
substitute for the record. Everything load-bearing lives in files named below;
read them rather than trusting this summary. Last updated 2 November 2026, after Sahil's commit `bda744b`.

---

## 1. Who you are working with, and the standing constraints

Devansh Agrawal (`dagraw2@illinois.edu`), UIUC, CS 598 Hardware and Software for
AI, Fall 2026. Teammate: Sahil Sashi. Repo:
`github.com/Devansh63/Speculate-Decoding-Ladder`, local clone on the Mac at
`CS 598 HS4AI/Speculate-Decoding-Ladder`, which is a connected folder, so you
can read and edit it directly with a device shell rather than staging files.

Four constraints that hold for every turn:

1. **Git attribution is Devansh, always.** Commits are authored
   `Devansh Agrawal <dagraw2@illinois.edu>`. No AI co-author trailers, no
   "Generated with" footers, no assistant name anywhere in a commit message or
   PR body. Three author identities already exist in the history (an Illinois
   address, a Gmail address, a GitHub noreply address); all are his, and all
   read "Devansh Agrawal". Do not try to tidy them.
2. **No em dashes in anything written for him.** Commas, colons, semicolons or
   separate sentences instead. This applies to emails, docs, slides and repo
   prose.
3. **Deliverables go in the connected folder**, not only into the chat. Finished
   files belong in `CS 598 HS4AI/` or inside the repo, under names that do not
   already exist.
4. **Do not claim anything you have not checked.** See section 3. This is the
   single most important habit on this project and the one its author cares
   about most.

---

## 2. The project in one page

**Question.** Speculative decoding has a cheap drafter propose k tokens that the
target verifies in one forward pass. vLLM and SGLang let an operator enable
exactly one drafter at launch and never revisit it. We measure where each
drafter stops paying as context length, serving load and hardware change, and
ask whether a per-request router can pick the right one.

**Three research questions.** RQ1: does acceptance decay with context at serving
concurrency, and at different rates for different mechanisms? RQ2: can the point
where a mechanism stops paying be predicted on hardware the model was not fitted
on? RQ3 (now the lead, pending the professor's sign-off on the reframing): can a
routing policy that picks the drafter from context length and load beat every
fixed drafter?

**Apparatus.** Target Qwen3-8B bf16, inside vLLM at pinned commit
`0.29.1rc1.dev345`, torch 2.13, stable V1 model runner forced, on NCSA Delta.
Three rungs:

| Rung | What it is | Drafts on |
|---|---|---|
| `off` | No speculation. The baseline every speedup divides by | nothing |
| `ngram` | Prompt lookup: match the recent prefix against the prompt. No model, no weights, CPU only | 13 to 21% of steps |
| `dflash` | Qwen3-8B-DFlash-b16, a small trained block-diffusion draft head | about 100% of steps |

Workload is LongBench-v2 QA bucketed by prompt length into 2K, 8K and 32K
tokens. Hardware is A40 48 GB and A100 40 GB, with H200 pending. Decoding is
greedy, seed 1234, 256 output tokens.

**Why the two rungs were chosen.** They are opposite in kind. DFlash *predicts*
from the target's hidden states and attends over the whole context, so
prediction gets harder and more expensive as context grows. n-gram *retrieves* a
literal repeat, so a longer prompt gives it more to match and costs it only CPU
scanning. Everything measured follows from that asymmetry, and it is the thing
no published work exploits.

---

## 3. How to work on this project

These are from `RULES.md`, which is 13 numbered rules each naming the cost of
its absence. Read the file; these are the ones you will trip over first.

- **Pin the model runner.** Without `VLLM_USE_V2_MODEL_RUNNER=0`, n-gram
  silently falls back to the stable runner while DFlash stays on V2. Two arms on
  two code paths are not comparable.
- **Control concurrency with `max_num_seqs`.** The first run let the engine
  choose and measured an arbitrary load point of up to 77 concurrent.
- **Report within-job ratios, never absolute throughput across jobs.** Ratios
  replicate across nodes to 0.03 to 0.08%. Absolute throughput varied 8.8%
  against one node. Comparing across jobs measures the cluster.
- **Analyse `results.jsonl` per record, split on `env.slurm_job`.** Never take
  provenance from `summarize.py`, which collapses duplicate keys and once hid a
  replicated result for four days.
- **A Slurm state of COMPLETED means the loop finished, not that measurements
  succeeded.** Job 22685764 is COMPLETED and wrote nothing.
- **`LADDER_VENV` and `LADDER_OUT` are mandatory on every job.** Omitting the
  first on an A100 reserves a node and dies at kernel launch.
- **A prediction counts only if it was committed before the measurement.**
- **Do not write an escape hatch you have not checked.** If you cannot verify a
  caveat, say you did not verify it.

**The recurring failure mode**, named in `HANDOFF.md` section 9 and now also as
threat 18: almost every error on this project came from trusting a proxy (a
kernel listing, a traceback line, a summary table, a search snippet, a single
data point) instead of testing the thing itself. If you find yourself about to
assert something from a plausible-looking intermediate, stop and open the thing.

---

## 4. The cost model

Everything is read through one decomposition. Speedup is acceptance divided by
cost, and two measurements forced two extra terms into it.

| Term | Meaning | Belongs to |
|---|---|---|
| `tau_cond` | Mean accepted draft length, conditional on a draft being proposed | the drafter and the text |
| `cov` | Coverage: fraction of decode steps on which a draft was proposed at all | the drafter and the text |
| `R` | Verification cost ratio: one speculative step against one plain decode step | the machine and its load |
| `R_draft` | Fixed cost of drafting, paid every step regardless of k | drafter plus machine |
| `c` | Marginal cost per drafted token | the machine |

`tau_eff = cov * tau_cond + (1 - cov)` and `R(k) = R_draft + R_verify(k)`.

**Two things to understand before touching any number.** First, engines report
acceptance conditional on a draft existing, so a low-coverage drafter like
n-gram is overstated by more than 2x without the coverage term; uncorrected,
n-gram appears to out-accept DFlash (2.889 against 2.543 at 2K) while running
38% slower, which is incoherent. Second, `R` is **derived** in this project by
the identity `R = tau_eff / speedup`, so any ratio of `R` between machines
carries the acceptance ratio as well as the cost ratio. That is the defect
behind C20 and threat 17, and it is why the replacement is two reported columns,
a speedup ratio and a tau ratio, rather than one machine factor.

---

## 5. What was measured, and the headline

A40 grid complete; A100 has the two held-out buckets; H200 has nothing (the
architecture guard in `phase0.sh` aborts on compute capability 90).

**The ranking inverts along the context axis (C8), at concurrency 1:**

| context | DFlash k=7 | n-gram k=5 |
|---|---|---|
| 2K | **1.86x** | 1.16x |
| 8K | **1.26x** | 1.14x |
| 32K | 0.83x | **1.08x** |

Same ordering on the A100 (DFlash 0.64x against n-gram 0.97x at 32K). Two jobs
on two nodes three days apart agree on the ratios to 0.08%. **Say the below-1.0x
clause out loud every time:** at 32K the trained head is slower than not
speculating at all.

**Why it inverts, in two compounding penalties:**

| penalty | DFlash | n-gram |
|---|---|---|
| first-position acceptance, 2K / 8K / 32K | 0.601, 0.451, 0.296 | 0.601, 0.563, 0.552 |
| fixed drafting cost `R_draft`, 2K to 32K | 0.299 to 0.673 | 0.071 to 0.139 |

**Both** drafters pay more to draft as context grows. Saying "n-gram's drafting
cost does not grow" contradicts C12. What changes is the gap, 0.228 to 0.534,
and the gap is what flips the ranking.

**Other findings worth knowing.** Context taxes drafting, not verification: `c`
stays between 0.009 and 0.015 while `R_draft` grows four to seven times it
(C9, C20). DFlash costs 2.42 to 2.54 GiB of the KV pool at every concurrency up
to 12, about a fifth of the 32K concurrency ceiling, and n-gram costs nothing
(C19). The engine is bit-deterministic at fixed config, 150 of 150 sequences
identical at 2K batch 1 A40 with prefix caching on, bounding its own noise at
0.023% per token, against which speculation diverges at 0.56 to 0.78% per token,
so this build is **not** bitwise lossless (C21). Yet accuracy does not move:
nine arms at 8K score identically across 120 graded answers (C22). Plain decode
is bandwidth-bound at batch 1: the A100 is 2.21x the A40 against a bandwidth
ratio of 2.23 (C18).

**The frozen A100 prediction did not pass (C4, C17, C20).** Fitted on A40 with a
single machine factor of 1.054 and committed before the runs, it landed 2 of 4
cells within 5%, median error 6.2%, every error optimistic, and the misses split
by rung: n-gram 2 of 2 within tolerance, DFlash 0 of 2. **A single scalar cannot
produce a rung-dependent error**, so there is no single machine factor. Note the
pre-registration had two bars and hit neither, so the accurate wording is
"neither passed nor failed as pre-registered", not "not passed". Note also that
the 1.054 factor was fitted at 2K on a job that ran **five** prompts and then
applied to cells of 25 and 50, which is why rerunning A100 2K at n=50 is the
only real test of whether that parameter was ever estimated.

---

## 6. Where the literature leaves us, as of 2 November 2026

Verified against primary sources. Full evidence in
`../Adaptive K and drafter routing.md` and in `paper/related-work.md` under
"Verification pass, 2 November 2026".

**Taken.** Workload-dependent drafter ranking, including n-gram beating trained
heads, is published (MLSys 2026, *Performance or Illusion?*, same model, same
engine, and it already proposes the oracle). A trained drafter below break-even
at long context is published (OWL, 0.81x). Cross-paradigm routing between an
autoregressive drafter and **DFlash** on **Qwen3-8B** is published (WhiFlash,
per step, batch 1, HuggingFace, no context axis). Load-aware adaptive k with an
off switch on vLLM is published (Nightjar, AdaSpec, DSpark).

**Narrowed.** Context-keyed k exists as vLLM RFC #48627, PRs #48944 and #54801,
SGLang PR #31716 and the `depthchart` repo, all by one contributor, all
unmerged as far as verified, and his own evidence ledger concedes the context
axis has not beaten a static schedule. His drafters were all cost-growing, so
there was no asymmetry to exploit. That is the opening, and it must be addressed
head-on.

**Also taken, from Sahil's reading.** BanditSpec's drafter pool is PLD, REST, a
suffix tree and EAGLE-2, so "first to route across kinds of drafter" is dead.
Per-step confidence truncation of DFlash blocks is taken by DSpark, DScale,
AdaFlash, D2SD and BlockPilot. Do not claim either.

**The claim as one sentence, after both checks.** A joint per-request policy
over rung and draft length, with off in the action set and in the oracle, keyed
on prompt length and online acceptance through a measured two-term cost model,
for a drafter pair whose acceptance and cost move in **opposite directions**
with context, inside a production serving engine from 2K to 32K, with the cost
model fitted on one GPU and validated on another. The phrase that carries it is
"opposite-signed", not "across kinds". Drop any clause and a published work
covers it. The cross-hardware clause is the cleanest novelty: no
speculative-decoding paper does held-out-GPU prediction at all, and the
calibration point comes from outside the field (NeuSight, ASPLOS 2025, 8.1%
average error on OOD GPUs).

**Two surveys exist and they disagree.** Sahil's 48-paper survey is recorded in
`notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md`; an independent
citation check is in `notes/2026-11-02-verification-of-new-findings.md`. The
check found three claims in the first that are wrong or unverifiable (a
SpecExtend figure refuted on three counts, an AdaSpec "Corollary 4.4" that
could not be found, and a Compact Memories pair that could not be verified) and
five works the survey did not reach, of which **WhiFlash** (arXiv:2606.07710,
routing per step between EAGLE-3 and **DFlash** on **Qwen3-8B**) is the nearest
prior art in existence. Read both notes. Neither supersedes the other: the
first is better on the engine, the second on the citations.

**Engine reality for implementation.** Read section 4 of
`notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md`; it is from the source
at `bf13ecc2f` and it is authoritative. The headline is that **less needs
building than was assumed**. The pinned build already takes a per-step K from
`dynamic_sd_lookup` and passes `SchedulerOutput.num_spec_tokens_to_schedule` to
every proposer, and both `ngram` and `dflash` honor it including 0. More
important, `SchedulerConfig.scheduler_cls` is honored unconditionally, so the
policy can live in a **scheduler subclass with no vLLM source edit and no change
to the shared venv**. Five constraints bind it: the subclass must extend
`AsyncScheduler` or async scheduling is lost, and under async the per-request
lever is truncating `request.spec_token_ids`; a custom scheduler that varies K
without setting `num_speculative_tokens_per_batch_size` bypasses the override
forcing PIECEWISE cudagraphs, so set a constant table anyway and run the fixed-k
baselines through the same path or the comparison is confounded by graph mode;
DFlash buffers and graph sizes are fixed at construction from
`num_speculative_tokens`, so configure 15 and shrink; the `custom_class`
proposer receives token ids only and **cannot host a composite drafter**, which
belongs in the runner; and K=0 is **not** free, because the drafter forward
still runs with one query token to keep its KV in sync. Two further facts:
per-request acceptance arrays already exist via
`ObservabilityConfig.per_request_spec_decode_metrics = "detailed"`, which is the
label source for both oracles at no extra instrumentation; and **vLLM issue
#47930 is a live hazard** - prefix caching plus skipped DFlash steps leaves
restored prefix tokens without drafter KV and collapses acceptance to about
1.02. Every filed run has `prefix_caching: true`, so the caching-off control is
now a prerequisite rather than a nicety. The maintainers' own design for n-gram
plus a model drafter in one engine is RFC #18633 and PR #24344, closed stale 29
Oct 2026; build along it.

---

## 7. Reading order

Read in this order before doing anything substantive. Stop when you have what
the task needs.

1. `RULES.md` (13 rules, each with the cost of its absence), always.
2. `notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md`: the current
   direction, the prior-work table, two withdrawn assumptions, the engine facts
   at `bf13ecc2f`, and the phase plan. **The most important single file for
   anyone picking this up now.**
3. `notes/2026-11-02-verification-of-new-findings.md`: the independent citation
   check of the above, the works its survey missed, and the six claimable
   clauses re-read.
4. `HANDOFF.md`: current state, and section 9 on the recurring failure mode.
5. `TODO.md`: the live queue, with the NEW FINDINGS section at the top.
6. `paper/claims.md` is the append-only register: every claim, its evidence, its
   falsifier, its status. This is the project's memory. **Append, never rewrite.**
7. `paper/threats.md`: 18 numbered threats. Note a pre-existing numbering
   collision at 9.
8. `paper/related-work.md`: corrected and verified 2 Nov 2026.
9. `paper/results-table.md` and `paper/conclusions.md`.
10. `LOG.md`: dated narrative, with a dead-ends table from line ~1047 recording
    roughly 24 wrong turns and what each actually was.
11. `code/`: 25 scripts, each with a header saying what it does and why. Key
   ones: `build_workload.py` (never pads context by repeating text, which would
   manufacture n-gram hits), `acceptance.py` (four acceptance metrics),
   `analyze.py`, `summarize.py` (last-run-wins, do not take provenance from it),
   `check_lossless.py`, `make_subsets.py`.
12. `results/a40`, `results/a100`, `results/a40_c10`: every row carries commit,
   GPU, node, job, runner, model revision, rung, bucket, seed.

**Companion documents in the course folder, one level up from the repo:**

- `Speculation Ladder - full project briefing.md`: the complete teaching
  document, 14 sections, written to survive cross-questions.
- `Adaptive K and drafter routing.md`: the 2 Nov literature review.
- `What each script does.md`: plain-language tour of `code/`.
- `Midpoint deck study guide.md`: per-slide teaching for the midpoint deck.
- `multi-drafter-design.md`: what vLLM would need for multiple drafters, read
  against the pinned commit.
- `Speculation_Ladder_Midpoint.pptx`: the midpoint deck with speaker notes.

---

## 8. Current state and what is next

**Nothing is blocked on engineering.** Every remaining measurement is a job
submission. What is missing is the router, every figure, the paper text,
confidence intervals and the restated cost model.

**The live plan is section 5 of
`notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md`, and `TODO.md`'s NEW
FINDINGS section is its queue.** It replaces "router in vLLM" from the
Checkpoint 1 deck. 100 GPU hours were granted on "Delta AI", partition
unconfirmed, and confirming whether that is DeltaAI (GH200, ARM host) or Delta
A100 time gates the sizing. Phases, with Sahil's estimates:

1. **Phase 1, no GPU.** Fit `R(k, L) = R_draft(L) + c(L) * k` and `tau(k, L)`
   per rung from the filed A40 sweeps; predict every k to 15 including 0, the
   best pair per bucket, and the static router's gain over fixed k=7; register
   as frozen predictions with k=15 at concurrency 1 as the held-out test.
   Recover the n-gram k-probe numbers (k = 1, 2, 7) from the Delta logs, since
   `results/a40` has k=5 only. Write the diagnosis job.
2. **Phase 2, no GPU.** Policy package under `code/`, the scheduler subclass,
   harness v4, unit tests against a fake scheduler, job script. Phase 2b is the
   keep-warm proposer patch and the composite proposer along PR #24344's design,
   in a project-owned venv from `rebuild_multiarch.sh`.
3. **Phase 3, diagnosis, about one GPU hour.** Per-step acceptance per bucket;
   one 32K DFlash step profiled into KV write, draft forward and verify;
   **prefix-caching-off control**; constant-table path parity at 2K; keep-warm
   against K=0 against off at 32K.
4. **Phase 4, sweep, about eight wall hours per GPU.** Both rungs, three
   buckets, concurrency 1, k in 0, 1, 2, 4, 7, 11, 15, per-request timing on,
   A40 and A100. Gives the joint per-request oracle.
5. **Phase 5, policy evaluation, about eight wall hours per GPU.** Static
   router, one-way or two-way ladder, goodput argmax, against the best fixed
   pair, off, and both oracles; two repeats; concurrency axis at 2K/8, 8K/4,
   32K/2.
6. **Phase 6.** Claims C23 onward, two figures, Checkpoint 2.

**Still queued from before and not superseded**, mostly free: swap in
`check_lossless.v3` and patch its stale verdict block, which prints a pre-C21
conclusion while comparing 100% against nan% (threat 14); the speculative-arm
reproducibility control, about 30 minutes, which separates "speculation
diverges" from "speculation diverges systematically" (threat 12); A100 2K at
n=50, about 10 minutes, the one point off the C20 collapse and the only test of
a parameter fitted on five prompts; disjoint prompt subsets for intervals
(`make_subsets.py` exists, zero subset jobs have run); MAXTOK=1 at 8K and 32K
for decode-only ratios; and the figures, which must be generated from
`results.jsonl` by a script. F3, predicted against measured on log-log, is the
lead figure and is not started; settle the C20 collapse figure as two series
before drawing (threat 17).

**Baselines and thresholds for any gain claim.** The baseline is the best single
rung-and-k pair **per bucket**, not a global fixed setting. The ceiling is the
joint per-request oracle including off. Reviewers treat under 5% as noise, 5 to
10% as needing intervals, and 10 to 20% as a normal headline. MetaSD and
HedgeSpec reach 85 to 97% of their oracles, which is the fraction ours will be
compared against.

**Known open defects.** The "503 graded answers" figure is wrong by about 4x and
should read 120 (n = 49/47/24); it is still wrong in `paper/claims.md` around
line 1327 and `paper/conclusions.md` around line 146. Batch-composition
circularity: every cell was measured with the whole batch on one arm, so a
router's mixed batches appear nowhere in the grid and the offline result is an
upper bound until bounded by three short jobs. Two KV anomalies to chase before
quoting absolute GiB. `env/gpu-support.md` has a wrong capacity table.

**Administrative.** The professor has asked for adaptive k to be part of the
project; the answer is in `paper/related-work.md` and the literature review. His
answer on the router reframing is still pending. GPU budget is disputed between
accounts and `sreport`, and the A100 multiplier was assumed 2x where Delta's
page says 1.0.

---

## 9. Things not to do

- Do not rewrite `paper/claims.md`. It is append-only by design, including its
  failed predictions and withdrawn claims. A register of only confirmations is
  not evidence of anything.
- Do not quote absolute throughput across jobs.
- Do not add a citation you have not opened. See threat 18.
- Do not "fix" the three git author identities.
- Do not report a job as successful because Slurm says COMPLETED.
- Do not assume K=0 costs the same as plain decode on DFlash. The drafter
  forward still runs with one query token to keep its KV in sync.
- Do not pad context by repeating text under any circumstances.
- Do not claim "first to route across kinds of drafter". BanditSpec's pool is
  PLD, REST, a suffix tree and EAGLE-2. The phrase is "opposite-signed".
- Do not repeat the SpecExtend 2.76x to 0.83x figure or the AdaSpec "Corollary
  4.4" without opening the papers first. Both failed verification on 2 Nov.
- Do not run a new DFlash comparison with prefix caching on until the
  caching-off control has run. vLLM issue #47930 collapses acceptance to about
  1.02 when drafter steps are skipped with caching on.
