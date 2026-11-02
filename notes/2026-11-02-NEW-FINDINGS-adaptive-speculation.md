# NEW FINDINGS, 2 Nov 2026: adaptive speculation, what prior work already owns, and the plan

**Everything in this note came from documents and source code, not from a
measurement.** Where it bears on a measured claim, the claim is named and the
measurement stands; what is questioned is the mechanism attributed to it.

Direction, as agreed 1 Nov: the paper's contribution is a **joint policy over
rung and draft length** (off, n-gram at k up to 11, DFlash at k up to 15),
chosen per request at admission and per step while decoding, inside vLLM V1 at
the pinned commit. The router from the proposal is the special case that picks
the rung once; adaptive k alone is the special case that holds the rung fixed.
A 100 GPU-hour allotment on "Delta AI" is available; whether that is DeltaAI
(Grace Hopper, ARM host) or Delta A100 time is unconfirmed and gates the
sizing below.

Sources: a 48-paper survey (1 Nov) and a seven-source prior-work check on
multi-drafter switching (1 and 2 Nov). Full text with URLs lives outside the
repo in Sahil's vault; the live summary is
<https://claude.ai/artifact/XcfDxUSnWPxmatB9WkiCf6>.

---

## 1. What the literature already settles

- **The inversion is expected, not anomalous.** AdaSpec (SoCC 2025, Cor. 4.4)
  derives that the optimal draft length falls with context when the drafter's
  own cost grows with context; MagicDec shows the opposite sign for drafters
  with bounded KV. The two rungs therefore have opposite signs by
  construction. n-gram sits at near-zero draft cost, so its k depends on
  acceptance alone.
- **Published analogues exist for trained heads, none for a block drafter.**
  SpecExtend: EAGLE-3 on Llama-3.1-8B 2.76x at 1K to 0.83x at 16K. "Strong
  Drafts Need Compact Memories" (Aug 2026): 0.92x to 1.06x at 8K to 32K. The
  DFlash paper trains at 3K and never evaluates past 4K; its Table 4 shows the
  Qwen3.5-27B drafter degrading past its training window and recovering after
  a 1.6K-sample long-context fine-tune. That fine-tune is the training-side
  fix our policy competes with.
- **Per-step confidence truncation of DFlash blocks is taken.** DSpark (shipped
  in vLLM main as adaptive verification), DScale, AdaFlash, D2SD, BlockPilot,
  all 2026, all at concurrency 8 or more on prompts under 4K; it saves
  verification cost only. Do not claim it.
- **DFlash-b16 is 1 anchor plus 15 mask slots.** The official vLLM launch uses
  `num_speculative_tokens: 15`. Our k=7 runs the head on an 8-token block; the
  paper's Table 8 puts train-16/infer-8 at roughly 1.2 fewer accepted tokens
  per step. The paper must say whether k=7 was deliberate. (The harness
  default is 15; every filed run used 7.)
- **External check on our 2K number.** H-Spec (Sep 2026) measured DFlash on
  Qwen3-8B in vLLM at batch 1 on an A100, k=7: 2.28x average, 2.17x on QA.
  Our 1.86x at 2K on A40 is in line.

## 2. What prior work already owns (the honest list)

| Paper | Owns | Does not do |
|---|---|---|
| MetaSD, ACL 2026 Findings (arXiv 2604.05417; same paper as "Multi-Drafter SD with Alignment Feedback") | per-step bandit over drafters; switching cost proportional to the KV gap; explore-then-commit with Sequential Halving; 85-95% of the per-task oracle | no off action; no length action; accepted-tokens objective, not time; prompts of a few hundred tokens; custom harness; no code |
| HedgeSpec / "Not-a-Bandit", ICLR 2026 (arXiv 2510.20064) | free feedback on unchosen drafters by prefilling the verified chunk into each; no-regret bound; 90-97% of per-domain oracle | same gaps; says PLD is a poor fit because one-hot proposals degenerate its acceptance estimate |
| BanditSpec, ICML 2025 (arXiv 2505.15141), read in full | per-round bandit over a drafter pool (PLD, REST, suffix tree, EAGLE-2) in one experiment and over EAGLE-1 lengths in another | off arm is in the text but **not in the released code or the oracle**; one length per step for the whole batch; prompts truncated at 256-512; no cost model; switching state never discussed |
| Nightjar (arXiv 2512.22420) | off arm with a profiled re-enable cost, inside vLLM | keyed on batch size; one drafter |
| SmartSpec / TurboSpec, AdaSpec | goodput argmax over k including 0 | batch-level; one drafter |
| vLLM main | batch-size-keyed `num_speculative_tokens_per_batch_size` table, K=0 allowed (PR #32374, Jun 2026) | no context key; no per-request choice |
| vLLM RFC #18633 / PR #24344 (closed stale 29 Oct 2026) | maintainers' own design for n-gram plus a model drafter in one engine: always run the model drafter's context pass, pick n-gram per request, skip the model draft when every request has an n-gram hit | never merged |

**Do not claim:** first online drafter selection, first switching-cost model,
first free feedback on unchosen drafters, first off arm, first bandit over
draft length, first DFlash truncation.

**Still claimable, clause by clause** (closest prior work in the table):
(a) the joint choice of drafter and length with off as a selectable action that
the oracle includes; (b) keyed on prompt length and online acceptance through a
measured cost model rather than accepted tokens alone; (c) inside a production
engine at batch 1 across 2K to 32K; (d) for a block-diffusion head and a
lookup drafter whose acceptance and cost move in opposite directions with
context; (e) an asymmetric switching cost, free downward to n-gram and a
keep-warm KV write upward to DFlash; (f) a per-request oracle over the joint
action set that includes off, stratified by prompt length.

## 3. Two things the plan assumed that the source code contradicts

1. **One-way descent is not forced.** The target's aux hidden states are
   indeed dropped each step (`ExecuteModelState` cleared at the top of
   `sample_tokens`). But the DFlash context-KV write is a separate cheap path:
   `precompute_and_store_context_kv` is a fused norm, one GEMM for all layers'
   K/V, K-norm and RoPE over only the tokens the target just processed, about
   250 MB of weight reads, under 1% of a step. A **keep-warm mode** that runs
   only that write on idle steps makes DFlash resumable with no catch-up. It
   needs a small proposer patch because at bf13ecc2f the write is reachable
   only through `propose()`. By contrast, K=0 through the existing dynamic-SD
   path still runs the full draft forward with one query token
   (`llm_base_proposer.py`, "prefill forward pass above already ran to keep
   the drafter KV cache in sync"). So the three DFlash idle modes to measure
   at 32K are keep-warm, K=0 and off, and two-way switching is on the table.
2. **The mechanism attributed to `R_draft` is in question; the measurement
   is not.** `notes/2026-10-08-drafter-cost-scaling.md` measured R_draft
   0.296 / 0.434 / 0.662 at 2K / 8K / 32K and explained it as the head
   attending over the full context. In the vLLM implementation the head never
   re-runs over the context; the only context-scaling term in the drafter is
   its attention reading about 20 KiB per token of draft KV, about 640 MiB at
   32K, on the order of 1 ms on an A40. The measured overhead per speculative
   step over plain decode is about 11, 20 and 52 ms at 2K, 8K and 32K, far
   larger than that. Candidates: the non-causal FlashAttention path for a
   handful of queries over 32K, or the verify pass. This is the same gap LOG
   already lists as "why both drafter costs are sublinear in context,
   unexplained." The 32K logs also show acceptance collapsing (mean accepted
   length 2.54 at 2K, 1.46 at 32K). Two consequences: the paper needs a
   profile of one 32K DFlash step split into KV write, draft forward and
   verify before it names a cause; and the policy's descent trigger should be
   acceptance-driven, which is observable online, with context length only a
   proxy.

## 4. Engineering facts at bf13ecc2f that bind the build

- The pinned build already has dynamic speculation: the scheduler takes a
  per-step K from `dynamic_sd_lookup` and passes
  `SchedulerOutput.num_spec_tokens_to_schedule` to every proposer; both
  `ngram` and `dflash` honor it, including 0. `SchedulerConfig.scheduler_cls`
  is honored unconditionally, so the policy can live in a scheduler subclass
  with no vLLM source edit and no change to the shared venv.
- Async scheduling defaults **on** for `dflash` and **off with a warning** for
  CPU `ngram`. The subclass must extend `AsyncScheduler` or async is lost.
  Under async, K is fixed before drafts exist; the per-request lever is
  truncating `request.spec_token_ids` in the scheduler.
- A custom scheduler that varies K without setting
  `num_speculative_tokens_per_batch_size` bypasses the override that forces
  PIECEWISE cudagraphs on V1. Set a constant table anyway, and run the fixed-k
  baselines through the same path, or the comparison is confounded by graph
  mode.
- DFlash buffers, warm-up and graph sizes are fixed at construction from
  `num_speculative_tokens`. K can only shrink from it: configure 15 and
  shrink.
- **Prefix caching plus skipped DFlash steps reproduces a known bug.** Issue
  #47930 / PR #47926: restored prefix tokens never receive drafter KV and
  acceptance collapses to about 1.02. Every filed run has
  `prefix_caching: true`. The prefix-caching-off control already in LOG's open
  items is now a prerequisite, not a nicety, and the existing DFlash numbers
  could carry a small version of this if prompts share an instruction prefix.
- The experimental `custom_class` proposer receives token ids only, no hidden
  states, so it cannot host a composite drafter. The composite goes in the
  runner: CPU n-gram drafts arrive after bookkeeping from lists, DFlash drafts
  before it from GPU tensors (the wall PR #24344 hit), so use
  `NgramProposerGPU` or merge after bookkeeping; `_draft_probs` rows must be
  nulled or set one-hot when an n-gram draft replaces a DFlash draft.
- Per-request acceptance arrays already exist in this build:
  `ObservabilityConfig.per_request_spec_decode_metrics = "detailed"` puts the
  ordered per-step proposed/accepted arrays on each `RequestOutput`. That is
  the label source for both oracles, at no extra instrumentation.
- PR #63373 (open) removes the V1 runner upstream. The pin is safe; the work
  will not be upstreamable on that runner.
- Issue #49112, cited in an outside summary as the "engineering wall", is an
  open stale issue about a hybrid draft model and changed nothing; its
  multi-group dependency PR #35062 is unmerged.

## 5. The plan, as it stands

Baseline for any gain: the best single rung-and-k pair per bucket, not a
global fixed setting. Ceiling: the joint per-request oracle including off.
Thresholds reviewers apply: under 5% is noise, 5-10% needs intervals, 10-20%
is a normal headline, more only where the fixed baseline is wrong, which is
the 32K case. MetaSD and HedgeSpec reach 85-97% of their oracles; that is the
fraction ours will be compared against.

1. **Phase 1, no GPU.** Fit the two-term step cost per rung and bucket from
   the existing A40 sweeps (DFlash k = 1, 2, 7 at 2K and 8K, k = 1, 2, 4, 7 at
   32K); predict speedup for every k up to 15 including 0, the best pair per
   bucket, and the static router's gain over fixed k=7; register them as
   frozen predictions with k=15 at concurrency 1 as the held-out test. Recover
   the n-gram k-probe numbers TODO cites (k = 1, 2, 7) but `results/` lacks.
   Write the diagnosis job.
2. **Phase 2, no GPU.** Policy package under `code/`, scheduler subclass,
   harness v4 (policy flags, detailed per-request metrics, per-step arrays
   written per request), unit tests against a fake scheduler, job script.
   Phase 2b: keep-warm proposer patch and the composite proposer along PR
   #24344's design, in a project-owned venv built with
   `rebuild_multiarch.sh`.
3. **Phase 3, diagnosis, about one GPU hour.** Per-step acceptance at each
   bucket; one 32K DFlash step profiled into KV write, draft forward, verify;
   prefix-caching-off control; constant-table path parity against the static
   path at 2K; keep-warm vs K=0 vs off at 32K.
4. **Phase 4, sweep completion, about eight wall hours per GPU.** Both rungs,
   three buckets, concurrency 1, k in 0, 1, 2, 4, 7, 11, 15, per-request
   timing on, A40 and A100. Gives the joint per-request oracle.
5. **Phase 5, policy evaluation, about eight wall hours per GPU.** Static
   router, one-way or two-way ladder, goodput argmax, against the best fixed
   pair, off, and both oracles; two repeats; concurrency axis at 2K/8, 8K/4,
   32K/2.
6. **Phase 6.** Claims C23 onward, two figures, Checkpoint 2.

Budget, charged GPU hours: smoke and diagnosis 2, sweep 12, evaluation 20,
concurrency axis 8, reserve 58 (covers a DeltaAI bring-up if that is where
the hours are).

## 6. Open decisions

- DeltaAI or Delta A100 for the 100 hours.
- Whether k=7 was deliberate over the native 15.
- One-way or two-way ladder, decided by the keep-warm measurement, not by
  assumption.
- Forward citations of BanditSpec to triage by title before related work is
  final: TapOut 2511.02017, CAST 2610.00321, TableTune 2610.38968, "Bandits
  in Production" 2609.01335, FASER 2604.20503, AdaSpark 2610.05774, MemSpec
  2608.10362.
