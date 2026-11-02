# Related work

What each neighbour does, and what it leaves open. The claim that survives all of
them: choosing among drafting mechanisms of different kinds, inside a serving
runtime at concurrency, from a fitted model of context and memory, with the
crossing point predicted on hardware not used for fitting.

| Work | What it does | What it leaves |
|---|---|---|
| MagicDec, arXiv 2408.11049 | Cost against context with a bounded drafter at batch 32 to 256. Measures acceptance (0.84 to 0.79 from 4K to 100K on PG-19) and notes the hardware direction | Never predicts where the inflection lands. One drafter class. PG-19 continuation is near best case for a sliding-window drafter |
| SmartSpec, arXiv 2406.14066, in vLLM | Per-request speculation length from 0 upward chosen by goodput. Linear cost model in context and batch | One method at a time. Acceptance from a moving average, no context model. Says outright that its coefficients must be re-profiled per hardware, which is the gap we fill |
| AdaServe, arXiv 2501.12162 (EuroSys 2026 claimed, **venue not verified**; the entry renders under two titles) | Per-request SLO-customized draft trees under a roofline token budget. 4.3x fewer SLO violations, 1.9x goodput. Scheduler costs 0.31 to 0.41 percent CPU | One draft model throughout. Context never varied. Single platform, profiled not predicted. Its Figure 12, accepted tokens falling as request rate rises, is independent evidence for our concurrency axis |
| AdaSpec, arXiv 2503.05096, ACM SoCC '25, DOI 10.1145/3772052.3772239 | Adaptive speculative length plus a confidence-prior verifier under SLOs. Skips drafting when the predicted efficiency gain is non-positive, and takes batch size and context length as system-configuration inputs, on vLLM | One drafter, per-hardware calibration. **Not two papers:** v1 of this arXiv entry was titled SpecServe and v2 (Jan 2026) is AdaSpec, so citing both double-counts it. A claimed "Corollary 4.4" deriving that optimal k falls with context does **not** appear in the fetched v2 text, which has only Lemma 4.1 and 4.2 |
| SAM-Decoding, arXiv 2411.10666 (ACL 2025 claimed, venue not verified) | Per-step switching between a suffix-automaton draft and EAGLE-2 by match length | Local signal, batch 1, no serving concurrency, no hardware model |
| Nightjar (2512.22420), DSpark (2607.05147), BanditSpec (2505.15141, ICML 2025), Not-a-Bandit / HedgeSpec (2510.20064, ICLR 2026), SpecRouter (2505.07680), FASER | Draft length or routing within one mechanism class, or across model sizes, from reactive signals. Nightjar's arm set includes gamma = 0 | No context model, no memory model, no cross-hardware prediction. **TurboSpec is not a separate work:** it is the SmartSpec arXiv entry rendering under a second title. DSpark's reported deployment is the DeepSeek-V4 serving system, not vLLM; the vLLM integration exists but is restricted to `method="dspark"`, so it is unavailable to us |
| Sequoia, arXiv 2402.12374 | Adjusts tree size and depth to hardware by grid search | **Read, 2 Nov 2026.** Section 3.3.1 states the procedure as measuring t(n) and c "empirically on the inference hardware being used," then grid-searching. So it re-profiles per hardware and never claims prediction onto an unprofiled GPU. A100 and L40 only. **Our cross-hardware transfer claim survives this.** Section 4.2.3 was not reached in three fetch attempts and is the one remaining place a refutation could hide |

**Gaps in the submitted proposal's reference list:** SmartSpec, AdaServe and
AdaSpec are not cited. They are the closest serving-side neighbours. Add them.

**Refresh protocol:** re-search before each milestone with at least four
phrasings. On this project more than twelve ideas looked novel under one phrasing
and turned out published under another. **SwitchSD does not exist.** Repeated targeted searching found no paper,
preprint or repo of that name. The id 2609.20186 belongs to "To Copy or Not to
Copy" (NeurIPS 2026), a per-step copy-versus-neural switch on a hidden-state
probe. ASPIRE (2609.17943) is real but is self-speculation, not multi-drafter,
and argues the opposite premise, that speedups grow with context; its venue is
disputed between COLM 2026 and an ICML 2026 virtual listing.


---

## Verification pass, 2 November 2026

Every entry above was re-checked against primary sources on this date. The full
evidence, with per-claim verification status, is in
`../../Adaptive K and drafter routing.md` (outside the repo, in the course
folder). Eight works below must be cited and differentiated; the first three are
the ones a reviewer reaches for first.

| Work | What it does | What it leaves |
|---|---|---|
| **WhiFlash**, arXiv 2606.07710 (Samsung AI Center Cambridge, Jun 2026) | Routes **per decoding step** between EAGLE-3 and **DFlash**, on target next-token entropy or an MLP over the target's final-layer hidden state predicting the acceptance-length gap. On Qwen3-8B: EAGLE-3 2.78x, DFlash 3.55x, WhiFlash-Neural 3.87x | **Closest prior art: same drafter, same target, same cross-paradigm idea.** Per step, not per request. No no-speculation action. Context length is not an evaluation axis (sequence length appears only in switching-cost profiling). HuggingFace Transformers 4.57.1, **batch 1 only**, no serving engine. Its Oracle-Prompt bound is over two always-speculate actions at batch 1, so it is not an oracle over our action set |
| **OWL**, arXiv 2510.07535 (Snowflake / SNU / CMU) | Long-context speculation. Publishes the sub-break-even collapse: EAGLE3 at **0.81x** generation speed (Llama-3.3-70B) and acceptance length **1.28** (Llama-3.1-8B) against 5.79 short-context, on LongSpecBench. HOWL is its hybrid variant, routing per step between SuffixDecoding-style and tree verification | **Cite as corroboration for our 0.83x, not as our discovery.** The two figures are separate rows and separate models; do not quote them as one point. Long context is a fixed regime, not a key. No ranking inversion across drafter kinds, and retrieval acceptance there falls with model size, a different axis |
| **Speculative Decoding: Performance or Illusion?**, arXiv 2601.11580, **MLSys 2026 oral** (Berkeley) | vLLM 0.10.1.1 study **including Qwen3-8B**. n-gram beats EAGLE and EAGLE-3 on InstructCoder (up to +53% at k=3, +100% at k=5, predicted by prompt-output BLEU-4 overlap above ~0.6). Drafting vs verification decomposition: verification 42 to 95%, n-gram drafting under 2%, EAGLE 12 to 20% at batch 1. **Already proposes the oracle** (up to 4.9x, 2.2x beyond best fixed) and asks for "an accurate yet lightweight predictor" | **Does not vary context length** (it buckets by generated-token position) and is **H100 only**. The context axis is the uncovered cell. Scope our "ranking inverts" claim to context length, since the workload-axis version is published. It also reports **every** SD variant beating no-SD; our 0.83x and OWL's 0.81x are counterexamples on the axis it does not vary |
| **Context-Aware Assistant Selection**, EMNLP 2024 Main, 2024.emnlp-main.332 | Selects among drafters **per request**, with a learned no-speculation arm | **The citation we cannot omit.** Keys on a query sentence embedding; length is never an input. gamma fixed at 7. Not a serving engine. Names per-step draft-length choice as future work. Have the answer ready for "why length, not content?": length is known at admission, costs nothing, needs no target-specific training, and moves the **cost** term rather than the acceptance term |
| **MetaSD**, arXiv 2604.05417 (paper title: "Multi-Drafter Speculative Decoding with Alignment Feedback") | UCB bandit over drafters, selection per SD cycle | Arms are all one kind (68M AR drafters, or EAGLE heads differing by fine-tuning). Featureless; contextual bandits are future work. **Its own tables show UCB trailing the best task-matched specialist on every task, and losing at long context (2.031 vs 2.044).** Direct evidence against featureless bandits in our regime |
| **SuffixDecoding / ArcticInference**, arXiv 2411.04975 | **Shipped vLLM plugin** selecting per decoding iteration between CPU suffix trees and a trained speculator | **The dangerous shipped-code citation.** Per iteration on likelihood and length, not load aware, `enable_suffix_decoding` is a static switch with no off action, and no context-length term anywhere in its config surface |
| **ReSpec** (2511.01282), **To Copy or Not to Copy** (2609.20186, NeurIPS 2026), **TAPS** (2603.27027, workshop) | Per-step selection between retrieval and a trained head, on target entropy, a hidden-state probe, or drafter confidence | Retrieval-vs-trained-head selection is therefore **not** novel in itself. All per step, intrinsic signals, batch 1, no context axis, no off action |
| **Engine code:** vLLM RFC #48627, PRs #48944 and #54801; SGLang PR #31716; `seongyun1104/depthchart` | Context-length-keyed draft length, including **K=0 above 128k** in #54801 and a (batch, ctx) table in #48944 | **All five are one contributor's line of work, not five signals.** All unmerged as far as could be verified, and `depthchart`'s own EVIDENCE_LEDGER states as of 2026-10-22 that "a performance advantage from adding the context axis is not established by this campaign." His drafters were all cost-growing, so there was no cost asymmetry to exploit. That is our opening, and it must be addressed head-on rather than ignored |

| **vLLM RFC #18633 / PR #24344** (closed stale 29 Oct 2026) | **The maintainers' own design** for n-gram plus a model drafter in one engine: always run the model drafter's context pass, pick n-gram per request, skip the model draft when every request has an n-gram hit | Never merged. More useful as a precedent than issue #6300, because it is a design rather than a feature request. Build along it. Found by Sahil's source reading, 2 Nov |
| **H-Spec** (Sep 2026) | External check on our own 2K number: DFlash on Qwen3-8B in vLLM, batch 1, A100, k=7, 2.28x average and 2.17x on QA | Not independently verified. Our 1.86x at 2K on A40 is in line with it |
| **BanditSpec**, arXiv 2505.15141, ICML 2025, read in full | Per-round bandit over a pool of **PLD, REST, a suffix tree and EAGLE-2**, and separately over EAGLE-1 lengths | **This kills "first to route across kinds of drafter."** Its off arm is in the text but not in the released code or the oracle; one length per step for the whole batch; prompts truncated at 256 to 512; no cost model; switching state never discussed |

**What survives, stated as a conjunction.** A **per-request** choice across
drafter **kinds of opposite cost structure**, keyed on **context length and
load**, with **no-speculation as an action**, inside a **serving engine**, plus a
cost model **fitted on one GPU and validated on another with a drafter-dependent
residual**. Drop any clause and a published work covers it.

**The cross-hardware clause is the cleanest novelty.** No speculative-decoding
paper found fits a cost model on one GPU, freezes it, and tests on another.
SmartSpec profiles per model-and-hardware; Sequoia re-profiles; *Decoding
Speculative Decoding* (NAACL 2025) says in print that its A100 conclusions "may
not hold" on H100 or V100 and does not test it; MagicDec is qualitative. The
calibration point comes from outside this literature: **NeuSight** (ASPLOS 2025,
DOI 10.1145/3669940.3707265) reports 8.1% average and 28.2% maximum error
predicting onto out-of-distribution GPUs, against our 2 of 4 cells within 5%.

**Two surveys, and they disagree.** Sahil's 48-paper survey is recorded in
`../notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md`. The independent
citation check is `../notes/2026-11-02-verification-of-new-findings.md`, which
found three of its claims wrong or unverifiable and five works it did not
reach. Neither note supersedes the other: the first is better on the engine,
the second on the citations. Read both before writing related work.

**Still unverified, needs a human.** Whether vLLM PR #54801 merged (one grep on
the pinned tree settles it); Sequoia Section 4.2.3; whether WhiFlash has a
published version at 2026.findings-emnlp.633; and whether "Windowed-MTP" exists
at all.
