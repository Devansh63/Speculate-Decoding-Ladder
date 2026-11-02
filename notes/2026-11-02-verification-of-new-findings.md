# Verification of the 2 Nov NEW FINDINGS note, 2 Nov 2026

**What this is.** An independent check of the citations in
`notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md` against primary
sources, plus the works its 48-paper survey did not reach. Done the same day,
from the papers and the repositories, not from search snippets. Full evidence,
with a verification status per work, is in `../Adaptive K and drafter
routing.md` in the course folder.

**What this is not.** It does not touch section 3 or section 4 of that note.
Those came from reading vLLM at `bf13ecc2f` and they supersede what was
believed before, including two things asserted in earlier sessions: that
per-request k has no supported route (the pinned build takes a per-step K that
both proposers honor, including 0, and `SchedulerConfig.scheduler_cls` is
honored unconditionally, so the policy can live in a scheduler subclass with no
source edit), and that a composite drafter could ride the `custom_class` hook
(it cannot, and the note gives the reason and the alternative). Section 4 is
the better record. Follow it.

Three results below, in descending order of how much they change the paper.

---

## 1. Three citations in section 1 are wrong or unverifiable

| Claim in the note | What the primary source shows |
|---|---|
| "AdaSpec (SoCC 2025, Cor. 4.4) derives that the optimal draft length falls with context when the drafter's own cost grows with context" | **No Corollary 4.4 was found.** arXiv:2503.05096v2 is the right paper (ACM SoCC '25, DOI 10.1145/3772052.3772239) but that section contains only Lemma 4.1 and Lemma 4.2, and the fetched HTML truncates inside Lemma 4.2's proof. Lemma 4.1 makes total time a quadratic in speculative length with average context entering only the linear and constant terms; Lemma 4.2 states diminishing marginal accepted tokens. Neither addresses a drafter whose cost grows with context. **Honest status: not found, not disproved.** A corollary could exist in the camera-ready beyond the truncation. Either way, do not rest the theory on it |
| "SpecExtend: EAGLE-3 on Llama-3.1-8B 2.76x at 1K to 0.83x at 16K" | **Refuted on three counts.** The targets are Vicuna-7B and LongChat-7B, not Llama-3.1-8B. The drafter is EAGLE, not EAGLE-3. The real Table 2 GovReport values are 2.57x (1K), 2.24x (2K), 1.81x (4K), 1.34x (8K), **1.61x (16K)**, with nothing below 1.0x at long context; the only sub-1x entries in the paper are 0.96x and 0.98x at **short** context. With SpecExtend applied, speedup **rises** with context, to 3.08x at 16K. The paper's whole message is that long-context speedup recovers once draft KV and attention are fixed, so it is currently cited against its own conclusion. Cite the ACL 2026 Findings version, 2026.findings-acl.2153 |
| "Strong Drafts Need Compact Memories (Aug 2026): 0.92x to 1.06x at 8K to 32K" | **Could not verify.** arXiv:2608.30252, EMNLP 2026 Findings. The abstract reports only maxima (up to 2.08x on Llama 3.1-8B, 3.33x on Llama 3.1-70B) with no per-length breakdown and no sub-1.0x baseline |

**A replacement for the lost theory, with better provenance.** The sign split
the note wants follows from two peer-reviewed sources that each hold the
other's variable fixed: Leviathan et al.'s closed form for optimal gamma from
acceptance and the draft-to-target cost ratio, and MagicDec (ICLR 2025,
arXiv:2408.11049), which ties speculation length to sequence length and batch
size and separates drafters whose cost grows with context from those whose cost
does not. Deriving `dk*/dL > 0` for n-gram and `dk*/dL < 0` for DFlash from
those two is a stronger move than citing a corollary nobody can open, and no
source found states the combination, so it is ours to claim.

**There is one genuine published analogue of the sub-break-even collapse**, and
it is not SpecExtend. OWL (arXiv:2510.07535, Snowflake / SNU / CMU) reports
EAGLE3 at **0.81x** generation speed on Llama-3.3-70B and accepted length
collapsing to **1.28** on Llama-3.1-8B against 5.79 short-context, on its
LongSpecBench. Those are two separate rows and two separate models; do not
quote them as one point. Cite OWL as corroboration for our 0.83x rather than
presenting the observation as new.

---

## 2. Five things the 48-paper survey did not reach, one of them serious

**WhiFlash, arXiv:2606.07710** (Samsung AI Center Cambridge, Jun 2026) routes
**per decoding step** between EAGLE-3 and **DFlash**, on **Qwen3-8B**, using
target next-token entropy or an MLP over the target's final-layer hidden state
predicting the acceptance-length gap. Reported: EAGLE-3 2.78x, DFlash 3.55x,
WhiFlash-Neural 3.87x on Qwen3-8B. It also has an **Oracle-Prompt** bound, that
is a per-request oracle, which WhiFlash-Entropy matches exactly and
WhiFlash-Neural exceeds.

This is the nearest prior art by a distance and it must be addressed in the
first paragraph of related work. What it does not do: it is per step, not per
request; it has **no off action**; context length is not an evaluation axis
(sequence length appears only in switching-cost profiling, 512 to 16,384
prefill); it runs in HuggingFace Transformers 4.57.1 at **batch 1** with no
serving engine; and its 3.55x DFlash number is a short-context number. Its
per-prompt oracle is over two always-speculate actions at batch 1, so it is not
an oracle over our action set. Prepare the answer to "a per-step policy already
matches the per-prompt oracle, so why per request?": because their oracle
cannot express "at 32K and under load, speculating at all is net negative".

**Speculative Decoding: Performance or Illusion?** (arXiv:2601.11580, **MLSys
2026 oral**, Berkeley) is a vLLM 0.10.1.1 study that **includes Qwen3-8B**. It
reports n-gram beating EAGLE and EAGLE-3 on InstructCoder (up to +53% at
proposal length 3, +100% at 5, predicted by prompt-output BLEU-4 overlap above
roughly 0.6), gives a drafting versus verification decomposition (verification
42 to 95%, n-gram drafting under 2%, EAGLE 12 to 20% at batch 1), **already
proposes the oracle** (up to 4.9x, 2.2x beyond the best fixed strategy), and
explicitly calls for "an accurate yet lightweight predictor" as future work.
That sentence is our motivation section, written by someone else. It does
**not** vary context length (it buckets by generated-token position) and it is
**H100 only**, so the context axis is the uncovered cell. Note also that it
reports **every** speculative variant beating no-speculation; our 0.83x and
OWL's 0.81x are counterexamples on the axis it does not vary, and that is the
cleanest framing available: its optimistic conclusion is an artifact of short
contexts.

**Context-Aware Assistant Selection** (EMNLP 2024 Main, 2024.emnlp-main.332)
selects among drafters **per request** and has a **learned no-speculation
arm**. It keys on a query sentence embedding from the target model; length is
never an input, and gamma is fixed at 7. It is not a serving engine, and it
names per-step draft-length choice as future work. This is the one that bears
on clause (a) below, and it cannot be omitted.

**Retrieval-versus-trained-head selection is not novel in itself.** Four works
do it, all per step on intrinsic signals: SAM-Decoding (arXiv:2411.10666, match
length), ReSpec (arXiv:2511.01282, target entropy), To Copy or Not to Copy
(arXiv:2609.20186, NeurIPS 2026, a linear probe on target hidden states), and
TAPS (arXiv:2603.27027, drafter confidence, workshop). ArcticInference ships
per-iteration selection between CPU suffix trees and a trained speculator as a
vLLM plugin today. None keys on context length, none has an off action.

**Context-keyed k exists in engine code, though weakly.** vLLM RFC #48627, PRs
#48944 and #54801 (the last adding `num_speculative_tokens_per_seq_len` with
K=0 above 128k), SGLang PR #31716, and the `seongyun1104/depthchart` lab are
**all one contributor's line of work**, all unmerged as far as could be
verified, and that contributor's own EVIDENCE_LEDGER states as of 2026-10-22
that "a performance advantage from adding the context axis is not established
by this campaign". His drafters were all cost-growing, so there was no
asymmetry to exploit. Address this head-on in the paper; it is both the nearest
negative result and our best motivation.

---

## 3. What the note's prior-work table gets right, and what it adds

Three items in section 2 of that note are better than anything in the external
review and should be treated as authoritative here:

- **BanditSpec's pool is PLD, REST, suffix tree and EAGLE-2.** That is routing
  across drafter kinds, read in full from the paper. Any claim phrased as "we
  are the first to route across kinds of drafter" is dead. The surviving
  distinction is the pair's **opposite-signed** acceptance and cost with
  respect to context, not that the kinds differ.
- **vLLM RFC #18633 and PR #24344** (closed stale 29 Oct 2026) are the
  maintainers' own design for n-gram plus a model drafter in one engine: always
  run the model drafter's context pass, pick n-gram per request, skip the model
  draft when every request has an n-gram hit. An external review that cited
  only issue #6300 missed this, and #24344 is the more useful precedent because
  it is a design rather than a feature request. Build along it.
- **H-Spec (Sep 2026)** gives an external check on our own number: DFlash on
  Qwen3-8B in vLLM at batch 1 on an A100, k=7, 2.28x average and 2.17x on QA.
  Not independently verified here.

One caution on MetaSD. The note lists it as ACL 2026 Findings; that venue could
not be confirmed, and the arXiv entry (2604.05417) is titled "Multi-Drafter
Speculative Decoding with Alignment Feedback", with MetaSD as the framework
name. Its own tables show UCB trailing the best task-matched specialist on
every task (Code 2.300 against 2.437, Translation 1.587 against 2.076, Math
2.280 against 2.454) and losing in its single long-context check (2.031 against
2.044). That is evidence **for** a feature-keyed policy over a featureless
bandit, and it is worth quoting in our favour rather than only as prior art.

---

## 4. The six clauses, re-read against all of the above

Section 2 of the note lists six still-claimable clauses. After verification:

| Clause | Status |
|---|---|
| (a) joint choice of drafter and length with off as a selectable action the oracle includes | **Narrow it.** Context-Aware Assistant Selection has per-request drafter selection with a learned no-speculation arm, and BanditSpec's text has an off arm. What survives is **joint over rung and length**, with off in **both** the action set and the oracle, keyed on a measured cost model |
| (b) keyed on prompt length and online acceptance through a measured cost model rather than accepted tokens alone | **Survives**, and it is the strongest clause. No router found keys on length; the engine-code line that does is unmerged and unproven |
| (c) inside a production engine at batch 1 across 2K to 32K | **Survives, with a named rival.** MLSys 2026 is in vLLM at batch 1 to 128 on Qwen3-8B but does not vary context; WhiFlash varies neither engine nor context |
| (d) for a block-diffusion head and a lookup drafter whose acceptance and cost move in opposite directions with context | **Survives and is the load-bearing clause.** WhiFlash pairs a block-diffusion head with an autoregressive head, not a lookup drafter, so the opposite-signed pair is still ours. Drop "across kinds" phrasing, keep "opposite-signed" |
| (e) asymmetric switching cost, free downward and a keep-warm KV write upward | **Survives.** Nothing comparable found; MetaSD's switching cost is proportional to the KV gap, which is the nearest, and section 3 of the note is the only source for the keep-warm path |
| (f) per-request oracle over the joint action set including off, stratified by prompt length | **Survives, but say "including off".** MLSys 2026 and WhiFlash both publish oracles; neither includes a no-speculation action, and neither stratifies by prompt length |

**The claim as one sentence, after both checks.** A joint per-request policy
over rung and draft length, with off in the action set and in the oracle, keyed
on prompt length and online acceptance through a measured two-term cost model,
for a drafter pair whose acceptance and cost move in opposite directions with
context, inside a production serving engine from 2K to 32K, with the cost model
fitted on one GPU and validated on another. Drop any clause and a published
work covers it.

**The cross-hardware clause deserves more weight than it is getting.** No
speculative-decoding paper found fits a cost model on one GPU, freezes it and
tests on another. SmartSpec profiles per model and hardware; Sequoia measures
t(n) and c "empirically on the inference hardware being used" then grid
searches, so it re-profiles rather than predicts; Decoding Speculative Decoding
(NAACL 2025) says in print that its A100 conclusions "may not hold" on H100 or
V100 and does not test it; MagicDec is qualitative. The only calibration point
comes from outside this literature: NeuSight (ASPLOS 2025, DOI
10.1145/3669940.3707265) reports 8.1% average and 28.2% maximum error
predicting onto out-of-distribution GPUs, against our 2 of 4 cells within 5%.

---

## 5. Still unverified, and the rule going forward

Four items need a human and only the first two can change a claim: whether vLLM
PR #54801 merged (one grep on the pinned tree settles it); what Sequoia Section
4.2.3 actually says, which is the one place the cross-hardware claim could be
refuted; whether WhiFlash has a published version at 2026.findings-emnlp.633;
and whether "Windowed-MTP" exists at all. Every GitHub state in the external
review was read from rendered HTML, which drops the open/merged/closed banner,
so treat PR and issue states as unverified unless the row says otherwise.

Threat 18 now records the rule: nothing enters `paper/related-work.md` unless
its full text was opened, and every entry carries an arXiv id or DOI plus a
verification status. This is the `HANDOFF.md` section 9 failure mode applied to
the literature rather than to the measurements, and a wrong citation is the one
class of error a reviewer catches for free.
