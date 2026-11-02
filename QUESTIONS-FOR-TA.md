# Questions for the TA

Living list. Add to it, reorder it, strike what gets answered, and record the
answer here rather than only in the meeting. Last updated 2 Nov 2026.

**One line of context to open with, if the TA has not seen the project:** we
measure when each speculative-decoding drafter stops paying as context length,
load and GPU change, inside vLLM on Qwen3-8B on Delta, and we are building a
per-request policy that picks the drafter and the draft length from those
signals. At batch 1 the two drafters invert: the trained head goes 1.86x at 2K
to 0.83x at 32K, which is slower than not speculating at all, while prompt
lookup goes 1.16x to 1.08x.

---

## 1. Is benchmark-to-benchmark the right train and test split for the router?

**Ask:** should we generate the routing data on one benchmark, fit the policy on
it, and verify on a different but comparable benchmark? Or is a held-out split
inside one benchmark enough?

**Why we are asking.** Everything we have is LongBench-v2, bucketed to 2K, 8K
and 32K, with 50 prompts per bucket at 2K and 8K and 25 at 32K. A policy fitted
on that could be learning LongBench-v2's prompt distribution rather than the
mechanism. The features we want to key on, context length and prompt
self-overlap, are properties of the corpus as much as of the request, and
prompt self-overlap is exactly what decides whether the lookup drafter fires at
all. So a within-benchmark split may not test what we claim.

**What we would want from the answer.** Whether the TA considers a second
corpus necessary for the claim to hold, and if so which one. Candidates we have
looked at are InfiniteBench, RULER, SCROLLS and LongBench-v1. A separate reason
to want a second corpus: the MLSys 2026 paper "Speculative Decoding:
Performance or Illusion?" reports that prompt-to-output BLEU-4 overlap above
roughly 0.6 predicts when n-gram beats a trained head, on InstructCoder. That is
a task-type signal on a corpus we do not have, and it is close to what we want
our router to learn.

**Related question worth asking in the same breath.** Does the router need to be
a learned model at all? Our current design is a fitted cost model with a
threshold, not a trained classifier. If the cost model predicts the crossing
point well, a learned router adds a training story, a train/test split problem
and a generalization claim, and it may add no accuracy. We would rather be told
now than after we build it.

---

## 2. Should a trained head go into the ablation, and as what?

**Ask:** is it worth spending GPU hours adding a third rung (a trained
autoregressive head such as EAGLE-3) as an ablation, or is depth on the two
rungs we have the better use of the budget?

**Why we are asking.** Our two rungs were chosen to be opposite in kind. DFlash
is a trained block-diffusion head that predicts from the target's hidden states
and drafts on essentially every step; n-gram prompt lookup is retrieval with no
weights that fires on 13 to 21% of steps. The measured consequence is that
their acceptance and their drafting cost move in opposite directions with
context, and that is the mechanism our whole claim rests on.

A third rung would test whether the effect is about the *kind* of drafter or
about DFlash specifically. That matters because the nearest prior art, WhiFlash,
routes between EAGLE-3 and DFlash on Qwen3-8B, so a reviewer will ask what
happens with a third mechanism in the pool.

**The cost side, which is the real question.** A third rung multiplies the grid:
three buckets by seven k values by two GPUs is already the Phase 4 sweep, and a
third rung adds roughly 50% to it. We would rather hear whether the ablation
buys enough to justify that, or whether a cheaper version suffices, for example
one bucket at one k on one GPU purely to show the ordering.

---

## 3. Is the framing right, and does the trained-head direction strengthen or dilute it?

**Ask:** we are building a routing model that switches between draft mechanisms
based on context length, load and potentially task type. Is that the right
contribution to lead with, and does adding trained heads to the rung set
strengthen the story or blur it?

**What to tell the TA, concretely.** We picked n-gram and DFlash because they
are opposite in kind, and we have measured evidence that they behave in opposite
directions: first-position acceptance falls 0.601 to 0.296 for DFlash across 2K
to 32K but only 0.601 to 0.552 for n-gram, while the fixed drafting cost grows
0.299 to 0.673 for DFlash against 0.071 to 0.139 for n-gram. Both penalties
compound against the trained head, which is why its line crosses below 1.0.
We are now exploring whether trained heads belong in the rung set as well, since
different heads may suit different kinds of data, and task type is a signal we
would like the router to use.

**The honest tension to put in front of the TA.** Adding more trained heads
pushes us toward the problem the literature has already solved. BanditSpec
routes over a pool of PLD, REST, a suffix tree and EAGLE-2, so routing across
kinds of drafter is taken, and WhiFlash routes between EAGLE-3 and DFlash. What
is still ours is narrower: a pair whose acceptance and cost move in **opposite
directions** with context, a policy keyed on context and load rather than on
observed acceptance, "do not speculate at all" as an action the policy and the
oracle both include, and a cost model fitted on one GPU and validated on
another. We want to know whether the TA thinks widening the rung set protects
that or dilutes it.

**And the uncomfortable number to be ready for.** Over our measured grid, a
hindsight oracle beats the best fixed drafter by about 6% on a uniform context
mix and about 10% on a long-context-heavy mix, but by 0% if the workload sits in
one regime. The penalty for picking the wrong fixed drafter is 17 to 30%. So the
result is the asymmetry, not the average gain, and we would like a read on
whether that framing is strong enough to carry the paper.

---

## 4. KV cache on DeltaAI: sizing, charging, and what breaks

**Ask:** how should we think about KV cache on DeltaAI's GH200 nodes, and what
goes wrong there that does not go wrong on Delta?

**What we already know, from NCSA's own documentation.** DeltaAI nodes carry 4
GH200 superchips, each with an H100 GPU holding 96 GB of HBM3 and a 72-core
Grace ARM CPU with 120 GB of LPDDR5X, of which about 110 GB is available to
jobs. The smallest allocatable unit is one superchip. **The charge unit is one
GH200 for one hour, with no multiplier by GPU type**, and a node-exclusive job
costs 4 of those per hour.

**Why this matters to us specifically.** The 32K regime is memory-bound for us:
on the A100 at 40 GB we fit about 4 concurrent 32K requests, and the trained
head costs 2.42 to 2.54 GiB of the KV pool at every concurrency we have
measured, which is about a fifth of our 32K concurrency ceiling. 96 GB per GPU
is the headroom we have been asking for. One observation at concurrency 32 shows
the head's KV charge rising to 4.0 GiB, so every high-batch estimate we have is
an upper bound.

**The specific things to ask.**

- **The charging trap.** NCSA documents that charges are based on the number of
  GH200s needed to satisfy the requested memory **or** cores, whichever is
  larger, and their own example is that requesting 220 GB of CPU memory with 1
  GPU allocates and charges 2 superchips. Does that mean a large `--mem` on a
  single-GPU vLLM job silently doubles the charge? That would halve an
  allocation without anyone noticing.
- **Does HBM count toward that memory term, or only CPU memory?** The
  documentation's only example is CPU memory, and we cannot tell.
- **Unified memory.** The GH200 presents CPU and GPU memory domains through
  NUMA, and NCSA's page notes the GPU NUMA ID as a new feature. Can vLLM's KV
  cache spill into or be backed by the Grace LPDDR5X, and if so is that a real
  option or a performance trap? For us this is the difference between 96 GB and
  something much larger for the 32K high-concurrency cell.
- **What `--mem` to request** for a single-GPU vLLM job so that we are charged 1
  superchip-hour rather than 2 or 3.

---

## 5. The ARM rebuild, and whether it breaks comparability with everything we have measured

**This slot was left blank; this is our suggestion for it, since it is the
largest thing we learned this week. Swap it out if something else is more
pressing.**

**Ask:** DeltaAI is ARM. Our entire measured record comes from an x86_64 vLLM
build. What is the right way to add a GH200 point without invalidating the A40
and A100 numbers?

**Why we are asking.** Our vLLM is a pinned multi-architecture build, commit
`0.29.1rc1.dev345` with torch 2.13, compiled for sm_80, sm_86 and sm_90 on an
x86_64 host, and it took real effort to get right: half the kernels had no
sm_80 build at one point and the diagnosis took three wrong guesses before a
bisection found it. DeltaAI's host CPU is ARM, so that venv cannot be used
there. The whole stack has to be rebuilt on aarch64.

**The risk is not the build time, it is the comparison.** Our method rule is
that every speedup is a ratio computed inside one job against that job's own
no-speculation arm, because absolute throughput moves by almost 9% across nodes
while ratios replicate to under a tenth of a percent. That protects us across
nodes of the same build. It does not obviously protect us across two different
builds of vLLM on two different CPU architectures, and our cross-hardware cost
model claim is precisely a claim about predicting from one machine onto another.
If the GH200 point comes from a different binary, we may not be able to tell a
hardware effect from a build effect.

**What we would want from the answer.** Whether to rebuild from the same pinned
commit on aarch64 and treat the build difference as a stated threat; or to
re-run one A40 or A100 cell under a freshly built binary to measure the build
effect directly before trusting any GH200 number; or whether the TA thinks the
cross-architecture point is not worth the hours at all and we should spend them
deepening the two platforms we have.

**One correction to carry into the meeting.** Our own notes and the Checkpoint 1
deck say "H200, billed at 3x". NCSA's DeltaAI accounting page sets one service
unit per GH200-hour with no multiplier by GPU type, and DeltaAI's GPUs are
H100s at 96 GB, not H200s. We should not repeat the 3x figure without checking
Delta's own accounting page, which is a separate system with separate rates.

---

## Candidates for a sixth question, if there is time

- **Intervals when the engine is deterministic.** Repeated runs at fixed
  configuration return bit-identical output, 150 of 150 sequences, so repeats
  measure nothing, and this build leaves per-request timing structurally empty,
  so the usual paired bootstrap is unavailable. Our plan is disjoint prompt
  subsets by document id. Is there a better approach?
- **A contaminated control.** vLLM issue #47930 says prefix caching plus skipped
  drafter steps leaves restored prefix tokens without drafter KV and collapses
  acceptance to about 1.02. Every run we have filed has prefix caching on. We
  have not yet run the caching-off control. How much should we worry before
  re-running the grid?
- **Both rungs are measured at non-optimal k.** DFlash ran at k=7 of a native
  15, and our own sweep suggests n-gram at k=7 beats the k=5 we filed. The sign
  of the inversion replicated across two nodes and two GPU types, but the
  magnitude is a statement about one arbitrary pair of settings. Is reporting
  the best k per rung per bucket the right fix, or does that overfit?
- **Scope.** The proposal was approved as a hardware-transfer study with the
  router as a stretch goal, and we now lead with the router. Has that been
  signed off?

---

## Answers

Record what the TA says here, dated, with who asked. An answer that only exists
in someone's memory is not an answer.

| Date | Question | What the TA said | What we changed |
|---|---|---|---|
| | | | |
