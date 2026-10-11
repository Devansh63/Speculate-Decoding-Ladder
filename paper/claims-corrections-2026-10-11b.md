# Corrections, 11 Oct 2026 (second review)

**This file is urgent and separate so that it lands before the A100 results
are read.** Fold it into `paper/claims.md` at the next opportunity; the
append-only rule applies to that file and nothing here edits an earlier
entry.

A second independent review, briefed only from `HANDOFF.md` and the
repository, produced four objections, each backed by arithmetic rather than
assertion. **All four are accepted.** Three of them concern entries made
earlier the same day.

---

## 1. WITHDRAWN: the capacity confound on the frozen prediction

The frozen prediction (`claims.md`, "Frozen prediction, 11 Oct 2026")
carries this clause:

> **Pre-registered confound.** The A100-SXM4-40GB has 40 GB of memory
> against the A40's 48. At 32K the concurrency ceiling is therefore lower and
> a long-context cell may be in a different memory regime. A large miss at
> 32K may be a capacity artifact rather than a failure of the cost model, so
> the prediction should be judged primarily on 8K.

**That clause is withdrawn.** It does not survive arithmetic. At concurrency
1, a 32K cell needs roughly:

| item | size |
|---|---|
| Qwen3-8B weights, bf16 | ~16.4 GB |
| target KV, 33,792 tokens at 144 KiB/token | ~4.9 GB |
| DFlash-b16 drafter weights | ~1.9 GB |
| drafter KV | ~0.7 GB |
| **total** | **~24 GB** |

Against 40 GB. It fits comfortably. **There is no capacity pressure at
concurrency 1**, and the prediction is a concurrency-1 prediction.

This was an escape hatch written into a pre-registration without checking
that it was physically possible. That is worse than having no confound
clause at all, because it pre-licenses explaining away a failure. The
original frozen text stands unedited, as the register's rules require; this
withdrawal is recorded against it.

**Timing matters and is recorded deliberately.** Jobs 22802261 (8K) and
22802262 (32K) were submitted before this was written and **no A100 cell at
full n has been read.** Withdrawing the clause now is housekeeping;
withdrawing it after seeing a 32K miss would have been moving the goalposts.

### Replacement reason to weight 8K

The instruction to judge primarily on 8K survives, but on a different and
better basis. The machine factor 1.054 was measured **only at 2K**. C9
establishes that `R`'s *composition* changes with context — `R_draft` grows
2.24x from 2K to 32K for DFlash — so applying a factor fitted at 2K to 32K
is a materially longer extrapolation than applying it to 8K.

Therefore:

- **8K is the nearer test** and carries the weight.
- **A 32K-specific miss is informative, not excusable.** If 8K lands and 32K
  misses, the reading is that the machine factor is **context-dependent** —
  that `R` is not purely a machine property, because the drafter's cost over
  the prefix may scale differently on different GPUs. That would weaken
  **C17**, not merely C4, and it is a result rather than an artifact.

---

## 2. C13 is wrong as stated

C13 says:

> The inversion is driven by the absolute drafting-cost gap, not by a
> difference in how the two drafters scale.

Tested by holding one factor at its 2K value and moving the other to its 32K
value, using measured `tau_eff` and `R` for DFlash k=7 at concurrency 1:

| counterfactual | speedup | against n-gram's 1.08x |
|---|---|---|
| 2K acceptance (tau_eff 2.536) with 32K cost (R 1.766) | **1.44x** | DFlash still wins — **no inversion** |
| 32K acceptance (tau_eff 1.459) with 2K cost (R 1.360) | **1.07x** | **tied — inversion on the cusp** |

**Moving cost alone does not flip the ranking. Moving acceptance alone
roughly does.**

The error was not in the cost arithmetic, which stands: `R_draft` does grow
2.24x against n-gram's 1.96x, and the absolute gap does widen from 0.225 to
0.523. The error was building a **cost-only causal story** while C1 and C2
were already measuring a large acceptance collapse — DFlash's `tau_cond`
falls 2.543 to 1.459 and its first-position acceptance 0.601 to 0.296, while
n-gram's `tau_cond` is flat (2.889 to 2.844) and its coverage *rises* 0.132
to 0.212. Both channels were in the data; only one went into the explanation.

**Corrected statement:** the inversion requires both channels, and the
acceptance channel is the larger contributor. DFlash pays roughly 2.2x more
to draft at 32K *and* gets drafts that are worth roughly 1.7x less, while
n-gram's yield is flat and its opportunities increase.

The decomposition is order-dependent, as any such counterfactual is. The
asymmetry survives either ordering: no ordering makes the cost change
sufficient on its own.

---

## 3. C17's mechanism is withdrawn; the observation stands

`notes/2026-10-11-hardware-transfer.md` and `HANDOFF.md` section 2.7 explain
the A100 result as: a faster machine wastes less memory bandwidth, so
speculation has less to reclaim.

**That explanation contradicts a roofline reading.** Ridge points on dense
bf16:

| GPU | peak bf16 | bandwidth | ridge point |
|---|---|---|---|
| A40 | ~74.8 TFLOPS | ~696 GB/s | ~107 FLOP/byte |
| A100-SXM4-40GB | ~312 TFLOPS | ~1555 GB/s | ~201 FLOP/byte |

The A100 has the **higher** ridge point, meaning more arithmetic intensity is
needed before compute binds — so there is *more* headroom to convert
bandwidth-bound decoding into compute-bound verification, and naive roofline
predicts speculation should gain **more** on A100. We measured **less**
(DFlash 1.89x to 1.79x, n-gram 1.14x to 1.06x).

**The observation stands; the mechanism does not.** Honest status: direction
measured, mechanism unexplained.

A candidate hypothesis, recorded as a hypothesis: at batch size 1, fixed
per-step overheads — kernel launch, CPU scheduling, drafter setup — are a
larger *fraction* of a faster GPU's step time, which would inflate `R` on the
faster machine without any bandwidth story. Testable by profiling a step, not
by more cells.

The spec-sheet figures above are from the review and remain unverified
against NVIDIA's published numbers. **That check is now a prerequisite for
any mechanism claim in the paper**, and it is free.

---

## 4. C8: the two realizations share one prompt set

Jobs 22713173 and 22714602 both used **the same 25 prompts with the same
seed**. They therefore bound **node variance and run-to-run engine
variance** — which is what they were cited for — and say **nothing about
sampling variance across prompts.**

At n=25, on a heavy-tailed quantity like long-prompt latency, that is a real
gap. The inversion's margin is ~30% and probably survives it, but "C8 is
replicated" must carry the qualifier: replicated across nodes and runs, not
across prompt samples.

**What would close it:** a 32K cell at concurrency 1 on a *different* prompt
subset — a different seed or a different slice of the bucket. Roughly 1
GPU-hour, and it tests something none of the existing repeats test.

---

## 5. Also corrected, from the first review

A separate review the same day produced four further items, recorded here for
completeness and detailed in the prompt issued to that reviewer.

- **C10 restated.** `check_lossless.py` compares the baseline against itself
  **at different concurrency** (line 111) while comparing speculative arms
  against the baseline **at matching concurrency** (line 117). The control
  therefore carries two sources of difference and the treatment one. The
  strong claim — speculation perturbs output less than the engine perturbs
  itself — is **not established**. What is supported: *changing concurrency
  perturbs output more than adding speculation does at fixed concurrency.*
  The strong form needs a same-config `off` vs `off` control, which does not
  exist: the identical-config pairs we have (22713173/22714602,
  22757102/22757105) each wrote to one directory and the per-request
  filenames do not encode the run, so the earlier job's token ids were
  overwritten.
- **C15 wording corrected twice over.** The original overclaim was "n-gram is
  faster at long context" (demoted). The replacement, "does not degrade with
  context", is **also wrong**: at matched k=5 the speedup falls from ~1.15 at
  2K to ~1.08 at 32K, about 7%, with two runs at each end agreeing to ~0.1%.
  Correct statement: n-gram degrades mildly with context, far less than
  DFlash, while its `tau_cond` is context-invariant and its coverage rises.
- **The falsification criterion had a gap.** Pass required all four cells
  within 5%; fail required median absolute relative error above 10%. With
  four cells, one badly missed cell among three good ones satisfies neither.
  Clarified: score **per cell with signed errors**, weight 8K per section 1
  above, and score the 32K *ordering* (DFlash below 1.0, n-gram above)
  separately from the 32K *magnitude*. Written before any A100 cell at full n
  was read.
- **The k-versus-load axis has no valid data.** `LOG.md` and
  `paper/results-table.md` cite job 22569403 as evidence that optimal k
  shrinks with load, while `results-table.md` simultaneously disqualifies
  that run — "those speedups measure the batching decision rather than the
  drafter". Both statements were written by the same author hours apart.
  **No valid load-varying k data exists at any bucket**, and C6's second axis
  is unsupported.

---

## 6. Review credit, and why it is recorded

Items 1–4 came from one external review and item 5 from another, both
briefed only from this repository. Neither had access to the reasoning that
produced the errors, which is the point.

This register keeps failed predictions on purpose. Errors caught from outside
belong in the same place and for the same reason: a record that contains only
the claims that survived is not evidence about the process that produced
them.

The pattern is also consistent with the one already documented in `LOG.md` —
conclusions drawn from a proxy rather than from the thing itself. Three of
the four items in section 1–4 were refuted by arithmetic that was available
the whole time: a memory budget, a counterfactual on two measured numbers,
and a ratio of two spec-sheet figures. None required a GPU.
