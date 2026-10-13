# Threats to validity

Written early on purpose: each one names the confound and the control, so the
control gets built into the experiment rather than bolted on at writing time.

**Workload favours copying.** LongBench-v2 QA answers quote the context, which
inflates n-gram acceptance. Already visible at 2K. Control: report crossings on
natural-text workloads, treat any extractive instruction as a labelled stress
case, and say which is which in every figure caption.

**Comparing rungs at different draft lengths.** A k=15 drafter has more room to
lose tokens than a k=5 one. Control: truncate every acceptance surface to a common
k before comparing, which is free because the per-position counts are recorded.

**One checkpoint per mechanism.** Each mechanism is represented by a single
download carrying its own training data and position range. A result about
"DFlash" may be a result about that checkpoint. Control: a second EAGLE-3 or
DFlash checkpoint in at least one cell, and language that says "this checkpoint"
unless the second one agrees.

**Context regime.** Qwen3-8B is native to 32K; 128K needs YaRN, which the DFlash
head never saw. Control: native and YaRN reported as separate populations, cost
model fitted on native, no decay statistic computed across the boundary.

**Model runner.** n-gram silently falls back to the stable runner while DFlash
stays on V2 unless pinned. Control: pinned and recorded per run, never pooled.

**Uncontrolled concurrency.** The first run measured whatever batch size the
engine chose, which is a high-load point where speculation loses by construction.
Control: `max_num_seqs` set explicitly for every cell.

**Single runs without intervals.** Every number so far is one run. Control: paired
bootstrap over prompts, and five repeats on the comparisons that carry a claim.
*Amended 11 Oct:* per-request timing is unavailable on this build, so the paired
bootstrap is impossible; the control is five **disjoint prompt subsets** per
bucket, each its own job, with the spread across subsets as the interval
(`notes/2026-10-11-a100-decision-tree.md`, section 3). Same-prompt repeats do not
count: they re-measure node noise, already shown at 0.1% on ratios.

**Buckets share documents.** All 25 ids in the 32K cell are among the 2K cell's
50, and the 2K and 8K prompt lists are identical in order, so 2K and 8K average
25 documents absent from 32K. A cross-bucket comparison is therefore partly a
composition difference. Control: report the matched-25 subset alongside the
full-50 figures for 2K and 8K; the per-request files carry the ids.

**Prediction that is really a measurement.** If the held-out platform is profiled
at depth before the prediction is frozen, the prediction is not out of sample.
Control: only bandwidth and one attention-kernel timing before the freeze; the
tag and commit hash go in the figure caption.

**9. τ does not transfer across machines, and the outputs differ.** A
`token_ids` diff of `perreq_ngram_32K_k5_n25_s1234_c1.json` between A40 and A100
shows **21 of 25 sequences differ**, per-token divergence hazard 0.89%. Four of
six machine-factor cells have |Δτ_eff| above 1% (2K n-gram −4.12%, 32K n-gram
−3.24%, 2K DFlash +1.14%, 8K n-gram +1.03%). The earlier claim that "τ
transfers within 1%" holds only for DFlash at 8K (−0.04%) and 32K (+0.97%) and
must be narrowed to those cells. **τ is to be reported per machine**; the
1.390-vs-1.345 discrepancy is not an error in either figure.

Divergence is front-loaded — 8 of the 21 first diverge within the first 16
tokens — so a single hazard rate under-predicts agreement (10.1% implied against
16.0% observed). Prefill numerics most likely differ and the first token follows.

**10. Sequence-level output agreement is a length artefact.** Agreement is the
per-token hazard raised to the output length, so at 256 tokens it mostly
measures how long the outputs are. C10's figures (12.7% baseline-against-itself,
19.0% speculative-against-baseline) are consistent with per-token rates of 0.80%
and 0.65%, which means the right statistic is the **per-token hazard** and on
that statistic speculation perturbs output *less* than the engine perturbs
itself across batch sizes. That is C10's withdrawn direction, so the metric —
not only the control — needs replacing. The same-configuration C10 control is
still required before any null exists.

**11. n-gram may have been run at a suboptimal k throughout.** The A40 n-gram
k-sweep gives 32K speedups of 1.0432 (k=1), 1.1320 (k=2) and 1.1809 (k=7)
against 1.078 for the k=5 used in the main grid and in the frozen prediction.
Coverage falls with k (0.308 → 0.260 → 0.195) while τ_cond rises (1.670 → 2.123
→ 3.073), and the product favours k=7. If this survives re-measurement on a
non-excluded node, every n-gram number in the paper is understated and the
frozen prediction tested a k that was not the rung's best.
