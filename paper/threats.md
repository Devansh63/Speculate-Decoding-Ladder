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
