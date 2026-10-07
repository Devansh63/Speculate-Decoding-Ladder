# TODO

Ordered by when it has to happen, not by how interesting it is.

For what was already done, see `LOG.md`. For what we learned, see
`notes/findings.md`.

---

## RESOLVED: the scope decision

**Option B. The context axis carries the paper.** The A100 rebuild drops from
urgent to optional.

The 32K cell (job 22713173) decided it. At concurrency 1 the ranking of the
rungs inverts across the context axis: DFlash 1.86x to 0.83x, n-gram 1.16x to
1.08x, so the cheap retrieval rung overtakes the trained head and the trained
head becomes actively harmful. Both predictions from the 8K note held. Details
in `notes/2026-10-06-32k-inversion.md`.

That is a stronger claim than the hardware axis was likely to produce, it is
already established on three points, and it costs nothing more to defend.

**Still tell the professor.** He approved a scope that included a hardware
comparison, and dropping it is his call as much as yours. Lead with the
inversion; it is a better result than what was promised, which makes the
conversation an easy one. He has also offered more compute, so the hardware
axis may still be affordable as a bonus rather than a core deliverable.

---

## Highest value remaining: the k=1 experiment

**DFlash at 32K may not be broken, just misconfigured.** Its acceptance by
position at 32K is `0.296, 0.087, 0.038, ...`, dead after position two, while
the fitted marginal cost is 0.109 per draft position. So position 2 already
costs more than it returns and **the optimal k is 1, not 7**. At k=1 the model
gives tau 1.296, R about 1.109, speedup about **1.17x**, which would beat
n-gram's 1.08x.

If it holds, the paper's prescription becomes rung *and* draft length, with k
shrinking as context grows. Prescriptive beats descriptive.

One job, concurrency 1 only, A40:

```bash
cd /projects/bikd/$USER/Speculative_Ladder/ladder/scripts
sbatch --time=01:00:00 --export=ALL,BUCKET=32K,LIMIT=25,CONC="1" phase0.sh
```

`phase0.sh` hardcodes `k=7` for DFlash, so this needs either a small edit to
parameterise k, or a direct `run_benchmark.py` call with `--k 1` and `--k 2`.
Two arms plus the shared baseline is about 20 minutes.

---

## Confidence intervals: the plan changed

```
per request timing available on 0/25 requests
```

This vLLM build does not populate per-request metrics on the V1 engine, so the
**paired bootstrap over prompts is not possible**. Harness v2's timing code is
inert.

The remaining path is **repeated runs**, estimating run-to-run variance.
Coarser, but the only option short of patching vLLM.

- [ ] Sahil's `SEED=5678` 2K repeat, already assigned. This is now foundational
      rather than optional.
- [ ] **Repeat the 32K cell at least twice.** The headline claim lives there
      and currently rests on 25 prompts in a single run. Highest priority after
      the k experiment.

---

## Immediate

- [ ] **Tell the professor** about the scope change and the inversion result.
- [ ] **Request an allocation top-up.** Roughly 1 to 2 GPU-hours remain.
- [ ] **Diagnose Sahil's 7-second failures** (22713193, 22713194). Both on
      different partitions, so not the A100 kernel problem. Seven seconds means
      setup. Check whether the log files exist at all; if Slurm could not create
      the output file, that is the answer.

---

## Backlog, ordered by value per GPU-hour

- [ ] **k sweep across buckets.** Finding 9 says optimal k shrinks with
      context, and the 2K k=15 result says it shrinks with load too. A small
      grid of k against bucket would turn two anecdotes into a surface.
- [ ] **n-gram coverage frontier.** Vary `prompt_lookup_min` (currently 3).
      n-gram's coverage rose on its own from 0.132 to 0.212 across the context
      range; forcing it higher trades acceptance for coverage. Needs a flag in
      `run_benchmark.py` first.
- [ ] **Drafter memory cost.** At 32K the DFlash arms got 20 to 25% less KV
      cache than the baseline because the head occupies GPU memory, and the
      concurrency-4 arm cleared its requested concurrency by 0.02. A rung that
      needs resident weights competes with the KV cache, and that cost appears
      nowhere in `tau / R`. Worth quantifying as a third axis of cost.
- [ ] **Prefix-caching-off control.** Splits the two causes of the 0.80%
      non-determinism floor. `--no-prefix-caching` is already in the harness.
- [ ] **Quality evaluation.** LongBench-v2 is multiple choice, so scoring is
      cheap. Does the 0.65% token divergence change any answer?

---

## Deferred, with reasons

| Item | Why |
|---|---|
| A100 rebuild | Root cause known: half the kernels in the vLLM extension have no sm_80 build (`env/a100-sm80-failure.md`). A rebuild with `8.0;8.6;9.0` should fix it and would also cover H200. Now optional, since the context axis carries the paper. Costs hours of compiling plus ~3 GPU-hours of A40 re-runs for a single-binary table. |
| H200 | Same rebuild. |
| Cerebras | `sdk.cerebras.ai` is the CSL kernel SDK, not an inference API. Belongs in the discussion as a prediction target: wafer-scale SRAM puts the ridge point an order of magnitude below a GPU's, so the R model predicts speculation is worthless there even at batch 1. Verify specs from Cerebras's own sheet first. |
| Second model family | Would address the single-model weakness but costs a full grid. |
| EAGLE-3 and self-speculative rungs | Stretch rungs. Not worth adding while the must-do rungs lack intervals. |
