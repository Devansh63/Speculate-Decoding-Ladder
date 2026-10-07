# TODO

Ordered by when it has to happen, not by how interesting it is.

For what was already done, see `LOG.md`. For what we learned, see
`notes/findings.md`.

---

## Blocked on: job 22713173, A40 32K

Running as of 6 Oct. Everything below the decision depends on what it shows.

**What to check when it lands:**

```bash
grep '^###' /projects/bikd/$USER/Speculative_Ladder/logs/phase0-22713173.out   # 9 lines = all arms ran
grep -i 'maximum concurrency' /projects/bikd/$USER/Speculative_Ladder/logs/phase0-22713173.out
module load cray-python/3.12.12
python3 /projects/bikd/$USER/Speculative_Ladder/ladder/scripts/summarize.py \
        /projects/bikd/$USER/Speculative_Ladder/results/a40
bash /projects/bikd/$USER/Speculative_Ladder/ladder/scripts/push_results.sh "A40 32K sweep, job 22713173"
```

The concurrency check matters more here than anywhere else: A40 fits about
160,912 tokens of KV, so a 32K request takes roughly a fifth of the cache and
the ceiling is near 4.9. The grid asks for 1, 2 and 4. If vLLM reports a
maximum below 4, the top row ran at a lower concurrency than its label claims.

**The two predictions it tests:**

1. n-gram overtakes DFlash at 32K. The ranking of the rungs inverts along the
   context axis.
2. DFlash falls below 1.0x even at concurrency 1, ceasing to pay at long
   context under any load.

---

## THE DECISION: rescue the hardware axis, or go deep on context?

This is the real open question and it is not technical. Both paths are
reasonable; the budget does not cover both.

### Option A: rebuild, restore the hardware axis

Rebuild vLLM from source at commit `bf13ecc2f` with
`TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`, then re-run the A40 cells so the whole
table comes from one binary.

- **Cost:** hours of login-node compiling (no GPU time), plus roughly 3
  GPU-hours of A40 re-runs, plus A100 cells on top.
- **Buys:** the hardware comparison the proposal committed to. A40 at ridge
  point 107 against A100 at 201, and possibly H200 at 206. Without it, the R
  half of `tau / R` is never varied by hardware, which is the whole reason for
  separating the two terms.
- **Risk:** the rebuild might not fix it. The failure is not understood yet,
  only localised. sm_80 cubins are present in the current binary and still do
  not execute. Rebuilding is the obvious move but it is not a guaranteed one.
- **Also:** re-running A40 invalidates nothing already learned, but it does
  mean the 2K, 8K and 32K cells are spent twice.

### Option B: abandon the hardware axis, go deep on context

Keep the current binary and A40 only. Spend the remaining budget on 32K, the
n-gram coverage frontier, the k sweep, and repeated runs for confidence
intervals.

- **Cost:** fits the remaining budget comfortably.
- **Buys:** depth on the axis where the results are strongest. Finding 7 (the
  rungs decaying in opposite directions) is more surprising than anything the
  hardware axis was likely to produce, and it is already half-established.
- **Gives up:** a committed deliverable. The proposal promised a hardware
  comparison and the professor approved it on that basis. Dropping it needs to
  be raised with him, not quietly reframed.
- **Mitigation:** the R model can *predict* other hardware without measuring
  it, and the Cerebras extrapolation is a natural discussion-section use of
  that. Prediction without validation is weaker than measurement, and a
  reviewer will say so.

### What decides it

**The 32K cell.** If n-gram overtakes DFlash there, the context axis carries
the paper on its own, the inversion is the headline result, and the rebuild
becomes optional rather than urgent. If the rungs hold their 8K ordering, the
context story is a decay curve rather than an inversion, which is weaker, and
the hardware axis is worth rescuing.

**Either way, tell the professor before committing.** He approved a scope that
included hardware. A change of that size is his call as much as yours, and he
has already said more compute is available, which may make the choice moot.

---

## Immediate

- [ ] **Cancel `scancel 22713396`** (ssashi, queued on A100). A100 cannot run
      this build; the job will burn a reservation and fail.
- [ ] **Diagnose Sahil's jobs 22713193 and 22713194.** Both FAILED in 7
      seconds with exit `0:53`, one on A100 and one on A40, so it is not the
      A100 kernel problem. Seven seconds means setup. Check whether the log
      files exist at all; if Slurm could not create the output file, that is
      the answer and it is a permissions issue on the logs directory, not the
      venv. The venv group theory was wrong: `id ssashi` shows `grp_202` is
      his primary group, so he could read it all along.
- [ ] **Request an allocation top-up.** Roughly 3 GPU-hours remain against a
      10-hour deposit, and Option A needs more than that. The professor has
      confirmed more is available. Ask before a job is refused, not after.
      Mention the balance lag while you are at it: `accounts` has not tracked
      actual usage closely.

---

## Backlog, ordered by value per GPU-hour

- [ ] **Confidence intervals.** Every number is a single run, and two measured
      speedups (1.02x, 1.07x) sit close enough to 1.0 that a reader will ask
      whether they differ from noise. Harness v2 records per-request latency,
      so a paired bootstrap is possible, but no data with it exists yet. This
      is the cheapest fix to the most likely reviewer objection.
- [ ] **n-gram coverage frontier.** Vary `prompt_lookup_min` (currently 3) to
      trade acceptance for coverage. n-gram's ceiling at 13% coverage is 1.25x
      even with free verification. At 8K its coverage already rose to 16%
      unassisted, which makes the frontier more interesting, not less. Cheap,
      concurrency-1 runs only, and as far as we have seen unpublished.
- [ ] **k sweep at high concurrency.** Fitting R's slope per draft position
      predicts the optimal k shrinks with load: about 7 at concurrency 1, 3 at
      32. Only the k=15 point supports the direction. One extra arm tests it.
- [ ] **Prefix-caching-off control.** Splits the two causes of the 0.80%
      non-determinism floor. `--no-prefix-caching` is already in the harness
      and writes to its own `_nopc` run id.
- [ ] **Quality evaluation.** Losslessness was assumed, so task accuracy was
      never measured. Finding 6 weakens that assumption. Does the 0.65% token
      divergence change any LongBench answer? LongBench-v2 is multiple choice,
      so this is cheap to score.
- [ ] **Count cubins per architecture** in the vLLM fat binary. Free, and it
      would show whether the build emitted sm_80 for only a subset of kernels,
      which is the leading explanation for the A100 failure.

---

## Deferred, with reasons

| Item | Why it is not being done |
|---|---|
| H200 | No sm_90 cubins and no PTX. Needs the same rebuild as A100; folded into Option A. |
| Cerebras | `sdk.cerebras.ai` is the CSL kernel SDK, not an inference API. Porting a serving stack is not a semester task. Belongs in the discussion as a prediction target: wafer-scale SRAM puts the ridge point roughly an order of magnitude below a GPU's, so the R model predicts speculation is worthless there even at batch 1. Verify the specs from Cerebras's own sheet before writing it. |
| Second model family | Would address the single-model weakness, but costs a full grid and the project has one GPU. |
| EAGLE-3 and self-speculative rungs | Stretch rungs in the proposal. Not worth adding while the must-do rungs lack confidence intervals. |
