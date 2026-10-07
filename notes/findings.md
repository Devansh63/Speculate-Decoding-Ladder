# Findings log

Append-only. One entry per thing learned, newest at the bottom, with the date and
how it was established. Anything here that came from a document rather than a
measurement says so.

For **what was run and what it cost**, see `LOG.md`. This file is for what we
learned.

## 30 Sep 2026

**The runner has to be pinned or the comparison is void.** vLLM chooses its model
runner automatically (`vllm/config/vllm.py:694`). If a configuration hits
something the V2 runner does not support, it falls back to the stable runner with
only a warning. n-gram is on that unsupported list; DFlash v1 is not. With default
settings the two must-do rungs would run on different code paths. Every run sets
`VLLM_USE_V2_MODEL_RUNNER=0` and `run_benchmark.py` refuses to start without it.
Established by reading the source at commit bf13ecc2.

**vLLM already records per-position acceptance.** `vllm:spec_decode_num_drafts`,
`num_draft_tokens`, `num_accepted_tokens` and `num_accepted_tokens_per_pos`, read
through `llm.get_metrics()`, but only if the engine is constructed with
`disable_log_stats=False`, which `LLM()` silently defaults to `True`. This removes
the planned metrics patch from the critical path.

**Engine startup dominates short runs.** 263 s without a drafter, 209 s with one,
mostly torch.compile and JIT kernel warmup. A nine-cell sweep pays that nine
times, which is why the compile cache is shared across runs and users.

**First measurement (job 22569403).** At 2K with uncontrolled concurrency every
rung is slower than no speculation (0.86x, 0.78x, 0.71x), while acceptance is
healthy (2.5 to 3.0). The cost term, not the acceptance term, is what sinks it at
high load. Concurrency must be controlled before any speedup number means
anything.

**n-gram out-accepts DFlash at 2K on this workload.** 2.979 against 2.545 at a
common k=5. Later qualified: that is *conditional* acceptance, and n-gram fires on
only 13% of steps. See 1 Oct.

**Charging on Delta is per reserved resource, and the A100 costs twice the A40.**
Slurm shows `billing=512` for one A40 and `billing=1024` for one A100 on the same
request shape. Whether the H200 is 1x or 3x is still open.

## 1 Oct 2026

**tau is invariant to serving load; all the variation in speedup is in R.**
DFlash mean accepted length across concurrency 1, 8, 32 and saturation: 2.543,
2.507, 2.470, 2.520. First-position acceptance: 0.601, 0.600, 0.597, 0.601.
Speedup over the same points: 1.86x, 1.49x, 1.03x, 0.86x. This is the
`Speedup ~= tau / R` decomposition measured rather than assumed, and it is what
makes tau portable across machines. Job 22598608.

**The decomposition needs a third term: coverage.** vLLM's mean accepted length
is conditional on a draft having been proposed. DFlash proposes on every step;
n-gram fires only on a prompt-lookup hit, measured at 13.2% of steps. So
`tau_eff = cov * tau_cond + (1 - cov) * 1`, verified as
`0.132 * 2.889 + 0.868 = 1.249` against a measured 1.250. Corrected for coverage,
n-gram's implied R is 1.082, *below* DFlash's 1.360, which is what the hardware
predicts for a CPU string lookup verifying 6 tokens instead of 8. n-gram is the
cheapest rung per pass and loses only because it is idle seven steps in eight.

**Published acceptance figures for lookup-based proposers are not comparable to a
model drafter's.** Follows from the above, and is a methodological point beyond
this project.

**The baseline knee explains the crossings.** No-speculation throughput at 2K:
32.6, 172.1, 335.5, 405.1 tok/s, which is 5.3x for 8x the requests, then 1.95x for
4x, then 1.21x. The GPU saturates between concurrency 8 and 32, exactly where
speculation stops paying. Speculation trades compute for memory traffic, so its
benefit is bounded by the slack below the roofline.

## 5 Oct 2026

**Exact-match losslessness is not testable on a continuous-batching server.**
Hashes showed 0 of 9 speculative arms reproducing the baseline. But the `off` arm
compared against itself at different concurrency, with no speculation anywhere,
agrees on only 12.7% of sequences, while speculative arms agree with the baseline
on 19.0%. **Speculation perturbs the output less than the engine perturbs
itself.** Implied per-token divergence: 0.80% baseline-to-baseline, 0.65%
speculative-to-baseline. Causes are batch-dependent reduction order in GPU kernels
and prefix caching. The claim must be framed relative to the engine's own
reproducibility floor, which is the honest version and arguably a result in its
own right.

**A sequence-level hash is the wrong unit.** One flipped token in a 256-token
generation changes the hash, so a 0.5% per-token rate produces ~72% mismatched
sequences. Report per-token agreement.

## 6 Oct 2026

**The rungs decay in opposite directions along the context axis.** From 2K to 8K,
DFlash's acceptance collapses (tau 2.543 to 1.953, first position 0.601 to 0.451,
D = 0.219) while n-gram's barely moves (D = 0.044) and its coverage *rises* (0.132
to 0.161). DFlash predicts, and prediction gets harder as the context to summarise
grows; n-gram retrieves, and a longer prompt holds more matchable spans. At 2K the
two had identical first-position acceptance; at 8K the cheap retrieval rung is
ahead, 0.563 against 0.451. The speedup gap at concurrency 1 fell from 0.70 to
0.12 in one bucket step, with no change to hardware or batch size. Job 22685911.

**The usable operating window shrinks with context.** n-gram's 1.0x crossing moved
from concurrency 14.4 to 8.9; DFlash's from above 32 to about 12. R rises because
verifying k+1 tokens costs more attention work at longer context, and the baseline
saturates earlier.

**`cuobjdump` listing an architecture does not prove the kernels run on it.** The
vLLM extension lists sm_80 cubins and still fails on A100 with
`cudaErrorNoKernelImageForDevice`, while plain torch runs fine on the same node.
The only reliable test is executing one custom op on the target GPU, which costs
about four minutes. See `env/a100-sm80-failure.md`.

**CUDA errors name the line where they were noticed, not where they happened.**
The first A100 traceback pointed at `output.fill_(0)`, which is a profiling
shortcut that skips attention entirely. It was a sticky error from an earlier
failed launch. `CUDA_LAUNCH_BLOCKING=1` helps but does not cover CUDA graph
capture. Bisect by layer instead.

## Resolved open questions

- ~~Where does each rung's speedup cross 1.0x on the concurrency axis?~~ At 2K:
  n-gram around 14, DFlash only at saturation. At 8K: n-gram 8.9, DFlash about 12.
- ~~Does acceptance decay with context, at different rates per rung?~~ Yes, and in
  opposite directions. D = 0.219 for DFlash, 0.044 for n-gram, 2K to 8K.
- ~~Does DFlash at k=7 dominate k=15 everywhere?~~ At saturation, k=15 bought 0.29
  effective tokens and cost 0.67 in R, so k=7 wins there. Draft positions 8 to 15
  contributed 0.156 tokens between them. Not yet tested at low concurrency.

## Open questions

- Does n-gram overtake DFlash at 32K? Extrapolation says yes. Job 22713173 tests
  it.
- Does DFlash fall below 1.0x even at concurrency 1 at 32K?
- Do prediction-based drafters generally decay faster than retrieval-based ones,
  or is this one drafter on one model family?
- Why do sm_80 kernels in the vLLM fat binary fail to execute? Counting cubins per
  architecture would show whether the build emitted sm_80 for only a subset.
- Is the H200 billed at 3x the A40, as the proposal assumes, or 1x as NCSA's
  published table says? Still unanswered and now moot until a rebuild.
- How much of the 0.80% non-determinism floor does prefix caching own? A
  caching-off control answers it.
- Does the 0.65% token divergence change any task answer? No quality evaluation
  has been run.
