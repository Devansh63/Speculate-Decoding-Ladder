# Findings log

Append-only. One entry per thing learned, newest at the bottom, with the date and
how it was established. Anything here that came from a document rather than a
measurement says so.

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

**First measurement (job 22569403).** See `results/2026-09-30_2K_a40.md`. At 2K
with uncontrolled concurrency every rung is slower than no speculation (0.86x,
0.78x, 0.71x), while acceptance is healthy (2.5 to 3.0). The cost term, not the
acceptance term, is what sinks it at high load. Concurrency must be controlled
before any speedup number means anything.

**n-gram out-accepts DFlash at 2K on this workload.** 2.979 against 2.545 at a
common k=5. Expected direction for extractive QA, but worth stating carefully:
LongBench-v2 answers quote the context, which favours copying. A non-extractive
workload may reverse it, and that reversal is itself a result.

**Charging on Delta is per reserved resource, and the A100 costs twice the A40.**
Slurm shows `billing=512` for one A40 and `billing=1024` for one A100 on the same
request shape. Whether the H200 is 1x or 3x is still open; a pending job's
`billing=` field answers it without spending anything.

## Open questions

- Where does each rung's speedup cross 1.0x on the concurrency axis at fixed
  context? That is the milestone.
- Does acceptance decay with prompt context (2K to 32K) and at different rates per
  rung? That is RQ1 and nothing measured so far speaks to it.
- Does DFlash at k=7 dominate k=15 everywhere, or only at high load?
- Is the H200 billed at 3x the A40, as the proposal assumes, or 1x as NCSA's
  published table says?
