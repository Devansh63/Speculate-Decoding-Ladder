# Activity log

What was run, what it cost, what broke, what was decided. Newest first.

For **what we learned**, see `notes/findings.md` and the dated notes in
`notes/`. This file is the record of activity, including the parts that did not
work.

---

## Job index

| Job | Date | GPU | Cell | State | Outcome |
|---|---|---|---|---|---|
| 22713396 | 6 Oct | A100 | 8K (ssashi) | PENDING | **cancel** — A100 cannot run this build |
| 22713194 | 6 Oct | A40 | ? (ssashi) | FAILED 00:00:07 | undiagnosed, exit 0:53 |
| 22713193 | 6 Oct | A100 | ? (ssashi) | FAILED 00:00:07 | undiagnosed, exit 0:53 |
| 22713175 | 6 Oct | A100 | diagnostic | COMPLETED 00:03:35 | **answered the A100 failure** |
| 22713173 | 6 Oct | A40 | 32K | RUNNING | the decisive cell |
| 22685911 | 5 Oct | A40 | 8K | COMPLETED 01:03:12 | **best result so far** |
| 22685764 | 6 Oct | A100 | 2K | COMPLETED 00:17:03 | **all 9 arms failed**, no results written |
| 22598608 | 1 Oct | A40 | 2K concurrency sweep | COMPLETED 00:52:39 | first valid measurement |
| 22569403 | 30 Sep | A40 | 2K, uncontrolled | COMPLETED | speedups void, acceptance usable |
| (six jobs) | 30 Sep | A40 | — | FAILED ~00:00:05 | `ModuleNotFoundError: vllm` |

A "COMPLETED" state means the shell script finished its loop, not that the
measurements succeeded. Job 22685764 is COMPLETED and produced nothing. Check
`results.jsonl` exists before believing a job worked.

---

## 6 Oct 2026

**A100 is blocked.** Job 22685764 lost all nine arms to
`cudaErrorNoKernelImageForDevice`. Diagnostic job 22713175 isolated it: plain
torch runs a bf16 matmul and `fill_` on the node without trouble, but vLLM's
own `rms_norm` custom op fails with no torch.compile and no CUDA graphs
involved, and both eager and compiled paths die. Node, driver and torch are
healthy; the vLLM extension does not work on sm_80. Full write-up in
`env/a100-sm80-failure.md`. Needs a from-source rebuild, which also covers
H200.

**Decision pending:** rebuild with `8.0;8.6;9.0` and re-run the A40 cells so
the whole table comes from one binary (about 3 GPU-hours plus hours of
compiling), or abandon the hardware axis and go deep on context. Holding until
32K lands, because if n-gram overtakes DFlash there the context axis carries
the paper on its own.

**8K analysed.** The rungs decay in opposite directions. DFlash D = 0.219,
n-gram D = 0.044, and n-gram's coverage *rises* from 0.132 to 0.161. At 2K the
two had identical first-position acceptance; at 8K n-gram is ahead, 0.563 to
0.451. Speedup gap at concurrency 1 fell from 0.70 to 0.12. Write-up in
`notes/2026-10-06-8k-context-decay.md`.

**32K submitted** (22713173) with grid 1/2/4, limit 25, 3h wall. Tests two
predictions: that n-gram overtakes DFlash, and that DFlash falls below 1.0x
even at concurrency 1.

**Team access sorted**, mostly. Shared venv copied into project space with
paths rewritten; group corrected from `grp_202` to `delta_bikd`; an ACL
granting `ssashi` read already existed on the project root. Sahil's two jobs
still failed in seven seconds each and remain undiagnosed.

**Cerebras.** The professor sent `sdk.cerebras.ai`. That is the CSL kernel
development SDK for the Wafer-Scale Engine, not an inference API: no models,
no endpoints, no decoding parameters. Porting a serving stack to it is not a
semester task. Its value is as a *prediction target*: wafer-scale keeps
weights in on-chip SRAM, so the ridge point is roughly an order of magnitude
below a GPU's, and our R model predicts speculation is worthless there even at
batch 1. That belongs in the discussion section, not the experiment plan.
Specs need verifying from Cerebras's own sheet before anything is written.

## 5 Oct 2026

**Losslessness investigated and resolved.** Hashing output token ids showed 0
of 9 speculative arms reproducing the baseline, about 80% of prompts
differing. The control settled it: the `off` arm compared against itself at
different concurrency, with no speculation anywhere, agrees on only 12.7% of
sequences, while speculative arms agree with the baseline on 19.0%.
Speculation perturbs output *less* than the engine perturbs itself. Write-up
in `notes/2026-10-05-losslessness-and-engine-nondeterminism.md`.

**Harness v2** installed before either queued job started, so both picked it
up without a cancel. Adds full output token ids (a SHA-1 cannot distinguish
one flipped token from a different answer), per-request timing for a paired
bootstrap, and a prefix-caching switch. All additions guarded so an
instrumentation failure records a null rather than losing the run.

**8K submitted** (22685911) and **A100 2K submitted** (22685764).

## 1 Oct 2026

**First valid measurement** (22598608, 52 minutes). 2K, three rungs,
concurrency 1/8/32. tau is flat across load (DFlash 2.543, 2.507, 2.470) while
speedup collapses (1.86x, 1.49x, 1.03x), so all the variation is in R.

**Coverage discovered.** n-gram drafts on only 13.2% of decode steps, so
vLLM's reported acceptance overstates what it delivers by more than a factor
of two. Corrected, n-gram's R is 1.082, *below* DFlash's 1.360, which is what
the hardware predicts. `Speedup ~= [cov * tau + (1 - cov)] / R`.

## 30 Sep 2026

**First run** (22569403). All prompts in flight at once, so vLLM batched as
wide as memory allowed and every rung lost to no speculation (0.86x, 0.78x,
0.71x) while acceptance was healthy. Diagnosed as a batching artifact;
concurrency control added.

**Six jobs failed in about five seconds each** with
`ModuleNotFoundError: No module named 'vllm'`. The scripts loaded the Python
module but never activated the venv.

**Environment established.** Qwen3-8B and the DFlash head downloaded,
LongBench-v2 bucketed at 2K/8K/32K, vLLM built from source, runner pin
verified by reading the source.

---

## Decision register

| Decision | Why | Date |
|---|---|---|
| Pin `VLLM_USE_V2_MODEL_RUNNER=0`, refuse to start without it | n-gram silently falls back to the stable runner while DFlash stays on V2, so the rungs would run on different code paths | 30 Sep |
| Control concurrency explicitly rather than letting vLLM batch freely | Otherwise speedup measures the batching choice, not the rung | 1 Oct |
| Shrink the concurrency grid per context bucket | vLLM accepts an impossible `max_num_seqs` and quietly schedules fewer, mislabelling the row | 5 Oct |
| Read the GPU from `nvidia-smi` at run time, not from the partition flag | A mislabelled result is harder to notice than a failed job and poisons the cross-hardware table | 5 Oct |
| Defer H200 rather than rebuild immediately | A rebuild would replace the binary that produced every existing measurement | 1 Oct |
| Keep prefix caching on | It is what a real server does; it cancels from within-bucket speedup ratios. Revisit before the headline table | 5 Oct |
| Store full token ids, not just a hash | A hash cannot distinguish one flipped token from a different answer, and that distinction is the losslessness result | 5 Oct |
| Do **not** switch A100 to `TRITON_ATTN` to make it run | A40 ran on FlashAttention; mixing backends would put the difference into R, which is the quantity being measured | 6 Oct |
| Keep the uncontrolled 30 Sep run in the results | It is the right-hand end of the load axis, not a mistake | 1 Oct |

---

## Dead ends, and what they cost

Recording these is cheaper than rediscovering them.

| What was believed | Why it was wrong | Cost |
|---|---|---|
| A100 is supported, because `cuobjdump` lists sm_80 | The glob `ls $VD/*.so` is not recursive and missed the FlashAttention extension in a subdirectory. Worse, sm_80 *was* listed in the main extension and the kernels still do not run: a listing is not proof of execution | one failed A100 job, ~17 min billed |
| The A100 failure is compile-cache poisoning from A40 runs | Per-config and `torch_aot_compile` hashes are disjoint across GPUs, and the A100 run logged nine saves and zero loads | ~20 min of analysis, no GPU |
| The failure is in FlashAttention, per the traceback | `output.fill_(0)` at `flash_attn.py:1211` is the profiling shortcut that skips attention entirely. The error was sticky, from an earlier failed launch. CUDA errors surface where they are noticed, not where they happen | sent the diagnosis down the wrong path twice |
| Sahil could not read the venv because its group was `grp_202` | `id ssashi` shows `grp_202` is his *primary* group. He could read it all along | a `chgrp` over 100k files; his failures are still undiagnosed |
| `same length` would distinguish a token flip from a different answer | Every generation hits the 256-token cap, so all lengths are equal by construction | one useless diagnostic, rewritten |

**The pattern in four of the five:** a conclusion drawn from a proxy — a file
listing, a directory name, a traceback line, a group name — instead of testing
the thing itself. The fix that worked was bisection: `diag_a100.sh` runs four
layers cheapest-first and the first failure names the culprit. Ten minutes of
GPU time, and it should have been the first move.

---

## Budget

Sub-allocation: 10 GPU-hours deposited per student. More available on request;
the professor has confirmed this.

| Spent on | Approx. |
|---|---|
| 30 Sep first run and failures | ~2 h |
| 22598608, 2K sweep | ~0.9 h |
| 22685911, 8K sweep | ~1.1 h |
| 22685764, A100, produced nothing | ~0.6 h (A100 bills ~2x) |
| 22713175, diagnostic | ~0.1 h |
| 22713173, 32K, in flight | ~2 h est. |
| **Remaining** | **~3 h**, before any top-up |

Delta's `accounts` balance has lagged visibly behind actual usage, so treat
these as estimates. **Request a top-up now**, not after a job is refused.

---

## Open items

- Sahil's jobs 22713193 and 22713194, FAILED in 7 s with exit `0:53`, cause
  unknown. His job 22713396 is queued on A100 and should be cancelled.
- Decide rebuild versus context-depth after 32K lands.
- No confidence intervals anywhere. Harness now records per-request timing;
  no data with it yet.
- Prefix-caching-off control not run.
- Allocation top-up not requested.
