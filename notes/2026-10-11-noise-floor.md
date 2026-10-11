# A 32K outlier, and what it says about the noise floor

**Source** job 22757472 (32K) and 22757473 (2K), compared against jobs
22713173 and 22714602 (32K) and 22598608 (2K). No GPU hours were spent to
learn this; it came out of a baseline arm that `kprobe_32k.sh` re-runs on
purpose.

The script's own comment anticipated exactly this check:

> This gives a second independent realization of the headline 32K
> concurrency-1 row.

It did, and the realization disagrees with the previous two.

## The measurement

The `off` arm at concurrency 1 — no speculation, no drafter, identical seed,
identical prompt count, identical code:

| bucket | realizations (tok/s) | spread |
|---|---|---|
| 2K | 32.6 (22598608), 32.4 (22757473) | **0.61%** |
| 32K | 14.9, 14.8 (22713173, 22714602 — per C11), 13.5 (22757472) | **8.8% low outlier** |

**This is the correction to the first version of this note**, which called 32K
irreproducible to ten percent. It is not. Two of three realizations agree to
0.67 percent, which is the same order as 2K. The third sits 8.8 percent below
both. That is an *occasional excursion*, not a wide distribution, and the two
have different causes and different fixes.

The derived quantity shows the same shape. Fitting `R(k)` for n-gram from the
new jobs and predicting k=5, against the k=5 value measured earlier:

| bucket | predicted R(5) | measured R(5) | miss |
|---|---|---|---|
| 2K | 1.086 | 1.083 | **0.3%** |
| 32K | 1.173 | 1.290 | **10%** |

The model is sound — at 2K it predicts a held-out point to three parts in a
thousand. The 32K miss is inherited from the excursion, not from the model.

## Why an excursion is worse than high variance, for planning

If 32K variance were uniformly 9 percent, N=3 would give a t-interval
half-width near 22 percent and we would simply need more repeats. An
occasional excursion is nastier: **a single excursion in three runs moves the
mean by about 3 percent and inflates the sample standard deviation enough to
make the interval meaningless**, and no feasible N fixes that without either
identifying the cause or adopting a robust estimator and justifying it.

So the correct next step is **not** to buy repeats. It is to find out what
makes a 32K run occasionally slow, which is free from logs already on disk.

## What this invalidates, and what it does not

Speedup is computed per arm against the `off` arm **from the same job**, in
the same process on the same node, so a slow node cancels out of the ratio.

**Unaffected — within-job comparisons:**

- **The inversion (C8).** DFlash 0.83x and n-gram 1.08x at 32K concurrency 1
  both come from job 22713173 against its own baseline. A 30 percent gap in a
  within-job ratio.
- **DFlash's 32K k-trend (C9).** Four monotonic points from job 22714602:
  R = 1.677, 1.704, 1.743, 1.766 at k = 1, 2, 4, 7.
- **Everything at 2K.**

**Affected:**

- **n-gram's 32K k coefficient.** Non-monotonic *within* its own job
  (R(1) = 1.156 > R(2) = 1.141): a 0.015 inversion against a 0.033 trend. Not
  resolved.
- **Cross-job 32K comparisons**, including DFlash's R_draft 0.662 (22714602)
  against n-gram's 0.139 (22757472). The gap is 4.8x, far outside any
  excursion of this size, so the qualitative conclusion stands; the values
  should not be quoted to three decimals.
- **Absolute throughput at 32K.** Report ratios.

## Suspected cause, in order

1. **KV cache allocation variance.** Logged cache sizes at 32K range from
   125,456 to 173,104 tokens across arms and jobs — a 28 percent spread. At
   33,792-token requests a small allocation leaves almost no headroom, and
   vLLM's response to running out is preemption and recompute, which is
   exactly a silent slowdown with no error. This also explains why 2K never
   shows it: a 2K request is a fortieth of the cache, so allocation variance
   cannot bite. **Leading candidate, and it predicts an on/off excursion
   rather than a smooth variance — which is what we observe.**
2. **n = 25.** Half of 2K's sample, on a heavier-tailed quantity. Would
   produce smooth variance, not an excursion.
3. **Node-to-node variation.** Not ruled out; the node is recorded in the
   logs and has not been checked.

Free to test, and worth doing before any repeats:

```bash
grep -i "preempt\|recompute\|cache full" logs/phase0-22713173.out | head
grep -i "GPU KV cache size" logs/kprobe32k-22757472.out
grep -i "GPU KV cache size" logs/kprobe32k-22714602.out
hostname -s   # and the node line at the top of each log
```

If the slow job's `off` arm got a materially smaller cache than the fast
jobs', candidate 1 is confirmed and the fix is a configuration change —
lowering `max_model_len` toward the actual bucket length, or pinning
`gpu_memory_utilization` — rather than more GPU hours.

## A data-handling problem found on the way

Jobs 22713173 and 22714602 both wrote records into
`results/a40/results.jsonl` with the same key `(32K, conc 1, off, k=0)`.
`summarize.py` keys on that tuple, so it collapses them and the table now
shows a single 32K baseline row. The two per-job values survive only because
C11 recorded them by hand at the time.

This is why the 8 Oct n-gram sweeps were sent to `results/a40_ngramk` and why
any repeat series must get its own output directory per run. A repeat written
into a shared directory is not a repeat; it is an overwrite.

## Consequences for the project

**The line "run-to-run variance is below 1 percent" is not wrong but is
incomplete**, and C11 has been amended rather than replaced. Two 32K
realizations do agree to 0.67 percent. A third does not. One replication
bounds nothing; it only fails to detect.

**Repeat counts and output paths must be set per bucket and per run.** And no
32K claim should be reported with an interval until the excursion is
explained, because an interval computed over a contaminated sample is worse
than an honest "measured, single run" — it looks like rigour.

## The pattern, again

`LOG.md`'s dead-ends table notes that most errors here came from trusting a
proxy instead of testing the thing. This note's own first version did it too:
I saw one disagreement and generalized it into a variance estimate, without
checking that the register already held two agreeing realizations. One
observation is not a distribution — which is the same error as fitting an
intercept to a single point, in a new costume.
