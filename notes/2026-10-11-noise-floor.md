# The noise floor is bucket-dependent: 0.6% at 2K, 8.8% at 32K

**Source** job 22757472 (32K) and 22757473 (2K), compared against jobs
22713173 (32K) and 22598608 (2K). No GPU hours were spent to learn this; it
came out of a baseline arm that `kprobe_32k.sh` re-runs on purpose.

The script's own comment anticipated this exactly:

> And since per-request timing came back 0/25 on this vLLM build, the paired
> bootstrap is impossible and repeated runs are the only route to confidence
> intervals. This gives a second independent realization of the headline 32K
> concurrency-1 row.

It did, and the realization disagrees with the first one.

## The measurement

The `off` arm at concurrency 1 — no speculation, no drafter, identical seed,
identical prompt count, identical code, identical venv:

| bucket | job A | tok/s | job B | tok/s | difference |
|---|---|---|---|---|---|
| 2K | 22598608 (1 Oct) | 32.6 | 22757473 (8 Oct) | 32.4 | **0.61%** |
| 32K | 22713173 (6 Oct) | 14.8 | 22757472 (8 Oct) | 13.5 | **8.8%** |

The derived quantity is worse. Fitting `R(k)` for n-gram from 22757472/3 and
predicting k=5, against the k=5 value measured in the older jobs:

| bucket | predicted R(5) | measured R(5) | miss |
|---|---|---|---|
| 2K | 1.086 | 1.083 | **0.3%** |
| 32K | 1.173 | 1.290 | **10%** |

So the model is fine — it predicts a held-out point at 2K to three parts in a
thousand. The problem is that **a 32K measurement is not reproducible across
jobs to better than about ten percent.**

## What this invalidates, and what it does not

This matters less than it first appears, because of how speedup is computed:
each arm is divided by the `off` arm **from the same job**, in the same
process, on the same node. Node-level slowness cancels.

**Unaffected — within-job comparisons:**

- **The inversion.** DFlash 0.83x and n-gram 1.08x at 32K concurrency 1 both
  come from job 22713173, against its own baseline. The gap is 30 percent
  against a within-job ratio. Safe.
- **DFlash's 32K k-trend.** Its four points from job 22714602 are monotonic:
  R = 1.677, 1.704, 1.743, 1.766 at k = 1, 2, 4, 7. Monotonicity across four
  points measured in one engine session is not threatened by cross-job
  variance.
- **Everything at 2K**, where cross-job agreement is 0.61 percent on
  throughput and 0.3 percent on R.

**Affected — anything at 32K spanning two jobs:**

- **n-gram's 32K k coefficient.** Non-monotonic *within* its own job
  (R(1) = 1.156 > R(2) = 1.141), with a 0.015 inversion against a 0.033
  trend. Not resolved. Marked as such in `paper/claims.md`.
- **The R_draft comparison at 32K.** DFlash's 0.662 comes from 22714602 and
  n-gram's 0.139 from 22757472 — different jobs. The gap is 4.8x, far outside
  a 10 percent band, so the qualitative statement survives; the two numbers
  should not be quoted to three decimals.
- **Any absolute throughput figure at 32K.** Report ratios, not tok/s.

## Suspected cause

Not established, and worth distinguishing from the alternatives before
anything is claimed:

1. **KV cache allocation variance.** Logged cache sizes at 32K range from
   125,456 to 173,104 tokens across arms and jobs — a 28 percent spread. A
   smaller cache at 32K is close to the point where sequences get preempted
   and recomputed, and 33,792-token requests leave little headroom. This is
   the leading candidate and it also explains why 2K is unaffected: a 2K
   request is a fortieth of the cache, so allocation variance is irrelevant
   there.
2. **n = 25.** The 32K bucket uses 25 prompts against 2K's 50. A long-prompt
   sample of 25 is a small sample of a heavy-tailed quantity.
3. **Node-to-node variation.** Not ruled out; the two 32K jobs ran three days
   apart and the node is not recorded in a way we have checked.

Candidate 1 is testable for free from logs already on disk, and should be
checked before any repeats are run, because if preemption is the cause then
the fix is a smaller `max_model_len` or fewer concurrent sequences rather
than more repeats:

```bash
grep -i "preempt\|recompute\|cache full" logs/phase0-22713173.out | head
grep -i "GPU KV cache size" logs/kprobe32k-22757472.out
```

## Consequences for the project

**The "run-to-run noise under 1 percent" line this log has been carrying
applies to 2K only**, and has been corrected wherever it appears. It was
derived from 2K data and generalized without justification — the same error
as assuming an intercept.

**Repeat counts must be set per bucket.** With sigma near 1 percent at 2K,
N=3 gives a t-interval half-width around 2.5 percent, which is plenty against
a 30 percent effect. With sigma near 9 percent at 32K, N=3 gives roughly 22
percent — wider than some of the effects we want to report. The long-context
bucket needs more repeats than the short one, which is the opposite of how
the GPU budget would prefer it, since 32K cells are also the expensive ones.

**What still has not been measured** is the noise on *speedup* at 32K, as
opposed to on the baseline. Speedup is a within-job ratio and should be far
more stable than the 8.8 percent above, but "should be" is what this note
exists to warn against. It requires repeats of a full cell — `off` plus both
speculative rungs in one job, three times — which is about 1.3 GPU-hours and
is now the single item blocking every 32K claim.

## The pattern, again

The dead-ends table in `LOG.md` notes that most errors here came from
trusting a proxy instead of testing the thing. This is the same shape in a
new place: a variance estimate measured at one operating point and assumed to
hold at another, where the physics (cache pressure) is entirely different.
The fix was free and took one arm of one job, because somebody wrote the
baseline re-measurement into the script on purpose.
