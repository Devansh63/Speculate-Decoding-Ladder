# C19 corrected: the drafter's weights are charged to the KV pool

Written 2026-10-13. Supersedes the magnitude paragraph of
`notes/2026-10-11-memory-tax-and-remaining-axes.md`, which is left in place unedited so the original
reasoning stays readable.

### C19 corrected, 13 Oct 2026

C19 as filed reported that DFlash costs ~2.4 GiB of KV pool and reduces the
33,792-token concurrency ceiling, and flagged an unexplained within-rung drift
of 1.6–1.9 GiB that made the magnitude doubtful at three significant figures.

An independent re-derivation from the engine logs first appeared to refute the
magnitude: pooling all 52 arms across 11 jobs gives a within-rung spread of
6.1 GiB against a between-rung difference of 2.6 GiB, i.e. not separable.
**That pooled comparison is invalid.** It mixes A40 (48 GB) with A100 (40 GB)
logs, so most of the "spread" is the difference between two cards.

Paired **within job** — same machine, same node, same engine build, same
profiling pass — across all 17 available (job, bucket, concurrency) triples
spanning 11 jobs, both GPUs, three buckets and concurrency 1 to 32:

| concurrency | n pairs | DFlash cost (GiB) | n-gram cost (GiB) | ceiling loss |
|---|---|---|---|---|
| 1 | 9 | **+2.42 to +2.47** | −0.01 to +0.00 | −21.3% to −24.2% |
| 2–12 | 4 | **+2.38 to +2.54** | +0.03 to +0.09 | −21.5% to −22.4% |
| 32 | 1 | **+4.01** | +0.28 | **−28.1%** |
| (no suffix, 30 Sep build) | 1 | +2.77 | +0.03 | −22.9% |

Representative concurrency-1 rows:

| job | GPU | bucket | off | n-gram | DFlash | cost | ceiling |
|---|---|---|---|---|---|---|---|
| 22598608 | A40 | 2K | 22.10 | 22.11 | 19.68 | +2.42 | 4.76 → 3.72 |
| 22685911 | A40 | 8K | 23.77 | 23.77 | 21.30 | +2.47 | 5.12 → 4.03 |
| 22713173 | A40 | 32K | 23.77 | 23.77 | 21.30 | +2.47 | 5.12 → 4.03 |
| 22757104 | A100 | 2K | 17.63 | 17.64 | 15.21 | +2.42 | 3.80 → 2.88 |
| 22802261/308 | A100 | 8K | 19.30 | 19.30 | 16.83 | +2.47 | 4.16 → 3.18 |
| 22802262/309 | A100 | 32K | 19.30 | 19.30 | 16.83 | +2.47 | 4.16 → 3.18 |

**The cost is stable at roughly 2.4–2.5 GiB for all concurrencies up to 12, on
both machines and at every bucket.** n-gram costs nothing measurable.

**The drift that motivated the original doubt cancels under pairing.** Job
22713173 at concurrency 1, 2 and 4: `off` 23.77 / 22.03 / 23.75, DFlash 21.30 /
19.65 / 21.24. Paired differences 2.47, 2.38, 2.51. vLLM profiles peak memory
with the configured `max_num_seqs`, so the shift is common-mode across arms in a
job and divides out.

#### New: the cost is itself concurrency-dependent above ~c=12

The single c=32 pair (job 22598608, 2K) shows DFlash costing **+4.01 GiB**, not
2.4 — a 63% increase over the same job's c=1 pair, and a ceiling loss of 28.1%
against 21.8%. n-gram also rises, from −0.01 to +0.28.

One observation, so this is a signal to test and not a result. But the direction
is what the mechanism predicts: a trained head maintains its own KV, which
scales with the number of resident sequences, so its memory charge should grow
with concurrency while its *weight* charge stays fixed. At c=1 the weights
dominate and the cost looks constant; at c=32 the head's own KV becomes visible.

**Consequence for the concurrency sweep.** Admissible concurrency cannot be
computed as (pool ÷ per-sequence KV) using the c=1 pool, because the pool itself
shrinks as concurrency rises in the speculative arms. Every planned high-batch
estimate is therefore an upper bound — including the H200 figure of roughly 22
concurrent 32K sequences, which assumed a fixed 18.3 GiB of weights. The sweep
must read achieved concurrency and the arm's actual pool from its own log rather
than projecting from c=1. Adding a c=16 and c=32 pair at 8K would turn this one
observation into a trend for the cost of two short jobs.

**Corrected framing.** Not a tax and not a third cost axis: **the drafter's
weights are charged to the KV pool.** vLLM profiles memory with both models
resident and gives the remainder to KV, so a second checkpoint comes straight
out of the pool. n-gram has no checkpoint and costs nothing, which is the
control that confirms the interpretation.

**What is publishable** is the consequence, not the GiB figure: a trained head
buys acceptance by surrendering roughly a fifth of the long-context concurrency
ceiling, and the loss is worse on the smaller card (−24% on the 40 GB A100
against −21% on the 48 GB A40) because the cost is absolute. The ceiling is a
property of the **rung**, not of machine-and-bucket alone.

Two anomalies left open rather than explained, and **excluded** from the figures
above:

* Job **22757105**'s n-gram arm reports 23.77 GiB against its own `off` arm's
  22.10, a cost of −1.67 GiB and a ceiling *gain* of 7.6% — the only negative in
  the set. 23.77 is exactly the pool the A40 8K/32K jobs report, so the arm
  looks mis-configured rather than mis-measured. It is the sole reason n-gram's
  raw mean across all 17 pairs reads −0.07 instead of about +0.03.
* The A40 2K jobs report 22.10–22.74 GiB where the A40 8K/32K jobs report 23.77
  at identical `max_model_len` of 33,792. Most likely an engine-build
  difference between the 30 Sep and 5 Oct runs, but unverified.

Neither affects the paired DFlash figures, because pairing is within job. Both
must be chased before any *absolute* GiB number is quoted — only the paired
differences are safe to publish.
