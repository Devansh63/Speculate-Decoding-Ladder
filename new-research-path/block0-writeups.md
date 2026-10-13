# Block 0 write-ups, 13 Oct 2026

Three append-ready sections and one correction. Each is scoped to what the
records in `results/a40`, `results/a40_ngramk` and `results/a100` actually
support, with job ids on every figure.

Order of operations:
1. Append **§A** to `paper/claims.md` (append-only register).
2. Append **§B** to `paper/claims.md` immediately after §A.
3. Replace the magnitude paragraph of C19 in `notes/2026-10-11-memory-tax-and-remaining-axes.md` with **§C**, and append §C's correction note to `claims.md`.
4. Add **§D** to `paper/threats.md`.

---

## §A — Frozen prediction scored (append to claims.md)

### Frozen prediction scored, 13 Oct 2026

Scored from records. Measured values are the mean over all realizations of each
cell at concurrency 1 in `results/a100` (harness `v2-2026-10-05`), paired to an
`off` baseline from the same job.

| cell | predicted | measured | rel. error | implied R | machine factor | Δτ_eff vs A40 |
|---|---|---|---|---|---|---|
| 8K n-gram k=5 | 1.09 | 1.0788 | −1.03% | 1.2022 | 1.0734 | +1.01% |
| 8K DFlash k=7 | 1.20 | 1.1054 | −7.89% | 1.7664 | 1.1440 | +0.28% |
| 32K n-gram k=5 | 1.02 | 0.9745 | −4.46% | 1.3804 | 1.0700 | −3.23% |
| 32K DFlash k=7 | 0.78 | 0.6432 | −17.53% | 2.2894 | 1.2963 | +0.93% |

Realizations (every cell ran twice, on different nodes):

| cell | job / node | speedup | R |
|---|---|---|---|
| 8K n-gram k=5 | 22802261 / gpua008 | 1.0780 | 1.2031 |
| 8K n-gram k=5 | 22802308 / gpua070 | 1.0796 | 1.2013 |
| 8K DFlash k=7 | 22802261 / gpua008 | 1.1066 | 1.7644 |
| 8K DFlash k=7 | 22802308 / gpua070 | 1.1041 | 1.7684 |
| 32K n-gram k=5 | 22802262 / gpua008 | 0.9742 | 1.3807 |
| 32K n-gram k=5 | 22802309 / gpua087 | 0.9747 | 1.3800 |
| 32K DFlash k=7 | 22802262 / gpua008 | 0.6437 | 2.2877 |
| 32K DFlash k=7 | 22802309 / gpua087 | 0.6428 | 2.2910 |

Cross-node spread on the ratios is 0.05%–0.23%, so every miss below is one to
two orders of magnitude larger than the timing noise.

**Cells within the 5% tolerance: 2 of 4. Median absolute relative error: 6.18%.**

**Verdict: does not pass. Formally ambiguous on failure.**

- The pass limb required *every* cell within 5%. Two exceeded it. Not a pass.
- The fail limb required median absolute error above 10% — it is 6.18%, so not
  by that limb — **or** "the 32K ordering does not reproduce".
- That phrase admits two readings and they disagree. **Relative order** (DFlash
  below n-gram) reproduced: 0.6432 against 0.9745. **Break-even side** (DFlash
  below 1.0 *and* n-gram above 1.0) failed: n-gram landed at 0.9745.

Recorded as **not passed, fail-leaning**, with the ambiguity named rather than
resolved. External review correction #3 closed one pass/fail gap on 11 Oct and
left this one. The wording must be fixed before the next prediction is frozen,
not after it is scored.

**The pattern splits by rung, not by bucket.** n-gram 2/2 within tolerance
(−1.03%, −4.46%); DFlash 0/2 (−7.89%, −17.53%). All four errors are negative:
every prediction was optimistic. That is bias, not noise, and it is the
rung-dependent machine factor appearing in the scoreboard.

**What the failure teaches, against the pre-registered diagnostic list.** The
11 Oct entry named three candidate explanations in order of testability:
context length (pointing at `R_draft`), coverage (a factor differing between
rungs at long context but not short), and memory capacity. The second is what
happened, and C20 below measures the first as its mechanism. The capacity confound is
dead on records: the engine logs report KV cache usage of 7.9% at 8K and 30.7%
at 32K at concurrency 1, nowhere near the ceiling.

**One parameter caveat that limits this entire scoreboard.** The machine factor
1.054 was fitted at 2K on job 22757104, which ran `LIMIT=5`. The prediction's
single parameter was estimated from five prompts and then applied to cells of
n=50 and n=25. A100 2K at n=50 is therefore not a gap in a table — it is the
only direct test of whether the parameter was ever estimated at all.

---

## §B — C20: the machine factor is rung-independent at short context and diverges with it (append to claims.md)

### C20, 13 Oct 2026 — the machine factor is not a machine property

The 11 Oct prediction assumed a single scalar machine factor relating A40 `R`
to A100 `R`. Measured per cell from records:

| bucket | n-gram k=5 | DFlash k=7 |
|---|---|---|
| 2K | 1.0456 | 1.0556 |
| 8K | 1.0715 | 1.1397 |
| 32K | 1.0703 | 1.2964 |

**At 2K the two rungs are indistinguishable (1.046 against 1.056). They diverge
monotonically with context.** The factor fitted at 2K looked like a machine
constant because 2K is the one bucket where the rungs agree.

This is the pre-registered **branch B** signature exactly as written — "a factor
that differs between rungs at long context but not short would mean the machine
and drafter terms are not separable." The terms do not separate.

#### The mechanism, measured

Fitting `R(k) = (1 + R_draft) + c·k` over the k-sweeps on A40 at concurrency 1:

| rung | bucket | k values | R_draft | c | R² | source job |
|---|---|---|---|---|---|---|
| DFlash | 2K | 1, 2, 7 | **+0.2988** | +0.0095 | 0.982 | 22737716 / 22598608 |
| DFlash | 8K | 1, 2, 7 | **+0.4447** | +0.0152 | 0.967 | 22737717 |
| DFlash | 32K | 1, 2, 4, 7 | **+0.6734** | +0.0138 | 0.944 | 22714602 |
| n-gram | 2K | 1, 2, 7 | **+0.0714** | +0.0030 | 1.000 | 22757473 (gpub091) |
| n-gram | 32K | 1, 2, 7 | **+0.1392** | +0.0069 | 0.799 | 22757472 (gpub073) |

Three things follow.

**1. The fixed drafting cost dominates the marginal verification cost.** At 32K,
k=7: `R_draft` = 0.673 against `c·k` = 0.097. The fixed term is seven times the
marginal one. At 2K it is four and a half times.

**2. Both terms grow with context, but the fixed term grows faster.** DFlash
`R_draft` +125% from 2K to 32K against `c` +45%; n-gram `R_draft` +95% against
`c` +130% (on a much smaller base). The growth of `R` with context is therefore
mostly in the per-step drafting cost, not in the cost of verifying extra
positions.

**3. This narrows the mechanism and rules one out.** LongSpec attributes
long-context speculative inefficiency to tree-attention verification falling
off optimised kernel paths for arbitrary attention masks. If that were the
dominant effect here, it would appear in `c`, which is the per-verified-position
term. It does not: `c` stays between 0.003 and 0.015 across a 16× context
change. The dominant effect is the drafter's own forward pass over a growing
prefix. The kernel hypothesis is not refuted, but it is second-order in this
data.

**n-gram's `R_draft` is not zero (0.071 at 2K, 0.139 at 32K), and that is not a
GPU cost.** The n-gram rung runs no model. What grows is prefix matching over a
longer prompt — CPU-side work charged to every decode step. So "the drafter is
free" is false even for the model-free rung, and the reason it still tracks the
machine factor closely is that its drafting cost is 4–9× smaller than DFlash's.

#### Status and the one point off the curve

Plotting the machine factor against `R_draft` gives a monotone relation in four
of five points:

| R_draft | machine factor | cell |
|---|---|---|
| 0.071 | 1.0456 | n-gram 2K |
| 0.139 | 1.0703 | n-gram 32K |
| 0.299 | **1.0556** | DFlash 2K |
| 0.445 | 1.1397 | DFlash 8K |
| 0.673 | 1.2964 | DFlash 32K |

The exception is DFlash 2K, which sits below the n-gram 32K point despite twice
the drafting cost. **Its A100 side is job 22757104 at `LIMIT=5`** — the same
five-prompt cell that supplied the frozen prediction's only parameter. The one
point that breaks the collapse is the one with five prompts.

**C20 is therefore recorded as supported but not established**, pending A100 2K
at n=50. If the collapse holds with that point replaced, the claim is that
`R_draft` relative to target cost is the governing variable and context length
is its proxy. If it does not hold, the two-channel decomposition still stands
and the collapse does not.

#### Scope

Concurrency 1 only. Two drafter families from two of roughly seven. bf16. One
target model. The n-gram k-sweep rows come from `gpub073` and `gpub091`;
`gpub073` is on the project's exclusion list, so n-gram's `R_draft` carries that
caveat and should be re-measured on a clean node before it appears in a figure.

---

## §C — C19 correction: the figure stands, the pooling was wrong

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

---

## §D — Threats to add (append to paper/threats.md)

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
