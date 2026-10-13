# Block 2 write-ups, 13 Oct 2026

Four sections. §E is the day's most consequential result and it reframes C10
rather than amending it.

Routing (handled by `b2_03_apply.sh`):
1. §E and §F append to `paper/claims.md`.
2. §G appends to `paper/claims.md` as a threat retirement.
3. §H appends to `paper/threats.md`.

---

## §E — C21: the engine is bit-deterministic at fixed configuration, which refutes C10's strong form

### C21, 13 Oct 2026 — there is no non-determinism floor at fixed concurrency

Jobs **22875382** (gpub015) and **22876140** (gpub036), A40, harness
`v3-2026-10-13`, 2K bucket, `off` rung, n=50, concurrency 1, prefix caching on
(read back off the engine, not assumed). Three realizations across two nodes,
written into one directory because the v3 per-request file name can no longer
overwrite itself.

| pair | identical | per-token hazard |
|---|---|---|
| 22875382 vs 22876140 | 50/50 (100%) | **0.000%** |
| 22875382 vs 22876140_r2 | 50/50 (100%) | **0.000%** |
| 22876140 vs 22876140_r2 | 50/50 (100%) | **0.000%** |

**150 of 150 sequences bit-identical, including across two different nodes.**

Zero events is not proof of zero rate. One pair covers 50 × 256 = 12,800 token
positions, so by the rule of three the true hazard is bounded above by roughly
**0.023%** per token at 95%. Across the three pairs the bound is tighter but
the pairs are not independent, so 0.023% is the figure to quote.

#### What this does to C10

C10 compared the baseline against itself at **different concurrencies** —
12.7% sequence agreement — and read the disagreement as engine
non-determinism. External review correction #1 (11 Oct) called the comparison
confounded. This measures the confound directly: **at matched concurrency the
engine does not disagree with itself at all.** The entire control effect was
concurrency.

Three consequences.

**1. C10's strong form is refuted, not merely unsupported.** "Speculative
decoding perturbs output less than the serving engine perturbs itself across
batch sizes" compares a speculation effect against a concurrency effect. At
matched concurrency the engine's self-perturbation is below 0.023%/tok, so
there is nothing for speculation to be smaller than.

**2. The speculative hazards already in `results/a40` are attributable to
speculation.** They were computed against a same-job baseline at matching
concurrency, so with the floor established they need no noise subtraction:

| 2K, c=1, A40 | hazard | multiple of the 0.023% bound |
|---|---|---|
| `off` vs `off`, same config | **0.000%** | — |
| n-gram k=5 | 0.556%/tok | ~24× |
| DFlash k=1 | 0.738%/tok | ~32× |
| DFlash k=2 | 0.781%/tok | ~34× |
| DFlash k=7 | 0.781%/tok | ~34× |

**3. Speculative decoding on this build is not bitwise lossless.** Under
greedy decoding a correct implementation accepts exactly the tokens plain
decoding would produce, provided the logits match bitwise. They do not:
verifying k+1 positions runs different kernel shapes from decoding one, so
near-ties resolve differently. The divergence is real, measurable, and
systematic in k for DFlash (0.738 → 0.781 from k=1 to k=2). "Lossless" holds
at the task level (C22 below) and fails at the bit level.

#### Scope, and the control that is still missing

Measured at 2K, concurrency 1, rung `off`, A40, prefix caching on. Determinism
at c=1 is expected to hold at 8K and 32K because nothing batches, but it is
**unverified**. Prefix caching being enabled is plausibly *why* it holds —
identical prompts in identical order produce identical cache hits — and the
variation of scheduling order at c>1 is the likely mechanism behind the whole
cross-concurrency effect. That is a hypothesis, not a measurement.

**The remaining gap, stated plainly: only the BASELINE has been shown
reproducible.** Two runs of `ngram` or `dflash` at one configuration have never
been compared. If a speculative arm is itself unstable run-to-run, some part of
its 0.556–0.781% divergence from the baseline is its own noise rather than a
systematic offset. Closing it costs one job:
`ARMS="ngram:5 dflash:7" REPEAT=2` at 2K c=1, about 30 minutes. **Until that
runs, C21 supports "the engine is deterministic" and "speculation diverges from
the baseline", but not yet "speculation diverges systematically".**

#### Also measured

The same four runs give a **timing** floor: 32.5698 and 32.5696 tok/s on
gpub015, 32.5703 and 32.5644 on gpub036 — a total spread of **0.018%** across
two nodes. Note that the A100 duplicate-pair ratio spreads were 0.05–0.23%,
which is larger, so those were not pure node variance either and something
further is unaccounted for there.

---

## §F — C22: speculation changes tokens without changing answers

### C22, 13 Oct 2026 — accuracy is invariant across rungs

LongBench-v2 is multiple choice, so the stored `token_ids` can be decoded and
the chosen letter scored against the answer key. Scored over `results/a40` with
`code/b2_01_score_accuracy.py`.

Extraction rests mostly on the confident patterns: 84.7% of generations begin
with the letter, a further 7.9% match "answer is X", 3.9% fall to the loosest
pattern and 3.5% are unparseable. **Unparseable generations are counted
separately and never scored as wrong.**

| bucket | n | `off` | n-gram k=5 | DFlash |
|---|---|---|---|---|
| 8K c=1 | 49 | 34.7% | 34.7% | 34.7% (k1, k2, k7) |
| 2K c=1 | 47 | 34.0% | 34.0% | 34.0% (k1), 36.2% (k2, k7) |
| 32K c=1 | 24 | 50.0% | 50.0% | 50.0% (k1, k7), 54.2% (k2, k4) |

At 8K, **nine arms score identically**. Every difference anywhere else is
exactly one prompt: 16 against 17 of 47, or 12 against 13 of 24.

**And the baseline differs from itself by the same one prompt**: `off` at 32K
reads 50.0% at c=1 and 54.2% at c=4. So no rung deviates by more than the
baseline's own variation across concurrency.

Per domain, grouped by (bucket, n) because only arms covering the same
documents are comparable:

| domain, bucket | n | spread across all arms |
|---|---|---|
| Code Repository Understanding, 2K | 3 | **0** (all 66.7%) |
| Code Repository Understanding, 8K | 3 | **0** (all 66.7%) |
| Code Repository Understanding, 32K | 2 | **0** (all 100.0%) |
| Long In-context Learning, 8K | 8 | **0** (all 37.5%) |
| Long In-context Learning, 2K | 8 | 1 prompt: `off`/n-gram/k1 at 50.0%, k2/k7 at 62.5% |

Four of five domain-bucket cells are perfectly invariant. The fifth differs by
one prompt out of eight, in DFlash only, with n-gram matching the baseline
exactly.

**Taken with C21 this is the paper's losslessness statement, and it is in two
halves that point opposite ways.** Speculation changes tokens at 24–34 times
the determinism floor, and does not change the answer on 503 graded questions.
The metric the C10 argument was conducted in — sequence identity — turns out
not to predict task outcome at all.

Counts are small. One flip moves 32K accuracy by 4.2 points, so a larger-n
cell is needed before the invariance is quoted with an interval. The claim as
recorded is that no difference exceeds the baseline's own variation, which is
what the data supports.

---

## §G — Threat retired: every arm reached its requested concurrency

### Concurrency labels verified, 13 Oct 2026

`phase0.sh`'s header warns: "vLLM accepts `--max-num-seqs 32` and then
schedules fewer sequences than asked, so the row gets labelled with a
concurrency it never ran at. That is worse than a crash." Checked against
every engine log with `code/b2_02_achieved_concurrency.py`, which reads
`Running: N reqs` and compares the peak to the requested value.

**54 arms parsed. Every arm with an explicit concurrency reached it at least
once.** c=32 at 2K reached 32, c=12 at 8K reached 12, c=4 at 32K reached 4. The
warning was prudent and never fired, because no bucket was ever asked for more
than it could hold — `phase0.sh`'s own capacity table (2K ~70, 8K ~19, 32K ~4.9
on an A40) was respected.

An earlier claim in this conversation that "a c=32 row cannot have run at 32"
was **wrong**: it applied the 32K per-sequence KV arithmetic to a 2K cell.
Recorded because the error is the kind that propagates.

Two things the data does say:

**Sustained concurrency is well below peak.** c=32 at 2K: peak 32, mean
**23.5**. c=4 at 32K: peak 4, mean 3.64. c=12 at 8K: peak 12, mean 11.33. A
figure with concurrency on an axis should use the mean, not the label.

**The 30 September job ran at up to 77 concurrent, unlabelled.** Job
**22569403**'s arms carry no `_c` in their run_id, so they took the engine
default: `off` peaked at 77, n-gram at 77, DFlash at 56. Those rows are not
comparable to anything labelled `c1`, which quantifies what had been described
only as "the uncontrolled run". **`--exclude-job 22569403` belongs in every
analysis that mixes concurrencies.**

---

## §H — Threats and notes to add (append to paper/threats.md)

**12. Only the baseline has been shown reproducible.** C21 establishes that two
`off` runs at one configuration are bit-identical. No speculative arm has been
run twice at one configuration, so the possibility that a speculative arm is
itself unstable run-to-run is open, and with it the possibility that part of
its measured divergence from the baseline is its own noise. One job closes it:
`ARMS="ngram:5 dflash:7" REPEAT=2` at 2K c=1.

**13. The determinism result is scoped to 2K, c=1, A40, prefix caching on.**
Determinism at c=1 should hold at longer buckets because nothing batches, and
caching-on is plausibly the reason it holds at all, but neither is measured.
Any claim that the engine is deterministic *in general* overreaches this data.

**14. `check_lossless.py`'s verdict block is stale.** The control and main
sections were updated on 13 Oct to separate same-config from cross-concurrency
pairs and to report the per-token hazard, but the closing VERDICT paragraph was
not. On the C10 control directory it printed "Both rates are similar, which is
consistent with a common cause in the engine" while comparing 100% against
`nan%`. It assumes the pre-C21 framing and must be rewritten to compare against
the same-config floor. Until then, ignore that paragraph.

**15. Per-cell cost estimates in the plan are too low.** Each 2K n=50 `off` arm
took 393s of generation, and two arms plus two model loads came to 17:54 of
wall clock — so roughly 9 minutes per arm on an A40, not the ~6 estimated from
A100 timings. A three-arm 8K job is 30–45 minutes. Block 3 needs rebudgeting
from A40 measurements rather than A100 ones.

**16. Accuracy differs by bucket in a direction difficulty cannot explain.**
32K scores 50.0% where 8K scores 34.7% on the same model and prompt style. A
longer context making the task easier is implausible; this is the
bucket-document-overlap confound (threat 3) appearing in task accuracy, and it
strengthens the case for per-domain subset runs since domain composition
differs by bucket.
