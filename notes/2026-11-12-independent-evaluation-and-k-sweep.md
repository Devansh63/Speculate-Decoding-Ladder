# 12 Nov 2026: an outside read of "is there a paper", and the k sweep that answers the TA

No jobs run, no GPU hours spent. Two things are recorded here.

1. An adversarial evaluation of the project, done on 11 Nov from the project
   brief alone and then checked against primary sources. Its verdict, what it
   found wrong in the brief, and where the repository already knew better.
2. The scripts for the sweep the TA asked for: every rung at every draft
   length, per bucket and per LongBench-v2 domain, on matched code paths.

Read with `notes/2026-11-02-NEW-FINDINGS-adaptive-speculation.md` and
`notes/2026-11-02-verification-of-new-findings.md`. Nothing here replaces
them.

Verification status of every external statement below: the arXiv abstract
page and the arXiv HTML full text were fetched on 11 Nov and read through a
summarising fetch tool, not by eye. That is weaker than the 2 Nov rule for
`paper/related-work.md` ("nothing enters unless its full text was opened"),
so **nothing from this note has been added to `paper/related-work.md`**. Open
the sources before any of it is cited.

---

## 1. Verdict

**As the work stands it is not a paper at any tier.** It is a strong course
project. The headline (the 32K inversion) rests on a stock drafter run far
outside its training length, one k per arm, no intervals, and comparisons
that cross scheduling modes. With the controls already queued it becomes a
credible workshop measurement note. The router should stop being the claimed
contribution.

| bar | answer |
|---|---|
| publication, today | none |
| publication, after the queued controls | workshop measurement note |
| course project | comfortably met |

The narrowest claim that still looks defensible: in vLLM on one 8B target, a
weight-free lookup drafter overtakes a stock DFlash head between 8K and 32K
prompt tokens, the flip decomposes into an acceptance channel and a cost
channel, and the cost channel does not transfer across GPUs by one scalar.
Per-request routing, the off action, selection across kinds of drafter, the
oracle, and long-context collapse are each taken by prior work already in
`paper/related-work.md`.

---

## 2. What the check found that the brief did not say

| brief | source |
|---|---|
| A long-context fine-tune "leaves the drafting-cost channel untouched" | Weaker than stated. The current `z-lab/Qwen3.5-27B-DFlash` model card describes a Z-Lab and Modal retrain at 40k sequence length **with sliding-window attention**. vLLM PR #61261 (open, 12 Oct 2026) adds `draft_context_window` to bound DFlash draft attention. Neither reports drafting cost against context, and the PR's window is 32K, so neither refutes our 32K number. Both show the cost channel is being worked on upstream. |
| RFC #48627 is about context-keyed draft length | Incomplete. The RFC itself reports speculation against no speculation at 0.79x to 1.10x across context 400 to 4000, crossing break-even near 1900, in vLLM (Gemma-4-31B, MTP drafter, H100). A trained drafter crossing break-even along the context axis inside vLLM is already written down. |
| DFlash Table 4 | Confirmed exactly. Base Qwen3.5-27B drafter trained at 3,072 tokens; after 1.6K LongAlign samples for 3 epochs, gov_report at 32K goes 2.09 to 3.56 and hotpotqa at 16K goes 3.61 to 6.05. The 32K row exists for gov_report only. |
| WhiFlash | Confirmed: Qwen3-8B, 3.87x against DFlash 3.55x, Transformers at batch 1 on H100, no off action, vLLM integration listed as future work, EMNLP 2026. Not confirmed: that its entropy router "matches its per-prompt oracle exactly". The fetched text has the neural router exceeding Oracle-Prompt on all three targets. |
| Berkeley "Performance or Illusion" says every variant beat no speculation | That is its introduction. Its body reports a wide tree below 1x at batch 64. Its 4.9x appears only in the introduction; its combined oracle picks per position, not per request. MLSys 2026 oral confirmed. |

Not checked at all: BanditSpec's drafter pool and the no-speculation arm in
Context-Aware Assistant Selection (their abstract pages do not say; the repo's
own full reads stand), SAM-Decoding, ReSpec, TAPS, "To Copy or Not to Copy",
ArcticInference, NeuSight, SmartSpec, Sequoia. One new lead, found and not
read: **MLSYSIM (arXiv:2607.02558)** models speculative decoding as a
compute-bound draft phase against a memory-bound verify phase. Read it before
the paper says nobody predicts across hardware.

---

## 3. Where the repository was already ahead of the brief

The evaluation was done cold, so three of its concerns were already answered
here. Recorded so nobody re-derives them.

- **Prefix caching (issue #47930).** The evaluation said exposure depends on
  whether prompts share a prefix within one server lifetime. The 2 Nov
  response already measured it: 1 prompt in 50 at 2K and 8K, none at 32K. The
  issue page adds one fact: a commenter reports the long-context drop
  persists with prefix caching off, on a different drafter, past 128K.
- **Different code paths.** The evaluation flagged the runner split. The repo
  had the sharper version: the runner is pinned, but async scheduling is not,
  and every filed n-gram ratio crosses scheduling modes. Section 5 below
  closes it in the harness.
- **Why drafting cost grows with context.** The evaluation said the result
  needs a mechanism. C9's falsifier is already registered.

---

## 4. Two numbers recomputed from the filed rows

Both are arithmetic on `results/a40/results.jsonl`, jobs 22598608, 22685911,
22713173 (the headline cells) and 22737716, 22737717, 22714602 (the k
sweeps). No GPU. Reproduce with:

```bash
python3 code/ksweep_table.py results/a40 --assume-pin-defaults --exclude-job 22569403
```

**4a. The oracle gain is larger than the brief's 6.3% once it is weighted by
time.** The brief averaged speedups across buckets. Throughput over a mix of
requests is total tokens over total time, and a 32K request takes about twice
as long as a 2K one, so the bucket where the fixed choice is wrong carries
more weight. Seconds per request at concurrency 1, 256 tokens, prefill
included:

| arm | 2K | 8K | 32K | sum | against off |
|---|---|---|---|---|---|
| off | 7.86 | 9.27 | 17.18 | 34.31 | 1.000x |
| n-gram k=5 | 6.80 | 8.12 | 15.93 | 30.85 | 1.112x |
| DFlash k=7 | 4.22 | 7.36 | 20.80 | 32.37 | 1.060x |
| best per bucket (DFlash, DFlash, n-gram) | 4.22 | 7.36 | 15.93 | 27.51 | 1.247x |
| DFlash gated off at 32K | 4.22 | 7.36 | 17.18 | 28.76 | 1.193x |

Three consequences.

- On a uniform mix the **best fixed rung by time is n-gram, not DFlash**
  (1.112x against 1.060x). The brief's table had DFlash as best fixed at
  1.317x because it averaged ratios.
- Best-per-bucket over the best fixed rung is **12.1%**, not 6.3%.
- The n-gram column crosses scheduling modes (section 3), so the clean
  version uses only off and DFlash: gating DFlash off at 32K is 1.193x
  against fixed DFlash at 1.059x, **12.6%**, with no n-gram in it at all.

These are single realizations with no interval. They are also end to end:
`elapsed_s` includes prefill, which at 32K with 256 output tokens is a large
share of the request. A decode-only split (`MAXTOK=1` subtraction) would
change every figure here.

**4b. In the filed sweeps the best DFlash k does not move.**

| bucket | k measured at concurrency 1 | best | next | gap |
|---|---|---|---|---|
| 2K | 1, 2, 7 | 7 at 1.864x | 2 | 27.0% |
| 8K | 1, 2, 7 | 7 at 1.261x | 2 | 15.1% |
| 32K | 1, 2, 4, 7 | 7 at 0.826x | 4 | 1.7%, a tie, and all below 1.0x |

The best k is the largest k tested in every bucket. That is the outcome the
TA described as "adaptive k does not matter": within what was measured, the
k decision is a constant and the only decision that pays is the gate at 32K.
It is not yet a finding, because k = 11 and 15 (the head's native block) have
never run at concurrency 1, and n-gram has k = 5 only in this repository. The
sweep in section 5 is what settles it.

---

## 5. The k sweep

Four new files under `code/`, all variants. No working original was edited.

| file | what it is |
|---|---|
| `run_benchmark.v4.py` | harness v3 plus `--async-scheduling`, `resolved.async_scheduling` in every row, rung `ngram_gpu`, `--domain`, `--per-request-spec`. Every default reproduces v3. |
| `make_domains.py` | login node. Slices the shared bucket files into one workload per LongBench-v2 domain, paired by document across buckets. |
| `ksweep.sh` | the job. One bucket, one domain, one concurrency; every rung at every k; both baselines. Exits non-zero if any arm produced no row. |
| `submit_ksweep.sh` | login node. Prints every job and a cost floor. Submits only with `GO=1`. |
| `ksweep_table.py` | no GPU. Matched-baseline speedups, best k per cell, and the verdict on whether adapting k buys anything. |
| `push_ksweep.sh` | login node. Copies `results/*_ksweep*` into the clone and pushes. |

**What "domain" means here.** LongBench-v2's own `domain` field, which
`build_workload.py` already keeps on every row. It is the only domain axis
the existing data supports without building a new workload. If the intent
was different datasets (code completion, summarisation, chat), that is a
separate workload build and none of this covers it.

**Arms in one job**, so every ratio is within-job:

```
off                        async at the engine default (on)
off --async-scheduling off the matched baseline for CPU n-gram
ngram      k = 1 2 5 7 11  CPU prompt lookup; the engine runs it sync
ngram_gpu  k = 1 2 5 7 11  GPU prompt lookup; allowed to run async
dflash     k = 1 2 4 7 11 15
```

**How to run it**, from your clone on Delta:

```bash
cd /projects/bikd/$USER/Speculative_Ladder/repo/code
python3 make_domains.py --check          # read the per-domain counts first
python3 make_domains.py
bash submit_ksweep.sh                    # three all-domain jobs, printed, not submitted
bash submit_ksweep.sh DOMAINS=each       # one job per domain per bucket, printed
bash submit_ksweep.sh GO=1               # submit
python3 ksweep_table.py /projects/bikd/$USER/Speculative_Ladder/results/a40_ksweep*
```

**Cost, stated before anything runs (RULES section 7).** The submitter builds
its figure from the filed A40 off arms (7.9, 9.3 and 17.2 seconds per prompt)
plus an assumed 3 minutes of load per arm. The default all-domain grid is 18
arms per job, three jobs: floor 9.2 GPU hours wall, 4.6 charged on A40 at
factor 0.5. Per-arm extrapolation has run 2x low three times here, so budget
about 9 charged. That is most of one student's 10 hour cap. The per-domain
grid at the default k lists is roughly four times that across six domains
and does not fit; cut the k lists for it (`K_DFLASH="7 11 15" K_NGRAM="5 7"
RUNGS="ngram_gpu dflash"`) or wait for the answer on the 100 hours.

**What has and has not been tested.** Tested on a laptop against a stub
engine: argument handling, run ids, the domain guard, the async guard, the
row and per-request files, the arm loop, the incomplete-job exit, the
submitter's plan, and the table on both stub rows and the filed A40 rows.
**Not tested: anything on Delta.** In particular, nobody has yet run
`ngram_gpu` at this pin, passed `async_scheduling=False` to an off arm, or
read `spec_decode_metrics` off a real output. The option names were read from
the vLLM source at `bf13ecc2f` (`vllm/engine/arg_utils.py` lines 679 and 764,
`vllm/config/speculative.py` line 66, `vllm/outputs.py` line 69), not from a
run. Run one 2K job with `LIMIT=5` before the grid.

Known gap: `check_lossless.v3.py` matches the rung in a per-request file name
with `[a-z0-9]+`, so it will not pair `ngram_gpu` files, and the `_d<domain>`
suffix is untested against it. It needs a variant before a lossless check on
sweep output.

---

## 6. What the evaluation says to do with five weeks

Keep: the two async arms (now inside the sweep), the prefix-caching-off
control (`PC=off` on the sweep), the k sweep per bucket, disjoint-subset
intervals, two or three cells at concurrency above 1.

Cut: the router implementation, mixed-batch experiments, the DeltaAI port,
and the switch to a 27B target. The 27B switch discards every filed number
and needs an aarch64 rebuild, since 27B in bf16 fits neither Delta GPU.

The one experiment that would most change the verdict: fine-tune the
Qwen3-8B DFlash head on about 1.6K LongAlign samples, as in DFlash section
5.4, and rerun 8K and 32K. Applying Table 4's recovery to our own tau (2.55
at 2K, 1.45 at 32K, against a step cost of about 1.37 and 1.74) puts DFlash
at 1.15x to 1.42x at 32K, above n-gram's 1.08x. That estimate crosses a model
and a benchmark, so it is a warning and not a result. If the inversion
survives the fine-tune, the cost channel alone carries it and that is a real
finding. If it does not, the 32K result is about a drafter's training length.
Whether DFlash training code is public was not checked.

**Hypothesis, not registered, on the hardware miss.** A100 over A40 speedup
ratios in the brief are 0.96 / 0.88 / 0.77 for DFlash and 0.91 / 0.95 / 0.90
for n-gram. That pattern is what you get if part of the drafting cost does
not scale with memory bandwidth while the target step does. If so, a two-term
cost (one scaled by the machine, one fixed) fits where one scalar could not.
It is written here as an idea to test, not as a prediction: it was formed
after seeing the A100 numbers, so it cannot be registered under RULES section
11 for those cells.
