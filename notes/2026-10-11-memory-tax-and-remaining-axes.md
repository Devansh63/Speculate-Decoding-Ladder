# The memory tax, and the axes the paper still needs

Written 11 Oct 2026. Two sections: a **result that was already sitting in the
committed logs**, and an inventory of what else speculative-decoding
performance depends on, what is measurable today, and what each experiment
would buy. Nothing here was run; the first section is extracted from
`results/logs/*.out.gz` and `results/a40/results.jsonl`.

---

## 1. Result: a rung's memory footprint is a third cost axis

Every vLLM arm prints its KV budget at startup, and the logs are committed.
Extracted per arm, in the order the arms ran:

| machine / job | bucket | arm | KV memory | KV tokens | max concurrency at 33,792 |
|---|---|---|---|---|---|
| A40, 22598608 | 2K | off | 22.10 GiB | 160,912 | **4.76x** |
| A40, 22598608 | 2K | n-gram k=5 | 22.11 GiB | 160,960 | 4.76x |
| A40, 22598608 | 2K | DFlash k=7 | **19.68 GiB** | **125,840** | **3.72x** |
| A40, 22598608 | 2K | DFlash k=7 @ c=32 | **17.95 GiB** | 114,752 | **3.40x** |
| A40, 22713173 | 32K | off | 23.77 GiB | 173,056 | **5.12x** |
| A40, 22713173 | 32K | n-gram k=5 | 23.77 GiB | 173,104 | 5.12x |
| A40, 22713173 | 32K | DFlash k=7 | **21.30 GiB** | **136,160** | **4.03x** |
| A100, 22757104 | 2K (n=5) | off | 17.63 GiB | 128,368 | **3.80x** |
| A100, 22757104 | 2K (n=5) | n-gram k=5 | 17.64 GiB | 128,416 | 3.80x |
| A100, 22757104 | 2K (n=5) | DFlash k=7 | **15.21 GiB** | **97,264** | **2.88x** |

Four findings, all from data already on disk:

1. **DFlash costs about 2.4 GiB of KV budget**, the same absolute amount on
   both machines (22.10 → 19.68 on A40; 17.63 → 15.21 on A100). It is its
   weights plus its own KV, and it buys nothing back.
2. **n-gram costs nothing** — 22.11 against 22.10 GiB. The rung already known
   to be cheap per step is also free in memory. That asymmetry is the ladder's
   whole shape, and it is now measured rather than assumed.
3. **The consequence is a lost concurrency ceiling, and it is
   rung-dependent.** At 32K, the A100 baseline admits 3.80 concurrent
   requests and the DFlash rung admits **2.88** — a 24% loss. On A40 the same
   comparison is 5.12 against 4.03. So the ceiling is not a property of
   machine-plus-bucket; it is a property of *the rung*. A grid of `1 2 4` is
   honest for `off` on A40 and dishonest for DFlash on A100, and the row would
   be labelled with a concurrency it never reached.
4. **Because the tax is absolute, it is relatively worse on the smaller
   card** — 11% of the KV pool on A40, 14% on A100 — which is exactly the
   direction that makes a ladder look good on a big accelerator and bad on a
   small one.

**Why this belongs in the paper.** The project's cost model is
`Speedup ≈ [cov·τ_cond + (1−cov)] / R`, and `R` is a *time* ratio. Memory
appears nowhere in it. But at serving load the binding constraint is how many
requests fit, and a rung that occupies memory reduces that number — so a
drafter can lose on throughput twice: once in `R`, and once by shrinking the
batch. The 32K inversion is currently attributed to drafting cost; part of the
gap is this, and it should be separated. It also gives the ladder's
per-request choice a third criterion alongside τ and R: **KV headroom**.

Caveat, stated plainly: the KV budget also drifts *within* a job for reasons
not yet explained — the DFlash arm reads 19.65 GiB at c=2 and 21.24 GiB at c=4
in job 22713173, above the c=1 value. Candidate causes are the profiling run
sizing with `max_num_seqs` and fragmentation. The *between-rung* difference is
consistent across every pair measured; the within-rung drift is not explained
and no number here should be quoted to better than a few percent.

### Doc correction this exposes

`env/gpu-support.md` gives A40 KV capacity as "~182,000 tokens" and reachable
concurrency at 32K as "~5". Measured: **160,912 tokens and 4.76**. The A100 row
says "~131,000" and "~4"; measured **128,368 and 3.80**. Both are close enough
to have gone unnoticed, and both are rounded in the direction that flatters a
grid. Its table should carry the measured values and, below it, the
rung-dependent ceiling from the table above.

---

## 2. Axes other than context length

Performance lives in `[cov·τ_cond + (1−cov)] / R`. Splitting the axes by what
they move:

| axis | moves | measured today | how to get it | cost |
|---|---|---|---|---|
| **Prompt/context length** | `τ` (trained head) and `R_draft`; not `τ` for n-gram | 3 buckets × 3 rungs | done; a 16K bucket locates the break-even | 1 job |
| **Serving load** | `R`, via batch composition, and the crossing | grids at all buckets | `--max-num-seqs` is a flag; the crossing points are interpolated, so denser sampling near 1.0x | 3–5 jobs |
| **Output length** | the whole payoff: speculation earns per *generated* token, and current ratios fold prefill in | one value, 256 | `--max-tokens` is a flag. `--max-tokens 1` measures prefill almost alone; subtract from 256 for decode-only | ~15 min |
| **Draft length k** | `R` linearly, `τ` sublinearly | sweeps at c=1 only | `--k` is a flag; missing: k×load surface, 8K n-gram | 2–4 jobs |
| **Drafter size / kind** | `R_draft` and the *machine scaling factor* | two rungs, one trained head | Qwen3-0.6B/1.7B/4B share the tokenizer (drop-in drafts); EAGLE3 head for Qwen3-8B exists | 3–6 jobs |
| **Task type / domain** | `τ`: code, prose and retrieval have different predictability | not at all | LongBench-v2 carries the category; **needs per-domain runs**, because acceptance is aggregate per arm | subsets are already the tooling |
| **Temperature** | `τ`; the literature's classic dependence | hardcoded 0.0 at `run_benchmark.py:242` | one-line edit + flag; T ∈ {0, 0.7} | 2 jobs |
| **n-gram match window** | `cov` — the coverage frontier | hardcoded `prompt_lookup_max: 5`, `min: 3` | one-line edit + flag; forcing coverage trades acceptance | 2–3 jobs |
| **Prefix caching** | prefill cost, hence `R` at short context | on for every cell | `--prefix-caching`/`--no-prefix-caching` flag exists | 1 job |
| **Prompt style** | prompt construction, hence n-gram's hit rate | `qa` only | `--prompt-style` flag exists, never swept | 1 job |
| **Memory headroom** | admissible concurrency (section 1) | **already in the logs** | extract per arm; no new runs | free |
| **Topology (TP degree)** | `R`, via an all-reduce every step | never | needs 2 GPUs; a separate population, not a grid row | see below |
| **Quality** | whether any of this changes answers | not at all | score stored `token_ids` against LongBench-v2 answers, login node | free |
| **Per-request acceptance** | τ by prompt, by task, by position | not stored | harness edit: counters are aggregate | edit |

Verified facts about the harness, since three of these hinge on them:
`temperature=0.0` and the n-gram lookup window are **hardcoded**; `--max-tokens`
and `--k` are **flags**; and the stored `timing` dict contains **only
`arrival_time`** — in every record, on every machine — so there is no
per-request latency and no per-request acceptance anywhere. Intervals must come
from the prompt dimension, and the prefill/decode split from `--max-tokens 1`.

## 3. Rungs worth adding

| rung | what it is | what it tests | cost to add |
|---|---|---|---|
| **Qwen3-0.6B / 1.7B / 4B** as drafters | same tokenizer and vocab as the target, so losslessness stays well defined | turns `R_draft` into a **dial**: if the machine scaling factor rises monotonically with drafter size, the "fixed per-step overhead" hypothesis becomes a prediction instead of an inference from two points | download; 4B is tight on A100 at 32K |
| **EAGLE3** (`Tengyunw/qwen3_8b_eagle3`, `RedHatAI/...eagle3`, `mobilint/EAGLE3-Qwen3-8B`) | trained head, no retraining needed | holds drafter cost roughly fixed and varies head quality, so it tests **τ**, not `R` — the orthogonal axis to the size sweep | drop-in, third rung |
| Medusa | none published for Qwen3-8B | — | skip |
| DFlash2, DSpark | runner-V2 only, and the harness refuses V2 | would change the code path, voiding comparability | skip for the must-do |

**One trap.** vLLM has an open issue where `SpeculativeConfig` can silently
override an explicitly requested method from `model_type` and path substrings
(#47486) — the same failure shape as the V2-runner fallback that already cost
this project a job: a flag you passed, a method you did not get, a row labelled
wrong. Registering a new rung must **assert the method from the engine's own
startup log**, not from the argument.

## 4. Two GPUs, and Hopper

- **Never tensor-parallel inside the measurement law.** TP adds an all-reduce
  to every decode step, which changes `R` — the quantity being measured. A TP=2
  cell is not comparable with the single-GPU grid, so using it to reach a bigger
  drafter would trade the paper's cleanest property for headroom the drafter
  size sweep does not need.
- **Best use of a second GPU: two independent single-GPU replicas.** It doubles
  cell throughput against the 17 Oct cutoff, and it makes the same-config
  control cheap — two `off` arms, same node, same hour, separate output
  directories, which is exactly what C10's strong form wants.
- **TP as its own labelled axis is a genuinely different and interesting
  question**: does the crossing move under TP? In a datacenter nobody serves an
  8B on one GPU at scale; they serve TP=2/4/8. If `R` is a property of the
  machine *plus the topology*, that is a stronger and more useful statement than
  "machine property", and it is exactly the kind of thing this project is
  equipped to measure. It must be published as a separate population.
- **H200's value** is a third point on the **bandwidth** axis (0.70 / 1.56 /
  4.8 TB/s; the ridge points are 201–215 and cannot discriminate), plus 141 GB
  of slack that no other card has: the 32K × high-concurrency quadrant that the
  A100 physically cannot host (2.88 concurrent DFlash requests, section 1). It
  is not needed for drafter compatibility — bf16 speculation runs on sm_80/86.

## 5. Datacenter shape: co-resident ladder, or replicas plus a router?

Three architectures, with what the measurements say about each:

1. **Co-resident ladder per replica** — every drafter loaded, switch per
   request. Costs the memory tax on every GPU (2.4 GiB per trained head,
   measured), which reduces admissible concurrency by up to 24% at 32K; and
   vLLM cannot switch a drafter mid-flight (one method per engine, #6300
   closed as not planned), so this needs engine work the project does not have.
2. **Specialised replicas + a router** — one drafter per replica, each stock
   vLLM. Buys back the memory tax and lets long-context traffic land where the
   KV fits. Pays a real, measurable cost the proposal does not mention:
   **prefix caching is per instance, and scattering requests across replicas
   fragments the cache**, so hit rate falls and prefill work rises. The
   committed logs already carry per-arm prefix-cache hit rates, so this cost is
   measurable from data on disk.
3. **Hybrid, which is where the physics points**: n-gram is free in memory and
   free per step, so it can be resident everywhere and a request that looks
   unprofitable can always be served; the trained head goes on a subset of
   replicas and is used where the prediction says it pays.

**Routing criteria, and which we can actually compute:**

| signal | available when | do we have it | note |
|---|---|---|---|
| prompt/context length | at admission | measured, bucket-level | the crossing axis today; per-request by length bin tomorrow |
| queue depth / load | at admission + online | measured, 3 levels | the crossing moves with load: 14.4 / 8.9 / 3.7 |
| prompt–prompt overlap (n-gram coverage proxy) | **at admission, from the prompt alone, no model call** | not computed yet; derivable from stored prompts offline | the strongest non-reactive signal, and free to test |
| observed acceptance (reactive) | after a few steps | needs per-request acceptance | the literature's baseline (BanditSpec, Not-a-Bandit); this project's reactive comparison |
| task type / domain | at admission | needs per-domain runs | acceptance is per-arm aggregate, so it needs cells, not re-analysis |
| KV headroom | at admission | measured, section 1 | a rung that costs 2.4 GiB changes which rung is even admissible under load |
| SLO / goodput | — | **not measurable** | per-request latency does not exist on this build; be honest that goodput-at-p99 cannot be reported |

**The free deliverable that makes the router claim real.** An **offline router
simulation** over the measured grid: for every cell we have, compare a
hindsight oracle, a context-and-load rule, a reactive acceptance rule, and each
fixed rung, on the same traffic mix. Zero GPU hours, recomputable as cells land,
and it produces the "would a per-request ladder beat a fixed rung, and by how
much" number the proposal promises — before any controller exists. It also
tells you which routing signal is worth instrumenting.

## 6. What the paper is still missing

- **The records from the A100 cells are not in this repository** — `results/`
  holds `a40/` and `logs/` only. Every branch read and every number quoted from
  those cells is currently unverifiable by a reader, which is the exact failure
  that `RULES.md` section 8 exists to prevent.
- Predicted-against-measured, per cell, with signed errors — the lead figure.
- A quality row: accuracy per rung per bucket beside the agreement rates.
- The memory axis (section 1) folded into the cost discussion.
- The `--max-tokens 1` split, so the headline ratios are decode-only.
- A limitations paragraph owning that every ratio to date is end-to-end wall
  clock, with prefill included.

> **Corrected 2026-10-13.** The magnitude in this note was re-derived by pairing within job; see `notes/2026-10-13-c19-correction.md`. The direction and the ceiling consequence stand; the pooled-across-jobs comparison that appeared to refute them was invalid.
