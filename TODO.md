# TODO

Ordered by when it has to happen, not by how interesting it is.

For what was already done, see `LOG.md`. For what we learned, see
`paper/claims.md`, `paper/threats.md` and the dated notes in `notes/`.

Rewritten 13 Oct 2026. Items struck below are struck because they were done or
superseded, with the reason kept — not deleted, so nobody re-derives them.

---

## RESOLVED: the scope decision

**Option B. The context axis carries the paper.** Unchanged, and strengthened
on 13 Oct: C20 shows the hardware term is not a single scalar either, which
makes the hardware axis a *result* rather than a side quest.

The 32K cell (job 22713173) decided it. At concurrency 1 the ranking of the
rungs inverts across the context axis. Details in
`notes/2026-10-06-32k-inversion.md`.

**Still tell the professor.** He approved a scope that included a hardware
comparison. He has since offered more compute and a Hopper, which changes the
calculus — see "Hopper" below.

---

## The three claims the paper now rests on

Recorded here so the queue below is read against them.

- **C20** — the machine factor is rung-independent at 2K (1.046 vs 1.056) and
  diverges with context (1.070 vs 1.296 at 32K). `R_draft` measured from the
  k-sweeps: DFlash 0.299 / 0.445 / 0.673, n-gram 0.071 / – / 0.139. The fixed
  drafting term dominates `c·k` by 4–7× and grows twice as fast.
  **Status: supported, not established** — the one point off the collapse is
  the n=5 2K cell.
- **C21** — the engine is bit-deterministic at fixed configuration (150/150,
  two nodes), so C10's control was measuring *concurrency*. Speculation's
  token divergence is real and 24–34× the floor: this build is **not bitwise
  lossless**.
- **C22** — accuracy is invariant across rungs. Speculation changes tokens and
  does not change answers.

---

## Immediate, no GPU

- [ ] **Patch `check_lossless`'s verdict block.** The control and main sections
      were rewritten on 13 Oct; the closing VERDICT paragraph was not, and on
      the C10 directory it printed "Both rates are similar, which is consistent
      with a common cause in the engine" while comparing 100% against `nan%`.
      It assumes the pre-C21 framing. Threat 14.
- [ ] **Swap in the v3 files.** `run_benchmark.v3.py`, `check_lossless.v3.py`
      and `phase0.v3.sh` are committed as candidates in `code/` and are not in
      use. `check_lossless` can be swapped now — it is pure analysis and was
      validated on `results/a40` with every number unchanged. `run_benchmark`
      needs the no-op proof below first.
- [ ] **Re-run the per-domain accuracy** with the bucket-aware grouping and
      fold the result into C22. The earlier pooled table made a bucket
      difference look like a two-prompt rung effect.

## Immediate, cheap GPU

- [ ] **The speculative-arm reproducibility control.** ~30 min. The single
      most valuable job remaining, because it is the difference between
      "speculation diverges from the baseline" and "speculation diverges
      *systematically*" — and the second is the claim the paper wants.
      Threat 12.
      ```bash
      S=/projects/bikd/dagraw2/Speculative_Ladder/ladder/scripts
      sbatch --export=ALL,BUCKET=2K,LIMIT=50,CONC=1,ARMS="ngram:5",REPEAT=2,\
      LADDER_RUN=$S/run_benchmark.v3.py,\
      LADDER_OUT=/projects/bikd/dagraw2/Speculative_Ladder/results/a40_c10 \
        $S/phase0.v3.sh
      ```
- [ ] **A100 2K at n=50.** ~10 min, and it tests the cost model's only
      parameter rather than filling a hole in a table: the 1.054 machine factor
      was fitted on job 22757104 at `LIMIT=5`, five prompts. It is also the one
      point that breaks C20's collapse.
- [ ] **The v3 no-op proof.** One 2K `off` arm under v3 against the existing v2
      row. Gate for swapping `run_benchmark.py`. Note the tolerance is the
      0.018% same-config spread, not zero, and cross-node not same-node.
- [ ] **`MAXTOK=1` at 8K and 32K.** ~16 min, and it redefines the dependent
      variable: `throughput_tok_s` is `gen_tokens / elapsed` and elapsed
      includes prefill — 823,170 prompt tokens at 32K. Subtracting the
      1-token run from the 256-token run gives the marginal per-token cost.
      **The intercept is arm-dependent** (a 1-token speculative run still pays
      one draft-and-verify cycle), so only the difference is comparable.
- [ ] **n-gram k sweep on a clean node.** The existing sweep gives 32K speedups
      of 1.043 (k=1), 1.132 (k=2), **1.181 (k=7)** against 1.078 for the k=5
      used in the main grid and the frozen prediction — but it ran on
      `gpub073`, which is on the exclusion list. If it holds, every n-gram
      number in the paper is understated and the prediction tested a
      suboptimal k. Threat 11.
      ```bash
      sbatch --exclude=gpub073 --export=ALL,BUCKET=32K,LIMIT=25,CONC=1,\
      ARMS="off:0 ngram:1 ngram:2 ngram:5 ngram:7" $S/phase0.v3.sh
      ```
- [ ] **8K at c=16 and c=32.** Two short jobs. The DFlash KV cost jumped from
      2.4 to 4.01 GiB at c=32 on a single observation. If that is a trend, every
      high-batch concurrency estimate including the H200 figure is an upper
      bound.
- [ ] **Prefix-caching-off control.** `--no-prefix-caching` exists, and
      `check_lossless.v3` can finally read the files it produces. Given C21,
      the question has sharpened: caching-on is plausibly *why* the engine is
      deterministic, so this measures whether determinism survives without it.

## The interval instrument

- [ ] **Disjoint prompt subsets**, by document id. Still the right route, and
      C21 makes the reason sharper than it was: run-to-run noise at fixed
      configuration is **zero**, so repeats measure nothing. Prompt-sampling
      variance is the only unmeasured term. `make_subsets.py` exists.
- [ ] **Loop-mode timing** (`LOOP=1`, v3, c=1 only) as the second route.
      Validate equivalence on one cell against the batched call before using
      it, and keep loop and batched rows as separate populations — it changes
      the arrival process. Do subsets first: no code change, no second
      population, and it covers the term that matters.
- [x] ~~Paired bootstrap over per-request latency~~ **Struck 13 Oct:**
      `request_timing()` asks for `first_token_time`, `last_token_time` and
      `finished_time` and this build populates none of them, so
      `per_request_timing` is structurally always 0. Loop mode is the
      replacement.
- [x] ~~Sahil's `SEED=5678` 2K repeat~~ **Struck 11 Oct:** `SEED` is inert
      under greedy and the harness always reads the first `LIMIT` lines.
- [x] ~~Repeat the 32K cell twice~~ **Superseded 11 Oct**, and confirmed
      13 Oct: three same-config realizations agree bit-for-bit, so repeats
      cannot measure sampling variance.

## Free analysis, no GPU

- [ ] **Per-domain τ.** Needs per-domain cells for acceptance; the join from
      per-request `id` to the dataset `_id` works and the accuracy path already
      uses it.
- [ ] **Offline router simulation** over the measured grid — oracle against a
      context-and-load rule against a reactive rule against each fixed rung.
      The one artifact that turns this from a characterization paper into a
      systems paper, and it costs nothing. Label interpolated cells as
      interpolated in every figure.
- [ ] **Chase two KV anomalies** before quoting any *absolute* GiB figure: job
      22757105's n-gram arm reports 23.77 GiB against its own `off` arm's
      22.10, the only negative cost in the set; and A40 2K jobs report
      22.10–22.74 where A40 8K/32K jobs report 23.77 at identical
      `max_model_len`. Neither affects the paired differences.

## Hopper

The professor has offered a Hopper and more hours. `phase0.sh` hard-aborts on
anything but compute capability 80 or 86 (`case "$CAP" in 80|86)`), so this
needs a guard variant, then a diagnostic — a kernel listing is not proof of
execution — then cells.

- [ ] **32K at c≈20.** In bf16: 141 GB × 0.90 − 18.3 GB of weights ≈ 108 GB of
      KV ÷ 4.86 GB per sequence ≈ 22 concurrent, and that is an upper bound
      because it subtracts weights but not activation and workspace. 5–6× the
      A100's 3.8, and the only reachable card for the regime MagicDec says
      matters. **FP8 KV would halve per-sequence KV and could reach batch 32**,
      at the cost of a dtype confound — worth stating as a route rather than
      declaring the regime closed.
- [ ] **FP8 target.** Tests the sharpest prediction C20 generates: making the
      target cheaper should *narrow* the margin. Note this is contested —
      QuantSpec exists and at least one popular account argues the combination
      is super-additive, so this is a disagreement to resolve rather than
      virgin ground.
- [ ] **Target-size sweep, 14B and 32B.** **Run it on n-gram as the clean
      arm** — it needs no checkpoint, so it transfers across target sizes with
      zero confound. A draft-model arm (Qwen3-0.6B/1.7B) is a second, labelled
      series; no trained head exists for those targets, so switching drafter
      family mid-axis would confound it.
- [ ] **Third machine point** at 8K/32K c=1 for C20's factor.

## Deferred, with reasons

| Item | Why |
|---|---|
| A100 rebuild | **Done.** The multi-architecture venv works; `results/a100` is committed with 15 rows. |
| Sahil's 7-second failures | **Done.** Named-ACL precedence, not the kernel problem. |
| Drafter memory as a third cost axis | **Done and reframed.** The drafter's weights are charged to the KV pool; see C19 and `notes/2026-10-13-c19-correction.md`. |
| Quality evaluation | **Done.** C22: accuracy is invariant. The 0.65% token divergence changes no answer. |
| `k=1` experiment / parameterising k | **Done.** `ARMS` in `phase0.v3.sh`; the A40 k-sweeps gave `R_draft` for C20. |
| n-gram coverage flag | **Done.** `PLMIN`/`PLMAX` in v3; the sweep itself is still queued. |
| EAGLE-3 | Checkpoints exist for Qwen3-8B (Tengyunw, RedHatAI, mobilint). It varies head quality at roughly fixed drafter cost, so it tests τ rather than R — a different axis from the drafter-size dial. Worth it only once the must-do rungs have intervals. |
| Drafter-size dial (0.6B/1.7B/4B) | The cleanest way to make `R_draft` a dial and generalise C20 past DFlash. **Run at 2K/8K, not 32K:** a 0.6B's own KV is ~78% of the 8B's per token, so at 32K it costs ~3.7 GB — more than DFlash's entire 2.4 GiB. The trained head is the memory-cheap rung; a small draft model is not. |
| Tensor parallelism inside the grid | An all-reduce every step changes `R`, the quantity being measured, so a TP=2 cell is not comparable. Worth doing as its own labelled population — nobody serves 8B on one GPU in a datacenter — but never mixed in. |
| Power sampling / tokens per joule | Passive and cheap, and the right column for a hardware course, but it blocks nothing. Last. |
| Cerebras | `sdk.cerebras.ai` is the CSL kernel SDK, not an inference API. Belongs in the discussion as a prediction target. Verify specs from Cerebras's own sheet first. |
| Second model family | Would address the single-model weakness but costs a full grid. |
