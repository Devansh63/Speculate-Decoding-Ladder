# Speculative Decoding Ladder

Measuring where each speculative decoding mechanism stops paying inside a serving runtime, and predicting that point on hardware the cost model was never fitted on.

**CS 598 Hardware & Software for AI, UIUC, Fall 2026** · Devansh Agrawal, Sahil Sashi
vLLM · Qwen3-8B · NVIDIA A40, A100, H200 (NCSA Delta)

## The problem

Speculative decoding lets a cheap drafter propose several tokens that the target model verifies in one forward pass, with output identical to ordinary decoding. Many drafting mechanisms exist and they differ by an order of magnitude in memory held, bytes read per drafted token, and accuracy. vLLM and SGLang let an operator enable exactly one at launch and never revisit it. Field reports show that fixed choice going from +129% throughput at 2K context to -51% at 30K.

The **ladder** is one server holding several drafters, with a controller assigning one per request as context grows and the server fills. Underneath it is a cost model: acceptance is a property of the model pair and workload, verification cost a property of the machine, so the point where a mechanism stops paying can be predicted on hardware it was never measured on.

## Status

Harness works. First measurements taken on an A40.

| | |
|---|---|
| Measurement harness | Working, on the stable vLLM model runner |
| Workloads | LongBench-v2 at 2K, 8K, 32K tokens of context |
| Rungs measured | no speculation, n-gram lookup, DFlash v1 (k=7 and k=15) |
| Context buckets measured | 2K |
| Concurrency axis | Not yet controlled, next run |

## First result, 2K context on A40

100 prompts, greedy, 256 generated tokens, uncontrolled concurrency.

| rung | k | tok/s | speedup | accepted length | truncated to k=5 | position 1 |
|---|---|---|---|---|---|---|
| off | 0 | 405.1 | 1.00 | 1.000 | | |
| DFlash | 7 | 348.5 | 0.86 | 2.520 | 2.407 | 0.601 |
| DFlash | 15 | 316.3 | 0.78 | 2.823 | 2.545 | 0.656 |
| n-gram | 5 | 289.0 | 0.71 | 2.979 | 2.979 | 0.608 |

Every speculative rung is slower than no speculation here, with healthy acceptance throughout. All 100 prompts were in flight at once, so the GPU was compute-bound and verification cost more than it saved. The cost term, not the acceptance term, is what sinks it at high load. Controlling concurrency is the next run.

Two things survive that caveat, since acceptance does not depend on how the batch was scheduled:

- **n-gram out-accepts DFlash at the same draft length** on this workload: 2.979 against 2.545 at k=5, with a far slower per-position decay (0.608, 0.457, 0.364, 0.297, 0.254 against 0.656, 0.401, 0.240, 0.148, 0.100). LongBench-v2 answers quote the context, which favours copying, and that is itself worth stating.
- **DFlash at k=15 wastes most of its drafts.** Acceptance falls below 0.03 past position 9, and k=7 is faster.

## Layout

```
code/     measurement harness, job scripts, workload builder
results/  one dated record per run, raw results.jsonl
notes/    findings log, lab notebook
paper/    claims and evidence, figures, related work, threats to validity
env/      cluster facts: partitions, storage, charging
```

## Rules that keep the numbers meaningful

1. **Pin the model runner.** Every run sets `VLLM_USE_V2_MODEL_RUNNER=0`. Without it, n-gram silently falls back to the stable runner while DFlash stays on the experimental one, and the two are no longer comparable.
2. **Control concurrency** with `--max-num-seqs`. A run that lets the engine choose measures an arbitrary load point.
3. **Truncate to a common draft length** before comparing rungs. A k=15 drafter has more room to lose tokens than a k=5 one.
4. **Never pool** across context regimes (native against YaRN) or across model runners.
5. **Greedy, fixed seed, capped generation.** Temperature 0, seed 1234, 256 tokens.
6. **Every run records metadata**: commit, GPU, runner, model revision, rung, bucket, seed. A number without its row in `results.jsonl` does not exist.
7. **No claim from a single run.** Intervals before conclusions.

## Reproducing

```bash
sbatch --export=ALL,BUCKET=2K,LIMIT=100 phase0_a40.sh
python3 summarize.py <results dir>
```

The no-speculation baseline runs first, then each rung. Buckets are separate jobs so a failure at 32K does not cost 2K and 8K.
