# Losslessness, and why the first check looked like a disaster

5 Oct 2026. A40 results, 2K bucket, jobs 22569403 and 22598608.

## What the first check said

Every speculative arm failed to reproduce the no-speculation output:

| cell | identical |
|---|---|
| 2K conc=1, dflash k=7 | 6/50 |
| 2K conc=1, ngram k=5 | 11/50 |
| 2K conc=8, dflash k=7 | 12/50 |
| 2K conc=8, ngram k=5 | 11/50 |
| 2K conc=32, dflash k=7 | 7/50 |
| 2K conc=32, ngram k=5 | 10/50 |
| 2K conc=auto, dflash k=7 | 18/100 |
| 2K conc=auto, dflash k=15 | 14/100 |
| 2K conc=auto, ngram k=5 | 23/100 |

0 of 9 arms passed. Read at face value: speculation is changing the output,
which would be a serious defect and would make every speedup in the project
unpublishable.

That reading was wrong, and the check as first written could not tell the
difference.

## The control

The comparison is only meaningful if the baseline reproduces itself. It does
not. Comparing the `off` arm against the `off` arm at different concurrency,
same 50 prompts, same seed, temperature 0, no speculation in either side:

```
off c=1  vs off c=8   :  7/50 identical  (14%)
off c=1  vs off c=32  :  5/50 identical  (10%)
off c=8  vs off c=32  :  7/50 identical  (14%)
```

**12.7% agreement, with no speculation involved at all.**

The speculative arms against the baseline at matching concurrency average
**19.0%**. Speculation agrees with the baseline *more often than the baseline
agrees with itself*. It cannot be the source of the divergence.

## Per-token rate

Sequence-level agreement is a bad unit. One flipped token anywhere in a 256
token generation changes the SHA-1, so a tiny per-token rate produces a huge
fraction of mismatched sequences. Treating the first divergence as a hazard
over 256 positions, `agreement = (1 - p)^256`:

| comparison | sequences identical | implied per-token rate |
|---|---|---|
| baseline vs baseline | 12.7% | 0.80% |
| speculative vs baseline | 19.0% | 0.65% |

About one token in 140 differs between two runs of the *same* configuration at
different batch sizes.

The hazard framing is the right one: divergence is absorbing, because once the
prefix differs the suffix is generated from a different context. So `p` is the
per-position probability of a flip *given an identical prefix so far*, which is
exactly the quantity of interest.

## Why the engine is not deterministic

Two mechanisms, both switched on by normal serving behaviour:

1. **Batch-dependent reduction order.** Matrix multiplications and softmax
   reductions split work differently depending on how many sequences are in
   the batch. Floating point addition is not associative, so logits move in
   their last bits. Greedy decoding takes an argmax, and on a near-tie a shift
   of one ULP flips the token. Everything after that point is generated from a
   different prefix.

2. **Prefix caching**, on in these runs at a 21 to 24% hit rate. Whether a
   request reuses cached keys and values or recomputes them depends on arrival
   order and on what else is resident. The cached and recomputed paths are not
   bitwise identical, so the same prompt can take either and diverge.

Neither has anything to do with drafting. Both change with concurrency, which
is precisely the axis this study varies.

## What this means for the paper

**Exact-match losslessness is not testable in a real continuous-batching
server.** The textbook statement, that speculative decoding returns exactly
what the target model would have returned, holds for the mathematics of the
verification step. It does not survive contact with an engine whose own output
depends on batch composition. Any paper claiming byte-exact equality on a
vLLM-class server either disabled batching, or did not check.

So the claim is framed relative to the engine's own noise floor:

> Speculative decoding does not measurably perturb the greedy decoding path.
> Across 2K-context runs on A40, speculative arms agreed with the
> no-speculation baseline on 19.0% of sequences, while the baseline agreed
> with itself across batch sizes on only 12.7%. The implied per-token
> divergence is 0.65% for speculation against 0.80% for the engine's own
> batch-size non-determinism. Speculation therefore sits below the
> reproducibility floor of the serving system it runs on.

That is a stronger and more useful statement than an exact-match claim,
because it is both true and measured. It also stands on its own as a finding:
anyone benchmarking LLM serving and comparing token-level outputs across
configurations is comparing against a baseline that does not reproduce itself.

## Follow-ups

1. **Store token ids, not just a hash.** 50 requests x 256 tokens is about
   60 KB per arm, which is nothing. It turns the per-token rate from an
   inference under an independence-style assumption into a direct measurement,
   and it locates the first divergence, which distinguishes late numerical
   noise from early systematic difference. This is now the top harness change,
   ahead of per-request timing.

2. **Prefix-caching-off control.** Re-run one cell with prefix caching
   disabled and repeat the baseline-vs-baseline comparison. If agreement jumps
   toward 100%, caching is the dominant cause and is worth disabling for the
   headline table. If it does not move, the cause is kernel reduction order,
   which cannot be switched off and must simply be reported.

3. **Decide whether the headline table runs with caching on or off.** Caching
   on is what a real server does and keeps the throughput numbers
   representative. Caching off is more reproducible. Current lean: keep it on,
   report the noise floor, and run the caching-off cell as a control rather
   than as the main configuration.

## What this does NOT affect

Nothing measured so far is invalidated. Throughput, mean accepted length,
acceptance by draft position, coverage and the draft counters do not depend on
*which* tokens were produced, only on how many were accepted and how quickly.
The tau-invariance finding, the concurrency crossing points and the coverage
result all stand unchanged.

No job needed cancelling. The check runs after the fact on files already
written and costs no GPU time.

## Method note for anyone repeating this

The first version of `check_lossless.py` reported whether mismatched
generations had the same length, on the theory that equal lengths point to a
mid-sequence token flip and unequal lengths to an early stop. That diagnostic
is worthless on this data: every generation runs to the 256 token cap, so all
lengths are equal by construction and the test can only ever say one thing.
The control, not the length, is what separates the hypotheses.
