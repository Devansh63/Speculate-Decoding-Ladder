# Ratios replicate to 0.1%; absolute throughput does not

**Third version of this note.** The first two were written from
`summarize.py` output and were both wrong. This one is written from the
stored per-record data, which was there the whole time. The sequence of
errors is recorded at the bottom, because it is the more useful part.

## What the stored records actually show

`results/a40/results.jsonl` holds 15 records at 32K. Every record carries
`env.slurm_job` and `env.host`, so each measurement can be attributed to a
job and a physical node. Splitting on that:

| | job 22713173, gpub019 | job 22714602, gpub047 |
|---|---|---|
| `off` c1 | 14.90 | 14.83 |
| `ngram` k=5 c1 | 16.07 | 15.98 |
| `dflash` k=7 c1 | 12.31 | 12.25 |
| **n-gram speedup** | 16.07/14.90 = **1.0785** | 15.98/14.83 = **1.0776** |
| **DFlash speedup** | 12.31/14.90 = **0.8262** | 12.25/14.83 = **0.8260** |

Two independent jobs, two different physical machines, three days apart. The
ratios agree to **0.08%** and **0.02%**. Absolute throughput agrees to 0.47%.

**The 32K inversion has been replicated since 7 Oct.** It was never a single
run. It went unnoticed because `summarize.py` keys on
`(bucket, conc, rung, k)` and both jobs produce the same keys, so it keeps one
record and prints one row — and the register has been describing the result
as unreplicated ever since.

## The one node that is different

A third realization, job 22757472 on **gpub073**, gives `off` = 13.5 tok/s —
8.8% below the two above, which agree with each other to 0.47%. Both of my
earlier explanations are refuted from the logs:

- **KV cache allocation: no.** The `off` arm was allocated 173,056 tokens in
  all three jobs, identical to the token.
- **Preemption: no.** Zero `preempt` or `recompute` lines in any of the three
  logs.

What remains is the node. And the interesting part is that the *ratio* moved
too: interpolating gpub073's n-gram k=2 to k=7 curve to k=5 gives about 1.16,
against 1.078 measured on both other nodes. So on gpub073 plain decoding was
slower **and speculation helped more**.

That is the direction a bandwidth-limited node would produce. Speculative
decoding trades memory-bound decode steps for a compute-bound verification
pass, so it pays off more when memory bandwidth rather than compute is the
binding constraint — which is the R model working as designed, on a machine
we did not choose to test.

**Consistent with, not established.** One node, one job, and we have no
visibility into what else was resident on that machine's other three GPUs.
`phase0.sh` requests `--gpus-per-node=1` on 4-GPU nodes, so co-tenancy is
possible but unobservable from our side.

## What is safe and what is not

**Safe — ratios, and anything within a job:**

- **C8, the inversion.** Replicated across two nodes to 0.1%. This is now the
  best-supported number in the project.
- **C9, DFlash's k-trend at 32K.** Four monotonic points within job 22714602:
  R = 1.677, 1.704, 1.743, 1.766 at k = 1, 2, 4, 7.
- Everything at 2K, where the cross-job baseline agrees to 0.61%.

**Not safe:**

- **C15, "n-gram is faster at 32K than at 2K."** 1.14x against 1.18x, a 4%
  claim, drawn from two different jobs — and the 32K half is gpub073, the
  node with the anomaly. Demote to "n-gram's speedup does not degrade with
  context", or drop.
- **n-gram's 32K k coefficient.** Non-monotonic within its own job
  (R(1) = 1.156 > R(2) = 1.141), and that job is gpub073.
- **Absolute throughput at 32K across jobs.** Report ratios.

## Practical rules this produces

1. **Report ratios, not throughput.** The ratio is the stable quantity by two
   orders of magnitude.
2. **Never compare two numbers from different `slurm_job` values** unless both
   are ratios and the difference is well above a few percent.
3. **Split by `env.slurm_job` before analysing.** The data supports it;
   `summarize.py` throws it away. The script should group by job, or at
   minimum warn on duplicate keys instead of silently keeping one.
4. **Every repeat gets its own output directory**, since the per-request
   filenames do not encode the run and would be overwritten.

Point 3 is the actionable software fix. Point 1 is the one that matters for
the paper.

## How this note was wrong twice

Worth recording in full, because the pattern repeated inside a single hour.

**Version 1** read the 13.5 against 14.8 and concluded "32K is irreproducible
to about 10%." That generalized one disagreement into a variance estimate.

**Version 2** corrected it to "an occasional excursion" after noticing that
C11 already recorded two agreeing realizations — but still described the
inversion as a single unreplicated run, still recommended adding node and job
id to the harness, and still treated the within-job status of the headline
numbers as unknown.

**Version 3** read the records. `env.host` and `env.slurm_job` have been
recorded since harness v2 on 5 Oct, so the attribution was always available.
The inversion was already replicated. The harness needed no change. Both
earlier versions argued from `summarize.py`'s collapsed table — a derived
view — while the per-record data sat in the same directory.

The dead-ends table in `LOG.md` says the recurring error here is trusting a
proxy instead of testing the thing. A summary table is a proxy for the data.
Three versions of a note about measurement error, all written without opening
the measurements.
