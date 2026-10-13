
### Frozen prediction scored, 2026-10-13

Scored from records, not from prose. Every measured value below is the
mean over all realizations of that cell at concurrency 1 found in `/projects/bikd/ssashi/Speculative_Ladder/results/a100`;
individual realizations and their jobs are listed under the table.

| cell | predicted | measured | rel. error | implied R | machine factor | d_tau vs A40 |
|---|---|---|---|---|---|---|
| 8K dflash k=7 | 1.20 | **1.1054** | -7.89% | 1.7664 | 1.1440 | +0.28% |
| 8K ngram k=5 | 1.09 | **1.0788** | -1.03% | 1.2022 | 1.0734 | +1.01% |
| 32K dflash k=7 | 0.78 | **0.6432** | -17.53% | 2.2894 | 1.2963 | +0.93% |
| 32K ngram k=5 | 1.02 | **0.9745** | -4.46% | 1.3804 | 1.0700 | -3.23% |

Realizations:

- `8K dflash k=7` job 22802261 on gpua008: speedup 1.1066, R 1.7644  (cov pinned by clamp)
- `8K dflash k=7` job 22802308 on gpua070: speedup 1.1041, R 1.7684  (cov pinned by clamp)
- `8K ngram k=5` job 22802261 on gpua008: speedup 1.0780, R 1.2031
- `8K ngram k=5` job 22802308 on gpua070: speedup 1.0796, R 1.2013
- `32K dflash k=7` job 22802262 on gpua008: speedup 0.6437, R 2.2877
- `32K dflash k=7` job 22802309 on gpua087: speedup 0.6428, R 2.2910
- `32K ngram k=5` job 22802262 on gpua008: speedup 0.9742, R 1.3807
- `32K ngram k=5` job 22802309 on gpua087: speedup 0.9747, R 1.3800

**Cells within 5% relative: 2 of 4. Median absolute relative error: 6.18%.**

Against the pre-registered criterion:

- **Pass** required *every* cell within 5% and the 32K ordering to hold. 2 cell(s) exceeded the tolerance, so it does not pass.
- **Fail** required median absolute error above 10% (it is 6.18%, so not by this limb) **or** the 32K ordering failing to reproduce.
- The 32K ordering is ambiguous as written. Relative order (DFlash below n-gram): **reproduced** (0.6432 vs 0.9745). Break-even side (DFlash below 1.0 *and* n-gram above 1.0): **failed**.

Recording the ambiguity rather than resolving it silently: the wording
"the 32K ordering does not reproduce" admits both readings, and the outcome
differs between them. The wording should be fixed before the next prediction
is frozen, not after.

**The pass/fail pattern splits by rung, not by bucket.**

- `dflash`: 0/2 within tolerance (-7.89%, -17.53%)
- `ngram`: 2/2 within tolerance (-1.03%, -4.46%)

The rung with no drafter forward pass landed; the rung with one did not.
That is the rung-dependent machine factor showing up in the scoreboard, and
it is the diagnosis -- not a repair of the model.

All 4 errors are negative: every prediction was optimistic. That is
bias, not noise.

Provenance: `/projects/bikd/ssashi/Speculative_Ladder/results/a100`, GPUs NVIDIA A100-SXM4-40GB, harness v2-2026-10-05.
Filters: harness=v2 exclude-host=- exclude-job=-.
