#!/usr/bin/env python3
"""Block 3, step 1 -- propagate the 13 Oct findings into the six files that
were left untouched while nine commits landed that day.

  paper/claims.md        1365 lines  C20 amended (the machine factor conflates
                                     acceptance with cost); C22 heading
  HANDOFF.md              691 lines  C10, C16/C17, the scored prediction, and
                                     the "signature of a machine property"
  paper/conclusions.md               losslessness section; three limitations
  paper/results-table.md             the hardware section's conclusion; A100 row
  paper/threats.md                   threats 9 and 10; new threat 17
  TODO.md                            the C20 and C22 summaries

Why a patcher and not an API push: reproducing a 77 KB file from a chat window
to send it through an API is how a transcription error enters the paper's spine
silently, with no copy to diff against. Every edit below is located by a unique
anchor taken from the file itself and asserted to match exactly once. A single
failure writes nothing. Edits already present are skipped, so a second run is
safe. Committing from here also keeps the author line on
Devansh Agrawal <dagraw2@illinois.edu>, which an API push cannot.

  python3 b3_01_patch_propagation.py                       # dry run
  python3 b3_01_patch_propagation.py --write               # write only
  DOCOMMIT=1 python3 b3_01_patch_propagation.py --write    # write and commit

R defaults to the Delta checkout; override with R=/path/to/repo.
"""
import os, subprocess, sys

R = os.environ.get("R", "/projects/bikd/dagraw2/Speculative_Ladder/repo")
WRITE = "--write" in sys.argv
DOCOMMIT = os.environ.get("DOCOMMIT") == "1"

EDITS = [
    ('paper/claims.md',
     '**At 2K the two rungs are indistinguishable (1.046 against 1.056). They diverge\nmonotonically with context.** The factor fitted at 2K looked like a machine\nconstant because 2K is the one bucket where the rungs agree.\n\nThis is the pre-registered **branch B** signature exactly as written — "a factor\nthat differs between rungs at long context but not short would mean the machine\nand drafter terms are not separable." The terms do not separate.\n',
     '**AMENDED hours later the same day, after external review. The factor above is\ncontaminated by acceptance.** This project defines `R` by the identity\n`R = tau_eff / speedup`, so\n\n```\nfactor = R_A100 / R_A40 = (tau_A100 / tau_A40) / (speedup_A100 / speedup_A40)\n```\n\nand threat 9, registered the same day, establishes that `tau` does **not**\ntransfer between machines — 21 of 25 sequences differ between A40 and A100. The\ntau term is therefore not a small correction that can be waved through. It is\nthe same channel C13\'s 11 Oct correction said must not be dropped, reappearing\non the other side of the equation.\n\nHolding acceptance fixed and asking what the cost ratio alone would be\n(`speedup_A40 / speedup_A100`):\n\n| bucket | rung | as registered | tau-free | d_tau |\n|---|---|---|---|---|\n| 2K | n-gram k=5 | 1.0456 | **1.0905** | -4.12% |\n| 2K | DFlash k=7 | 1.0556 | **1.0437** | +1.14% |\n| 8K | n-gram k=5 | 1.0715 | 1.0605 | +1.04% |\n| 8K | DFlash k=7 | 1.1397 | 1.1402 | -0.04% |\n| 32K | n-gram k=5 | 1.0703 | 1.1061 | -3.24% |\n| 32K | DFlash k=7 | 1.2964 | 1.2840 | +0.97% |\n\n`registered = tau-free x (1 + d_tau)` holds to four decimal places in every\ncell, so this is arithmetic rather than interpretation.\n\n**Withdrawn 1 — "at 2K the two rungs are indistinguishable."** Tau-free they\ndiffer by **4.7 points** rather than 1.0, **and in the opposite order**: n-gram\n1.0905 above DFlash 1.0437. The rungs looked equal at 2K only because n-gram\'s\ntau fell 4.12% on A100 while DFlash\'s rose 1.14%, and those cancel a\n4.7-point difference in the cost term.\n\n**Withdrawn 2 — the branch-B label.** Branch B was pre-registered as "a factor\nthat differs between rungs at long context but not short". Tau-free the rungs\ndiffer at *every* bucket, so the observed pattern is in none of the three\npre-registered branches, nor in the "broader than B" reading proposed and\nretracted earlier the same day. That the pre-registered diagnostic list does\nnot contain the pattern that occurred is itself the outcome, and is recorded as\nsuch rather than forced into a label.\n\n**What survives is stronger.** C20\'s conclusion — the machine factor is not a\nproperty of the machine — is untouched. As registered it rested on the rungs\nagreeing at 2K and diverging after. Tau-free they disagree at *every* bucket,\nwhich is the stronger statement: there is no context at which a single scalar\nrelates A40 cost to A100 cost across rungs. The `R_draft` fits below are\nunaffected, being fitted within one machine across k.\n\n**The fix is two columns, not a different scalar.** The tau-free factor is\nitself a **counterfactual** — it asks what the factor would be if tau\ntransferred, and threat 9 says it does not. Replacing one contaminated scalar\nwith one counterfactual scalar is not an improvement. Report both channels and\nstop claiming a single machine factor exists, which is what this claim\'s own\ntitle asserts and what its evidence section then contradicted by quoting one\nnumber per cell.\n\n**Aggregation, omitted from the table as first published.** `R` is averaged\nover same-cell realizations on both sides before the ratio is taken. An\nexternal reviewer recomputing 8K n-gram obtained 1.0704 against the published\n1.0715 by choosing a single A40 realization as the reference. Both are\ndefensible; only one is what the table reports. Any future factor table states\nits aggregation.\n\nFull working in `notes/2026-10-13-c20-amendment.md`. The conflation, the 2K\nreversal and the aggregation omission were all found by external review of this\nrepository\'s 13 Oct commits, not by the work that produced them.\n\nAs first registered, and now withdrawn, kept for the record:\n\n> **At 2K the two rungs are indistinguishable (1.046 against 1.056). They\n> diverge monotonically with context.** The factor fitted at 2K looked like a\n> machine constant because 2K is the one bucket where the rungs agree.\n>\n> This is the pre-registered **branch B** signature exactly as written — "a\n> factor that differs between rungs at long context but not short would mean\n> the machine and drafter terms are not separable." The terms do not separate.\n',
     'claims.md C20: amendment inserted, two statements withdrawn'),
    ('paper/claims.md',
     "| R_draft | machine factor | cell |\n|---|---|---|\n| 0.071 | 1.0456 | n-gram 2K |\n| 0.139 | 1.0703 | n-gram 32K |\n| 0.299 | **1.0556** | DFlash 2K |\n| 0.445 | 1.1397 | DFlash 8K |\n| 0.673 | 1.2964 | DFlash 32K |\n\nThe exception is DFlash 2K, which sits below the n-gram 32K point despite twice\nthe drafting cost. **Its A100 side is job 22757104 at `LIMIT=5`** — the same\nfive-prompt cell that supplied the frozen prediction's only parameter. The one\npoint that breaks the collapse is the one with five prompts.\n",
     "| R_draft | machine factor | tau-free factor | cell |\n|---|---|---|---|\n| 0.071 | 1.0456 | 1.0905 | n-gram 2K |\n| 0.139 | 1.0703 | 1.1061 | n-gram 32K |\n| 0.299 | **1.0556** | **1.0437** | DFlash 2K |\n| 0.445 | 1.1397 | 1.1402 | DFlash 8K |\n| 0.673 | 1.2964 | 1.2840 | DFlash 32K |\n\nThe exception is DFlash 2K, which sits below the n-gram 32K point despite twice\nthe drafting cost. **Its A100 side is job 22757104 at `LIMIT=5`** — the same\nfive-prompt cell that supplied the frozen prediction's only parameter. The one\npoint that breaks the collapse is the one with five prompts.\n\n**On the tau-free column the collapse is worse, not better.** 1.0905 at\n`R_draft` = 0.071 exceeds 1.0437 at 0.299, so monotonicity fails at two points\nrather than one, and the lowest-cost cell carries the third-highest factor.\nWhatever the collapse figure ends up plotting, it cannot be the registered\nfactor against a within-machine cost fit: that puts acceptance on one axis only.\n",
     'claims.md C20: tau-free column added to the collapse table'),
    ('paper/claims.md',
     '**C20 is therefore recorded as supported but not established**, pending A100 2K\nat n=50.',
     '**C20 is therefore recorded as supported but not established**, now for two\nreasons rather than one: the n=5 point below, and the restatement of the factor\nas a two-channel decomposition required by the amendment above, which must be\nsettled **before** the collapse figure is drawn rather than after. Pending\nA100 2K at n=50.',
     "claims.md C20: second reason for 'not established'"),
    ('paper/claims.md',
     '### C22, 13 Oct 2026 — accuracy is invariant across rungs\n',
     '### C22, 13 Oct 2026 — no accuracy difference exceeds the baseline\'s own variation\n\n**This heading was first written "accuracy is invariant across rungs", which\noverclaims at n = 24 to 49.** The body always said the weaker thing; the\nheading is what would have been quoted. Corrected in place because a heading is\na label, not a belief.\n',
     'claims.md C22: heading de-overclaimed'),
    ('HANDOFF.md',
     'coverage. That is the signature of a machine property. The A100 is 2.2x faster',
     'coverage. **That reading is withdrawn (13 Oct).** The 5.5% is a ratio of `R`,\nand `R` is defined here as `tau_eff / speedup`, so it carries the cross-machine\nacceptance ratio as well as the cost ratio. Tau-free the two rungs differ by\n4.7 points at 2K and in the reverse order — n-gram 1.0905 against DFlash\n1.0437 — so the agreement that looked like a machine property is two\ncancelling errors. See C20 and `notes/2026-10-13-c20-amendment.md`. The A100 is 2.2x faster',
     "HANDOFF §2.3: 'signature of a machine property' withdrawn"),
    ('HANDOFF.md',
     '- **C10** (changing concurrency perturbs output more than adding speculation\n  does at fixed concurrency) — **measured in this narrow form only.** The\n  strong form, that speculation perturbs output less than the engine perturbs\n  itself, is not established by the design. Needs a same-config control and an\n  interval; `code/c10_control.py` does the bootstrap, and `code/c10_control.sh`\n  runs the control.',
     '- **C10** (changing concurrency perturbs output more than adding speculation\n  does at fixed concurrency) — **the strong form is REFUTED, 13 Oct.** The\n  same-config control ran (jobs 22875382, 22876140): 150 of 150 sequences\n  bit-identical across two nodes, so at matched concurrency the engine does\n  not disagree with itself at all, and the entire control effect was\n  concurrency. There is nothing for speculation to be smaller than. The\n  speculative arms diverge at 0.556–0.781% per token against a floor bounded\n  at 0.023%, so this build is **not bitwise lossless**. See C21.',
     'HANDOFF §4.2: C10 strong form refuted by C21'),
    ('HANDOFF.md',
     "- **C16, C17** (tau is machine-invariant; R scales by a single rung-independent\n  factor) — measured, one pair, n=5. **C17's mechanism is unexplained**: the\n  bandwidth story is withdrawn, and so is the ridge-point argument against it\n  (the A40 figure was wrong). Both are now restated as transfer across a\n  **bandwidth** axis, not a compute-to-bandwidth ratio axis.",
     "- **C16, C17** (tau is machine-invariant; R scales by a single rung-independent\n  factor) — **both overtaken on 13 Oct.** C16 is **refuted at the sequence\n  level**: 21 of 25 outputs differ between A40 and A100 at the same cell, so\n  tau does not transfer and the 0.8% agreement in the n=5 table is agreement\n  in a mean, not invariance (threat 9). C17 is **falsified as stated** by C20:\n  the factor differs between rungs at every bucket once the acceptance channel\n  is removed, and the 2K agreement that motivated it was two cancelling\n  errors. The replacement is not a better scalar but a **two-column\n  decomposition** — a speedup ratio and a tau ratio, reported separately.\n  C17's mechanism was already unexplained (the bandwidth story is withdrawn,\n  and so is the ridge-point argument against it, the A40 figure being wrong);\n  the amendment removes the quantity that mechanism was meant to explain. See\n  `notes/2026-10-13-c20-amendment.md`.",
     'HANDOFF §4.2: C16/C17 overtaken by threat 9 and C20'),
    ('HANDOFF.md',
     '**The test** is `code/submit_a100_predict.sh`, one command, ~1.5 GPU-hours.\nThis is the highest-value GPU time available to the project.',
     "**The test RAN on 13 Oct** — jobs **22802261** (8K) and **22802262** (32K),\nboth COMPLETED — and was scored from records rather than prose.\n\n| cell | predicted | measured | rel. error |\n|---|---|---|---|\n| 8K, n-gram k=5 | 1.09x | 1.0789x | -1.03% |\n| 8K, DFlash k=7 | 1.20x | 1.1054x | -7.88% |\n| 32K, n-gram k=5 | 1.02x | 0.9744x | -4.49% |\n| 32K, DFlash k=7 | 0.78x | 0.6432x | -17.5% |\n\n**Verdict: neither pass nor fail as pre-registered.** Two of four cells are\nwithin 5%, median absolute error 6.18%, under the 10% fail threshold. The 32K\nordering holds — n-gram 0.974 above DFlash 0.643 — so the inversion survived\nonto a second machine, which was the prediction's substance. **Every cell is\noptimistic**, and the misses split cleanly **by rung**: n-gram 2 of 2 within\n5%, DFlash 0 of 2. A single machine factor cannot produce a rung-dependent\nerror, which is where C20 came from, and the amendment to C20 shows the factor\nalso conflates acceptance with cost. The remaining GPU test is **A100 2K at\nn=50**, because the factor's only parameter was fitted on five prompts.",
     'HANDOFF §5: prediction scored, no longer framed as pending'),
    ('paper/results-table.md',
     '**Acceptance transfers to under one percent; the cost ratio rises 5.5%, and by\nthe same amount for both rungs** — rungs that differ by 4x in drafting cost\nand 7x in coverage. That is what a machine property looks like.',
     '**Acceptance transfers to under one percent; the cost ratio rises 5.5%, and by\nthe same amount for both rungs** — rungs that differ by 4x in drafting cost\nand 7x in coverage.\n\n**"That is what a machine property looks like" was the conclusion drawn here,\nand it is WITHDRAWN (13 Oct).** Two corrections, in opposite directions.\n\n*The acceptance row does not mean what it says.* Agreement to 0.8% is agreement\nin a **mean**. At the sequence level 21 of 25 outputs differ between the two\nmachines at the same cell, so tau is not machine-invariant; the mean is stable\nwhile the thing it averages is not (threat 9).\n\n*The cost row is not a cost row.* `R` is defined in this project as\n`tau_eff / speedup`, so a ratio of `R` carries the cross-machine acceptance\nratio inside it. Removing that channel — asking what the ratio would be at\nfixed acceptance — gives **1.0905 for n-gram against 1.0437 for DFlash at 2K**:\n4.7 points apart and in the **reverse** order. The two rungs agreed only\nbecause n-gram\'s tau fell 4.12% on A100 while DFlash\'s rose 1.14%. See C20 and\n`notes/2026-10-13-c20-amendment.md`. Report the speedup ratio and the tau ratio\nas two columns; there is no single machine factor to quote.',
     "results-table: 'what a machine property looks like' withdrawn"),
    ('paper/results-table.md',
     '| A100-SXM4-40GB | sm_80 | **working** since the 8 Oct multi-architecture rebuild | one cell: 2K, batch 1, n=5 (job 22757104) |',
     "| A100-SXM4-40GB | sm_80 | **working** since the 8 Oct multi-architecture rebuild | 2K batch 1 n=5 (job 22757104), plus the 8K and 32K prediction grid of 13 Oct (jobs 22802261, 22802262) — 15 rows in `results/a100`. Still missing: **2K at n=50**, the cell the machine factor's only parameter was fitted on |",
     'results-table: A100 coverage row updated to the 13 Oct grid'),
    ('paper/conclusions.md',
     'All measurements: Qwen3-8B on A40, LongBench-v2, greedy decoding, vLLM pinned to\nthe stable model runner. Jobs 22598608 (2K), 22685911 (8K), 22713173 (32K),\n22714602 (k sweep and replication).',
     'All measurements: Qwen3-8B, LongBench-v2, greedy decoding, vLLM pinned to the\nstable model runner. A40 jobs 22598608 (2K), 22685911 (8K), 22713173 (32K),\n22714602 (k sweep and replication). A100 jobs 22757104 (2K, n=5), 22802261 (8K)\nand 22802262 (32K). Determinism control: A40 jobs 22875382 and 22876140 under\nharness `v3-2026-10-13`.\n\nUpdated 13 Oct 2026. The hardware axis is no longer empty, the losslessness\nsection below is rewritten, and two of the five "cannot claim" items have been\nclosed.',
     'conclusions: header provenance updated'),
    ('paper/conclusions.md',
     '## Losslessness, honestly stated\n\nSpeculative decoding is supposed to return exactly what the target model would\nhave returned. On a continuous-batching server that is not testable.\n\nComparing the no-speculation arm **against itself** at different concurrency,\nsame prompts, same seed, no speculation anywhere: the outputs agree on only\n**12.7%** of sequences. Speculative arms against the baseline agree on **19.0%**.\n\n**Changing concurrency perturbs output more than adding speculation does at\nfixed concurrency.** The stronger reading — that speculation perturbs output\nless than the engine perturbs itself — does not follow from this design: the\ncontrol varies batch size *and* nondeterminism, while the treatment varies\nspeculation alone (register, C10, corrected 11 Oct). Implied\nper-token divergence is 0.80% baseline-to-baseline and 0.65%\nspeculative-to-baseline. The causes are batch-dependent reduction order in GPU\nkernels and prefix caching.\n\nThe defensible claim is therefore relative: speculative decoding does not\nmeasurably perturb the greedy decoding path beyond the reproducibility floor of\nthe serving system it runs on. Any paper claiming byte-exact equality on a\nvLLM-class server either disabled batching or did not check.',
     '## Losslessness, in two halves\n\nSpeculative decoding is supposed to return exactly what the target model would\nhave returned. Measured, this build fails that at the bit level and passes it\nat the task level, and the two halves point in opposite directions.\n\n**At the bit level it is not lossless.** First the floor: three realizations of\nthe no-speculation arm at one configuration, across two nodes, are **150 of 150\nsequences bit-identical**. Zero events in 12,800 token positions bounds the\ntrue per-token hazard at about **0.023%** (rule of three, 95%). Against that\nfloor the speculative arms diverge at **0.556%** per token for n-gram k=5 and\n**0.781%** for DFlash — 24 to 34 times the bound, and systematic in k\n(0.738% at k=1 rising to 0.781% at k=2). Verifying k+1 positions runs different\nkernel shapes from decoding one, so near-ties resolve differently. The\ndivergence is real, measurable and attributable to speculation (C21).\n\n**At the task level it is lossless.** LongBench-v2 is multiple choice, so the\nstored tokens can be scored against the answer key. At 8K **nine arms score\n34.7%, identically**. Every difference at 2K and 32K is exactly one prompt —\nand the baseline differs from itself by exactly one prompt across concurrency.\nNo rung deviates by more than the baseline\'s own variation, over 503 graded\nquestions (C22).\n\n**This replaces the framing this section used to carry**, which was built on a\ncontrol comparing the baseline against itself at *different* concurrency: 12.7%\nsequence agreement against 19.0% for speculative arms, read as the engine\nperturbing itself more than speculation does. At matched concurrency the engine\ndoes not disagree with itself at all, so **the entire control effect was\nconcurrency**, and the sentence this section ended on — that speculation does\nnot perturb the greedy path beyond the serving system\'s reproducibility floor —\nis **withdrawn**. The floor is zero and speculation sits 24 to 34 times above\nit.\n\nWhat replaces it is sharper than what it replaces. Sequence identity, the\nmetric the entire losslessness argument was conducted in, **does not predict\ntask outcome**: speculation changes tokens at 24-34x the floor and changes no\nanswer. A paper claiming byte-exact equality on a vLLM-class server either\ndisabled batching or did not check. A paper claiming task-level equivalence can\nsupport it — and should say which of the two it means, because they are not the\nsame claim and this data separates them.\n\n**The control still missing.** No speculative arm has been run twice at one\nconfiguration, so some part of the 0.556-0.781% could be that arm\'s own\nrun-to-run noise rather than a systematic offset. Until that job runs, the\nsupported statements are "the engine is deterministic" and "speculation\ndiverges from the baseline", but not yet "speculation diverges systematically"\n(threats, 12).',
     'conclusions: losslessness section rewritten for C21 and C22'),
    ('paper/conclusions.md',
     "- **Hardware portability.** The project's original purpose was to show tau\n  transfers across machines while R does not. The vLLM build has no usable sm_80\n  kernels, so A100 fails outright and H200 cannot run. **Every number comes from\n  one GPU.** The decomposition is demonstrated across load and context, never\n  across hardware. This is the largest limitation and a rebuild would fix it.\n- **Confidence intervals.** Per-request timing is unavailable on this build, so\n  the planned paired bootstrap is impossible. One replication bounds run-to-run\n  variance below 1%, far under the effects claimed, but a bound is not an\n  interval.",
     "- **Hardware portability — partly answered, and not in the direction planned.**\n  The multi-architecture rebuild landed on 8 Oct, so A100 works and\n  `results/a100` holds 15 rows; H200 is blocked only by `phase0.sh`'s\n  `80|86` architecture guard. What the second platform showed is that the\n  original framing was wrong on both halves: **tau does not transfer** (21 of 25\n  sequences differ between machines at the same cell, even though the means\n  agree to 0.8%), and **R does not scale by one factor** (the rungs differ at\n  every bucket once acceptance is separated out). The hardware term is a result\n  rather than a side quest, but it is **one platform pair**, the 2K cell is\n  n=5, and a third point is needed before any of it is a trend.\n- **Confidence intervals.** Still none, and the reason has changed twice.\n  Per-request timing is not merely unavailable but **structurally absent**: the\n  harness asks the engine for `first_token_time`, `last_token_time` and\n  `finished_time`, this build populates none of them, so the planned paired\n  bootstrap cannot be computed at all. Nor can repeats supply one — at fixed\n  configuration the engine is bit-deterministic, so repeated whole runs measure\n  zero. The only unmeasured variance left is **prompt sampling**, and the route\n  to an interval is disjoint prompt subsets by document id.",
     'conclusions: hardware and interval limitations corrected'),
    ('paper/conclusions.md',
     '- **Answer quality.** Losslessness was assumed, so task accuracy was never\n  scored. Given the 0.65% token divergence, that assumption is weaker than\n  planned. LongBench-v2 is multiple choice, so this is cheap to fix.',
     "- **Answer quality — closed.** Scored on 13 Oct: no rung's accuracy differs\n  from the baseline by more than the baseline's own variation across\n  concurrency, nine arms identical at 8K, every other difference exactly one\n  prompt (C22). The caveat that remains is **n**: 24 to 49 prompts per bucket,\n  where one flip moves 32K accuracy by 4.2 points, so the invariance has no\n  interval and should not be quoted with one.",
     'conclusions: answer quality closed by C22'),
    ('TODO.md',
     '- **C20** — the machine factor is rung-independent at 2K (1.046 vs 1.056) and\n  diverges with context (1.070 vs 1.296 at 32K). `R_draft` measured from the\n  k-sweeps: DFlash 0.299 / 0.445 / 0.673, n-gram 0.071 / – / 0.139. The fixed\n  drafting term dominates `c·k` by 4–7× and grows twice as fast.\n  **Status: supported, not established** — the one point off the collapse is\n  the n=5 2K cell.',
     '- **C20** — there is no single machine factor. **Amended the same day**: the\n  factor as first registered is `(tau ratio) / (speedup ratio)`, so it carried\n  acceptance as well as cost. Tau-free the rungs differ at *every* bucket, and\n  at 2K by 4.7 points in the **reverse** order (n-gram 1.0905 vs DFlash\n  1.0437) — so "rung-independent at 2K" and the branch-B label are both\n  withdrawn. The conclusion survives and strengthens. `R_draft` from the\n  k-sweeps is unaffected: DFlash 0.299 / 0.445 / 0.673, n-gram 0.071 / – /\n  0.139, the fixed drafting term dominating `c·k` by 4–7× and growing twice as\n  fast. **Status: supported, not established** — the n=5 2K cell is off the\n  collapse, and the factor needs restating as two columns before any figure.\n  See `notes/2026-10-13-c20-amendment.md`.',
     'TODO: C20 summary amended'),
    ('TODO.md',
     '- **C22** — accuracy is invariant across rungs. Speculation changes tokens and\n  does not change answers.',
     '- **C22** — no accuracy difference exceeds the baseline\'s own variation.\n  Speculation changes tokens and does not change answers. Stated that way\n  rather than as "invariant": n is 24 to 49 per bucket and one flip moves 32K\n  by 4.2 points.',
     'TODO: C22 summary de-overclaimed'),
    ('TODO.md',
     '| Quality evaluation | **Done.** C22: accuracy is invariant. The 0.65% token divergence changes no answer. |',
     "| Quality evaluation | **Done.** C22: no difference exceeds the baseline's own variation. The token divergence — 0.556-0.781%, not the 0.65% figure, which came from the confounded C10 control — changes no answer. |",
     'TODO: quality row, stale 0.65% figure replaced'),
    ('paper/threats.md',
     'Divergence is front-loaded — 8 of the 21 first diverge within the first 16\ntokens — so a single hazard rate under-predicts agreement (10.1% implied against\n16.0% observed). Prefill numerics most likely differ and the first token follows.',
     'Divergence is front-loaded — 8 of the 21 first diverge within the first 16\ntokens — so a single hazard rate under-predicts agreement (10.1% implied against\n16.0% observed). **Stated as what is measured: two rates, not one mechanism.**\nThe position distribution is consistent with an elevated early hazard followed\nby a lower steady one, and "prefill numerics differ and the first token\nfollows" is one explanation among several; nothing here distinguishes it from a\nuniform hazard plus a short high-variance transient. The modelling consequence\nholds either way — a single rate is the wrong summary — and the mechanism is\nuntested.',
     'threat 9: front-loading restated as a two-rate observation'),
    ('paper/threats.md',
     "and 0.65%, which means the right statistic is the **per-token hazard** and on\nthat statistic speculation perturbs output *less* than the engine perturbs\nitself across batch sizes. That is C10's withdrawn direction, so the metric —\nnot only the control — needs replacing. The same-configuration C10 control is\nstill required before any null exists.",
     "and 0.65%, which means the right statistic is the **per-token hazard** rather\nthan sequence agreement.\n\n**Resolved 13 Oct, and it went further than this entry expected.** The\nsame-configuration control ran: the engine's own hazard at matched concurrency\nis **zero** (150/150 identical, bounded at 0.023%), so the null exists and\nthere is nothing for speculation to be below. The reading recorded here — that\non the per-token statistic speculation perturbs output less than the engine\nperturbs itself — is **withdrawn with the rest of C10's strong form**. Against\na same-job baseline at matched concurrency the speculative hazards are\n0.556–0.781%/tok, 24–34× the floor (C21). The metric point stands and is now\nthe more useful half: sequence agreement is a length artefact, the per-token\nhazard is the statistic, and C22 shows neither predicts task outcome.",
     'threat 10: resolved by C21, withdrawn direction removed'),
    ('paper/threats.md',
     '**16. Accuracy differs by bucket in a direction difficulty cannot explain.**\n32K scores 50.0% where 8K scores 34.7% on the same model and prompt style. A\nlonger context making the task easier is implausible; this is the\nbucket-document-overlap confound (threat 3) appearing in task accuracy, and it\nstrengthens the case for per-domain subset runs since domain composition\ndiffers by bucket.\n',
     "**16. Accuracy differs by bucket in a direction difficulty cannot explain.**\n32K scores 50.0% where 8K scores 34.7% on the same model and prompt style. A\nlonger context making the task easier is implausible; this is the\nbucket-document-overlap confound (threat 3) appearing in task accuracy, and it\nstrengthens the case for per-domain subset runs since domain composition\ndiffers by bucket.\n\n**17. The machine factor conflated cost with acceptance — resolved, with one\nthing still open.** `R` is defined here as `tau_eff / speedup`, so every ratio\nof `R` between machines carries the cross-machine acceptance ratio as well as\nthe cost ratio. C20 was registered on such ratios, which is why its 2K cells\nagreed: n-gram's tau fell 4.12% on A100 while DFlash's rose 1.14%, cancelling a\n4.7-point gap in the cost term. Removing that channel reverses the 2K ordering\n(n-gram 1.0905 against DFlash 1.0437) and widens the gap. The arithmetic is\nsettled and recorded in C20 and `notes/2026-10-13-c20-amendment.md`; two\nstatements were withdrawn and the conclusion strengthened.\n\n**What is still open is the figure.** The planned collapse plot puts the\nregistered factor on y and a within-machine `R_draft` fit on x, so y is partly\nacceptance and x is not. The tau-free factor is a counterfactual (it asks what\nthe factor would be if tau transferred, and threat 9 says it does not), so\nsubstituting it is not a fix either. Either plot two series — the speedup ratio\nand the tau ratio — or plot the counterfactual and label it as one. **Settle\nthis before drawing, not after.**\n",
     'threats: new entry 17, appended after 16'),
]

if not os.path.isdir(os.path.join(R, ".git")):
    sys.exit(f"ERROR: {R} is not a git repo. Set R=/path/to/repo")

texts, applied, skipped, fail = {}, [], [], []
for path, old, new, label in EDITS:
    full = os.path.join(R, path)
    if path not in texts:
        if not os.path.isfile(full):
            fail.append(f"  [missing file] {path}: {label}")
            continue
        texts[path] = open(full, encoding="utf-8").read()
    if new in texts[path]:
        skipped.append(f"  skip  {path}: {label} (already applied)")
        continue
    n = texts[path].count(old)
    if n != 1:
        fail.append(f"  [{n} matches, expected 1] {path}: {label}")
        continue
    texts[path] = texts[path].replace(old, new, 1)
    applied.append(f"  ok    {path}: {label}")

print("\n".join(applied + skipped) or "  nothing matched")

if fail:
    print("\nANCHOR FAILURE -- nothing written:")
    print("\n".join(fail))
    print("\nThese files are not what this script was written against, most")
    print("likely because they moved on after 13 Oct. Re-derive the anchors")
    print("from the files rather than loosening them.")
    sys.exit(2)

if not applied:
    print("\nAll 20 edits already present. Nothing to do.")
    sys.exit(0)

if not WRITE:
    print(f"\nDRY RUN: {len(applied)} edits would apply, "
          f"{len(skipped)} already present. Re-run with --write.")
    sys.exit(0)

for path, text in texts.items():
    open(os.path.join(R, path), "w", encoding="utf-8").write(text)
print(f"\nwrote {len(texts)} files")
print(subprocess.run(["git", "-C", R, "diff", "--stat"],
                     capture_output=True, text=True).stdout)

MSG = """paper: propagate C20's amendment, C21 and C22 into the six stale files

LOG.md and TODO.md were updated on 13 Oct; the files the paper is actually
written from were not. Nine commits landed while HANDOFF.md, paper/claims.md,
paper/conclusions.md, paper/results-table.md and paper/threats.md went on
asserting things the same day's measurements had refuted. This closes that gap,
which the register's own propagation rule exists to prevent.

The substantive change is C20's amendment. R is defined in this project by the
identity R = tau_eff / speedup, so a ratio of R between machines carries the
cross-machine acceptance ratio as well as the cost ratio -- the same channel
C13's 11 Oct correction said must not be dropped, reappearing on the other side
of the equation. Removing it reverses the 2K ordering: n-gram 1.0905 against
DFlash 1.0437, 4.7 points apart rather than 1.0 and in the opposite direction.
So "at 2K the two rungs are indistinguishable" is withdrawn, and so is the
branch-B label, because tau-free the rungs differ at every bucket and the
pre-registered diagnostic list contains no branch for that. C20's conclusion
survives and strengthens: there is no context at which one scalar relates A40
cost to A100 cost across rungs.

The fix is two columns, not a different scalar. The tau-free factor is itself a
counterfactual -- it asks what the factor would be if tau transferred, and
threat 9 says it does not. New threat 17 records that the collapse figure has
to be settled before it is drawn, since as registered its y-axis is partly
acceptance while its x-axis is a pure within-machine cost fit. On the tau-free
column the collapse is worse, not better: monotonicity fails at two points
rather than one.

Also propagated. conclusions.md's losslessness section still said the outputs
agree on 12.7% of sequences and that speculation does not perturb the greedy
path beyond the serving system's reproducibility floor; C21 measured that floor
at zero (150/150 identical across two nodes) and speculation at 24-34x it, so
that sentence is withdrawn and the section is rewritten in two halves -- not
lossless at the bit level, lossless at the task level, with sequence identity
shown not to predict task outcome. Two of its five "cannot claim" items close
(hardware portability, answer quality) and the interval item's reason is
corrected: per-request timing is not merely unavailable but structurally
absent, and repeats cannot supply an interval because the engine is
deterministic, which leaves prompt sampling as the only unmeasured term.
HANDOFF's frozen prediction is no longer framed as pending work -- it ran,
scored 2 of 4 cells within 5% with a median 6.18% error, every cell optimistic,
and the misses split by rung, which is where C20 came from.

claims.md is append-only by its own header rule, so the withdrawn statements
are kept in a blockquote beneath the amendment rather than deleted.

Mechanically: anchor-based patching, 20 anchors each asserted to match exactly
once, nothing written on any failure, and already-applied edits skipped so a
second run is safe."""

if not DOCOMMIT:
    print("Review the diff, then re-run with DOCOMMIT=1 to commit, or commit")
    print("by hand. The prepared message is in this script as MSG.")
    sys.exit(0)

subprocess.run(["git", "-C", R, "add", "-A"], check=True)
subprocess.run(["git", "-C", R,
                "-c", "user.name=Devansh Agrawal",
                "-c", "user.email=dagraw2@illinois.edu",
                "commit", "-m", MSG], check=True)
print(subprocess.run(["git", "-C", R, "log", "-1",
                      "--format=%h | %an <%ae> | %s"],
                     capture_output=True, text=True).stdout)
print(f"Push with:  git -C {R} push origin HEAD")
