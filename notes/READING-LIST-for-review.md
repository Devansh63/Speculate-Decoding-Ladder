# Reading list for an external evaluator

What to read, in what order, and what to skip. Written for an AI agent or a
human reviewer with no prior context on the project.

Repo: https://github.com/Devansh63/Speculate-Decoding-Ladder

Raw URLs are given because an agent with web access can fetch them directly
without navigating GitHub's HTML. Replace nothing; these are literal.

```
RAW = https://raw.githubusercontent.com/Devansh63/Speculate-Decoding-Ladder/main
```

---

## Tier 1: the argument. Read all of these.

About 9,000 words total. Everything the evaluation depends on is here.

| File | What it is |
|---|---|
| `notes/PROJECT-BRIEF-for-review.md` | **Start here.** Self-contained statement of the question, the metric framework, what was measured, six findings, nine known weaknesses, and the specific questions we want answered. |
| `notes/2026-10-01-a40-2k-concurrency-sweep.md` | The only complete measurement cell, with full derivations. Where tau-invariance, the coverage correction and the crossing points come from. |
| `notes/2026-10-05-losslessness-and-engine-nondeterminism.md` | The output-stability investigation. Why an apparent catastrophe turned out to be an engine property, and what we think may be claimed as a result. |
| `paper/claims.md` | What the project intends to argue. **Challenge this hardest**, it asserts more than has been shown. |
| `paper/threats.md` | Threats to validity as we currently see them. Tell us what is missing from the list. |

```
$RAW/notes/PROJECT-BRIEF-for-review.md
$RAW/notes/2026-10-01-a40-2k-concurrency-sweep.md
$RAW/notes/2026-10-05-losslessness-and-engine-nondeterminism.md
$RAW/paper/claims.md
$RAW/paper/threats.md
```

## Tier 2: can the numbers be trusted? Read if evaluating methodology.

This is where a measurement artifact would hide. The central questions are
whether coverage is reconstructed soundly from aggregate counters, whether
speedup is computed against the right baseline, and whether acceptance is
compared across rungs fairly.

| File | What to check |
|---|---|
| `code/run_benchmark.py` | How the spec-decode counters are read and differenced. Whether the warmup is correctly excluded. Whether the runner pin does what the comments claim. |
| `code/summarize.py` | The coverage reconstruction (`derive()`), the matched-concurrency baseline, the common-k truncation, the implied-R calculation. |
| `code/check_lossless.py` | Whether the baseline-against-itself control is a valid control. |
| `results/a40/summary-2026-10-01.txt` | Raw tool output behind every number in the brief, including the draft counters. |
| `results/a40/results.jsonl` | The underlying records, one per arm. |

```
$RAW/code/run_benchmark.py
$RAW/code/summarize.py
$RAW/code/check_lossless.py
$RAW/results/a40/summary-2026-10-01.txt
$RAW/results/a40/results.jsonl
```

## Tier 3: context. Read if something in tier 1 seems unmotivated.

| File | What it explains |
|---|---|
| `env/gpu-support.md` | Why only A40 and A100 are usable, how that was established with `cuobjdump`, and the KV-capacity arithmetic that sets the concurrency grid per context bucket. |
| `env/delta.md` | Cluster, allocation and billing facts. Explains the budget constraint. |
| `paper/related-work.md` | What we believe the related work is. Likely incomplete; corrections wanted. |
| `paper/figures.md` | Planned figures. Useful for judging whether the evidence supports the intended presentation. |
| `notes/findings.md` | Running log, partly superseded by the two dated notes. |
| `README.md` | Repo orientation. |

```
$RAW/env/gpu-support.md
$RAW/env/delta.md
$RAW/paper/related-work.md
$RAW/paper/figures.md
$RAW/notes/findings.md
$RAW/README.md
```

## Tier 4: skip these.

Operational, with no bearing on whether the science is sound. Reading them
dilutes attention.

- `notes/teammate-quickstart.md`, `notes/github-from-delta.md`,
  `notes/lab-notebook-template.md`
- `code/phase0.sh`, `code/phase0_a40.sh`, `code/phase0_a100.sh`,
  `code/push_results.sh`, `code/share_with_team.sh`, `code/build_workload.py`
- `results/logs/*`

One exception: `code/phase0.sh` is worth a glance if you want to confirm that
every arm within a sweep really does see identical prompts, seed and engine
configuration, since that is load-bearing for the whole comparison.

---

## Not in this repo

These exist but are not tracked here and must be supplied separately if the
evaluation needs them.

| Document | Why it matters |
|---|---|
| **`CS598_Proposal_SpeculationLadder_v9_revised.pdf`** | The approved course proposal, revised after instructor feedback. Defines the committed scope, the must-do versus stretch split, the early milestone and its numeric gate, and the evaluation plan. **Attach this if you want an opinion on whether the project is on track against what was promised**, as opposed to whether the work is good in the abstract. |
| `speculation-ladder-plan-REALIGNED-partial.md` | Long implementation plan, roughly 128,000 words, day by day. Mostly superseded by what actually happened. Probably not worth an evaluator's time. |
| `speculation-ladder-HANDOFF.md` | Context transfer document written for a different assistant session. Overlaps heavily with the project brief; the brief is newer and better. |
| Slurm logs | Committed under `results/logs/` as gzip. Provenance only. |

## A note on how to read the brief

Section 7 of `PROJECT-BRIEF-for-review.md` lists nine weaknesses we already
know about. Confirming them back to us has little value. The useful output is:

- a weakness we did **not** list
- a finding that is actually a measurement artifact
- a claim in `paper/claims.md` that the data does not support
- prior work that already establishes something we present as new, in
  particular the coverage correction in Finding 2 and the non-determinism
  result in Finding 6
- a sharper version of the central claim

Disagreement is the deliverable. An evaluation that agrees with the brief
throughout tells us nothing we did not already believe.
