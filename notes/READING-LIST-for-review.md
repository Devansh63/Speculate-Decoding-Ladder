# Reading list for an external evaluator

What to read, in what order, and what to skip. Written for an AI agent or a
human reviewer with no prior context on the project.

Repo: https://github.com/Devansh63/Speculate-Decoding-Ladder

Raw URLs are given because an agent with web access can fetch them directly
without navigating GitHub's HTML.

```
RAW = https://raw.githubusercontent.com/Devansh63/Speculate-Decoding-Ladder/main
```

---

## Tier 1: the argument. Read all of these.

| File | What it is |
|---|---|
| `notes/PROJECT-BRIEF-for-review.md` | **Start here.** Self-contained statement of the question, the metric framework, what was measured, the findings, the known weaknesses, and the specific questions we want answered. |
| `notes/2026-10-01-a40-2k-concurrency-sweep.md` | The 2K cell with full derivations. Source of tau-invariance, the coverage correction, and the concurrency crossings. |
| `notes/2026-10-06-8k-context-decay.md` | The 8K cell. **The most important result.** The two rungs decay in opposite directions along the context axis, and the ranking is on course to invert. |
| `notes/2026-10-05-losslessness-and-engine-nondeterminism.md` | Output stability. Why an apparent catastrophe was an engine property, and what may honestly be claimed. |
| `paper/claims.md` | What the project intends to argue. **Challenge this hardest.** |
| `paper/threats.md` | Threats to validity as we see them. Tell us what is missing. |

```
$RAW/notes/PROJECT-BRIEF-for-review.md
$RAW/notes/2026-10-01-a40-2k-concurrency-sweep.md
$RAW/notes/2026-10-06-8k-context-decay.md
$RAW/notes/2026-10-05-losslessness-and-engine-nondeterminism.md
$RAW/paper/claims.md
$RAW/paper/threats.md
```

## Tier 2: can the numbers be trusted?

Where a measurement artifact would hide. The questions: is coverage
reconstructed soundly from aggregate counters, is speedup computed against the
right baseline, is acceptance compared across rungs fairly.

| File | What to check |
|---|---|
| `code/run_benchmark.py` | How the spec-decode counters are read and differenced; whether warmup is excluded; what the runner pin does. |
| `code/summarize.py` | The coverage reconstruction in `derive()`, the matched-concurrency baseline, the common-k truncation, the implied-R calculation. |
| `code/check_lossless.py` | Whether the baseline-against-itself control is a valid control. |
| `results/a40/summary.txt` | Raw tool output behind every number quoted. |
| `results/a40/results.jsonl` | The underlying records, one per arm. |

```
$RAW/code/run_benchmark.py
$RAW/code/summarize.py
$RAW/code/check_lossless.py
$RAW/results/a40/summary.txt
$RAW/results/a40/results.jsonl
```

## Tier 3: context and engineering.

| File | What it explains |
|---|---|
| `env/a100-sm80-failure.md` | Why the A100 half of the grid is missing, how it was diagnosed, and the rebuild decision pending. |
| `env/gpu-support.md` | Which GPUs the build covers, and the KV-capacity arithmetic that sets the concurrency grid per bucket. |
| `env/delta.md` | Cluster, allocation and billing facts. Explains the budget constraint. |
| `paper/related-work.md` | What we believe the related work is. Likely incomplete. |
| `paper/figures.md` | Planned figures. |
| `notes/findings.md` | Running log, partly superseded by the dated notes. |
| `README.md` | Repo orientation. |

## Tier 4: skip these.

Operational, no bearing on whether the science is sound.

- `notes/teammate-quickstart.md`, `notes/github-from-delta.md`,
  `notes/lab-notebook-template.md`
- `code/phase0_a40.sh`, `code/phase0_a100.sh`, `code/push_results.sh`,
  `code/share_with_team.sh`, `code/build_workload.py`, `code/diag_a100.sh`
- `results/logs/*`

One exception: `code/phase0.sh` is worth a glance to confirm that every arm
within a sweep sees identical prompts, seed and engine configuration, which is
load-bearing for the whole comparison.

---

## Not in this repo

| Document | Why it matters |
|---|---|
| **`CS598_Proposal_SpeculationLadder_v9_revised.pdf`** | The approved course proposal. Defines committed scope, the must-do versus stretch split, the early milestone and its numeric gate. **Attach it if you want an opinion on whether the project is on track against what was promised**, as opposed to whether the work is good in the abstract. |
| `speculation-ladder-plan-REALIGNED-partial.md` | Long implementation plan, mostly superseded by events. Probably not worth an evaluator's time. |
| `speculation-ladder-HANDOFF.md` | Context transfer for a different assistant session. The brief is newer and better. |

## How to read the brief

Section 7 lists the weaknesses we already know about. Confirming them back has
little value. The useful output is:

- a weakness we did **not** list
- a finding that is actually a measurement artifact
- a claim in `paper/claims.md` the data does not support
- prior work that already establishes something presented as new, especially
  the coverage correction, the opposite-direction context decay, and the
  non-determinism result
- a sharper version of the central claim

Disagreement is the deliverable. An evaluation that agrees throughout tells us
nothing we did not already believe.
