# Related work

What each neighbour does, and what it leaves open. The claim that survives all of
them: choosing among drafting mechanisms of different kinds, inside a serving
runtime at concurrency, from a fitted model of context and memory, with the
crossing point predicted on hardware not used for fitting.

| Work | What it does | What it leaves |
|---|---|---|
| MagicDec, arXiv 2408.11049 | Cost against context with a bounded drafter at batch 32 to 256. Measures acceptance (0.84 to 0.79 from 4K to 100K on PG-19) and notes the hardware direction | Never predicts where the inflection lands. One drafter class. PG-19 continuation is near best case for a sliding-window drafter |
| SmartSpec, arXiv 2406.14066, in vLLM | Per-request speculation length from 0 upward chosen by goodput. Linear cost model in context and batch | One method at a time. Acceptance from a moving average, no context model. Says outright that its coefficients must be re-profiled per hardware, which is the gap we fill |
| AdaServe, arXiv 2501.12162, EuroSys 2026 | Per-request SLO-customized draft trees under a roofline token budget. 4.3x fewer SLO violations, 1.9x goodput. Scheduler costs 0.31 to 0.41 percent CPU | One draft model throughout. Context never varied. Single platform, profiled not predicted. Its Figure 12, accepted tokens falling as request rate rises, is independent evidence for our concurrency axis |
| AdaSpec / SpecServe, arXiv 2503.05096 | Adaptive speculative length plus a confidence-prior verifier under SLOs | One drafter, per-hardware calibration |
| SAM-Decoding, ACL 2025 | Per-step switching between a suffix-automaton draft and EAGLE-2 by match length | Local signal, batch 1, no serving concurrency, no hardware model |
| Nightjar, DSpark, BanditSpec, SpecRouter, FASER, TurboSpec | Draft length or routing within one mechanism class, or across model sizes, from reactive signals | No context model, no memory model, no cross-hardware prediction |
| Sequoia | Adjusts tree size to hardware specifications | Read directly before claiming novelty on the hardware side. Nearest thing to hardware-aware configuration, but optimizes for a measured platform rather than predicting an unmeasured one |

**Gaps in the submitted proposal's reference list:** SmartSpec, AdaServe and
AdaSpec are not cited. They are the closest serving-side neighbours. Add them.

**Refresh protocol:** re-search before each milestone with at least four
phrasings. On this project more than twelve ideas looked novel under one phrasing
and turned out published under another. Two September 2026 papers, SwitchSD
(2609.20186) and ASPIRE (2609.17943), also narrow the claim and need reading.
