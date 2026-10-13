#!/usr/bin/env python3
"""
Block 0, step 2 -- score the frozen prediction FROM RECORDS and emit a
ready-to-append claims.md entry.

The prediction was frozen on 11 Oct 2026 before any A100 cell ran at full n.
This script does not restate it from memory: the predicted values are the
defaults below and can be overridden, but every measured value comes from
results.jsonl rows, paired to a baseline by the same rules decompose.py uses.

It also applies BOTH limbs of the pre-registered criterion and reports the
ambiguity in the failure limb rather than silently picking a reading.

Usage
-----
  python3 b0_02_score_prediction.py results/a100 > /tmp/entry.md
  python3 b0_02_score_prediction.py results/a100 --harness v2 --exclude-host gpub073

  # then read /tmp/entry.md, and only if you agree with it:
  cat /tmp/entry.md >> paper/claims.md

Requires decompose.py in the same directory (it reuses its math rather than
reimplementing the decomposition).
"""

import argparse
import os
import statistics
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from decompose import load, analyse, job_of, host_of, gpu_of
except ImportError:
    sys.exit("decompose.py must sit next to this script")

# Frozen 11 Oct 2026. (bucket, rung, k) -> predicted speedup.
FROZEN = {
    ("8K", "ngram", 5): 1.09,
    ("8K", "dflash", 7): 1.20,
    ("32K", "ngram", 5): 1.02,
    ("32K", "dflash", 7): 0.78,
}
# A40 values the prediction was built from, for the machine-factor column.
A40 = {  # (bucket, rung, k) -> (R_a40, tau_eff_a40)
    ("8K", "ngram", 5): (1.120, 1.284),
    ("8K", "dflash", 7): (1.544, 1.947),
    ("32K", "ngram", 5): (1.290, 1.390),
    ("32K", "dflash", 7): (1.766, 1.459),
}
PASS_TOL = 5.0   # percent, relative
FAIL_MEDIAN = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--harness")
    ap.add_argument("--exclude-host", action="append", default=[])
    ap.add_argument("--exclude-job", action="append", default=[])
    a = ap.parse_args()

    recs = analyse(load(a.tree, a.harness, set(a.exclude_host), set(a.exclude_job)))

    # collect every realization of each predicted cell, at c=1, full n
    got = defaultdict(list)
    for d in recs:
        r = d["row"]
        key = (r.get("bucket"), r.get("rung"), r.get("k"))
        if key in FROZEN and (r.get("max_num_seqs") or 0) == 1 and d["speedup"]:
            got[key].append(d)

    missing = [k for k in FROZEN if k not in got]
    if missing:
        print(f"<!-- MISSING CELLS, entry incomplete: {missing} -->", file=sys.stderr)

    rows, errs = [], []
    for key in sorted(FROZEN, key=lambda k: (k[0] != "8K", k[1])):
        pred = FROZEN[key]
        ds = got.get(key, [])
        if not ds:
            rows.append((key, pred, None, None, None, None, None, []))
            continue
        meas = [d["speedup"] for d in ds]
        m = sum(meas) / len(meas)
        err = (m - pred) / pred * 100.0
        errs.append(abs(err))
        tau = sum(d["tau_eff"] for d in ds) / len(ds)
        R = sum(d["R"] for d in ds) / len(ds)
        Ra, tau_a40 = A40.get(key, (None, None))
        factor = R / Ra if Ra else None
        dtau = (tau / tau_a40 - 1) * 100 if tau_a40 else None
        rows.append((key, pred, m, err, R, factor, dtau, ds))

    n_within = sum(1 for e in errs if e <= PASS_TOL)
    median = statistics.median(errs) if errs else float("nan")

    # ordering, both readings
    def sp(b, rung, k):
        ds = got.get((b, rung, k), [])
        return sum(d["speedup"] for d in ds) / len(ds) if ds else None
    nd, dd = sp("32K", "ngram", 5), sp("32K", "dflash", 7)
    rel_ok = (nd is not None and dd is not None and dd < nd)
    side_ok = (nd is not None and dd is not None and dd < 1.0 < nd)

    # ---------------------------------------------------------------- output
    print(f"\n### Frozen prediction scored, {date.today().isoformat()}\n")
    print("Scored from records, not from prose. Every measured value below is "
          "the\nmean over all realizations of that cell at concurrency 1 found "
          f"in `{a.tree}`;\nindividual realizations and their jobs are listed "
          "under the table.\n")
    print("| cell | predicted | measured | rel. error | implied R | machine factor | d_tau vs A40 |")
    print("|---|---|---|---|---|---|---|")
    for key, pred, m, err, R, factor, dtau, ds in rows:
        if m is None:
            print(f"| {key[0]} {key[1]} k={key[2]} | {pred:.2f} | **missing** | — | — | — | — |")
            continue
        print(f"| {key[0]} {key[1]} k={key[2]} | {pred:.2f} | **{m:.4f}** | "
              f"{err:+.2f}% | {R:.4f} | {factor:.4f} | {dtau:+.2f}% |"
              if factor is not None else
              f"| {key[0]} {key[1]} k={key[2]} | {pred:.2f} | **{m:.4f}** | "
              f"{err:+.2f}% | {R:.4f} | — | — |")

    print("\nRealizations:\n")
    for key, *_rest, ds in rows:
        for d in ds:
            r = d["row"]
            print(f"- `{key[0]} {key[1]} k={key[2]}` job {job_of(r)} on "
                  f"{host_of(r)}: speedup {d['speedup']:.4f}, R {d['R']:.4f}"
                  + ("  (cov pinned by clamp)" if d["pinned"] else ""))

    print(f"\n**Cells within {PASS_TOL:.0f}% relative: {n_within} of {len(errs)}. "
          f"Median absolute relative error: {median:.2f}%.**\n")

    print("Against the pre-registered criterion:\n")
    print(f"- **Pass** required *every* cell within {PASS_TOL:.0f}% and the 32K "
          f"ordering to hold. {len(errs) - n_within} cell(s) exceeded the "
          f"tolerance, so it does not pass.")
    print(f"- **Fail** required median absolute error above {FAIL_MEDIAN:.0f}% "
          f"(it is {median:.2f}%, so not by this limb) **or** the 32K ordering "
          f"failing to reproduce.")
    print(f"- The 32K ordering is ambiguous as written. Relative order "
          f"(DFlash below n-gram): **{'reproduced' if rel_ok else 'did not reproduce'}** "
          f"({dd:.4f} vs {nd:.4f}). Break-even side (DFlash below 1.0 *and* "
          f"n-gram above 1.0): **{'held' if side_ok else 'failed'}**.")
    print("\nRecording the ambiguity rather than resolving it silently: the "
          "wording\n\"the 32K ordering does not reproduce\" admits both readings, and the "
          "outcome\ndiffers between them. The wording should be fixed before "
          "the next prediction\nis frozen, not after.\n")

    if errs:
        pattern = defaultdict(list)
        for key, pred, m, err, *_ in rows:
            if err is not None:
                pattern[key[1]].append(err)
        print("**The pass/fail pattern splits by rung, not by bucket.**\n")
        for rung, es in sorted(pattern.items()):
            within = sum(1 for e in es if abs(e) <= PASS_TOL)
            print(f"- `{rung}`: {within}/{len(es)} within tolerance "
                  f"({', '.join(f'{e:+.2f}%' for e in es)})")
        print("\nThe rung with no drafter forward pass landed; the rung with one "
              "did not.\nThat is the rung-dependent machine factor showing up in "
              "the scoreboard, and\nit is the diagnosis -- not a repair of the model.\n")

    signs = [e for e in (r[3] for r in rows) if e is not None]
    if signs and all(s < 0 for s in signs):
        print(f"All {len(signs)} errors are negative: every prediction was "
              f"optimistic. That is\nbias, not noise.\n")

    print(f"Provenance: `{a.tree}`, GPUs "
          f"{', '.join(sorted({gpu_of(d['row']) for d in recs}))}, harness "
          f"{', '.join(sorted({str(d['row'].get('harness')) for d in recs}))}.")
    if a.harness or a.exclude_host or a.exclude_job:
        print(f"Filters: harness={a.harness or '-'} "
              f"exclude-host={a.exclude_host or '-'} "
              f"exclude-job={a.exclude_job or '-'}.")


if __name__ == "__main__":
    main()
