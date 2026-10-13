#!/usr/bin/env python3
"""
Block 1 item 8, the missing half -- extract ACHIEVED concurrency from the
engine logs. Zero GPU.

The harness records what max_num_seqs was REQUESTED, and v3 adds what the
engine RESOLVED. Neither says how many requests actually ran at once.

phase0.sh's own header warns about this: "vLLM accepts --max-num-seqs 32 and
then schedules fewer sequences than asked, so the row gets labelled with a
concurrency it never ran at. That is worse than a crash." At 32K a sequence
needs about a fifth of the A40 cache, so a c=32 row cannot have run at 32 --
it queued.

vLLM's logger prints, roughly once a second:
  Avg prompt throughput: ..., Running: 3 reqs, Waiting: 7 reqs, ...

This reads those, attributes them to the arm that follows (the harness prints
"=== <run_id> ===" after each arm finishes), and reports the max and the
generation-weighted mean of Running alongside the requested concurrency. A row
whose achieved concurrency was never read should not appear in a figure that
has concurrency on an axis.

Usage
-----
  python3 b2_02_achieved_concurrency.py LOGFILE [LOGFILE ...]
  python3 b2_02_achieved_concurrency.py /projects/.../logs/phase0-*.out
"""

import gzip
import io
import os
import re
import sys
from collections import defaultdict

RUNNING = re.compile(r"Running:\s*(\d+)\s*reqs?.*?Waiting:\s*(\d+)\s*reqs?")
RUNID = re.compile(r"^===\s+(\S+)\s+===\s*$")
# "_c12" at the end of a run_id, before any tag suffix
REQ_C = re.compile(r"_c(\d+)")


def opener(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), errors="replace")
    return open(path, errors="replace")


def parse(path):
    """Yield (run_id, samples) where samples is a list of (running, waiting)."""
    pending = []
    with opener(path) as fh:
        for line in fh:
            m = RUNNING.search(line)
            if m:
                pending.append((int(m.group(1)), int(m.group(2))))
                continue
            m = RUNID.match(line.strip())
            if m:
                yield m.group(1), pending
                pending = []
    if pending:
        yield "(trailing samples, no run_id followed)", pending


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)

    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  skip (missing): {p}", file=sys.stderr)
            continue
        job = re.search(r"phase0-(\d+)", os.path.basename(p))
        job = job.group(1) if job else "?"
        for run_id, samples in parse(p):
            req = REQ_C.search(run_id)
            rows.append(dict(
                job=job, run_id=run_id,
                requested=int(req.group(1)) if req else None,
                n=len(samples),
                mx=max((s[0] for s in samples), default=None),
                mean=(sum(s[0] for s in samples) / len(samples)) if samples else None,
                waitmax=max((s[1] for s in samples), default=None),
            ))

    if not rows:
        sys.exit("\nno 'Running: N reqs' lines found in any log.\n"
                 "Either these logs predate the engine logger line, or "
                 "disable_log_stats was on.")

    print(f"\n{'job':>9} {'run_id':>36} {'req':>4} {'max':>4} {'mean':>6} "
          f"{'wait':>5} {'samples':>8} {'verdict':>22}")
    bad = 0
    for r in sorted(rows, key=lambda r: (r["run_id"], r["job"])):
        if r["n"] == 0:
            verdict = "NO SAMPLES"
        elif r["requested"] is None:
            verdict = "no _c in run_id"
        elif r["requested"] == 0:
            verdict = "engine default"
        elif r["mx"] is None:
            verdict = "?"
        elif r["mx"] < r["requested"]:
            verdict = f"NEVER REACHED ({r['mx']}<{r['requested']})"
            bad += 1
        else:
            verdict = "reached"
        mean_txt = "--" if r["mean"] is None else "%.2f" % r["mean"]
        print("%9s %36s %4s %4s %6s %5s %8d %22s" % (
            r["job"], r["run_id"], r["requested"], r["mx"], mean_txt,
            r["waitmax"], r["n"], verdict))

    print(f"\n{len(rows)} arm(s) parsed.")
    if bad:
        print(f"{bad} arm(s) NEVER REACHED their requested concurrency. Those "
              f"rows are\nlabelled with a concurrency they did not run at. They "
              f"must not appear on a\nconcurrency axis without the achieved "
              f"value shown instead.")
    else:
        print("Every arm with an explicit concurrency reached it at least once.")
    print("\nNote: 'max' is the peak of a ~1 Hz sample, so it is a lower bound on "
          "what\nwas momentarily resident; 'mean' is the better figure for a "
          "sustained rate.\n'wait' is the peak queue depth -- a large wait beside "
          "a small max is the\nsignature of asking for more concurrency than the "
          "KV cache can hold.")


if __name__ == "__main__":
    main()
