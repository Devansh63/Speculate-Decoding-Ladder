#!/usr/bin/env python3
"""
Block 0, step 4 -- re-derive the KV-pool numbers from the engine logs so C19
can be rewritten from records.

What C19 should say after this runs:

  "The drafter's weights are charged to the KV pool." Not a third cost axis, not
  a tax. vLLM profiles memory with the models loaded and gives the remainder to
  the KV cache, so a second checkpoint comes straight out of the pool. n-gram
  has no model and costs nothing; a trained head costs its weights, and the
  consequence -- fewer concurrent long-context requests -- is the part worth
  publishing.

And what it should NOT say: a figure to three significant digits. If the
within-rung spread this script reports is the same order as the between-rung
difference, the difference is quotable only as "about 2 GiB".

vLLM prints the numbers we need at engine init, before each arm runs:
  "Available KV cache memory: 19.68 GiB"
  "GPU KV cache size: 123,456 tokens"
  "Maximum concurrency for 33,792 tokens per request: 4.76x"
and the harness prints "=== <run_id> ===" after that arm finishes. So init
lines are attributed forward to the next run_id marker.

Usage
-----
  python3 b0_04_kv_extract.py /projects/bikd/dagraw2/Speculative_Ladder/logs/phase0-*.out
  python3 b0_04_kv_extract.py results/logs/a100/*.gz
"""

import gzip
import io
import os
import re
import sys
from collections import defaultdict

PATTERNS = {
    "kv_gib":   re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB"),
    "kv_tok":   re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens"),
    "maxconc":  re.compile(r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x"),
    "blocks":   re.compile(r"#\s*GPU blocks:\s*([0-9,]+)"),
}
RUNID = re.compile(r"^===\s+(\S+)\s+===\s*$")


def opener(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), errors="replace")
    return open(path, errors="replace")


def parse(path):
    """Yield (run_id, {field: value}) in file order."""
    pending = {}
    with opener(path) as fh:
        for line in fh:
            for field, pat in PATTERNS.items():
                m = pat.search(line)
                if m:
                    if field == "maxconc":
                        pending["ctx_tokens"] = int(m.group(1).replace(",", ""))
                        pending["maxconc"] = float(m.group(2))
                    else:
                        v = m.group(1).replace(",", "")
                        pending[field] = float(v) if "." in v else int(v)
            m = RUNID.match(line.strip())
            if m:
                if pending:
                    yield m.group(1), dict(pending)
                else:
                    yield m.group(1), {}
                pending = {}
    if pending:
        yield "(unattributed: engine init with no following run)", pending


def rung_of(run_id):
    return run_id.split("_", 1)[0] if "_" in run_id else run_id


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
        n = 0
        for run_id, d in parse(p):
            if d:
                rows.append(dict(job=job, run_id=run_id, **d))
                n += 1
        print(f"  {os.path.basename(p)}: {n} arm(s) with KV lines", file=sys.stderr)

    if not rows:
        print("\nSTOP: no KV lines found in any log.")
        print("Either these logs predate the lines, or vLLM's wording differs in")
        print("this build. Check one log by hand before concluding anything:")
        print("  grep -iE 'kv cache|concurrency|GPU blocks' <a log> | head -20")
        sys.exit(2)

    print(f"\n{'job':>9} {'run_id':>34} {'KV GiB':>8} {'KV tok':>10} "
          f"{'ctx':>7} {'maxconc':>8}")
    for r in sorted(rows, key=lambda r: (r["run_id"], r["job"])):
        print(f"{r['job']:>9} {r['run_id']:>34} "
              f"{r.get('kv_gib', float('nan')):>8.2f} "
              f"{r.get('kv_tok', 0):>10} {r.get('ctx_tokens', 0):>7} "
              f"{r.get('maxconc', float('nan')):>8.2f}")

    # ---- the only comparison that matters: spread WITHIN a rung vs BETWEEN
    by_rung = defaultdict(list)
    for r in rows:
        if "kv_gib" in r:
            by_rung[rung_of(r["run_id"])].append(r["kv_gib"])

    print(f"\n{'rung':>10} {'n':>3} {'min GiB':>9} {'max GiB':>9} "
          f"{'spread':>8} {'mean':>8}")
    stats = {}
    for rung, vs in sorted(by_rung.items()):
        spread = max(vs) - min(vs)
        stats[rung] = (sum(vs) / len(vs), spread)
        print(f"{rung:>10} {len(vs):>3} {min(vs):>9.2f} {max(vs):>9.2f} "
              f"{spread:>8.2f} {sum(vs) / len(vs):>8.2f}")

    if "off" in stats:
        base, _ = stats["off"]
        print("\n  between-rung differences, against 'off':")
        for rung, (mean, _) in sorted(stats.items()):
            if rung == "off":
                continue
            print(f"    {rung:>10}: {base - mean:+.2f} GiB")

        # The spread can only be estimated from rungs measured more than once.
        # With one run per rung the spread is 0.00 by construction, and
        # comparing against it would certify pure noise as significant.
        repeated = {r: s for r, (m, s) in stats.items()
                    if len([v for v in by_rung[r]]) >= 2}
        FLOOR = 0.25  # GiB; below this, treat as indistinguishable regardless

        print("\n  Reading it:")
        if not repeated:
            print("    Within-rung spread is UNESTIMATED -- every rung has one run"
                  "\n    in this data, so the spread is 0.00 by construction and no"
                  "\n    significance call is possible. Report the direction only.")
            print("    To estimate it, parse several jobs that ran the same rung,"
                  "\n    or pass more logs to this script.")
        else:
            worst = max(repeated.values())
            thresh = max(2 * worst, FLOOR)
            print(f"    largest within-rung spread (from {len(repeated)} repeated "
                  f"rung(s)): {worst:.2f} GiB")
            print(f"    threshold for a quotable difference: {thresh:.2f} GiB")
            sig = [r for r, (m, _) in stats.items()
                   if r != "off" and abs(base - m) > thresh]
            ins = [r for r, (m, _) in stats.items()
                   if r != "off" and abs(base - m) <= thresh]
            if sig:
                print(f"    SEPARABLE: {', '.join(sorted(sig))} -- quotable as a "
                      f"direction and a\n               rough magnitude, not a "
                      f"three-digit figure.")
            if ins:
                print(f"    NOT SEPARABLE: {', '.join(sorted(ins))} -- within noise. "
                      f"Report as\n                   'no measurable cost', with no "
                      f"figure at all.")
        print("    Either way: do not quote a GiB figure to three significant")
        print("    digits when the spread is the same order as the difference.")
        print("    The publishable consequence is the concurrency ceiling in the")
        print("    'maxconc' column, which is a property of the RUNG, not of")
        print("    machine-and-bucket alone.")


if __name__ == "__main__":
    main()
