#!/usr/bin/env python3
"""Read results.jsonl and print the Phase 0 table.

Speedup is computed against the 'off' arm in the same bucket. Acceptance is
also reported truncated to a common draft length, because a rung running at
k=15 has more room to lose tokens than one at k=5 and the raw numbers are
not comparable.

    python3 summarize.py /projects/bikd/$USER/Speculative_Ladder/results/a40
"""

import json
import sys
from collections import defaultdict


def al_at_k(acc_by_pos, k):
    """Mean accepted length if the drafter had stopped at k positions."""
    if not acc_by_pos:
        return None
    return 1.0 + sum(acc_by_pos[:k])


def main():
    path = sys.argv[1].rstrip("/") + "/results.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        sys.exit("no rows")

    # Last run wins if a cell was repeated.
    latest = {}
    for r in rows:
        latest[(r["bucket"], r["rung"], r.get("k", 0), r.get("max_num_seqs", 0))] = r

    baseline = {}
    for (bucket, rung, k, conc), r in latest.items():
        if rung == "off":
            baseline[(bucket, conc)] = r["throughput_tok_s"]

    def bucket_key(b):
        return int(b.replace("K", "")) if b.replace("K", "").isdigit() else 0

    print(f"{'bucket':>7} {'conc':>5} {'rung':>8} {'k':>3} {'n':>4} "
          f"{'tok/s':>8} {'speedup':>8} {'AL':>6} {'AL@5':>6} {'pos1':>6}")
    print("-" * 72)
    for key in sorted(latest, key=lambda t: (bucket_key(t[0]), t[3], t[1], t[2])):
        bucket, rung, k, conc = key
        r = latest[key]
        s = r.get("spec", {})
        acc = s.get("acceptance_by_position") or []
        base = baseline.get((bucket, conc))
        speed = r["throughput_tok_s"] / base if base else float("nan")
        al = s.get("mean_accepted_length", 1.0)
        al5 = al_at_k(acc, 5)
        print(f"{bucket:>7} {conc:>5} {rung:>8} {k:>3} {r['limit']:>4} "
              f"{r['throughput_tok_s']:>8.1f} {speed:>8.2f} "
              f"{al:>6.3f} {(al5 if al5 else float('nan')):>6.3f} "
              f"{(acc[0] if acc else float('nan')):>6.3f}")

    print("\nacceptance by draft position")
    for key in sorted(latest, key=lambda t: (bucket_key(t[0]), t[3], t[1], t[2])):
        acc = latest[key].get("spec", {}).get("acceptance_by_position")
        if acc:
            print(f"  {key[0]:>5} c={key[3]:<3} {key[1]:>7} k={key[2]:<3} " +
                  ", ".join(f"{p:.3f}" for p in acc))

    # Decay across buckets, the number the milestone gate is written against.
    print("\ndecay D = 1 - AL(long) / AL(2K), at a common truncation of k=5")
    by_rung = defaultdict(dict)
    for (bucket, rung, k, conc), r in latest.items():
        if rung == "off":
            continue
        acc = r.get("spec", {}).get("acceptance_by_position")
        if acc:
            by_rung[(rung, k, conc)][bucket] = al_at_k(acc, 5)
    for (rung, k, conc), buckets in sorted(by_rung.items()):
        if "2K" not in buckets:
            continue
        short = buckets["2K"]
        for b in sorted(buckets, key=bucket_key):
            if b == "2K":
                continue
            d = 1.0 - buckets[b] / short
            print(f"  {rung} k={k} c={conc}: 2K {short:.3f} -> {b} {buckets[b]:.3f}   D = {d:+.3f}")

    print("\nNote: single runs, no confidence intervals. The milestone needs a "
          "paired bootstrap over prompts before any claim is made.")


if __name__ == "__main__":
    main()
