#!/usr/bin/env python3
"""
Block 0, step 3 -- resolve the 32K n-gram tau gap (1.390 on A40 vs 1.345 on
A100) by diffing the stored token_ids, and produce the per-token divergence
rate while we are in there.

Two jobs, one script:

1. The tau question. n-gram proposals are a deterministic function of the text,
   but acceptance rides on target logits, so the prior is "numerics or
   bookkeeping", not "one figure is wrong". If the two machines produced even
   one different token on a prompt, every downstream n-gram match changes and a
   3% tau gap needs no further explanation.

2. The C10 metric question. Sequence-level agreement is a bad statistic: it is
   the per-token divergence rate raised to the power of the output length, so
   it mostly measures how long the outputs are. This script reports the
   per-token HAZARD -- the probability a token diverges given that everything
   before it matched -- which is comparable across cells and output lengths. It
   then checks that hazard against the observed sequence-agreement rate, so the
   two numbers can be seen to be consistent (or not).

Usage
-----
  # cross-machine, same cell
  python3 b0_03_tau_diff.py \
      /projects/bikd/dagraw2/Speculative_Ladder/results/a40 \
      /projects/bikd/ssashi/Speculative_Ladder/results/a100 \
      --pattern 'perreq_ngram_32K_k5_*_c1.json'

  # all cells that exist on both sides
  python3 b0_03_tau_diff.py TREE_A TREE_B

Prints nothing it cannot support: if one side has no per-request files it says
so and exits, rather than reporting an empty agreement rate as agreement.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter


def read_perreq(path):
    """Return {id: token_ids} for one perreq file, skipping records without ids."""
    with open(path) as fh:
        d = json.load(fh)
    recs = d if isinstance(d, list) else d.get("records", d)
    if not isinstance(recs, list):
        return {}, "unrecognised structure"
    out, no_tok = {}, 0
    for r in recs:
        rid = r.get("id")
        tok = r.get("token_ids")
        if rid is None:
            continue
        if tok is None:
            no_tok += 1
            continue
        out[rid] = list(tok)
    note = f"{no_tok} record(s) had no token_ids" if no_tok else ""
    return out, note


def compare(a, b):
    """
    Compare two {id: token_ids} maps.

    Returns a dict with the sequence-agreement rate, the per-token hazard, and
    the first-divergence position distribution.
    """
    common = sorted(set(a) & set(b))
    if not common:
        return None

    identical = 0
    survived = 0          # total positions that matched before any divergence
    diverged = 0          # sequences that diverged at all
    first_pos = []
    len_mismatch = 0
    lengths = []

    for rid in common:
        xa, xb = a[rid], b[rid]
        n = min(len(xa), len(xb))
        lengths.append(n)
        if len(xa) != len(xb):
            len_mismatch += 1
        d = next((i for i in range(n) if xa[i] != xb[i]), None)
        if d is None and len(xa) == len(xb):
            identical += 1
            survived += n
        else:
            diverged += 1
            pos = d if d is not None else n   # same prefix, different length
            first_pos.append(pos)
            survived += pos

    hazard = diverged / survived if survived else float("nan")
    mean_len = sum(lengths) / len(lengths)
    predicted_agree = (1.0 - hazard) ** mean_len if hazard == hazard else float("nan")

    return dict(n=len(common), identical=identical, diverged=diverged,
                agree=identical / len(common), hazard=hazard,
                mean_len=mean_len, predicted_agree=predicted_agree,
                first_pos=first_pos, len_mismatch=len_mismatch,
                only_a=len(set(a) - set(b)), only_b=len(set(b) - set(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree_a")
    ap.add_argument("tree_b")
    ap.add_argument("--pattern", default="perreq_*.json",
                    help="glob applied in both trees (default: all perreq files)")
    args = ap.parse_args()

    fa = {os.path.basename(p): p for p in glob.glob(os.path.join(args.tree_a, args.pattern))}
    fb = {os.path.basename(p): p for p in glob.glob(os.path.join(args.tree_b, args.pattern))}

    print(f"tree A: {args.tree_a}   files matching {args.pattern}: {len(fa)}")
    print(f"tree B: {args.tree_b}   files matching {args.pattern}: {len(fb)}")

    if not fa or not fb:
        side = "A" if not fa else "B"
        print(f"\nSTOP: tree {side} has no per-request files matching the pattern.")
        print("Nothing can be concluded about output divergence from this pair.")
        print("If the A100 tree has no perreq files, that is the answer to the")
        print("tau question for now: the diff is not available, so the 1.390 vs")
        print("1.345 gap stays open and must not be used in a table.")
        sys.exit(2)

    shared = sorted(set(fa) & set(fb))
    if not shared:
        print(f"\nNo filename appears in both trees.")
        print(f"  only in A: {sorted(set(fa) - set(fb))[:6]}")
        print(f"  only in B: {sorted(set(fb) - set(fa))[:6]}")
        sys.exit(2)

    for name in shared:
        ma, na = read_perreq(fa[name])
        mb, nb = read_perreq(fb[name])
        print(f"\n=== {name} ===")
        print(f"  A: {len(ma)} record(s){' -- ' + na if na else ''}")
        print(f"  B: {len(mb)} record(s){' -- ' + nb if nb else ''}")
        res = compare(ma, mb)
        if res is None:
            print("  no ids in common -- cannot compare (different prompt subsets?)")
            continue
        if res["only_a"] or res["only_b"]:
            print(f"  NOTE ids only in A: {res['only_a']}, only in B: {res['only_b']} "
                  f"-- comparing the {res['n']} in common")
        print(f"  sequences compared        : {res['n']}")
        print(f"  byte-identical outputs    : {res['identical']} "
              f"({res['agree'] * 100:.1f}%)")
        print(f"  diverged                  : {res['diverged']}"
              + (f"  (of which {res['len_mismatch']} also differ in length)"
                 if res["len_mismatch"] else ""))
        print(f"  mean compared length      : {res['mean_len']:.1f} tokens")
        print(f"  per-token hazard          : {res['hazard'] * 100:.4f}% "
              f"per token")
        print(f"  agreement implied by that : {res['predicted_agree'] * 100:.1f}% "
              f"(observed {res['agree'] * 100:.1f}%)")
        if res["first_pos"]:
            c = Counter(min(p // 16, 15) for p in res["first_pos"])
            print("  first divergence position (16-token bins):")
            for b in sorted(c):
                lo = b * 16
                bar = "#" * min(c[b], 40)
                print(f"    {lo:>4}-{lo + 15:<4} {c[b]:>4} {bar}")

        print("\n  Reading it:")
        if res["agree"] == 1.0:
            print("    Outputs are identical. The tau gap is NOT explained by")
            print("    output divergence -- it is a bookkeeping or aggregation")
            print("    difference, and the two recorded figures must be")
            print("    reconciled by hand before either is used.")
        elif res["diverged"]:
            print("    Outputs diverge. A single different token changes every")
            print("    downstream n-gram match, so a few-percent tau difference")
            print("    needs no further explanation. Report tau per machine; do")
            print("    not treat one figure as the error.")
        print("    The per-token hazard is the figure to quote for output")
        print("    agreement -- NOT the sequence rate, which is the hazard")
        print("    raised to the output length and so mostly measures length.")


if __name__ == "__main__":
    main()
