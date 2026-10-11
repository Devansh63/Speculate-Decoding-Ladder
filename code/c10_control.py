#!/usr/bin/env python3
"""Same-config control for C10, with a paired bootstrap over prompts.

    python3 c10_control.py --dirs R1 R2
    python3 c10_control.py --a R1/perreq_off_2K_k0_n50_s1234_c1.json \
                           --b R2/perreq_off_2K_k0_n50_s1234_c1.json \
                           --c results/a40/perreq_off_2K_k0_n50_s1234_c1.json \
                           --d results/a40/perreq_off_2K_k0_n50_s1234_c8.json

What this exists to fix
-----------------------
C10 as originally written compared the `off` arm against the `off` arm at a
*different concurrency* (check_lossless.py: "off c=1 vs off c=8") with a
speculative arm against the baseline at *matching* concurrency. The control
varied two things — engine non-determinism and batch size — while the treatment
varied one. The two rates are not comparable and the strong claim does not
follow. See paper/claims.md, "External review, 11 Oct 2026", item 1.

It is worse than the confound: the surviving 2K `c=1` perreq file is not even
from the same job as its `c=8` and `c=32` siblings. It carries harness-v2
fields (`timing`, `token_ids`) while they carry harness-v1 fields only, and the
2K grid was job 22598608 on 1 Oct under v1 — so a later job (22737716, 7 Oct)
overwrote it. This script reports that provenance mismatch when it sees it,
because a cross-job comparison adds run-to-run and node variance to a
measurement whose whole purpose is to bound engine variance.

--a/--b are the missing control: two `off` runs at *identical* config (same
bucket, n, seed, concurrency), differing only in process and output directory.

--c/--d are optional and put an interval on the *difference* between the two
rates, on one common resample of prompt ids — the paired construction
HANDOFF.md section 6.2 asks for.

Comparison basis
----------------
`token_ids` when **both** files carry them, otherwise `sha1` for both. Mixing
the two (a v2 file against a v1 file) silently yields 0% agreement, which is
how this script first lied on real data: the c1 file has token ids and the c8
file does not.

Why a bootstrap is legitimate here, when it was not for speedup
--------------------------------------------------------------
The paired bootstrap was ruled out for speedup and tau because per-request
timing is unavailable on this vLLM build. Agreement is not a timing quantity:
each prompt contributes one binary outcome and the token ids are on disk, so
resampling prompts with replacement is valid. This is the free interval the
register says it cannot get, for the one claim where it is obtainable.

Stdlib only. Deterministic: SEED fixes the resamples.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

SEED = 20261011
BOOT = 10000
V2_MARKERS = ("token_ids", "timing", "finish_reason")


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)}


def basis_of(d):
    """token_ids only if every record carries them, else sha1."""
    return "token_ids" if all("token_ids" in r for r in d.values()) else "sha1"


def shape_of(d):
    keys = set()
    for r in d.values():
        keys |= set(r)
    return keys


def provenance_warning(name_a, a, name_b, b):
    """A file carrying v2 fields compared against one that does not is almost
    certainly a different job's output, not just a different run."""
    va = any(k in shape_of(a) for k in V2_MARKERS)
    vb = any(k in shape_of(b) for k in V2_MARKERS)
    if va != vb:
        print(f"  PROVENANCE: {name_a} carries harness-v2 fields "
              f"({', '.join(sorted(shape_of(a) & set(V2_MARKERS)))}) and "
              f"{name_b} does not "
              f"({', '.join(sorted(shape_of(b))) or 'none'}).")
        print("              These are not two runs of one harness revision, so "
              "any difference between them includes a harness change. Check the "
              "records in results.jsonl for env.slurm_job and env.host before "
              "quoting this number.")


def find_off(d):
    hits = [n for n in sorted(os.listdir(d))
            if n.startswith("perreq_") and n.endswith(".json")]
    if len(hits) != 1:
        sys.exit(f"expected exactly one perreq_*.json in {d}, found {len(hits)}: {hits}")
    return os.path.join(d, hits[0])


def seq_key(rec, basis):
    return rec["token_ids"] if basis == "token_ids" else rec["sha1"]


def compare(a, b, ids, basis):
    """(identical prompts, differing token positions, total token positions,
    first-divergence index per differing prompt)."""
    same = 0
    diff_positions = 0
    total_positions = 0
    first_div = []
    for i in ids:
        ta, tb = seq_key(a[i], basis), seq_key(b[i], basis)
        if ta == tb:
            same += 1
            continue
        if basis == "sha1":
            first_div.append(-1)      # position unknown without token ids
            continue
        n = min(len(ta), len(tb))
        fd = next((j for j in range(n) if ta[j] != tb[j]), n)
        first_div.append(fd)
        diff_positions += sum(1 for j in range(n) if ta[j] != tb[j])
        diff_positions += abs(len(ta) - len(tb))
        total_positions += max(len(ta), len(tb))
    return same, diff_positions, total_positions, first_div


def rate_of(a, b, ids, basis):
    same, _, _, _ = compare(a, b, ids, basis)
    return same / len(ids)


def boot_ci(a, b, ids, basis, rng):
    n = len(ids)
    out = sorted(rate_of(a, b, [ids[rng.randrange(n)] for _ in range(n)], basis)
                 for _ in range(BOOT))
    return out[int(0.025 * BOOT)], out[int(0.975 * BOOT)]


def boot_diff(a, b, c, d, ids, basis, rng):
    n = len(ids)
    out = []
    for _ in range(BOOT):
        res = [ids[rng.randrange(n)] for _ in range(n)]
        out.append(rate_of(a, b, res, basis) - rate_of(c, d, res, basis))
    out.sort()
    return out[int(0.025 * BOOT)], out[int(0.975 * BOOT)]


def report(label, a, b, ids, basis, rng):
    same, diff_pos, tot_pos, first_div = compare(a, b, ids, basis)
    n = len(ids)
    pct = 100.0 * same / n
    lo, hi = boot_ci(a, b, ids, basis, rng)
    print(f"{label}   [basis: {basis}]")
    print(f"  prompts compared          {n}")
    print(f"  identical sequences       {same}/{n} = {pct:.1f}%"
          f"   95% CI [{100*lo:.1f}%, {100*hi:.1f}%]  ({BOOT} resamples)")
    if tot_pos:
        print(f"  differing token positions {diff_pos}/{tot_pos} = "
              f"{100.0*diff_pos/tot_pos:.2f}%  (over prompts that differ)")
    known = [x for x in first_div if x >= 0]
    if known:
        c = Counter(known)
        top = ", ".join(f"pos {k}: {v}" for k, v in sorted(c.items())[:5])
        print(f"  first divergence position among differing prompts: {top}"
              f"{' ...' if len(c) > 5 else ''}")
    return pct


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs=2, metavar=("A", "B"),
                    help="two directories, each holding exactly one perreq_*.json")
    ap.add_argument("--a", help="control run 1 (perreq JSON)")
    ap.add_argument("--b", help="control run 2 (perreq JSON)")
    ap.add_argument("--c", help="optional comparison run 1")
    ap.add_argument("--d", help="optional comparison run 2")
    ap.add_argument("--json-out", help="also write the summary as JSON here")
    args = ap.parse_args()

    if args.dirs:
        a_path, b_path = find_off(args.dirs[0]), find_off(args.dirs[1])
    elif args.a and args.b:
        a_path, b_path = args.a, args.b
    else:
        sys.exit("need --dirs A B or --a/--b")

    A, B = load(a_path), load(b_path)
    ids = sorted(set(A) & set(B))
    if len(ids) < 10:
        sys.exit(f"only {len(ids)} shared prompt ids between the control pair")
    if len(ids) != len(A) or len(ids) != len(B):
        print(f"warning: {len(A)} and {len(B)} prompts loaded, {len(ids)} shared")

    ba, bb = basis_of(A), basis_of(B)
    basis = "token_ids" if ba == bb == "token_ids" else "sha1"
    if ba != bb:
        print(f"note: A has {ba}, B has {bb} — falling back to sha1 for both. "
              f"A mixed-basis comparison is invalid.")

    print("=" * 74)
    print("SAME-CONFIG CONTROL — two `off` runs, identical flags, separate dirs")
    print("=" * 74)
    print(f"  A  {a_path}")
    print(f"  B  {b_path}")
    provenance_warning("A", A, "B", B)
    rng = random.Random(SEED)
    rate_ab = report("CONTROL: A against B", A, B, ids, basis, rng)

    summary = {"a": a_path, "b": b_path, "prompts": len(ids), "basis": basis,
               "identical_pct": round(rate_ab, 3)}

    if args.c and args.d:
        C, D = load(args.c), load(args.d)
        ids2 = sorted(set(C) & set(D) & set(ids))
        if len(ids2) < 10:
            sys.exit(f"only {len(ids2)} shared prompt ids across all four files")
        bcd = basis_of(C) == basis_of(D) == "token_ids"
        basis2 = "token_ids" if bcd else "sha1"
        print()
        print("=" * 74)
        print("COMPARISON PAIR (optional) — e.g. a concurrency change, or spec vs base")
        print("=" * 74)
        print(f"  C  {args.c}")
        print(f"  D  {args.d}")
        provenance_warning("C", C, "D", D)
        rng = random.Random(SEED)
        rate_cd = report("COMPARISON: C against D", C, D, ids2, basis2, rng)
        delta = rate_ab - rate_cd
        lo = hi = float("nan")
        if basis == basis2:
            lo, hi = boot_diff(A, B, C, D, ids2, basis, random.Random(SEED))
            print()
            print(f"  DIFFERENCE  control {rate_ab:.1f}%  minus  comparison "
                  f"{rate_cd:.1f}%  =  {delta:+.1f} points")
            print(f"              95% CI [{100*lo:+.1f}, {100*hi:+.1f}] points, "
                  f"paired on prompt ids")
            print("              interval excludes zero: the two rates differ"
                  if (lo > 0 or hi < 0) else
                  "              interval spans zero: indistinguishable at this n")
        else:
            print()
            print(f"  DIFFERENCE  control {rate_ab:.1f}%  minus  comparison "
                  f"{rate_cd:.1f}%  =  {delta:+.1f} points")
            print("              no interval: the two pairs do not share a "
                  "comparison basis")
        summary.update({"c": args.c, "d": args.d, "comparison_pct": round(rate_cd, 3),
                        "diff_points": round(delta, 3),
                        "diff_ci_points": [round(100 * lo, 3), round(100 * hi, 3)]})

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
