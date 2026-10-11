#!/usr/bin/env python3
"""Slice the bucket files into disjoint prompt subsets, one directory each.

    module load cray-python/3.12.12
    python3 make_subsets.py                 # default: subsets 1..4 of every bucket
    python3 make_subsets.py --subsets 5 6   # add more later; nothing is redone

Login node only, no GPU. Why this exists: run_benchmark.py reads the first
LIMIT lines of the bucket file and SEED is inert under greedy decoding, so the
only way to measure a different prompt sample is a different file. phase0.sh
builds the dataset path as $DATA/longbench_v2_${BUCKET}.jsonl, so each subset
is a directory holding the file under the canonical name, and the job points
LADDER_DATA at that directory:

    LADDER_DATA=/projects/bikd/$USER/Speculative_Ladder/data/sub1

Subset 0 is the existing measurement, lines 1..n of the shared file, and is
never written here. Subset s holds lines s*n+1 .. (s+1)*n. n is the LIMIT the
existing cells used: 50 at 2K and 8K, 25 at 32K. A subset at a different n is
not comparable with subset 0 (RULES section 6).

Every subset is checked for disjointness by _id against every other subset
including subset 0, and the script refuses to write if any overlap exists.
"""

import argparse
import json
import os
import sys

SHARED = "/projects/bikd/dagraw2/Speculative_Ladder/ladder/data/longbench_v2"
N_PER_BUCKET = {"2K": 50, "8K": 50, "32K": 25}


def read_ids(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["_id"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SHARED, help="directory holding longbench_v2_<B>.jsonl")
    p.add_argument("--dst", default="/projects/bikd/%s/Speculative_Ladder/data" % os.environ.get("USER", "USER"))
    p.add_argument("--buckets", nargs="+", default=list(N_PER_BUCKET))
    p.add_argument("--subsets", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--n", type=int, default=None, help="override n for every bucket (not comparable with subset 0)")
    a = p.parse_args()

    if 0 in a.subsets:
        sys.exit("subset 0 is the shared file; it is read in place, not copied")

    for b in a.buckets:
        src = os.path.join(a.src, "longbench_v2_%s.jsonl" % b)
        n = a.n or N_PER_BUCKET.get(b)
        if n is None:
            sys.exit("no default n for bucket %s; pass --n" % b)
        lines = [l for l in open(src, encoding="utf-8") if l.strip()]
        total = len(lines)
        ids_all = [json.loads(l)["_id"] for l in lines]
        if len(set(ids_all)) != total:
            sys.exit("%s: duplicate _id inside the shared file, fix that first" % src)

        plan = {}
        for s in a.subsets:
            lo, hi = s * n, (s + 1) * n
            if hi > total:
                print("[%s] subset %d needs lines %d..%d but the bucket has %d; skipped"
                      % (b, s, lo + 1, hi, total))
                continue
            plan[s] = lines[lo:hi]

        # Disjointness against subset 0 and each other, by _id, before writing.
        seen = {i: 0 for i in ids_all[:n]}
        for s, chunk in plan.items():
            for l in chunk:
                i = json.loads(l)["_id"]
                if i in seen:
                    sys.exit("[%s] _id %s in subset %d already in subset %d; refusing to write"
                             % (b, i, s, seen[i]))
                seen[i] = s

        for s, chunk in plan.items():
            d = os.path.join(a.dst, "sub%d" % s)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, "longbench_v2_%s.jsonl" % b)
            if os.path.exists(out):
                have = read_ids(out)
                want = [json.loads(l)["_id"] for l in chunk]
                if have == want:
                    print("[%s] subset %d exists and matches, left alone: %s" % (b, s, out))
                    continue
                sys.exit("[%s] %s exists with different contents; refusing to overwrite" % (b, out))
            with open(out, "w", encoding="utf-8") as f:
                f.writelines(chunk)
            print("[%s] subset %d: lines %d..%d of %d -> %s" % (b, s, s * n + 1, (s + 1) * n, total, out))

    print("\nsubmit one job per (bucket, subset), e.g.\n"
          "  sbatch --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,"
          "LADDER_DATA=%s/sub1,"
          "LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/a40_sub1,"
          "BUCKET=2K,LIMIT=50,CONC=1 phase0.sh" % a.dst)


if __name__ == "__main__":
    main()
