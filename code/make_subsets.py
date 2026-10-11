#!/usr/bin/env python3
"""Slice the bucket files into disjoint prompt subsets, PAIRED BY DOCUMENT across buckets.

    module load cray-python/3.12.12
    python3 make_subsets.py                 # default: subsets 1..4
    python3 make_subsets.py --subsets 5 6   # add more later; nothing is redone
    python3 make_subsets.py --check         # print the plan, write nothing

Login node only, no GPU.

Why subsets at all: run_benchmark.py reads the first LIMIT lines of the bucket
file and SEED is inert under greedy decoding, so the only way to measure a
different prompt sample is a different file. phase0.sh builds the dataset path
as $DATA/longbench_v2_${BUCKET}.jsonl, so each subset is a directory holding
the file under the canonical name, and the job points LADDER_DATA at it:

    LADDER_DATA=/projects/bikd/$USER/Speculative_Ladder/data/sub1

Why paired by document: the shared bucket files are built in dataset order
and every document in the 32K file is also in the 2K and 8K files (the 2K and
8K lists are identical, line for line). The context axis is therefore a
within-document comparison: subset 0 (the existing cells) holds the same 50
documents at 2K and 8K, and 22 of its 25 documents at 32K are among those 50.
If subsets were cut by line position independently per bucket, that pairing
would break inside each subset and the interval on the decay D would mix in
document variance. So:

  - a subset is a BLOCK of 50 consecutive documents in the 2K ordering;
  - its 2K and 8K files are that block (n=50, same documents, same order);
  - its 32K file is the first 25 documents of the block that reach 32K
    tokens, taken from the shared 32K file so the text is the shared
    truncation (n=25, all 25 inside the block).

That is a tighter pairing than subset 0's 22 of 25. A block normally holds
about 38 documents that reach 32K (387 of 503); if a block holds fewer than
25 the script says so and skips that bucket for that block.

Subset 0 is never written here: it is the first 50 lines of the shared 2K
and 8K files and the first 25 of the shared 32K file, read in place. Every
document already in subset 0 at a bucket is excluded from every other subset
at that bucket, so the subsets are disjoint from subset 0 as well as from
each other. The three subset-0 32K documents that fall outside block 0 are
therefore skipped in whichever block they land.

n must match what subset 0 used (RULES section 6): 50 at 2K and 8K, 25 at 32K.
"""

import argparse
import json
import os
import sys

SHARED = "/projects/bikd/dagraw2/Speculative_Ladder/ladder/data/longbench_v2"
BLOCK = 50                      # documents per subset in the 2K ordering
N = {"2K": 50, "8K": 50, "32K": 25}
BUCKETS = ["2K", "8K", "32K"]


def read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append((json.loads(line)["_id"], line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SHARED)
    p.add_argument("--dst", default="/projects/bikd/%s/Speculative_Ladder/data" % os.environ.get("USER", "USER"))
    p.add_argument("--subsets", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--check", action="store_true", help="print the plan and write nothing")
    a = p.parse_args()
    if 0 in a.subsets:
        sys.exit("subset 0 is the shared file, read in place, never copied")

    files = {b: read(os.path.join(a.src, "longbench_v2_%s.jsonl" % b)) for b in BUCKETS}
    for b, rows in files.items():
        ids = [i for i, _ in rows]
        if len(set(ids)) != len(ids):
            sys.exit("%s: duplicate _id inside the shared file" % b)
    ids2k = [i for i, _ in files["2K"]]
    if ids2k != [i for i, _ in files["8K"]]:
        sys.exit("2K and 8K id lists differ; the pairing assumption in this script does not hold, stop")
    if not set(i for i, _ in files["32K"]) <= set(ids2k):
        sys.exit("32K ids are not a subset of 2K ids; the pairing assumption does not hold, stop")
    line32 = dict(files["32K"])
    line2k = dict(files["2K"])
    line8k = dict(files["8K"])

    # Subset 0 at each bucket: what the existing cells measured. Excluded everywhere else.
    sub0 = {b: set(i for i, _ in files[b][:N[b]]) for b in BUCKETS}
    assigned = {b: {i: 0 for i in sub0[b]} for b in BUCKETS}

    plan = {}
    for s in a.subsets:
        block = ids2k[s * BLOCK:(s + 1) * BLOCK]
        if len(block) < BLOCK:
            print("[sub%d] needs documents %d..%d of the 2K ordering, only %d exist; skipped"
                  % (s, s * BLOCK + 1, (s + 1) * BLOCK, len(ids2k)))
            continue
        plan[s] = {}
        for b in ("2K", "8K"):
            plan[s][b] = [i for i in block if i not in sub0[b]]
            if len(plan[s][b]) != N[b]:
                sys.exit("[sub%d] %s: block overlaps subset 0 (%d docs); blocks must start past it"
                         % (s, b, BLOCK - len(plan[s][b])))
        long = [i for i in block if i in line32 and i not in sub0["32K"]]
        if len(long) < N["32K"]:
            print("[sub%d] 32K: only %d documents in this block reach 32K and are not in subset 0; "
                  "need %d; 32K skipped for this subset" % (s, len(long), N["32K"]))
        else:
            plan[s]["32K"] = long[:N["32K"]]

    # Disjointness, by _id, per bucket, against subset 0 and every other subset.
    for s, per in plan.items():
        for b, ids in per.items():
            for i in ids:
                if i in assigned[b]:
                    sys.exit("[sub%d] %s: _id %s already in subset %d; refusing" % (s, b, i, assigned[b][i]))
                assigned[b][i] = s

    for s, per in sorted(plan.items()):
        for b, ids in per.items():
            inside = sum(1 for i in ids if i in set(ids2k[s * BLOCK:(s + 1) * BLOCK]))
            print("[sub%d] %-3s n=%d  documents %s..%s of the 2K ordering; %d of %d inside the block"
                  % (s, b, len(ids), ids2k.index(ids[0]) + 1, ids2k.index(ids[-1]) + 1, inside, len(ids)))
    if a.check:
        print("\n--check: nothing written")
        return

    src_line = {"2K": line2k, "8K": line8k, "32K": line32}
    for s, per in sorted(plan.items()):
        d = os.path.join(a.dst, "sub%d" % s)
        os.makedirs(d, exist_ok=True)
        for b, ids in per.items():
            out = os.path.join(d, "longbench_v2_%s.jsonl" % b)
            if os.path.exists(out):
                have = [i for i, _ in read(out)]
                if have == ids:
                    print("[sub%d] %s exists and matches, left alone" % (s, b))
                    continue
                sys.exit("[sub%d] %s exists with different contents; refusing to overwrite" % (s, out))
            with open(out, "w", encoding="utf-8") as f:
                for i in ids:
                    f.write(src_line[b][i])
            print("[sub%d] wrote %s" % (s, out))

    print("\nverify before submitting (one ls before twelve jobs):\n"
          "  for s in %s; do ls -l %s/sub$s/; done\n"
          "then one job per (bucket, subset), e.g.\n"
          "  sbatch --exclude=gpub073 --export=ALL,LADDER_VENV=$SHARED/vllm_env_multiarch,"
          "LADDER_DATA=%s/sub1,LADDER_OUT=/projects/bikd/$USER/Speculative_Ladder/results/a40_sub1,"
          "BUCKET=2K,LIMIT=50,CONC=1 phase0.sh"
          % (" ".join(str(s) for s in sorted(plan)), a.dst, a.dst))


if __name__ == "__main__":
    main()
