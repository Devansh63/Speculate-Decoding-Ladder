#!/usr/bin/env python3
"""Slice the bucket files into one workload per LongBench-v2 domain, paired by document.

    module load cray-python/3.12.12
    python3 make_domains.py --check     # print the plan, write nothing
    python3 make_domains.py             # write every domain that has enough documents

Login node only, no GPU.

Why: every filed cell reads the first LIMIT lines of the shared bucket file,
which is LongBench-v2 in dataset order, so each cell is whatever mix of domains
those lines happen to hold. Whether the best rung and the best draft length
depend on the kind of text (prose, code, dialogue, tables) cannot be read off
such a cell. build_workload.py already keeps `domain` and `sub_domain` on every
row; this script only regroups rows that exist.

Layout, the same trick make_subsets.py uses: one directory per domain holding
the file under the canonical name, so the job points LADDER_DATA at it and
nothing else changes.

    <dst>/domains/<slug>/longbench_v2_{2K,8K,32K}.jsonl
    <dst>/domains/<slug>/manifest.json

Pairing: the 2K and 8K files hold the same documents, and every 32K document
is one of them, so the context axis stays a within-document comparison inside
a domain. The documents are chosen like this, in shared-file order:

  1. the first --n32 documents of the domain that reach 32K tokens;
  2. then the domain's remaining documents, until --n are held.

The 32K file is group 1; the 2K and 8K files are groups 1 and 2 together.

These are NOT repeats of the existing cells and NOT disjoint from them. A
domain cell and the all-domain cell of the same bucket share documents. Never
pool them, and never read a domain cell as an interval on the all-domain one.
Harness v4 puts the domain in run_id for exactly that reason.

A domain with fewer than --min documents at a bucket is skipped at that bucket
and the script says so. Run --check first and read the counts: the yield per
domain at 32K is not known until the shared files are counted.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

SHARED = "/projects/bikd/dagraw2/Speculative_Ladder/ladder/data/longbench_v2"
BUCKETS = ["2K", "8K", "32K"]


def slug(name):
    """'Single-Document QA' -> 'single_document_qa'. Same rule as harness v4."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in (name or ""))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append((d["_id"], d.get("domain", ""), line))
    return rows


def plan_domain(ids_short, ids_long, n, n32):
    """Pick documents for one domain. Returns (short_ids, long_ids)."""
    long_set = set(ids_long)
    long_pick = [i for i in ids_short if i in long_set][:n32]
    rest = [i for i in ids_short if i not in set(long_pick)]
    short_pick = long_pick + rest[:max(0, n - len(long_pick))]
    return short_pick, long_pick


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SHARED)
    p.add_argument("--dst", default="/projects/bikd/%s/Speculative_Ladder/data"
                   % os.environ.get("USER", "USER"))
    p.add_argument("--n", type=int, default=25, help="documents per domain at 2K and 8K")
    p.add_argument("--n32", type=int, default=12, help="documents per domain at 32K")
    p.add_argument("--min", type=int, default=10,
                   help="skip a domain at a bucket if it has fewer documents than this")
    p.add_argument("--domains", nargs="*", default=None,
                   help="slugs to write; default every domain found")
    p.add_argument("--check", action="store_true", help="print the plan and write nothing")
    a = p.parse_args()
    if a.n32 > a.n:
        sys.exit("--n32 cannot exceed --n: the 32K documents are a subset of the 2K ones")

    files = {b: read(os.path.join(a.src, "longbench_v2_%s.jsonl" % b)) for b in BUCKETS}
    for b, rows in files.items():
        ids = [i for i, _, _ in rows]
        if len(set(ids)) != len(ids):
            sys.exit("%s: duplicate _id inside the shared file" % b)
    ids2k = [i for i, _, _ in files["2K"]]
    if ids2k != [i for i, _, _ in files["8K"]]:
        sys.exit("2K and 8K id lists differ; the pairing assumption does not hold, stop")
    if not set(i for i, _, _ in files["32K"]) <= set(ids2k):
        sys.exit("32K ids are not a subset of 2K ids; the pairing assumption does not hold, stop")

    line = {b: {i: ln for i, _, ln in files[b]} for b in BUCKETS}
    by_dom = OrderedDict()
    for i, dom, _ in files["2K"]:
        by_dom.setdefault(dom, {"short": [], "long": []})["short"].append(i)
    for i, dom, _ in files["32K"]:
        if dom not in by_dom:
            sys.exit("32K has domain %r that 2K does not; stop" % dom)
        by_dom[dom]["long"].append(i)

    print("%-44s %6s %6s   slug" % ("domain", "2K/8K", "32K"))
    for dom, v in by_dom.items():
        print("%-44s %6d %6d   %s" % (dom or "(empty)", len(v["short"]), len(v["long"]), slug(dom)))
    print()

    slugs = [slug(d) for d in by_dom]
    if len(set(slugs)) != len(slugs):
        sys.exit("two domains map to one slug; stop")
    if a.domains:
        unknown = [s for s in a.domains if s not in slugs]
        if unknown:
            sys.exit("unknown domain slug(s): %s" % ", ".join(unknown))

    plan = OrderedDict()
    for dom, v in by_dom.items():
        s = slug(dom)
        if not s or (a.domains and s not in a.domains):
            continue
        short_pick, long_pick = plan_domain(v["short"], v["long"], a.n, a.n32)
        per = {}
        if len(short_pick) >= a.min:
            per["2K"] = short_pick
            per["8K"] = short_pick
        else:
            print("[%s] 2K/8K: only %d documents, below --min %d; skipped" % (s, len(short_pick), a.min))
        if len(long_pick) >= a.min:
            per["32K"] = long_pick
        else:
            print("[%s] 32K: only %d documents reach 32K, below --min %d; skipped"
                  % (s, len(long_pick), a.min))
        if per:
            plan[s] = (dom, per)

    for s, (dom, per) in plan.items():
        for b in BUCKETS:
            if b in per:
                note = ""
                if b == "32K":
                    inside = sum(1 for i in per[b] if i in set(per.get("2K", [])))
                    note = "  (%d of %d also in this domain's 2K/8K file)" % (inside, len(per[b]))
                print("[%s] %-3s n=%d%s" % (s, b, len(per[b]), note))
    if a.check:
        print("\n--check: nothing written")
        return

    root = os.path.join(a.dst, "domains")
    for s, (dom, per) in plan.items():
        d = os.path.join(root, s)
        os.makedirs(d, exist_ok=True)
        for b, ids in per.items():
            out = os.path.join(d, "longbench_v2_%s.jsonl" % b)
            if os.path.exists(out):
                have = [i for i, _, _ in read(out)]
                if have == ids:
                    print("[%s] %s exists and matches, left alone" % (s, b))
                    continue
                sys.exit("[%s] %s exists with different contents; refusing to overwrite" % (s, out))
            with open(out, "w", encoding="utf-8") as f:
                for i in ids:
                    f.write(line[b][i])
            print("[%s] wrote %s" % (s, out))
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"domain": dom, "slug": s, "src": a.src,
                       "n": {b: len(ids) for b, ids in per.items()},
                       "ids": per}, f, indent=1)

    print("\nnext, still on the login node (prints the jobs and their cost, submits nothing):\n"
          "  bash submit_ksweep.sh DOMAINS=each")


if __name__ == "__main__":
    main()
