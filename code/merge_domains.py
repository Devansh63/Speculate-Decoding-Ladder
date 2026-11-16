#!/usr/bin/env python3
"""Merge make_domains.py's per-domain files into one domain-balanced workload
per bucket, for the dtai2 domain jobs. Login node only, no GPU.

On a DeltaAI login node, with deltaai_env.sh sourced (the venv's python, since
the login node's own python3 may be old):
    "$LADDER_VENV/bin/python3" make_domains.py --src "$LADDER_DATA" --dst "$LADDER_ROOT/data" --n 50 --n32 25 --check
    "$LADDER_VENV/bin/python3" make_domains.py --src "$LADDER_DATA" --dst "$LADDER_ROOT/data" --n 50 --n32 25
    "$LADDER_VENV/bin/python3" merge_domains.py --check
    "$LADDER_VENV/bin/python3" merge_domains.py

Why --n 50 --n32 25, twice make_domains.py's defaults: the domain jobs are
also router training data, and at the defaults the whole set holds about 200
distinct documents. A domain with fewer documents simply contributes all it
has; one with fewer than --min (10) at a bucket is skipped there.

Why one file instead of one cell per domain. ksweep.sh starts a fresh engine
for every arm, so DOMAINS=each pays one model load per arm per domain per
bucket (about 72 loads for six domains). One balanced file per bucket pays one
load per arm, and every per-request record already carries its prompt's
`domain` (run_benchmark.v5.py writes it), so the domain split is read off the
per-request files afterwards.

Order. Rows are interleaved round-robin across domains, each domain in its own
file order, so any prefix of the merged file is as balanced as it can be and a
short LIMIT still covers every domain. make_domains.py writes the same
documents in the same order to a domain's 2K and 8K files, so the merged 2K
and 8K files hold the same documents in the same order too, and the 32K file
holds a subset of them.

What this is not. The merged files are not the all-domain cells' prompt set
(those read the first 50 or 25 lines of the shared files, which are mostly
prose question answering). Rows from the merged files go to their own results
directories and are never pooled with any other cell.

Output, next to make_domains.py's:
    <root>/domains_balanced/longbench_v2_{2K,8K,32K}.jsonl
    <root>/domains_balanced/manifest.json   counts, ids and input checksums

Written 16 Nov 2026 for the dtai2 job set. Tested on synthetic files only.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict

BUCKETS = ["2K", "8K", "32K"]


def default_root():
    root = os.environ.get("LADDER_ROOT")
    return os.path.join(root, "data") if root else None


def read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append((d["_id"], d.get("domain", ""), line if line.endswith("\n") else line + "\n"))
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=default_root(),
                   help="the directory make_domains.py wrote domains/ into "
                        "(default $LADDER_ROOT/data)")
    p.add_argument("--out", default=None,
                   help="default <root>/domains_balanced")
    p.add_argument("--check", action="store_true", help="print the plan and write nothing")
    a = p.parse_args()
    if not a.root:
        sys.exit("no --root and no LADDER_ROOT; source deltaai_env.sh first")
    src = os.path.join(a.root, "domains")
    out = a.out or os.path.join(a.root, "domains_balanced")
    if not os.path.isdir(src):
        sys.exit(f"no {src}. Run make_domains.py with --dst {a.root} first.")

    slugs = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    if not slugs:
        sys.exit(f"no domain directories under {src}")

    plan = OrderedDict()
    inputs = {}
    for b in BUCKETS:
        per = OrderedDict()
        for s in slugs:
            f = os.path.join(src, s, f"longbench_v2_{b}.jsonl")
            if os.path.isfile(f):
                rows = read(f)
                doms = {d for _, d, _ in rows}
                if len(doms) != 1:
                    sys.exit(f"{f} holds domains {sorted(doms)}; expected one")
                per[s] = rows
                inputs[f"{s}/{os.path.basename(f)}"] = sha256(f)
        merged, i = [], 0
        while any(i < len(r) for r in per.values()):
            for s, r in per.items():
                if i < len(r):
                    merged.append(r[i])
            i += 1
        ids = [x[0] for x in merged]
        if len(set(ids)) != len(ids):
            sys.exit(f"{b}: a document appears in two domains; stop")
        plan[b] = (per, merged)

    if [x[0] for x in plan["2K"][1]] != [x[0] for x in plan["8K"][1]]:
        sys.exit("merged 2K and 8K documents differ; make_domains.py's pairing did not hold, stop")
    if not {x[0] for x in plan["32K"][1]} <= {x[0] for x in plan["2K"][1]}:
        sys.exit("merged 32K documents are not a subset of 2K; stop")

    print("%-44s %5s %5s %5s" % ("domain slug", "2K", "8K", "32K"))
    for s in slugs:
        print("%-44s %5d %5d %5d" % (s, *[len(plan[b][0].get(s, [])) for b in BUCKETS]))
    print("%-44s %5d %5d %5d" % ("merged", *[len(plan[b][1]) for b in BUCKETS]))
    if a.check:
        print("\n--check: nothing written")
        return

    os.makedirs(out, exist_ok=True)
    for b in BUCKETS:
        f = os.path.join(out, f"longbench_v2_{b}.jsonl")
        body = "".join(x[2] for x in plan[b][1])
        if os.path.exists(f):
            with open(f, encoding="utf-8") as g:
                if g.read() == body:
                    print(f"{f} exists and matches, left alone")
                    continue
            sys.exit(f"{f} exists with different contents; refusing to overwrite")
        with open(f, "w", encoding="utf-8") as g:
            g.write(body)
        print(f"wrote {f}  ({len(plan[b][1])} prompts)")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as g:
        json.dump({
            "written_by": "merge_domains.py",
            "src": src,
            "order": "round-robin across domains, each in its own file order",
            "n": {b: len(plan[b][1]) for b in BUCKETS},
            "n_by_domain": {b: {s: len(r) for s, r in plan[b][0].items()} for b in BUCKETS},
            "ids": {b: [x[0] for x in plan[b][1]] for b in BUCKETS},
            "input_sha256": inputs,
        }, g, indent=1)
    print(f"\nnext: bash submit_dtai2.sh BUNDLES=\"dom-2K dom-8K dom-32K\"   (plan only until GO=1)")


if __name__ == "__main__":
    main()
