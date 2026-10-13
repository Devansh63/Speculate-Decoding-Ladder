#!/usr/bin/env python3
"""
Block 2 -- score task accuracy from the stored token ids, per rung and per
LongBench-v2 domain. Zero GPU.

Why this is worth doing
-----------------------
Every losslessness number in this project so far is about whether two runs
produced the same TOKENS. That is the wrong question for the claim the paper
wants to make. LongBench-v2 is multiple choice, so two generations can differ
in hundreds of tokens and still pick the same letter. Token identity is a
proxy; the answer is the thing.

It is also far more robust to the engine non-determinism that dominates token
agreement: the measured baseline-against-itself hazard is about 0.75% per
token, which destroys sequence identity over 256 tokens while leaving the
chosen letter untouched most of the time.

What the null is, and why this script will not state one
--------------------------------------------------------
Speculative decoding is supposed to be lossless, so accuracy should be
statistically indistinguishable across rungs. "Indistinguishable" needs a
floor, and the floor is the baseline's disagreement with ITSELF under the same
configuration -- which does not exist in this repository yet (check_lossless.v3
reports "NO same-configuration pair exists"). So this script reports accuracies
and their differences and explicitly refuses to call a difference significant.
When the same-config control lands, compare against it.

Honesty about extraction
------------------------
The model is prompted with "Answer the question and explain your reasoning."
so the letter has to be pulled out of free text. Four patterns are tried in
descending confidence and THE TIER IS REPORTED, because an accuracy number
that depends on a loose regex needs its interpretation budget shown. Anything
unmatched is counted as unparseable, never guessed, and never silently scored
as wrong.

Needs the venv (transformers, for the tokenizer). The tokenizer path is read
out of results.jsonl rather than hardcoded.

Usage
-----
  source /projects/bikd/dagraw2/Speculative_Ladder/vllm_env_multiarch/bin/activate
  python3 b2_01_score_accuracy.py RESULTS_DIR [--data DATASET_DIR]

  python3 b2_01_score_accuracy.py /projects/bikd/dagraw2/Speculative_Ladder/results/a40
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

DEFAULT_DATA = "/projects/bikd/dagraw2/Speculative_Ladder/ladder/data/longbench_v2"

# Descending confidence. The tier each answer came from is reported.
PATTERNS = [
    ("starts-with-letter", re.compile(r"^\s*\(?([ABCD])\b")),
    ("answer-is",          re.compile(r"answer\s+is\s*\(?([ABCD])\b", re.I)),
    ("letter-punct",       re.compile(r"^\s*\(?([ABCD])\s*[\).:\-]")),
    ("first-standalone",   re.compile(r"\b([ABCD])\b")),
]
SCAN = 240   # characters of the generation searched for the loosest pattern


def extract(text):
    """Return (letter, tier) or (None, 'unparseable'). Never guesses."""
    head = text[:SCAN]
    for tier, pat in PATTERNS:
        m = pat.search(head if tier != "first-standalone" else head)
        if m:
            return m.group(1).upper(), tier
    return None, "unparseable"


def load_dataset(data_dir):
    """id -> {answer, domain, sub_domain, bucket}."""
    meta = {}
    files = sorted(glob.glob(os.path.join(data_dir, "longbench_v2_*.jsonl")))
    if not files:
        sys.exit(f"no longbench_v2_*.jsonl in {data_dir}")
    for f in files:
        bucket = os.path.basename(f).replace("longbench_v2_", "").replace(".jsonl", "")
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_id" not in d:
                continue
            meta[d["_id"]] = {
                "answer": (d.get("answer") or "").strip().upper()[:1],
                "domain": d.get("domain", "?"),
                "sub_domain": d.get("sub_domain", "?"),
                "bucket": bucket,
            }
    return meta


def tokenizer_for(results_dir):
    """Read the model path out of results.jsonl and load its tokenizer."""
    p = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(p):
        sys.exit(f"no results.jsonl in {results_dir} -- needed for the model path")
    paths = []
    for line in open(p):
        line = line.strip()
        if line:
            mp = json.loads(line).get("model")
            if mp and mp not in paths:
                paths.append(mp)
    if not paths:
        sys.exit("no 'model' field in any row")
    if len(paths) > 1:
        print(f"WARNING: {len(paths)} different model paths in this tree; "
              f"using the first:\n  " + "\n  ".join(paths), file=sys.stderr)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not importable -- activate the venv first:\n"
                 "  source .../vllm_env_multiarch/bin/activate")
    print(f"tokenizer: {paths[0]}", file=sys.stderr)
    return AutoTokenizer.from_pretrained(paths[0], trust_remote_code=True)


NAME = re.compile(
    r"^perreq_(?P<rung>[a-z0-9]+)_(?P<bucket>[^_]+)_k(?P<k>\d+)"
    r"_n(?P<n>\d+)_s(?P<s>\d+)(?:_c(?P<c>\d+))?"
    r"(?P<tags>(?:_(?!j(?:\d|nojob))[A-Za-z][A-Za-z0-9-]*)*)"
    r"(?:_j(?P<job>\d+|nojob))?\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--by-domain", action="store_true",
                    help="also break accuracy down by LongBench-v2 domain")
    a = ap.parse_args()

    meta = load_dataset(a.data)
    print(f"dataset: {len(meta)} records with an answer key", file=sys.stderr)
    tok = tokenizer_for(a.results_dir)

    files = sorted(glob.glob(os.path.join(a.results_dir, "perreq_*.json")))
    if not files:
        sys.exit(f"no perreq_*.json in {a.results_dir}")

    rows = []
    per_domain = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # dom->arm->[ok,tot]
    skipped_names = []

    for path in files:
        name = os.path.basename(path)
        m = NAME.match(name)
        if not m:
            skipped_names.append(name)
            continue
        g = m.groupdict()
        # Bucket MUST be in the arm key. Without it the per-domain table pools
        # 2K, 8K and 32K into one row per rung, and an arm that ran at only one
        # bucket (dflash k4 is 32K-only) gets compared against an arm that ran
        # at three -- different documents, same row. The n column was the only
        # thing exposing it.
        arm = f"{g['bucket']}/{g['rung']}_k{g['k']}_c{g['c'] or 0}{g['tags'] or ''}"

        recs = json.load(open(path))
        if isinstance(recs, dict):
            recs = recs.get("records", [])

        no_tokens = 0
        tiers = defaultdict(int)
        ok = scored = unmatched = nokey = 0
        for r in recs:
            tids = r.get("token_ids")
            if not tids:
                no_tokens += 1
                continue
            info = meta.get(r.get("id"))
            if not info or not info["answer"]:
                nokey += 1
                continue
            text = tok.decode(tids, skip_special_tokens=True)
            letter, tier = extract(text)
            tiers[tier] += 1
            if letter is None:
                unmatched += 1
                continue
            scored += 1
            hit = (letter == info["answer"])
            ok += hit
            if a.by_domain:
                d = per_domain[info["domain"]][arm]
                d[0] += hit
                d[1] += 1

        rows.append(dict(name=name, arm=arm, bucket=g["bucket"],
                         job=g["job"] or "", n=len(recs), no_tokens=no_tokens,
                         nokey=nokey, unmatched=unmatched, scored=scored, ok=ok,
                         tiers=dict(tiers)))

    if skipped_names:
        print(f"\nWARNING {len(skipped_names)} file(s) did not match the name "
              f"pattern:", file=sys.stderr)
        for n in skipped_names[:10]:
            print(f"  {n}", file=sys.stderr)

    # ---------------------------------------------------------------- report
    print(f"\n{'bucket':>6} {'arm':>24} {'job':>9} {'scored':>7} {'acc':>7} "
          f"{'unparsed':>9} {'no-tok':>7} {'no-key':>7}")
    for r in sorted(rows, key=lambda r: (str(r["bucket"]), r["arm"], r["job"])):
        acc = f"{100.0 * r['ok'] / r['scored']:.1f}%" if r["scored"] else "--"
        print(f"{str(r['bucket']):>6} {r['arm']:>24} {r['job']:>9} "
              f"{r['scored']:>7} {acc:>7} {r['unmatched']:>9} "
              f"{r['no_tokens']:>7} {r['nokey']:>7}")

    print("\nextraction tiers (how much interpretation each number rests on):")
    agg = defaultdict(int)
    for r in rows:
        for t, c in r["tiers"].items():
            agg[t] += c
    total = sum(agg.values()) or 1
    for t, c in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {t:>18}: {c:>5}  ({100.0 * c / total:.1f}%)")
    if agg.get("first-standalone", 0) > 0.15 * total:
        print("  NOTE: more than 15% fell to the loosest pattern. Treat the "
              "absolute\n        accuracies as soft; the DIFFERENCES between arms "
              "are still usable\n        because the same extractor is applied to "
              "every arm.")

    # ---------------------------------------------------- differences by cell
    print("\ndifference against the 'off' arm, same bucket and concurrency:")
    by_cell = defaultdict(dict)
    for r in rows:
        if not r["scored"]:
            continue
        conc = r["arm"].split("_c")[-1]
        by_cell[(r["bucket"], conc, r["job"])][r["arm"]] = r
    any_pair = False
    for key in sorted(by_cell, key=lambda t: (str(t[0]), str(t[1]), str(t[2]))):
        arms = by_cell[key]
        base = next((v for k, v in arms.items() if k.startswith("off_")), None)
        if not base:
            continue
        for arm, r in sorted(arms.items()):
            if arm.startswith("off_"):
                continue
            any_pair = True
            a_base = 100.0 * base["ok"] / base["scored"]
            a_arm = 100.0 * r["ok"] / r["scored"]
            print(f"  {str(key[0]):>4} c={key[1]:<3} job {key[2] or '?':>9}  "
                  f"{arm:>22}: {a_arm:5.1f}% vs off {a_base:5.1f}%  "
                  f"({a_arm - a_base:+.1f} pts, n={r['scored']})")
    if not any_pair:
        print("  no (speculative, off) pair in the same bucket/concurrency/job")

    print("\n  NOT a significance test. Speculative decoding is supposed to be")
    print("  lossless, so these differences should be indistinguishable from the")
    print("  baseline's disagreement with ITSELF under the same configuration --")
    print("  a floor this repository does not yet have. Do not call any gap")
    print("  above meaningful until the same-config control exists.")

    if a.by_domain and per_domain:
        print("\n" + "=" * 70)
        print("accuracy by LongBench-v2 domain")
        print("=" * 70)
        for dom in sorted(per_domain):
            print(f"\n{dom}")
            # Group by (bucket, n). Grouping by n alone still pools buckets --
            # 2K's eight Long-ICL documents are not 8K's eight -- and the
            # reported "spread" then measures a bucket difference rather than a
            # rung difference. That is exactly how a 25-point spread between
            # 2K and 8K got read as a two-prompt rung effect.
            by_n = defaultdict(list)
            for arm in sorted(per_domain[dom]):
                ok, tot = per_domain[dom][arm]
                if tot:
                    bkt = arm.split("/", 1)[0]
                    by_n[(bkt, tot)].append((arm, ok, tot))
            for gk in sorted(by_n, key=lambda t: (str(t[0]), -t[1])):
                tot = gk[1]
                group = by_n[gk]
                accs = [100.0 * ok / t for _, ok, t in group]
                spread = max(accs) - min(accs)
                flag = ""
                if len(group) > 1 and spread > 0:
                    flips = round(spread / 100.0 * tot)
                    flag = f"   <- spread {spread:.1f} pts = {flips} prompt(s)"
                print(f"  {gk[0]}  n={tot}{flag}")
                for arm, ok, t in group:
                    print(f"    {arm:>28}: {100.0 * ok / t:5.1f}%")
        print("\n  Grouped by (bucket, n): only arms covering the same documents")
        print("  are comparable. Pooling buckets makes a bucket difference look")
        print("  like a rung effect. Counts are small, so a one-prompt spread is")
        print("  a lead for a per-domain run, not a result -- and it needs the")
        print("  same-config floor before it means anything at all.")


if __name__ == "__main__":
    main()
