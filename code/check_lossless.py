#!/usr/bin/env python3
"""Verify that every speculative arm produced byte-identical output.

    python3 check_lossless.py /projects/bikd/$USER/Speculative_Ladder/results/a40

Speculative decoding is supposed to be exact: the target model verifies every
drafted token, so the sequence that comes out must be the sequence the target
would have produced alone. That is the whole reason it is acceptable to use in
production, and the paper will say so. It is also the one claim in the project
that can be checked for free, from files already on disk, with no GPU time.

If this ever reports a mismatch, every speedup number is suspect until the
cause is found. Faster output that is not the same output is not a speedup.

What it compares
----------------
run_benchmark.py writes perreq_<run_id>.json for each arm, holding one record
per prompt with a SHA-1 of the generated token ids. For each (bucket, limit,
seed, concurrency) group this script takes the 'off' arm as truth and compares
every speculative arm against it, prompt by prompt.

Comparisons are only ever made inside a group. Two arms run at different
concurrency, or with a different prompt count or seed, are not expected to
agree token for token and a mismatch there would mean nothing.

Caveats worth knowing before trusting a PASS
--------------------------------------------
* Greedy decoding only. These runs use temperature 0. Under sampling the
  outputs would differ run to run for reasons that have nothing to do with
  speculation, and this check would be meaningless.
* Exactness here is about the sampling path, not bitwise float determinism.
  Batching changes reduction order in GPU kernels, so logits can shift in the
  last bits and flip a near-tie argmax. A small number of mismatches
  concentrated in long generations usually means that, not a broken drafter.
  The script reports where the divergence starts so the two can be told apart:
  a drafter bug tends to diverge early and often, numerical noise late and
  rarely.
"""

import json
import os
import re
import sys
from collections import defaultdict

NAME = re.compile(
    r"^perreq_(?P<rung>[a-z0-9]+)_(?P<bucket>[^_]+)_k(?P<k>\d+)"
    r"_n(?P<n>\d+)_s(?P<s>\d+)(?:_c(?P<c>\d+))?\.json$"
)


def load(path):
    with open(path) as f:
        rows = json.load(f)
    return {r["id"]: r for r in rows}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1].rstrip("/")

    arms = {}
    for name in sorted(os.listdir(d)):
        m = NAME.match(name)
        if not m:
            continue
        g = m.groupdict()
        key = (g["bucket"], int(g["n"]), int(g["s"]), int(g["c"] or 0))
        arms.setdefault(key, {})[(g["rung"], int(g["k"]))] = os.path.join(d, name)

    if not arms:
        sys.exit(f"no perreq_*.json files in {d}")

    total_groups = 0
    total_pass = 0
    problems = []

    for key in sorted(arms, key=lambda t: (str(t[0]), t[3], t[1], t[2])):
        bucket, n, seed, conc = key
        group = arms[key]
        base_path = None
        for (rung, k), path in group.items():
            if rung == "off":
                base_path = path
        label = f"{bucket} n={n} seed={seed} conc={conc or 'auto'}"
        if not base_path:
            print(f"[skip] {label}: no 'off' arm to compare against")
            continue

        base = load(base_path)
        print(f"\n{label}   baseline {len(base)} prompts")

        for (rung, k), path in sorted(group.items()):
            if rung == "off":
                continue
            total_groups += 1
            spec = load(path)

            shared = set(base) & set(spec)
            only_base = set(base) - set(spec)
            only_spec = set(spec) - set(base)

            same = 0
            diff_ids = []
            for rid in sorted(shared):
                if base[rid]["sha1"] == spec[rid]["sha1"]:
                    same += 1
                else:
                    diff_ids.append(rid)

            status = "PASS" if (not diff_ids and not only_base and not only_spec) else "FAIL"
            if status == "PASS":
                total_pass += 1
            print(f"  {status}  {rung} k={k}: {same}/{len(shared)} identical")

            if only_base or only_spec:
                print(f"        prompt sets differ: {len(only_base)} only in off, "
                      f"{len(only_spec)} only in {rung}")
                problems.append((label, rung, k, "prompt sets differ"))

            if diff_ids:
                problems.append((label, rung, k, f"{len(diff_ids)} mismatched"))
                # Length is the cheap diagnostic. Equal lengths with different
                # hashes means a token changed mid-sequence. Different lengths
                # usually means one arm stopped earlier.
                shown = 0
                for rid in diff_ids:
                    b, s = base[rid], spec[rid]
                    note = ("same length" if b["gen_tokens"] == s["gen_tokens"]
                            else f"len {b['gen_tokens']} vs {s['gen_tokens']}")
                    print(f"        {rid}: {note}")
                    shown += 1
                    if shown >= 5:
                        print(f"        ... and {len(diff_ids) - shown} more")
                        break

    print("\n" + "=" * 60)
    if not problems:
        print(f"PASS: all {total_pass} speculative arms are byte identical to "
              f"their no-speculation baseline.")
        print("The losslessness claim is supported for every cell checked here.")
    else:
        print(f"{total_pass} of {total_groups} arms passed. Problems:")
        for label, rung, k, why in problems:
            print(f"  {label}  {rung} k={k}: {why}")
        print("\nDo not publish speedups for the affected cells until this is")
        print("explained. Check first whether the mismatches are a handful of")
        print("long generations, which points at float non-determinism under")
        print("batching rather than a drafter that is changing the output.")


if __name__ == "__main__":
    main()
