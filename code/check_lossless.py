#!/usr/bin/env python3
"""Check output stability: speculative arms against the baseline, AND the
baseline against itself.

    python3 check_lossless.py /projects/bikd/$USER/Speculative_Ladder/results/a40

Why the second half matters more than the first
-----------------------------------------------
The obvious test is whether a speculative arm reproduces the no-speculation
output. If it does not, the natural conclusion is that speculation changed the
answer, which would be a serious bug: speculative decoding is only acceptable
because the target model verifies every drafted token.

That conclusion is only valid if the baseline is itself reproducible. It may
not be. Changing how many requests share a batch changes the order of
reductions inside GPU kernels, so logits move in their last bits, and a greedy
argmax on a near-tie can flip. Prefix caching adds a second path to the same
problem: whether a request reuses cached keys and values or recomputes them
depends on arrival order, and the two are not bitwise identical.

So this script runs the control first. It compares the 'off' arm against the
'off' arm at different concurrency: same prompts, same seed, no speculation
anywhere. Any disagreement there is pure engine non-determinism, and it bounds
how much of the speculative mismatch can be blamed on speculation.

  baseline unstable  ->  the mismatch is the engine, and the losslessness
                         claim has to be stated as greedy-path equality up to
                         floating point, which is the honest version anyway
  baseline stable    ->  speculation really is changing the output, and no
                         speedup in this project is publishable until it is
                         explained

A length comparison is useless on this data, incidentally. Every generation
runs to the 256 token cap, so all sequences are the same length by
construction and the only thing that can differ is content. Diagnosing where
they diverge needs the token ids themselves, which the harness does not yet
store; it keeps only a SHA-1. Fixing that is the follow-up.
"""

import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

NAME = re.compile(
    r"^perreq_(?P<rung>[a-z0-9]+)_(?P<bucket>[^_]+)_k(?P<k>\d+)"
    r"_n(?P<n>\d+)_s(?P<s>\d+)(?:_c(?P<c>\d+))?\.json$"
)


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)}


def agree(a, b):
    """Returns (identical, shared, same_length_count)."""
    shared = set(a) & set(b)
    same = sum(1 for i in shared if a[i]["sha1"] == b[i]["sha1"])
    samelen = sum(1 for i in shared if a[i]["gen_tokens"] == b[i]["gen_tokens"])
    return same, len(shared), samelen


def conc_label(c):
    return "auto" if not c else str(c)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1].rstrip("/")

    # (bucket, n, seed) -> (rung, k, conc) -> path
    tree = defaultdict(dict)
    for name in sorted(os.listdir(d)):
        m = NAME.match(name)
        if not m:
            continue
        g = m.groupdict()
        tree[(g["bucket"], int(g["n"]), int(g["s"]))][
            (g["rung"], int(g["k"]), int(g["c"] or 0))] = os.path.join(d, name)

    if not tree:
        sys.exit(f"no perreq_*.json files in {d}")

    # ---------------------------------------------------------------- control
    print("=" * 70)
    print("CONTROL: is the no-speculation baseline reproducible across batching?")
    print("Same prompts, same seed, no speculation. Any disagreement here is")
    print("engine non-determinism and has nothing to do with speculation.")
    print("=" * 70)

    control_total = 0
    control_same = 0
    for key in sorted(tree, key=lambda t: (str(t[0]), t[1], t[2])):
        bucket, n, seed = key
        offs = {c: p for (rung, k, c), p in tree[key].items() if rung == "off"}
        if len(offs) < 2:
            print(f"\n{bucket} n={n} seed={seed}: only {len(offs)} baseline arm, "
                  f"no control possible")
            continue
        print(f"\n{bucket} n={n} seed={seed}")
        loaded = {c: load(p) for c, p in offs.items()}
        for c1, c2 in combinations(sorted(loaded), 2):
            same, shared, samelen = agree(loaded[c1], loaded[c2])
            control_total += shared
            control_same += same
            pct = 100.0 * same / shared if shared else float("nan")
            print(f"  off c={conc_label(c1):<4} vs off c={conc_label(c2):<4}: "
                  f"{same}/{shared} identical ({pct:.0f}%)   "
                  f"lengths equal on {samelen}/{shared}")

    # ------------------------------------------------------------- comparison
    print("\n" + "=" * 70)
    print("MAIN: speculative arms against the baseline at matching concurrency")
    print("=" * 70)

    spec_total = 0
    spec_same = 0
    for key in sorted(tree, key=lambda t: (str(t[0]), t[1], t[2])):
        bucket, n, seed = key
        by_conc = defaultdict(dict)
        for (rung, k, c), p in tree[key].items():
            by_conc[c][(rung, k)] = p
        for c in sorted(by_conc):
            arms = by_conc[c]
            base_p = next((p for (rung, k), p in arms.items() if rung == "off"), None)
            if not base_p:
                continue
            base = load(base_p)
            print(f"\n{bucket} n={n} seed={seed} conc={conc_label(c)}   "
                  f"baseline {len(base)} prompts")
            for (rung, k), p in sorted(arms.items()):
                if rung == "off":
                    continue
                same, shared, _ = agree(base, load(p))
                spec_total += shared
                spec_same += same
                pct = 100.0 * same / shared if shared else float("nan")
                print(f"  {rung:>7} k={k:<3}: {same}/{shared} identical ({pct:.0f}%)")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if control_total == 0:
        print("No control available. Run the 'off' arm at two concurrency levels")
        print("on the same prompts before drawing any conclusion from the")
        print("speculative comparison above.")
        return

    c_rate = 100.0 * control_same / control_total
    s_rate = 100.0 * spec_same / spec_total if spec_total else float("nan")
    print(f"baseline vs baseline : {control_same}/{control_total} identical ({c_rate:.0f}%)")
    print(f"speculative vs base  : {spec_same}/{spec_total} identical ({s_rate:.0f}%)")
    print()

    if c_rate > 99.0 and s_rate < 95.0:
        print("The baseline reproduces itself but the speculative arms do not.")
        print("That points at speculation genuinely altering the output, which")
        print("would be a real defect. Do not publish any speedup until it is")
        print("explained. First things to try: disable prefix caching, then")
        print("re-check; and store full token ids so the first point of")
        print("divergence can be located.")
    elif c_rate < 95.0:
        print("The baseline does not reproduce itself across batching, so the")
        print("mismatch in the speculative arms cannot be attributed to")
        print("speculation. This is engine level non-determinism: reduction")
        print("order inside GPU kernels changes with batch composition, and")
        print("prefix caching adds a second path to the same effect. A single")
        print("flipped token anywhere in a 256 token sequence changes the hash,")
        print("so even a very small per-token flip rate produces a large")
        print("fraction of mismatched sequences.")
        print()
        print("What to do about it:")
        print("  1. Re-run one cell with prefix caching disabled and compare")
        print("     the control rate. If it jumps, caching was the main cause.")
        print("  2. Store full token ids, not just a hash, so the fraction of")
        print("     differing TOKENS can be reported instead of the fraction of")
        print("     differing sequences. Per token is the meaningful number.")
        print("  3. State the claim honestly in the paper: speculative decoding")
        print("     preserves the greedy decoding path up to floating point")
        print("     non-determinism that the serving engine already exhibits")
        print("     without speculation. Quote both rates side by side.")
    else:
        print("Both rates are similar, which is consistent with a common cause")
        print("in the engine rather than in speculation. Store full token ids")
        print("and report per token agreement before making any claim.")


if __name__ == "__main__":
    main()
