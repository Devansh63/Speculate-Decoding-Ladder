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

# Tail is extensible on purpose. The old pattern required ".json" straight
# after the optional _c<n>, so it would silently skip any "_nopc" file the
# harness writes, and every v3 "_j<jobid>" file.
# Tags are alphanumeric segments (no dots: the harness writes _T0p7, not
# _T0.7, so this stays unambiguous). The job id is always last.
NAME = re.compile(
    r"^perreq_(?P<rung>[a-z0-9]+)_(?P<bucket>[^_]+)_k(?P<k>\d+)"
    r"_n(?P<n>\d+)_s(?P<s>\d+)(?:_c(?P<c>\d+))?"
    r"(?P<tags>(?:_(?!j(?:\d|nojob)|r\d)[A-Za-z][A-Za-z0-9-]*)*)"
    r"(?:_j(?P<job>\d+|nojob))?(?:_r(?P<rep>\d+))?\.json$"
)


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)}


def agree(a, b):
    """Returns (identical, shared, same_length_count, per_token_hazard).

    The hazard is the probability that a token differs GIVEN that every token
    before it matched: diverging sequences divided by total matched positions.

    It exists because sequence-level agreement is a bad statistic. Agreement is
    roughly the hazard raised to the output length, so at 256 tokens it mostly
    measures how long the outputs are, not how much the engine perturbs them.
    C10's 12.7% and 19.0% correspond to hazards of about 0.80% and 0.65% -- and
    on the hazard, the ordering between them reverses.

    Returns nan for the hazard when token ids are unavailable (files written
    with --no-token-ids carry only a sha1, which cannot locate a divergence).
    """
    shared = set(a) & set(b)
    same = sum(1 for i in shared if a[i]["sha1"] == b[i]["sha1"])
    samelen = sum(1 for i in shared if a[i]["gen_tokens"] == b[i]["gen_tokens"])

    survived = 0
    diverged = 0
    usable = 0
    for i in shared:
        xa, xb = a[i].get("token_ids"), b[i].get("token_ids")
        if xa is None or xb is None:
            continue
        usable += 1
        nmin = min(len(xa), len(xb))
        first = next((p for p in range(nmin) if xa[p] != xb[p]), None)
        if first is None and len(xa) == len(xb):
            survived += nmin
        else:
            diverged += 1
            survived += first if first is not None else nmin
    hazard = (diverged / survived) if (usable and survived) else float("nan")
    return same, len(shared), samelen, hazard


def conc_label(c):
    return "auto" if not c else str(c)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1].rstrip("/")

    # (bucket, n, seed, tags) -> (rung, k, conc, job) -> path
    #
    # tags are in the OUTER key so a configuration variant (_nopc, _T0p7,
    # _loop) forms its own group and can never be compared against a normal
    # run. job is in the INNER key so two runs of the same cell are two
    # realizations instead of one silently overwriting the other.
    tree = defaultdict(dict)
    skipped = []
    for name in sorted(os.listdir(d)):
        if not name.startswith("perreq_") or not name.endswith(".json"):
            continue
        m = NAME.match(name)
        if not m:
            skipped.append(name)
            continue
        g = m.groupdict()
        # Realization id is job plus repeat. Two runs of one cell inside a
        # single job share the job id, so job alone collapses them back into
        # one entry -- which is the bug this whole change exists to stop.
        realization = (g.get("job") or "") + \
                      (("r" + g["rep"]) if g.get("rep") else "")
        tree[(g["bucket"], int(g["n"]), int(g["s"]), g.get("tags") or "")][
            (g["rung"], int(g["k"]), int(g["c"] or 0),
             realization)] = os.path.join(d, name)

    if skipped:
        # Loud, because a silent skip here is how a "_nopc" control would have
        # produced files that nothing read and nothing reported missing.
        print(f"WARNING: {len(skipped)} perreq file(s) did not match the name "
              f"pattern and were NOT analysed:")
        for nm in skipped[:10]:
            print(f"  {nm}")
        print()

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
    same_cfg_pairs = 0
    for key in sorted(tree, key=lambda t: (str(t[0]), t[1], t[2], t[3])):
        bucket, n, seed, tags = key
        tagtxt = f" tags={tags}" if tags else ""
        offs = {(c, job): p for (rung, k, c, job), p in tree[key].items()
                if rung == "off"}
        if len(offs) < 2:
            print(f"\n{bucket} n={n} seed={seed}{tagtxt}: only {len(offs)} "
                  f"baseline arm, no control possible")
            continue
        print(f"\n{bucket} n={n} seed={seed}{tagtxt}")
        loaded = {ck: load(p) for ck, p in offs.items()}
        for a1, a2 in combinations(sorted(loaded), 2):
            (c1, j1), (c2, j2) = a1, a2
            same, shared, samelen, hazard = agree(loaded[a1], loaded[a2])
            pct = 100.0 * same / shared if shared else float("nan")
            hz = f"{hazard * 100:.3f}%/tok" if hazard == hazard else "n/a"
            if c1 == c2:
                # THE control: same configuration, different run. Everything
                # that disagrees here is engine non-determinism, with nothing
                # else varying. This is the floor every other number needs.
                same_cfg_pairs += 1
                control_total += shared
                control_same += same
                print(f"  SAME-CONFIG c={conc_label(c1):<4} job {j1 or '?'} vs "
                      f"{j2 or '?'}: {same}/{shared} identical ({pct:.0f}%)   "
                      f"hazard {hz}   lengths equal on {samelen}/{shared}")
            else:
                # Kept for continuity, but it varies run AND concurrency, so it
                # is not a control. External review correction #1, 11 Oct.
                print(f"  confounded  c={conc_label(c1):<4} vs c={conc_label(c2):<4}"
                      f": {same}/{shared} identical ({pct:.0f}%)   hazard {hz}"
                      f"   [varies two things -- not the control]")

    if same_cfg_pairs == 0:
        print("\nNO same-configuration pair exists in this directory, so there "
              "is no\nnon-determinism floor and no null for any agreement number "
              "below. Run\nthe same cell twice into the same directory; the v3 "
              "job id in the file\nname keeps both.")

    # ------------------------------------------------------------- comparison
    print("\n" + "=" * 70)
    print("MAIN: speculative arms against the baseline at matching concurrency")
    print("=" * 70)

    spec_total = 0
    spec_same = 0
    for key in sorted(tree, key=lambda t: (str(t[0]), t[1], t[2], t[3])):
        bucket, n, seed, tags = key
        tagtxt = f" tags={tags}" if tags else ""
        by_conc = defaultdict(dict)
        for (rung, k, c, job), p in tree[key].items():
            by_conc[c][(rung, k, job)] = p
        for c in sorted(by_conc):
            arms = by_conc[c]
            offs = [(job, p) for (rung, k, job), p in arms.items() if rung == "off"]
            if not offs:
                continue
            for (rung, k, job), p in sorted(arms.items()):
                if rung == "off":
                    continue
                # Pair with the baseline from the SAME job where one exists;
                # otherwise say which baseline was used. Mixing a speculative
                # arm with a baseline from another job compares two things.
                pick = next((bp for bj, bp in offs if bj == job), None)
                tier = "same-job"
                if pick is None:
                    pick = sorted(offs)[0][1]
                    tier = "cross-job"
                base = load(pick)
                same, shared, _, hazard = agree(base, load(p))
                spec_total += shared
                spec_same += same
                pct = 100.0 * same / shared if shared else float("nan")
                hz = f"{hazard * 100:.3f}%/tok" if hazard == hazard else "n/a"
                print(f"\n{bucket} n={n} seed={seed}{tagtxt} "
                      f"conc={conc_label(c)} job {job or '?'}")
                print(f"  {rung:>7} k={k:<3}: {same}/{shared} identical "
                      f"({pct:.0f}%)   hazard {hz}   baseline {tier}")

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
