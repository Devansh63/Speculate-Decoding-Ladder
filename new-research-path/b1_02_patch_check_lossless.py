#!/usr/bin/env python3
"""
Block 1, step 2 -- patch check_lossless.py to survive the v3 file name, stop
silently skipping files, and expose the same-config control.

MUST land in the same commit as the harness v3 patch. The v3 harness writes
perreq_<run_id>_j<jobid>.json; the current NAME regex requires ".json"
immediately after the optional _c<n>, so every v3 file would fail to match and
line 80 would `continue` past it. The tool would report "no perreq files" or
quietly analyse none of the new ones.

Three problems, one of which predates v3
----------------------------------------
1. **Latent, and it would have bitten the next control run.** run_benchmark.py
   line 220 appends "_nopc" to run_id when --no-prefix-caching is passed,
   producing perreq_off_8K_k0_n50_s1234_c1_nopc.json. The regex cannot match
   that. Verified 13 Oct: results/a40 contains zero _nopc files, so no data has
   been lost -- the prefix-caching determinism control has simply never been
   run. But the first time anyone ran it, this tool would have skipped the
   files in silence and reported nothing missing.

2. **The job id would collide in the dict, not on disk.** The inner key is
   (rung, k, conc) with one path per key. Two jobs of the same cell map to the
   same key and `sorted(os.listdir)` lets the lexically last job id win,
   silently. Fixing the file name without fixing the key just relocates the
   overwrite.

3. **The control is confounded, and the fix is free.** `offs` is keyed by
   concurrency alone, so a control pair only forms between two baselines at
   DIFFERENT concurrency -- which varies two things at once and is exactly what
   external review correction #1 flagged about C10. With the job in the key,
   two `off` arms at the SAME concurrency from different jobs form the
   same-config control that has been on the to-do list since 5 Oct.

Four edits
----------
A. NAME regex accepts tag suffixes (_nopc, _T0p7, _pl3-7, _loop) and an
   optional trailing _j<jobid>.
B. Tags join the OUTER key, so a _nopc run is its own group and can never be
   compared against a normal run. Job joins the INNER key, so realizations are
   distinct.
C. The control splits into SAME-config pairs (the real control, reported
   separately) and cross-concurrency pairs (kept, labelled confounded).
D. agree() also returns the per-token divergence hazard, and the MAIN loop
   prefers a baseline from the same job. Sequence-level agreement is the hazard
   raised to the output length, so it mostly measures output length; the hazard
   is the statistic that compares across cells.

Usage
-----
  python3 b1_02_patch_check_lossless.py /path/to/check_lossless.py
  python3 b1_02_patch_check_lossless.py ... --write
"""

import argparse
import difflib
import os
import sys

EDITS = [

    ("A. NAME regex: tag suffixes and an optional job id",
     '''NAME = re.compile(
    r"^perreq_(?P<rung>[a-z0-9]+)_(?P<bucket>[^_]+)_k(?P<k>\\d+)"
    r"_n(?P<n>\\d+)_s(?P<s>\\d+)(?:_c(?P<c>\\d+))?\\.json$"
)''',
     '''# Tail is extensible on purpose. The old pattern required ".json" straight
# after the optional _c<n>, so it would silently skip any "_nopc" file the
# harness writes, and every v3 "_j<jobid>" file.
# Tags are alphanumeric segments (no dots: the harness writes _T0p7, not
# _T0.7, so this stays unambiguous). The job id is always last.
NAME = re.compile(
    r"^perreq_(?P<rung>[a-z0-9]+)_(?P<bucket>[^_]+)_k(?P<k>\\d+)"
    r"_n(?P<n>\\d+)_s(?P<s>\\d+)(?:_c(?P<c>\\d+))?"
    r"(?P<tags>(?:_(?!j(?:\\d|nojob)|r\\d)[A-Za-z][A-Za-z0-9-]*)*)"
    r"(?:_j(?P<job>\\d+|nojob))?(?:_r(?P<rep>\\d+))?\\.json$"
)'''),

    ("B. tags in the outer key, job in the inner key",
     '''    # (bucket, n, seed) -> (rung, k, conc) -> path
    tree = defaultdict(dict)
    for name in sorted(os.listdir(d)):
        m = NAME.match(name)
        if not m:
            continue
        g = m.groupdict()
        tree[(g["bucket"], int(g["n"]), int(g["s"]))][
            (g["rung"], int(g["k"]), int(g["c"] or 0))] = os.path.join(d, name)''',
     '''    # (bucket, n, seed, tags) -> (rung, k, conc, job) -> path
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
        realization = (g.get("job") or "") + \\
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
        print()'''),

    ("C. split the control into same-config and cross-concurrency",
     '''    control_total = 0
    control_same = 0
    for key in sorted(tree, key=lambda t: (str(t[0]), t[1], t[2])):
        bucket, n, seed = key
        offs = {c: p for (rung, k, c), p in tree[key].items() if rung == "off"}
        if len(offs) < 2:
            print(f"\\n{bucket} n={n} seed={seed}: only {len(offs)} baseline arm, "
                  f"no control possible")
            continue
        print(f"\\n{bucket} n={n} seed={seed}")
        loaded = {c: load(p) for c, p in offs.items()}
        for c1, c2 in combinations(sorted(loaded), 2):
            same, shared, samelen = agree(loaded[c1], loaded[c2])
            control_total += shared
            control_same += same
            pct = 100.0 * same / shared if shared else float("nan")
            print(f"  off c={conc_label(c1):<4} vs off c={conc_label(c2):<4}: "
                  f"{same}/{shared} identical ({pct:.0f}%)   "
                  f"lengths equal on {samelen}/{shared}")''',
     '''    control_total = 0
    control_same = 0
    same_cfg_pairs = 0
    for key in sorted(tree, key=lambda t: (str(t[0]), t[1], t[2], t[3])):
        bucket, n, seed, tags = key
        tagtxt = f" tags={tags}" if tags else ""
        offs = {(c, job): p for (rung, k, c, job), p in tree[key].items()
                if rung == "off"}
        if len(offs) < 2:
            print(f"\\n{bucket} n={n} seed={seed}{tagtxt}: only {len(offs)} "
                  f"baseline arm, no control possible")
            continue
        print(f"\\n{bucket} n={n} seed={seed}{tagtxt}")
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
        print("\\nNO same-configuration pair exists in this directory, so there "
              "is no\\nnon-determinism floor and no null for any agreement number "
              "below. Run\\nthe same cell twice into the same directory; the v3 "
              "job id in the file\\nname keeps both.")'''),

    ("D1. agree() reports the per-token hazard",
     '''def agree(a, b):
    """Returns (identical, shared, same_length_count)."""
    shared = set(a) & set(b)
    same = sum(1 for i in shared if a[i]["sha1"] == b[i]["sha1"])
    samelen = sum(1 for i in shared if a[i]["gen_tokens"] == b[i]["gen_tokens"])
    return same, len(shared), samelen''',
     '''def agree(a, b):
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
    return same, len(shared), samelen, hazard'''),

    ("D2. MAIN prefers a same-job baseline and reports the hazard",
     '''    spec_total = 0
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
            print(f"\\n{bucket} n={n} seed={seed} conc={conc_label(c)}   "
                  f"baseline {len(base)} prompts")
            for (rung, k), p in sorted(arms.items()):
                if rung == "off":
                    continue
                same, shared, _ = agree(base, load(p))
                spec_total += shared
                spec_same += same
                pct = 100.0 * same / shared if shared else float("nan")
                print(f"  {rung:>7} k={k:<3}: {same}/{shared} identical ({pct:.0f}%)")''',
     '''    spec_total = 0
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
                print(f"\\n{bucket} n={n} seed={seed}{tagtxt} "
                      f"conc={conc_label(c)} job {job or '?'}")
                print(f"  {rung:>7} k={k:<3}: {same}/{shared} identical "
                      f"({pct:.0f}%)   hazard {hz}   baseline {tier}")'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to check_lossless.py")
    ap.add_argument("--write", action="store_true",
                    help="write check_lossless.v3.py beside the original")
    a = ap.parse_args()

    src = open(a.path).read()
    text, problems = src, []
    for desc, anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            problems.append(f"  {desc}: anchor found {n} times, expected 1")
            continue
        text = text.replace(anchor, repl, 1)

    print(f"patching {a.path}  ({len(src.splitlines())} lines)")
    if problems:
        print("\nANCHOR FAILURES -- nothing written:")
        print("\n".join(problems))
        sys.exit(2)
    print(f"all {len(EDITS)} anchors matched exactly once")

    # Write BEFORE printing the diff: piping stdout into head closes the pipe
    # mid-diff and everything after it is skipped silently.
    if a.write:
        dest = os.path.join(os.path.dirname(os.path.abspath(a.path)),
                            "check_lossless.v3.py")
        open(dest, "w").write(text)
        print(f"wrote {dest}")
        print("Run it on results/a40 BEFORE swapping it in: the WARNING list "
              "should be\nempty and every existing number unchanged, with only "
              "the new labels and the\nhazard column added.")
    else:
        print("(--write to produce check_lossless.v3.py; nothing written)")

    print(f"{len(src.splitlines())} -> {len(text.splitlines())} lines; diff follows")
    sys.stdout.writelines(difflib.unified_diff(
        src.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile="check_lossless.py", tofile="check_lossless.v3.py", n=3))


if __name__ == "__main__":
    main()
