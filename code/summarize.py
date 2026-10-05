#!/usr/bin/env python3
"""Read results.jsonl and print the Phase 0 tables.

    python3 summarize.py /projects/bikd/$USER/Speculative_Ladder/results/a40

Speedup is computed against the 'off' arm in the same bucket AND at the same
concurrency. Comparing against a differently batched baseline is how the first
run produced three rungs that all appeared to lose.

Three things here are easy to get wrong and are handled explicitly.

1. Acceptance is truncated to a common draft length before any cross-rung
   comparison. A rung running at k=15 has more positions in which to lose
   tokens than one at k=5, so raw mean accepted length is not comparable.

2. mean_accepted_length from vLLM is CONDITIONAL on a draft having been
   proposed. A proposer that only drafts sometimes (n-gram fires only when
   prompt lookup finds a match) will show a flattering tau that it does not
   actually deliver. Coverage is reconstructed here: every step without a
   draft emits exactly one token, so

       steps_without_draft = gen_tokens - drafts * tau_cond
       total_steps         = drafts + steps_without_draft
       coverage            = drafts / total_steps
       tau_eff             = gen_tokens / total_steps

   tau_eff is the number that belongs in Speedup ~= tau / R. tau_cond is a
   property of the drafter when it fires; tau_eff is what the serving system
   experiences.

3. R is reported as tau_eff / speedup, which is the verification cost ratio
   implied by the measurement rather than a separately measured quantity.
   Label it as implied R in the paper, not as measured R.
"""

import json
import sys
from collections import defaultdict


def al_at_k(acc_by_pos, k):
    """Mean accepted length if the drafter had stopped at k positions."""
    if not acc_by_pos:
        return None
    return 1.0 + sum(acc_by_pos[:k])


def conc_of(rec):
    """Requested concurrency. Rows from before --max-num-seqs existed have none."""
    c = rec.get("max_num_seqs")
    return c if c else 0


def conc_label(c):
    return "auto" if not c else str(c)


def bucket_key(b):
    s = b.replace("K", "")
    return int(s) if s.isdigit() else 0


def derive(rec):
    """Coverage, effective tau and implied R for one row."""
    spec = rec.get("spec", {}) or {}
    drafts = spec.get("drafts") or 0
    tau_cond = spec.get("mean_accepted_length") or 1.0
    gen = rec.get("gen_tokens") or 0

    out = {
        "drafts": drafts,
        "tau_cond": tau_cond,
        "coverage": None,
        "tau_eff": None,
    }
    if rec.get("rung") == "off" or drafts <= 0 or gen <= 0:
        out["coverage"] = 0.0 if rec.get("rung") == "off" else None
        out["tau_eff"] = 1.0 if rec.get("rung") == "off" else None
        return out

    # Tokens emitted by drafted steps, and by plain steps.
    from_drafted = drafts * tau_cond
    plain = gen - from_drafted
    if plain < 0:
        # Rounding, or a counter that moved between snapshots. Treat as full
        # coverage rather than inventing negative steps.
        plain = 0.0
    total_steps = drafts + plain
    if total_steps <= 0:
        return out
    out["coverage"] = drafts / total_steps
    out["tau_eff"] = gen / total_steps
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1].rstrip("/") + "/results.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        sys.exit("no rows in " + path)

    # Last run wins if a cell was repeated.
    latest = {}
    for r in rows:
        latest[(r["bucket"], conc_of(r), r["rung"], r.get("k", 0))] = r

    baseline = {}
    for (bucket, conc, rung, k), r in latest.items():
        if rung == "off":
            baseline[(bucket, conc)] = r["throughput_tok_s"]

    order = sorted(latest, key=lambda t: (bucket_key(t[0]), t[1], t[2], t[3]))

    print(f"{'bucket':>7} {'conc':>5} {'rung':>8} {'k':>3} {'n':>4} "
          f"{'tok/s':>8} {'speedup':>8} {'tauC':>6} {'tauE':>6} {'cov':>6} "
          f"{'R':>6} {'AL@5':>6} {'pos1':>6}")
    print("-" * 94)
    for key in order:
        bucket, conc, rung, k = key
        r = latest[key]
        spec = r.get("spec", {}) or {}
        acc = spec.get("acceptance_by_position") or []
        d = derive(r)

        base = baseline.get((bucket, conc))
        speed = r["throughput_tok_s"] / base if base else float("nan")
        al5 = al_at_k(acc, 5)
        tauE = d["tau_eff"]
        R = (tauE / speed) if (tauE and speed == speed and speed) else float("nan")

        def fmt(x, w=6, p=3):
            return f"{x:>{w}.{p}f}" if isinstance(x, float) and x == x else f"{'':>{w}}"

        print(f"{bucket:>7} {conc_label(conc):>5} {rung:>8} {k:>3} {r['limit']:>4} "
              f"{r['throughput_tok_s']:>8.1f} {speed:>8.2f} "
              f"{fmt(d['tau_cond'])} {fmt(tauE)} {fmt(d['coverage'])} "
              f"{fmt(R)} {fmt(al5)} "
              f"{fmt(acc[0] if acc else float('nan'))}")

    print("\ntauC  mean accepted length, conditional on a draft being proposed")
    print("tauE  effective tokens per decode step, counting steps with no draft")
    print("cov   fraction of decode steps on which a draft was proposed")
    print("R     implied verification cost ratio, tauE / speedup")

    print("\ndraft counters")
    print(f"{'bucket':>7} {'conc':>5} {'rung':>8} {'k':>3} "
          f"{'drafts':>9} {'drafted':>10} {'accepted':>10} {'gen_tok':>9} {'acc_rate':>9}")
    print("-" * 74)
    for key in order:
        bucket, conc, rung, k = key
        if rung == "off":
            continue
        r = latest[key]
        s = r.get("spec", {}) or {}
        print(f"{bucket:>7} {conc_label(conc):>5} {rung:>8} {k:>3} "
              f"{s.get('drafts', 0):>9} {s.get('draft_tokens', 0):>10} "
              f"{s.get('accepted_tokens', 0):>10} {r.get('gen_tokens', 0):>9} "
              f"{s.get('draft_acceptance_rate', float('nan')):>9.3f}")

    print("\nacceptance by draft position")
    for key in order:
        bucket, conc, rung, k = key
        acc = (latest[key].get("spec", {}) or {}).get("acceptance_by_position")
        if acc:
            print(f"  {bucket:>5} c={conc_label(conc):<5} {rung:>7} k={k:<3} " +
                  ", ".join(f"{p:.3f}" for p in acc))

    # Concurrency sensitivity: the headline of Phase 0. How fast does each
    # rung lose its advantage as the batch grows, with tau held essentially
    # constant? A rung whose tauC barely moves while its speedup collapses is
    # evidence that the effect is entirely in R.
    print("\nconcurrency sensitivity, per bucket and rung")
    curves = defaultdict(dict)
    for key in order:
        bucket, conc, rung, k = key
        if rung == "off" or not conc:
            continue
        r = latest[key]
        base = baseline.get((bucket, conc))
        if not base:
            continue
        d = derive(r)
        curves[(bucket, rung, k)][conc] = (
            r["throughput_tok_s"] / base, d["tau_cond"], d["tau_eff"])
    for (bucket, rung, k), by_conc in sorted(curves.items(), key=lambda t: (bucket_key(t[0][0]), t[0][1], t[0][2])):
        parts = []
        for c in sorted(by_conc):
            sp, tc, te = by_conc[c]
            parts.append(f"c={c}: {sp:.2f}x (tauC {tc:.2f})")
        print(f"  {bucket:>5} {rung:>7} k={k:<3} " + "   ".join(parts))
        # Where does it cross 1.0x? Linear interpolation between the bracketing
        # concurrency levels. Reported as a range, because it is one.
        cs = sorted(by_conc)
        for a, b in zip(cs, cs[1:]):
            sa, sb = by_conc[a][0], by_conc[b][0]
            if sa >= 1.0 > sb:
                frac = (sa - 1.0) / (sa - sb) if sa != sb else 0.0
                x = a + frac * (b - a)
                print(f"        crosses 1.00x between concurrency {a} and {b}, "
                      f"interpolated at about {x:.1f}")
                break
        else:
            if by_conc[cs[-1]][0] > 1.0:
                print(f"        still above 1.00x at the widest concurrency tested ({cs[-1]})")
            elif by_conc[cs[0]][0] < 1.0:
                print(f"        already below 1.00x at the narrowest concurrency tested ({cs[0]})")

    # Decay across buckets, the number the milestone gate is written against.
    print("\ndecay D = 1 - AL(long) / AL(2K), at a common truncation of k=5, "
          "matched on concurrency")
    by_rung = defaultdict(dict)
    for key in order:
        bucket, conc, rung, k = key
        if rung == "off":
            continue
        acc = (latest[key].get("spec", {}) or {}).get("acceptance_by_position")
        if acc:
            by_rung[(rung, k, conc)][bucket] = al_at_k(acc, 5)
    printed = False
    for (rung, k, conc), buckets in sorted(by_rung.items(), key=lambda t: (t[0][0], t[0][1], t[0][2])):
        if "2K" not in buckets or len(buckets) < 2:
            continue
        short = buckets["2K"]
        for b in sorted(buckets, key=bucket_key):
            if b == "2K":
                continue
            print(f"  {rung} k={k} c={conc_label(conc)}: "
                  f"2K {short:.3f} -> {b} {buckets[b]:.3f}   D = {1.0 - buckets[b]/short:+.3f}")
            printed = True
    if not printed:
        print("  (needs at least two context buckets at the same concurrency)")

    print("\nNote: single runs, no confidence intervals. The milestone needs a "
          "paired bootstrap over prompts before any claim is made.")


if __name__ == "__main__":
    main()
