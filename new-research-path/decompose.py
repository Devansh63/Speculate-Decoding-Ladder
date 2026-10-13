#!/usr/bin/env python3
"""
decompose.py -- recover cov, tau_cond, tau_eff, R and the machine factor from
results.jsonl rows, with explicit baseline provenance.

Why this exists: summarize.py reports throughput. The paper needs the three-term
decomposition

    speedup  ~=  [ cov * tau_cond + (1 - cov) ] / R

so that drafter quality (tau) and machine cost (R) are separated. This script
derives each term per row and says, for every speedup it prints, exactly which
baseline row it was paired against.

v2, 13 Oct 2026 -- fixes from two independent reviews:
  * decode_only now requires the SAME HOST on both sides of the division.
    Prefill and per-token cost are machine-dependent; dividing across nodes was
    a silent error.
  * --harness filters rows by harness version. Mixing pre-v2 rows in silently
    was how a disqualified run reached a fit.
  * --exclude-host / --exclude-job drop known-bad cells loudly instead of
    averaging them in.
  * fit_k prints n and refuses to report R^2 for a two-point fit. Two points
    always fit a line exactly; printing R^2 = 1.0000 manufactures confidence.
  * cov is marked as PINNED when the clamp fires -- it is an artifact of
    truncation, not a measurement.
  * decode_only's intercept is labelled "intercept", not "prefill": at
    max_tokens=1 a speculative arm still pays one draft-and-verify cycle, so
    the intercept is arm-dependent. Only the DIFFERENCE between two token
    counts gives marginal decode cost.

Usage
-----
  python3 decompose.py results/a40
  python3 decompose.py results/a40 --compare .../results/a100
  python3 decompose.py results/a40_ngramk --fit-k
  python3 decompose.py results/a100 --decode-only
  python3 decompose.py results/a40 --harness v2 --exclude-host gpub073

Conventions honoured
--------------------
* A row is keyed on (bucket, rung, k, limit, max_num_seqs, max_tokens,
  prompt_style, prefix_caching) AND its slurm job. Repeated cells are never
  silently merged -- every one is printed.
* A speculative row is paired with an 'off' row from the SAME job where
  possible, then the same host, then anything matching the cell shape. The
  pairing tier is printed. Tier 'loose' results should not go in a table
  without a note.
* COMPLETED does not mean measured. Rows with gen_tokens == 0 are dropped loudly.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

CELL_FIELDS = ("bucket", "limit", "max_num_seqs", "max_tokens",
               "prompt_style", "prefix_caching")


# ---------------------------------------------------------------- loading

def job_of(r):
    return (r.get("env") or {}).get("slurm_job", "?")


def host_of(r):
    return ((r.get("env") or {}).get("host") or "?").split(".")[0]


def gpu_of(r):
    return (r.get("env") or {}).get("gpu", "?")


def cell_of(r):
    return tuple(r.get(f) for f in CELL_FIELDS)


def load(path, harness=None, excl_hosts=(), excl_jobs=()):
    """Read results.jsonl from a directory or a file path, with filters."""
    if os.path.isdir(path):
        path = os.path.join(path, "results.jsonl")
    if not os.path.exists(path):
        sys.exit(f"no results.jsonl at {path}")

    rows, bad = [], 0
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            r["_src"], r["_line"] = path, n
            rows.append(r)
    if bad:
        print(f"  WARNING {path}: {bad} unparseable line(s)", file=sys.stderr)

    # harness versions present -- a tree with more than one is a mixing hazard
    versions = sorted({str(r.get("harness")) for r in rows})
    if len(versions) > 1:
        print(f"  NOTE {path}: {len(versions)} harness versions present: "
              f"{', '.join(versions)}", file=sys.stderr)
        if harness is None:
            print("  WARNING rows from different harness versions are being "
                  "mixed. Pass --harness to filter.", file=sys.stderr)

    kept, drops = [], defaultdict(int)
    for r in rows:
        if not r.get("gen_tokens"):
            drops["gen_tokens == 0 (COMPLETED but produced nothing)"] += 1
            continue
        if harness and not str(r.get("harness", "")).startswith(harness):
            drops[f"harness != {harness}*"] += 1
            continue
        if host_of(r) in excl_hosts:
            drops[f"excluded host {host_of(r)}"] += 1
            continue
        if job_of(r) in excl_jobs:
            drops[f"excluded job {job_of(r)}"] += 1
            continue
        kept.append(r)

    for reason, n in sorted(drops.items()):
        print(f"  DROPPED {n:>3} row(s): {reason}", file=sys.stderr)
    if not kept:
        sys.exit(f"every row in {path} was filtered out")
    return kept


# ---------------------------------------------------------------- the math

def terms(r):
    """
    Return (cov, tau_cond, tau_eff, pinned, note).

    Derivation, using only quantities the engine reports:

      drafts          = decode steps on which a draft was proposed
      accepted_tokens = draft tokens accepted (the bonus token is NOT included,
                        which is why drafts + accepted == drafts * tau_cond)
      tokens on drafted steps     = drafts + accepted_tokens
      tokens on non-drafted steps = gen_tokens - the above
      total steps = drafts + non-drafted steps
      cov      = drafts / total steps
      tau_cond = 1 + accepted_tokens / drafts   (== mean_accepted_length)
      tau_eff  = cov * tau_cond + (1 - cov)

    Edge case that really happens: for a drafter proposing on every step,
    drafts + accepted can EXCEED gen_tokens, because the final draft of each
    request is truncated at max_tokens. It fires on real rows -- A100 8K DFlash
    k=7 overruns by 38 tokens, A40 2K DFlash k=15 by 205.

    When that happens we clamp the non-drafted count at 0, which FORCES cov to
    exactly 1.000. That is an artifact of truncation, not a measurement: the
    true cov is <= 1 and is not resolvable from aggregate counters. Such rows
    are marked PINNED and cov must not be quoted from them.
    """
    spec = r.get("spec") or {}
    gen = r["gen_tokens"]
    drafts = spec.get("drafts") or 0
    accepted = spec.get("accepted_tokens") or 0

    if drafts == 0:
        return 0.0, 1.0, 1.0, False, ""

    plain_tokens = gen - (drafts + accepted)
    pinned, note = False, ""
    if plain_tokens < 0:
        pinned = True
        note = f"cov PINNED by clamp (+{-plain_tokens} tok over gen_tokens)"
        plain_tokens = 0

    steps = drafts + plain_tokens
    cov = drafts / steps if steps else 0.0
    tau_cond = 1.0 + accepted / drafts

    reported = spec.get("mean_accepted_length")
    if reported and abs(reported - tau_cond) > 1e-6:
        note = (note + "; " if note else "") + \
               f"tau_cond {tau_cond:.4f} != reported {reported:.4f}"

    return cov, tau_cond, cov * tau_cond + (1.0 - cov), pinned, note


def pick_baseline(row, offs):
    """
    Choose an 'off' row to pair with. Returns (baseline_row, tier) where tier is
    'job' | 'host' | 'loose', or (None, None).
    """
    same_cell = [o for o in offs if cell_of(o) == cell_of(row)]
    if not same_cell:
        return None, None
    for tier, pred in (("job", lambda o: job_of(o) == job_of(row)),
                       ("host", lambda o: host_of(o) == host_of(row))):
        hit = [o for o in same_cell if pred(o)]
        if hit:
            return hit[0], tier
    return same_cell[0], "loose"


def analyse(rows):
    offs = [r for r in rows if r.get("rung") == "off"]
    out = []
    for r in rows:
        cov, tau_cond, tau_eff, pinned, note = terms(r)
        base, tier = pick_baseline(r, offs)
        speedup = R = None
        if base and base.get("throughput_tok_s"):
            speedup = r["throughput_tok_s"] / base["throughput_tok_s"]
            if speedup > 0:
                R = tau_eff / speedup
        out.append(dict(row=r, cov=cov, tau_cond=tau_cond, tau_eff=tau_eff,
                        pinned=pinned, speedup=speedup, R=R,
                        base=base, tier=tier, note=note))
    return out


# ---------------------------------------------------------------- reporting

HDR = (f"{'bucket':>6} {'rung':>7} {'k':>3} {'c':>3} {'mtok':>5} "
       f"{'n':>4} {'job':>9} {'node':>8} {'tok/s':>8} {'speedup':>8} "
       f"{'cov':>7} {'tau_c':>6} {'tau_e':>6} {'R':>6} {'pair':>5}")


def report(recs, title):
    print(f"\n=== {title} ===")
    print(HDR)

    def sort_key(d):
        r = d["row"]
        order = {"off": 0, "ngram": 1, "dflash": 2}
        return (str(r.get("bucket")), order.get(r.get("rung"), 9),
                r.get("k") or 0, r.get("max_num_seqs") or 0, job_of(r))

    for d in sorted(recs, key=sort_key):
        r = d["row"]
        sp = f"{d['speedup']:.4f}" if d["speedup"] is not None else "--"
        RR = f"{d['R']:.4f}" if d["R"] is not None else "--"
        cov = f"{d['cov']:.3f}" + ("*" if d["pinned"] else " ")
        print(f"{str(r.get('bucket')):>6} {str(r.get('rung')):>7} "
              f"{r.get('k') or 0:>3} {r.get('max_num_seqs') or 0:>3} "
              f"{r.get('max_tokens'):>5} {r.get('limit'):>4} "
              f"{job_of(r):>9} {host_of(r):>8} "
              f"{r['throughput_tok_s']:>8.2f} {sp:>8} "
              f"{cov:>7} {d['tau_cond']:>6.3f} {d['tau_eff']:>6.3f} "
              f"{RR:>6} {str(d['tier'] or '--'):>5}")
        if d["note"]:
            print(f"{'':>14} -> {d['note']}")

    if any(d["pinned"] for d in recs):
        print("\n  * cov pinned at 1.000 by draft truncation -- an artifact, "
              "not a measurement. Do not quote cov from these rows.")
    loose = [d for d in recs if d["tier"] == "loose"]
    if loose:
        print(f"\n  WARNING {len(loose)} row(s) paired 'loose' -- the baseline "
              f"came from a different job AND a different node. Do not table "
              f"these without a note.")
    print(f"\n  GPUs: {', '.join(sorted({gpu_of(d['row']) for d in recs}))}")
    print(f"  harness: {', '.join(sorted({str(d['row'].get('harness')) for d in recs}))}")


def compare(a_recs, b_recs, a_name, b_name):
    """Machine factor = R_b / R_a for cells present in both."""
    def index(recs):
        ix = defaultdict(list)
        for d in recs:
            r = d["row"]
            if r.get("rung") == "off" or d["R"] is None:
                continue
            ix[(r.get("bucket"), r.get("rung"), r.get("k"),
                r.get("max_num_seqs"), r.get("max_tokens"))].append(d)
        return ix

    ia, ib = index(a_recs), index(b_recs)
    keys = sorted(set(ia) & set(ib), key=lambda k: (str(k[0]), str(k[1]), k[2] or 0))
    if not keys:
        print("\n=== machine factor ===\n  no cells in common")
        return

    print(f"\n=== machine factor: R({b_name}) / R({a_name}) ===")
    print(f"{'bucket':>6} {'rung':>7} {'k':>3} {'c':>3} "
          f"{'R_a':>7} {'R_b':>7} {'factor':>7} {'tau_a':>7} {'tau_b':>7} {'d_tau':>7}")
    for k in keys:
        da, db = ia[k], ib[k]
        Ra = sum(d["R"] for d in da) / len(da)
        Rb = sum(d["R"] for d in db) / len(db)
        ta = sum(d["tau_eff"] for d in da) / len(da)
        tb = sum(d["tau_eff"] for d in db) / len(db)
        dt = (tb / ta - 1.0) * 100 if ta else float("nan")
        print(f"{str(k[0]):>6} {str(k[1]):>7} {k[2] or 0:>3} {k[3] or 0:>3} "
              f"{Ra:>7.4f} {Rb:>7.4f} {Rb/Ra:>7.4f} "
              f"{ta:>7.4f} {tb:>7.4f} {dt:>+6.2f}%")
        if len(da) > 1 or len(db) > 1:
            print(f"{'':>14} -> averaged over {len(da)} x {a_name}, "
                  f"{len(db)} x {b_name} run(s)")
        if abs(dt) > 1.0:
            print(f"{'':>14} -> d_tau {dt:+.2f}% exceeds 1%. tau should transfer; "
                  f"check token_ids for output divergence before using this cell.")
    print("\n  tau should transfer across machines (small d_tau); R should not."
          "\n  A factor that differs BY RUNG is the drafter-cost signal.")


def fit_k(recs):
    """
    Fit R(k) = a + c*k per (bucket, rung, concurrency). Intercept a = 1 + R_draft,
    so R_draft = a - 1 is the fixed per-step drafting cost and c is the marginal
    cost of each extra verified position.
    """
    groups = defaultdict(list)
    for d in recs:
        r = d["row"]
        if r.get("rung") == "off" or d["R"] is None or not r.get("k"):
            continue
        groups[(r.get("bucket"), r.get("rung"), r.get("max_num_seqs"))].append(
            (r["k"], d["R"]))

    print("\n=== fit R(k) = (1 + R_draft) + c*k ===")
    any_fit = False
    for key in sorted(groups, key=lambda t: (str(t[0]), str(t[1]))):
        pts = sorted(groups[key])
        ks = sorted({k for k, _ in pts})
        label = f"  bucket={str(key[0]):>4} rung={str(key[1]):>7} c={key[2] or 0}"
        if len(ks) < 2:
            print(f"{label}  only k={ks} -- need >=2 distinct k, skipped")
            continue
        any_fit = True
        n = len(pts)
        sx = sum(k for k, _ in pts); sy = sum(v for _, v in pts)
        sxx = sum(k * k for k, _ in pts); sxy = sum(k * v for k, v in pts)
        c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a = (sy - c * sx) / n

        # R^2 is meaningless with two points -- a line through two points is
        # exact by construction. Refuse to print it rather than imply a fit.
        if len(ks) > 2:
            ybar = sy / n
            ss_tot = sum((v - ybar) ** 2 for _, v in pts)
            ss_res = sum((v - (a + c * k)) ** 2 for k, v in pts)
            r2 = f"{1 - ss_res / ss_tot:.4f}" if ss_tot > 1e-12 else "n/a"
        else:
            r2 = "n/a (2 points: a line is exact by construction)"

        flag = "  <-- NOT PHYSICAL" if a < 1.0 else ""
        print(f"{label}  k={ks} n={n}  R_draft={a - 1:+.4f}  c={c:+.4f}  "
              f"R2={r2}{flag}")
    if not any_fit:
        print("  nothing fittable -- run a k-sweep (>=2 distinct k per cell) first")
    else:
        print("\n  R_draft is the x-axis of the collapse figure. A negative "
              "R_draft is\n  not physical and means the linear form is wrong "
              "for that cell.")


def decode_only(recs):
    """
    Pair rows differing ONLY in max_tokens, on the SAME HOST, to split the
    fixed cost from the marginal decode cost:

        elapsed(T) = intercept + T_total * d

    With two points, d = (e2 - e1) / (g2 - g1) and intercept = e1 - g1*d.

    The intercept is NOT pure prefill: at max_tokens=1 a speculative arm still
    pays one draft-and-verify cycle, so the intercept is arm-dependent. Only d,
    the marginal per-token cost, is comparable across arms -- and d_off / d_spec
    is the decode-only speedup, free of the prefill dilution that pulls every
    end-to-end ratio toward 1.0.
    """
    groups = defaultdict(list)
    for d in recs:
        r = d["row"]
        groups[(r.get("bucket"), r.get("rung"), r.get("k"), r.get("limit"),
                r.get("max_num_seqs"), host_of(r))].append(r)

    print("\n=== fixed / marginal split (needs >=2 max_tokens per cell, same host) ===")
    per_cell = {}
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        pts = sorted({(r["gen_tokens"], r["elapsed_s"]) for r in rs})
        if len(pts) < 2:
            continue
        (g1, e1), (g2, e2) = pts[0], pts[-1]
        if g2 == g1:
            continue
        dd = (e2 - e1) / (g2 - g1)
        per_cell[key] = dd
        print(f"  bucket={str(key[0]):>4} rung={str(key[1]):>7} k={key[2] or 0} "
              f"c={key[4] or 0} node={key[5]:>8}  intercept={e1 - g1 * dd:8.2f}s  "
              f"marginal={dd * 1000:7.3f}ms/tok  (T={g1} and {g2})")

    if not per_cell:
        print("  no cell has two different max_tokens values on the same host.")
        print("  Run the same cell with --max-tokens 1 to get this.")
        return

    print("\n  decode-only speedup (marginal_off / marginal_spec), same host only:")
    found = False
    for key, dd in sorted(per_cell.items(), key=lambda kv: str(kv[0])):
        if key[1] == "off":
            continue
        # bucket, limit, concurrency AND host must all match -- per-token cost
        # is machine-dependent, so dividing across nodes is meaningless.
        cand = [v for k, v in per_cell.items()
                if k[1] == "off" and k[0] == key[0] and k[3] == key[3]
                and k[4] == key[4] and k[5] == key[5]]
        if not cand:
            print(f"  bucket={str(key[0]):>4} rung={str(key[1]):>7} k={key[2] or 0}  "
                  f"-- no 'off' arm on node {key[5]}, skipped")
            continue
        found = True
        print(f"  bucket={str(key[0]):>4} rung={str(key[1]):>7} k={key[2] or 0} "
              f"node={key[5]:>8}  decode-only speedup = {cand[0] / dd:.4f}")
    if not found:
        print("  no speculative arm had a same-host 'off' arm to divide by.")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", help="results dir (or results.jsonl) to analyse")
    ap.add_argument("--compare", metavar="TREE2",
                    help="second results dir; prints the machine factor R2/R1")
    ap.add_argument("--fit-k", action="store_true",
                    help="fit R(k) to recover R_draft and the per-position cost")
    ap.add_argument("--decode-only", action="store_true",
                    help="split fixed from marginal cost using paired max_tokens rows")
    ap.add_argument("--harness", metavar="PREFIX",
                    help="keep only rows whose harness starts with PREFIX "
                         "(e.g. v2 or v3). Strongly recommended on a mixed tree.")
    ap.add_argument("--exclude-host", action="append", default=[], metavar="NODE",
                    help="drop rows from this node (repeatable; e.g. gpub073)")
    ap.add_argument("--exclude-job", action="append", default=[], metavar="JOBID",
                    help="drop rows from this slurm job (repeatable)")
    args = ap.parse_args()

    excl_h, excl_j = set(args.exclude_host), set(args.exclude_job)
    if excl_h or excl_j:
        print(f"  exclusions: hosts={sorted(excl_h) or '-'} "
              f"jobs={sorted(excl_j) or '-'}", file=sys.stderr)

    def name(p):
        return os.path.basename(os.path.abspath(p.rstrip("/")))

    a = analyse(load(args.tree, args.harness, excl_h, excl_j))
    report(a, name(args.tree))
    if args.fit_k:
        fit_k(a)
    if args.decode_only:
        decode_only(a)

    if args.compare:
        b = analyse(load(args.compare, args.harness, excl_h, excl_j))
        report(b, name(args.compare))
        if args.fit_k:
            fit_k(b)
        if args.decode_only:
            decode_only(b)
        compare(a, b, name(args.tree), name(args.compare))


if __name__ == "__main__":
    main()
