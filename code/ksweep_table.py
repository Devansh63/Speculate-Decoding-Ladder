#!/usr/bin/env python3
"""Read k-sweep results and answer one question: does the best k move?

    python3 ksweep_table.py results/a40_ksweep results/a40_ksweep_d_*
    python3 ksweep_table.py results/a40 --assume-pin-defaults     # the filed v2/v3 rows
    python3 ksweep_table.py results/a40_ksweep --tie 3

No GPU. Login node or a laptop.

Rules this obeys (RULES sections 10 and 11, and the 2 Nov async decision):

  - Every record is kept and the unit is the JOB. A speedup is a ratio of two
    throughputs measured in the same job, at the same bucket, domain,
    concurrency, prompt count, generation cap, prefix-caching state,
    temperature and timing mode.
  - A speedup is printed only against an `off` arm whose RESOLVED
    async-scheduling state matches the arm's. A CPU n-gram arm (sync at the
    pin) is compared with the sync off arm ksweep.sh runs for it, never with
    the async one. Where no matched baseline exists the ratio is not printed.
  - Rows from harness v2 and v3 carry no resolved async state. They get no
    ratio unless --assume-pin-defaults is given, which fills in what
    vllm/config/vllm.py does at bf13ecc2f when the option is unset (on for off,
    dflash and ngram_gpu; off for CPU ngram) and marks the state "inferred".
    Under that flag the filed CPU n-gram rows correctly show NO speedup,
    because no sync baseline was ever filed for them.
  - Nothing is averaged across jobs. A cell measured in more than one job is
    listed once per job.
  - A difference smaller than --tie percent (default 5) is called a tie.
    There are no intervals on any of this until the disjoint-subset jobs run,
    so "best k" below means "best k in this one realization".

Four tables:

  1. arms: every record, with its matched speedup, tau and coverage.
  2. best k: per (gpu, domain, bucket, concurrency, rung, job), the best k,
     the runner-up, and whether the gap clears the tie threshold.
  3. does k need to adapt: per rung, the best single k across all cells
     against the best k per cell. This is the TA's question. If the per-cell
     optimum buys less than the tie threshold over one fixed k, adaptive k for
     that rung is not supported by this data.
  4. what a policy could win: best fixed (rung, k) across cells, best k per
     cell within the best rung, and best (rung, k) per cell with off allowed.
     Request-weighted time, restricted to arms that share one async state.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import derive, conc_of, conc_label, bucket_key  # noqa: E402

PIN_DEFAULT_ASYNC = {"off": True, "dflash": True, "ngram_gpu": True, "ngram": False}


def gpu_tag(rec):
    g = ((rec.get("env") or {}).get("gpu") or "").upper()
    for t in ("A40", "A100", "H200", "H100"):
        if t in g:
            return t.lower()
    return g.lower() or "unknown"


def job_of(rec):
    return str((rec.get("env") or {}).get("slurm_job") or "none")


def async_state(rec, assume):
    """(state, how) where state is True, False or None."""
    res = rec.get("resolved") or {}
    if res.get("async_scheduling") is not None:
        return bool(res["async_scheduling"]), "resolved"
    if assume and rec.get("rung") in PIN_DEFAULT_ASYNC:
        return PIN_DEFAULT_ASYNC[rec["rung"]], "inferred"
    return None, "unknown"


def pc_state(rec):
    res = rec.get("resolved") or {}
    if res.get("prefix_caching") is not None:
        return bool(res["prefix_caching"])
    return rec.get("prefix_caching")


def config_sig(rec):
    """Everything besides rung, k and async that must match inside a ratio."""
    return (rec.get("limit"), rec.get("max_tokens"), pc_state(rec),
            rec.get("temperature", 0.0), bool(rec.get("loop_timing")),
            rec.get("prompt_style"), rec.get("per_request_spec", "none"))


def load(dirs, assume, exclude_jobs):
    rows = []
    for d in dirs:
        path = os.path.join(d, "results.jsonl")
        if not os.path.exists(path):
            print("no results.jsonl in %s" % d, file=sys.stderr)
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if job_of(r) in exclude_jobs:
                continue
            r["_dir"] = d
            r["_async"], r["_async_how"] = async_state(r, assume)
            r["_domain"] = r.get("domain") or "all"
            r["_cell"] = (gpu_tag(r), r["_domain"], r.get("bucket"), conc_of(r), job_of(r))
            rows.append(r)
    return rows


def attach_speedups(rows):
    """Set r['_speedup'] against the matched off arm of the same job, or None."""
    base = defaultdict(list)
    for r in rows:
        if r.get("rung") == "off":
            base[(r["_cell"], config_sig(r), r["_async"])].append(r)
    dup = []
    for r in rows:
        r["_speedup"] = None
        r["_why"] = ""
        if r["_async"] is None:
            r["_why"] = "async state unknown"
            continue
        cands = base.get((r["_cell"], config_sig(r), r["_async"]), [])
        if not cands:
            r["_why"] = "no off arm with async=%s in this job" % ("on" if r["_async"] else "off")
            continue
        if len(cands) > 1:
            dup.append((r["_cell"], r["_async"], len(cands)))
        b = cands[0]
        if r is b:
            r["_speedup"] = 1.0
        elif b.get("throughput_tok_s"):
            r["_speedup"] = r.get("throughput_tok_s", 0.0) / b["throughput_tok_s"]
    return sorted(set(dup))


def a_label(r):
    if r["_async"] is None:
        return "?"
    return ("on" if r["_async"] else "off") + ("*" if r["_async_how"] == "inferred" else "")


def cell_sort(c):
    gpu, dom, bucket, conc, job = c
    return (gpu, dom != "all", dom, bucket_key(bucket or ""), conc, job)


def table_arms(rows):
    print("=" * 100)
    print("1. arms. speedup is against the off arm of the same job with the same async state.")
    print("   async: on / off as the engine resolved it; * = inferred from the pin's defaults.")
    print("=" * 100)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["_cell"]].append(r)
    for c in sorted(by_cell, key=cell_sort):
        gpu, dom, bucket, conc, job = c
        rs = by_cell[c]
        host = (rs[0].get("env") or {}).get("host", "")
        print("\n%s  bucket %s  domain %s  conc %s  job %s  %s  n=%s"
              % (gpu, bucket, dom, conc_label(conc), job, host.split(".")[0], rs[0].get("limit")))
        print("  %-10s %3s %-5s %9s %8s %8s %8s  %s"
              % ("rung", "k", "async", "tok/s", "speedup", "tau_c", "cover", ""))
        for r in sorted(rs, key=lambda x: (x.get("rung") != "off", x.get("rung"),
                                           x.get("k") or 0, str(x["_async"]))):
            d = derive(r)
            sp = "%.3f" % r["_speedup"] if r["_speedup"] is not None else "-"
            tau = "%.3f" % d["tau_cond"] if r.get("rung") != "off" else ""
            cov = "%.3f" % d["coverage"] if d["coverage"] is not None and r.get("rung") != "off" else ""
            print("  %-10s %3s %-5s %9.1f %8s %8s %8s  %s"
                  % (r.get("rung"), r.get("k") or 0, a_label(r),
                     r.get("throughput_tok_s", 0.0), sp, tau, cov, r["_why"]))


def best_per_cell(rows):
    """{(cell, rung): sorted [(speedup, k, row)] best first} for spec arms with a ratio."""
    out = defaultdict(list)
    for r in rows:
        if r.get("rung") != "off" and r["_speedup"] is not None:
            out[(r["_cell"], r["rung"])].append((r["_speedup"], r.get("k") or 0, r))
    for key in out:
        out[key].sort(key=lambda t: -t[0])
    return out


def table_best(best, tie):
    print("\n" + "=" * 100)
    print("2. best k per cell and rung. A gap under %.0f%% is a tie: there are no intervals yet." % tie)
    print("=" * 100)
    print("  %-5s %-34s %-4s %-4s %-10s %6s %8s %8s %8s  %s"
          % ("gpu", "domain", "bkt", "conc", "rung", "best k", "speedup", "next k", "gap %", ""))
    for (c, rung) in sorted(best, key=lambda k: (cell_sort(k[0]), k[1])):
        gpu, dom, bucket, conc, job = c
        lst = best[(c, rung)]
        s0, k0, _ = lst[0]
        if len(lst) > 1:
            s1, k1, _ = lst[1]
            gap = (s0 / s1 - 1.0) * 100.0
            flag = "tie" if gap < tie else ""
            nxt, gp = str(k1), "%.1f" % gap
        else:
            nxt, gp, flag = "-", "-", "only one k measured"
        if s0 < 1.0:
            flag = (flag + "; " if flag else "") + "every k below 1.0x: off wins"
        print("  %-5s %-34s %-4s %-4s %-10s %6s %8.3f %8s %8s  %s"
              % (gpu, dom[:34], bucket, conc_label(conc), rung, k0, s0, nxt, gp, flag))


def latest_cells(rows):
    """One job per (gpu, domain, bucket, conc): the most recent. Returns (map, notes)."""
    seen = defaultdict(dict)
    for r in rows:
        gpu, dom, bucket, conc, job = r["_cell"]
        t = (r.get("env") or {}).get("time", "")
        cur = seen[(gpu, dom, bucket, conc)].get(job, "")
        seen[(gpu, dom, bucket, conc)][job] = max(cur, t)
    pick, notes = {}, []
    for key, jobs in seen.items():
        j = max(jobs, key=lambda x: jobs[x])
        pick[key] = j
        if len(jobs) > 1:
            notes.append("%s %s %s conc %s: measured in jobs %s; using %s (latest). Not averaged."
                         % (key[0], key[1], key[2], conc_label(key[3]), ", ".join(sorted(jobs)), j))
    return pick, notes


def table_adaptive(rows, tie):
    print("\n" + "=" * 100)
    print("3. does k need to adapt? Per rung: one fixed k for every cell against the best k per cell.")
    print("   Loss is time-weighted over cells, one request per cell per unit of traffic.")
    print("=" * 100)
    pick, notes = latest_cells(rows)
    for n in notes:
        print("  NOTE " + n)
    groups = defaultdict(lambda: defaultdict(dict))  # (gpu, conc, rung) -> cell4 -> k -> row
    for r in rows:
        gpu, dom, bucket, conc, job = r["_cell"]
        if pick.get((gpu, dom, bucket, conc)) != job:
            continue
        if r.get("rung") == "off" or r["_speedup"] is None:
            continue
        groups[(gpu, conc, r["rung"])][(dom, bucket)][r.get("k") or 0] = r
    if not groups:
        print("  no spec arm has a matched baseline; nothing to compare")
    for (gpu, conc, rung) in sorted(groups):
        cells = groups[(gpu, conc, rung)]
        print("\n  %s  conc %s  %s  (%d cells)" % (gpu, conc_label(conc), rung, len(cells)))
        if len(cells) < 2:
            print("    one cell only: the question needs at least two")
            continue
        bestk = {}
        for cell in sorted(cells, key=lambda c: (c[0] != "all", c[0], bucket_key(c[1]))):
            ks = cells[cell]
            k_star = max(ks, key=lambda k: ks[k]["_speedup"])
            bestk[cell] = k_star
            print("    %-34s %-4s best k %-3s %.3fx   measured k: %s"
                  % (cell[0][:34], cell[1], k_star, ks[k_star]["_speedup"],
                     " ".join(str(k) for k in sorted(ks))))
        common = set.intersection(*(set(v) for v in cells.values()))
        if not common:
            print("    no k was measured in every cell; cannot score a fixed k")
            continue
        if len(common) < 2:
            print("    only k=%s was measured in every cell; a verdict needs at least two"
                  % " ".join(str(k) for k in sorted(common)))
            continue

        def t(row):  # seconds per request in that job
            return row["elapsed_s"] / max(1, row.get("limit") or 1)

        t_cell_best = sum(t(cells[c][bestk[c]]) for c in cells)
        scores = sorted((sum(t(cells[c][k]) for c in cells), k) for k in common)
        t_fixed, k_fixed = scores[0]
        gain = (t_fixed / t_cell_best - 1.0) * 100.0
        distinct = sorted(set(bestk.values()))
        print("    best k per cell takes the values: %s" % " ".join(str(k) for k in distinct))
        print("    best single k across cells: %s.  Per-cell k over that fixed k: %+.1f%%"
              % (k_fixed, gain))
        if len(distinct) == 1:
            print("    VERDICT: the best k is the same in every cell. Adaptive k has nothing to adapt to.")
        elif gain < tie:
            print("    VERDICT: the best k moves, but adapting it buys under %.0f%%. Not supported." % tie)
        else:
            print("    VERDICT: adapting k buys %.1f%% over the best fixed k. Needs intervals before it is a claim." % gain)


def table_policy(rows, tie):
    print("\n" + "=" * 100)
    print("4. what a policy could win, arms sharing one async state only. Time-weighted over cells.")
    print("   This is an oracle over CELLS (bucket and domain), not over requests.")
    print("=" * 100)
    pick, _ = latest_cells(rows)
    groups = defaultdict(lambda: defaultdict(dict))  # (gpu, conc, async) -> cell -> (rung,k) -> row
    for r in rows:
        gpu, dom, bucket, conc, job = r["_cell"]
        if pick.get((gpu, dom, bucket, conc)) != job or r["_async"] is None:
            continue
        if r["_speedup"] is None:
            continue
        groups[(gpu, conc, r["_async"])][(dom, bucket)][(r.get("rung"), r.get("k") or 0)] = r
    shown = False
    for (gpu, conc, a) in sorted(groups, key=lambda g: (g[0], g[1], not g[2])):
        cells = groups[(gpu, conc, a)]
        if len(cells) < 2:
            continue
        common = set.intersection(*(set(v) for v in cells.values()))
        if ("off", 0) not in common:
            continue
        shown = True

        def t(row):
            return row["elapsed_s"] / max(1, row.get("limit") or 1)

        t_off = sum(t(cells[c][("off", 0)]) for c in cells)
        fixed = sorted((sum(t(cells[c][arm]) for c in cells), arm) for arm in common)
        t_fix, arm_fix = fixed[0]
        rung_fix = arm_fix[0]
        t_rung_k = sum(min(t(row) for arm, row in cells[c].items() if arm[0] == rung_fix)
                       for c in cells)
        t_oracle = sum(min(t(row) for row in cells[c].values()) for c in cells)
        print("\n  %s  conc %s  async %s  (%d cells, %d arms common to all)"
              % (gpu, conc_label(conc), "on" if a else "off", len(cells), len(common)))
        print("    off everywhere                         1.000x")
        print("    best fixed arm: %-10s k=%-3s         %.3fx" % (arm_fix[0], arm_fix[1], t_off / t_fix))
        print("    same rung, best k per cell             %.3fx   (%+.1f%% over best fixed)"
              % (t_off / t_rung_k, (t_fix / t_rung_k - 1) * 100))
        print("    best rung and k per cell, off allowed  %.3fx   (%+.1f%% over best fixed)"
              % (t_off / t_oracle, (t_fix / t_oracle - 1) * 100))
        for c in sorted(cells, key=lambda c: (c[0] != "all", c[0], bucket_key(c[1]))):
            arm = min(cells[c], key=lambda x: t(cells[c][x]))
            print("      %-34s %-4s -> %s k=%s" % (c[0][:34], c[1], arm[0], arm[1]))
        if (t_fix / t_oracle - 1) * 100 < tie:
            print("    Under %.0f%%: a per-cell policy is not distinguishable from the best fixed arm here." % tie)
    if not shown:
        print("  needs at least two cells that share an off arm and an async state")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--tie", type=float, default=5.0, help="percent below which a difference is a tie")
    p.add_argument("--assume-pin-defaults", action="store_true",
                   help="for v2/v3 rows: fill in the pin's default async state, marked inferred")
    p.add_argument("--exclude-job", action="append", default=[],
                   help="drop a job, e.g. 22569403 (the uncontrolled-concurrency run)")
    a = p.parse_args()

    rows = load(a.dirs, a.assume_pin_defaults, set(a.exclude_job))
    if not rows:
        sys.exit("no records")
    dup = attach_speedups(rows)
    table_arms(rows)
    for cell, st, n in dup:
        print("\nNOTE %s: %d off arms with async=%s and one config in this job; the first is the baseline."
              % (" ".join(str(x) for x in cell), n, "on" if st else "off"))
    best = best_per_cell(rows)
    table_best(best, a.tie)
    table_adaptive(rows, a.tie)
    table_policy(rows, a.tie)
    unknown = sum(1 for r in rows if r["_async"] is None)
    if unknown:
        print("\n%d record(s) carry no resolved async state and got no ratio. They are v2/v3 rows;"
              " see --assume-pin-defaults." % unknown)


if __name__ == "__main__":
    main()
