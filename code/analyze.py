#!/usr/bin/env python3
"""Per-record analysis of results.jsonl: provenance, speedups, prediction scoring, subset spreads.

    python3 analyze.py results/a40 results/a100 ...        # any number of result dirs
    python3 analyze.py results/a100 --predict 8K:dflash:7=1.20 8K:ngram:5=1.09 \
                                              32K:dflash:7=0.78 32K:ngram:5=1.02

Why this exists (RULES section 10): summarize.py keys on (bucket, conc, rung,
k) and keeps one record per key, so two jobs measuring the same cell render as
one row. Here every record is kept, the unit of analysis is the JOB, and every
speedup is a within-job ratio against the `off` arm measured in that job at
the same concurrency. Nothing is ever compared across jobs except ratios.

Derived quantities use summarize.derive, so tau_eff, coverage and implied R
are the same arithmetic the register has used since 1 Oct.

Three tables:

1. records: one line per record, grouped by job, with host, GPU, dataset
   subset, and the within-job speedup / tau / R.
2. prediction scoring (with --predict): per cell, predicted, measured, signed
   relative error, per job. Never a median. If a cell was measured in more
   than one job, every realization is listed.
3. spreads: for each (gpu, bucket, conc, rung, k) measured in more than one
   job, the realizations side by side with min, max, and whether they came
   from different prompt subsets (dataset path) or the same prompts (node /
   run-to-run only). Different-subset spread is the interval RULES section
   11 asks for; same-subset spread is the noise floor.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import derive, conc_of, conc_label, bucket_key  # noqa: E402


# Nodes with a recorded anomaly: gpub073 ran 8.8% low on absolute throughput
# with identical KV allocation and no preemption (job 22757472). Subset jobs
# exclude it at submission (--exclude=gpub073); this flags it if it slips in.
ANOMALOUS = {"gpub073"}


def load(dirs):
    rows = []
    for d in dirs:
        path = d.rstrip("/") + "/results.jsonl"
        if not os.path.exists(path):
            print("no results.jsonl in %s" % d, file=sys.stderr)
            continue
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                r["_dir"] = d
                rows.append(r)
    return rows


def job_of(r):
    return str((r.get("env") or {}).get("slurm_job") or "none")


def host_of(r):
    return ((r.get("env") or {}).get("host") or "?").split(".")[0]


def gpu_of(r):
    g = (r.get("env") or {}).get("gpu") or "?"
    return g.replace("NVIDIA ", "")


def subset_of(r):
    """Which prompt sample: the parent directory of the dataset file."""
    ds = r.get("dataset") or ""
    d = os.path.basename(os.path.dirname(ds))
    return "sub0" if d == "longbench_v2" else (d or "?")


def within_job(rows):
    """Attach speedup and R to every record, against the off arm of its own job."""
    base = {}
    for r in rows:
        if r["rung"] == "off":
            base[(job_of(r), r["bucket"], conc_of(r))] = r["throughput_tok_s"]
    for r in rows:
        b = base.get((job_of(r), r["bucket"], conc_of(r)))
        d = derive(r)
        r["_speed"] = r["throughput_tok_s"] / b if b else None
        r["_tau_cond"] = d["tau_cond"]
        r["_tau_eff"] = d["tau_eff"]
        r["_cov"] = d["coverage"]
        r["_R"] = (d["tau_eff"] / r["_speed"]) if (d["tau_eff"] and r["_speed"]) else None
    return rows


def f(x, p=3, w=7):
    return ("%*.*f" % (w, p, x)) if isinstance(x, (int, float)) and x == x else " " * w


def table_records(rows):
    print("records, grouped by job (speedup and R are within-job)")
    print("%-9s %-8s %-14s %-8s %4s %5s %-7s %3s %4s %9s %8s %7s %7s %6s %7s"
          % ("job", "host", "gpu", "subset", "bkt", "conc", "rung", "k", "n",
             "tok/s", "speedup", "tauC", "tauE", "cov", "R"))
    print("-" * 124)
    key = lambda r: (job_of(r), bucket_key(r["bucket"]), conc_of(r), r["rung"], r.get("k", 0))
    for r in sorted(rows, key=key):
        print("%-9s %-8s %-14s %-8s %4s %5s %-7s %3s %4s %9.2f %8s %7s %7s %6s %7s"
              % (job_of(r), host_of(r), gpu_of(r)[:14], subset_of(r)[:8], r["bucket"],
                 conc_label(conc_of(r)), r["rung"], r.get("k", 0), r.get("limit", ""),
                 r["throughput_tok_s"], f(r["_speed"], 4, 8), f(r["_tau_cond"]),
                 f(r["_tau_eff"]), f(r["_cov"], 3, 6), f(r["_R"])))
    missing = [r for r in rows if r["rung"] != "off" and r["_speed"] is None]
    if missing:
        print("\n%d record(s) have no off arm in their own job at the same concurrency; "
              "no speedup is computable for them:" % len(missing))
        for r in missing:
            print("  job %s %s %s k=%s c=%s" % (job_of(r), r["bucket"], r["rung"],
                                                r.get("k"), conc_label(conc_of(r))))


def parse_predict(items):
    """'8K:dflash:7=1.20' -> ((bucket, rung, k), 1.20)"""
    out = {}
    for it in items or []:
        lhs, val = it.split("=")
        b, rung, k = lhs.split(":")
        out[(b, rung, int(k))] = float(val)
    return out


def table_predict(rows, pred, conc=1):
    if not pred:
        return
    print("\nfrozen prediction, scored per cell at concurrency %d (signed error = measured/predicted - 1)" % conc)
    print("%-4s %-7s %3s %9s %9s %8s %-9s %-8s %-14s %5s"
          % ("bkt", "rung", "k", "predicted", "measured", "error", "job", "host", "gpu", "n"))
    print("-" * 90)
    seen = set()
    for (b, rung, k), p in sorted(pred.items(), key=lambda t: (bucket_key(t[0][0]), t[0][1])):
        hits = [r for r in rows if r["bucket"] == b and r["rung"] == rung
                and r.get("k", 0) == k and conc_of(r) == conc and r["_speed"]]
        if not hits:
            print("%-4s %-7s %3d %9.2f %9s %8s   (no record)" % (b, rung, k, p, "", ""))
            continue
        for r in sorted(hits, key=job_of):
            err = r["_speed"] / p - 1.0
            flag = "" if abs(err) <= 0.05 else ("  >5%" if abs(err) <= 0.10 else "  >10%")
            print("%-4s %-7s %3d %9.2f %9.4f %+7.1f%% %-9s %-8s %-14s %5s%s"
                  % (b, rung, k, p, r["_speed"], 100 * err, job_of(r), host_of(r),
                     gpu_of(r)[:14], r.get("limit", ""), flag))
            seen.add((b, job_of(r)))
    # 32K ordering, scored separately from magnitude.
    for b in sorted({b for (b, _, _) in pred}, key=bucket_key):
        for j in sorted({j for (bb, j) in seen if bb == b}):
            # Compare at the k the prediction names, not whichever k came first.
            want = {rg: kk for (bb, rg, kk) in pred if bb == b}
            cell = {r["rung"]: r["_speed"] for r in rows
                    if r["bucket"] == b and job_of(r) == j and conc_of(r) == conc
                    and r["_speed"] and r.get("k", 0) == want.get(r["rung"])}
            df = [cell["dflash"]] if "dflash" in cell else []
            ng = [cell["ngram"]] if "ngram" in cell else []
            if df and ng:
                print("%s job %s ordering: dflash %.3f %s 1.0, ngram %.3f %s 1.0 -> %s"
                      % (b, j, df[0], "<" if df[0] < 1 else ">=", ng[0], ">" if ng[0] > 1 else "<=",
                         "n-gram above and DFlash below" if (df[0] < 1 < ng[0]) else "ordering NOT as predicted"))


def table_spreads(rows):
    cells = defaultdict(list)
    for r in rows:
        if r["_speed"] is None:
            continue
        cells[(gpu_of(r), r["bucket"], conc_of(r), r["rung"], r.get("k", 0))].append(r)
    multi = {k: v for k, v in cells.items() if len({job_of(r) for r in v}) > 1}
    if not multi:
        print("\nno cell has been measured in more than one job; no spread to report")
        return
    print("\nspreads: cells measured in more than one job")
    for (gpu, b, c, rung, k), rs in sorted(multi.items(), key=lambda t: (t[0][0], bucket_key(t[0][1]), t[0][2], t[0][3], t[0][4])):
        subs = {subset_of(r) for r in rs}
        kind = ("%d prompt subsets: interval" % len(subs)) if len(subs) > 1 else "same prompts: node/run noise only"
        sp = [r["_speed"] for r in rs]
        print("  %s %s c=%s %s k=%s  [%s]" % (gpu, b, conc_label(c), rung, k, kind))
        for r in sorted(rs, key=job_of):
            print("      job %-9s %-8s %-8s speedup %.4f  tauE %s  R %s"
                  % (job_of(r), host_of(r), subset_of(r)[:8], r["_speed"], f(r["_tau_eff"]), f(r["_R"])))
        print("      min %.4f  max %.4f  range %.2f%% of mean" % (min(sp), max(sp), 100 * (max(sp) - min(sp)) / (sum(sp) / len(sp))))
        if 2 <= len(subs) < 5:
            print("      (%d subsets; the plan calls for five before this is quoted as an interval)" % len(subs))
        # Does the node explain any of it? Realizations on one host differ only
        # in prompts (if the subsets differ) or in nothing but run noise (if
        # they do not), so a within-host range is the node-free spread.
        by_host = defaultdict(list)
        for r in rs:
            by_host[host_of(r)].append(r)
        hosts = ", ".join("%s (%d)" % (h, len(v)) for h, v in sorted(by_host.items()))
        print("      hosts: %s" % hosts)
        for h, v in sorted(by_host.items()):
            if len(v) > 1:
                vs = [r["_speed"] for r in v]
                what = "prompt-only" if len({subset_of(r) for r in v}) > 1 else "run-noise-only"
                print("      within %s (%s): min %.4f max %.4f range %.2f%%"
                      % (h, what, min(vs), max(vs), 100 * (max(vs) - min(vs)) / (sum(vs) / len(vs))))
        bad = [r for r in rs if host_of(r) in ANOMALOUS]
        for r in bad:
            print("      WARNING: job %s ran on %s, a node with a known anomaly (LOG.md, 11 Oct); "
                  "report it separately" % (job_of(r), host_of(r)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--predict", nargs="*", help="cells as BUCKET:RUNG:K=VALUE")
    p.add_argument("--conc", type=int, default=1, help="concurrency the prediction is scored at")
    a = p.parse_args()
    rows = within_job(load(a.dirs))
    if not rows:
        sys.exit("no records")
    table_records(rows)
    table_predict(rows, parse_predict(a.predict), a.conc)
    table_spreads(rows)


if __name__ == "__main__":
    main()
