#!/usr/bin/env python3
"""Token acceptance per record, from the draft counters already in results.jsonl.

    python3 acceptance.py results/a40                 # every speculative record
    python3 acceptance.py results/a40 --conc 1        # concurrency 1 only
    python3 acceptance.py results/a40 --positions     # add acceptance by draft position

No GPU and no job: every run since 1 Oct stores vLLM's spec-decode counters in
the record's `spec` field. This only reads them. Login node or laptop.

Four numbers, because "acceptance rate" means different things and the
differences are large here:

  acc_rate  accepted draft tokens / proposed draft tokens. vLLM's
            draft_acceptance_rate. Falls as k grows even when the drafter is
            unchanged, because later positions rarely survive, so it is NOT
            comparable across different k.
  tauC      mean tokens emitted per decode step on which a draft was proposed,
            including the bonus token (1 + accepted per draft). Conditional on
            a draft existing.
  tauE      mean tokens emitted per decode step over ALL steps, counting steps
            where the drafter proposed nothing as one token. This is the number
            that goes into Speedup ~= tau / R. Same arithmetic as summarize.py.
  AL@5      tauC truncated to the first five draft positions, the register's
            common basis for comparing rungs run at different k (rule 2).

coverage is the fraction of decode steps on which a draft was proposed. n-gram
proposes on 13 to 21% of steps, so its tauC overstates what it delivers by
more than 2x; tauE corrects for that.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import derive, conc_of, conc_label, bucket_key, al_at_k  # noqa: E402


def job_of(r):
    return str((r.get("env") or {}).get("slurm_job") or "none")


def gpu_of(r):
    return ((r.get("env") or {}).get("gpu") or "?").replace("NVIDIA ", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--conc", type=int, default=None, help="only this concurrency")
    p.add_argument("--positions", action="store_true", help="print acceptance by draft position")
    a = p.parse_args()

    rows = []
    for d in a.dirs:
        path = d.rstrip("/") + "/results.jsonl"
        if not os.path.exists(path):
            print("no results.jsonl in %s" % d, file=sys.stderr)
            continue
        rows += [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("rung") != "off" and (r.get("spec") or {}).get("drafts")]
    if a.conc is not None:
        rows = [r for r in rows if conc_of(r) == a.conc]
    if not rows:
        sys.exit("no speculative records")

    rows.sort(key=lambda r: (gpu_of(r), bucket_key(r["bucket"]), conc_of(r), r["rung"], r.get("k", 0), job_of(r)))
    print("%-9s %-12s %4s %5s %-7s %3s %4s %9s %9s %8s %6s %6s %6s %6s %6s"
          % ("job", "gpu", "bkt", "conc", "rung", "k", "n", "drafted", "accepted",
             "acc_rate", "tauC", "tauE", "cov", "AL@5", "pos1"))
    print("-" * 118)
    for r in rows:
        s = r["spec"]
        d = derive(r)
        acc = s.get("acceptance_by_position") or []
        rate = s.get("draft_acceptance_rate")
        if rate is None and s.get("draft_tokens"):
            rate = s.get("accepted_tokens", 0) / s["draft_tokens"]
        print("%-9s %-12s %4s %5s %-7s %3s %4s %9d %9d %8.3f %6.3f %6.3f %6.3f %6s %6s"
              % (job_of(r), gpu_of(r)[:12], r["bucket"], conc_label(conc_of(r)), r["rung"],
                 r.get("k", 0), r.get("limit", ""), s.get("draft_tokens", 0),
                 s.get("accepted_tokens", 0), rate or float("nan"), d["tau_cond"],
                 d["tau_eff"] or float("nan"), d["coverage"] or float("nan"),
                 "%.3f" % al_at_k(acc, 5) if acc else "", "%.3f" % acc[0] if acc else ""))
        if a.positions and acc:
            print("%-9s acceptance by position: %s" % ("", ", ".join("%.3f" % x for x in acc)))

    print("\nacc_rate  accepted / proposed draft tokens (vLLM draft_acceptance_rate); not comparable across k")
    print("tauC      tokens per drafted step, incl. the bonus token; conditional on a draft")
    print("tauE      tokens per step over all steps; the tau in Speedup ~= tau / R")
    print("cov       fraction of steps with a draft;  AL@5  tauC truncated to 5 positions")


if __name__ == "__main__":
    main()
