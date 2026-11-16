#!/usr/bin/env python3
"""Run harness v5 (or the variant named in LADDER_NSYS_HARNESS) under Nsight
Systems. Used only by dtai2_bundle.sh's profile bundle, which hands it to
ksweep.sh as LADDER_RUN and points LADDER_NSYS_HARNESS at harness v5c.

ksweep.sh refuses any harness file that does not contain the string
`HARNESS_VERSION = "v5`, so this wrapper carries that marker below. It
measures nothing itself: it re-executes
    nsys profile ... python3 run_benchmark.v5.py <the same arguments>
and run_benchmark.v5.py writes the row exactly as it always does. Those rows
were taken under the profiler, so the profile bundle sends them to their own
directory (results/gh200v2_nsys); they are never pooled with unprofiled rows.

One report per arm: <outdir>/nsys/<rung>_<bucket>_k<k>_j<jobid>.nsys-rep.
Read them later with `nsys stats` or the Nsight Systems GUI. With CUDA graph
node tracing on, kernels inside captured graphs show up individually, which
is what attributing DFlash's drafting time needs (context-KV write, draft
forward, verify).

Written 15 Nov 2026. Not run on DeltaAI yet.
"""
import os
import shutil
import sys

HARNESS_VERSION = "v5-nsys-wrapper"  # marker for ksweep.sh; rows say v5 itself


def flag_value(args, flag, default="na"):
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # LADDER_NSYS_HARNESS: profile another v5 variant (dtai2_bundle.sh sets it
    # to run_benchmark.v5c.py). Default: v5.
    harness = os.environ.get("LADDER_NSYS_HARNESS") or os.path.join(here, "run_benchmark.v5.py")
    if not os.path.isfile(harness):
        sys.exit(f"ABORT: no harness at {harness}")
    nsys = shutil.which("nsys")
    if nsys is None:
        sys.exit("ABORT: nsys is not on PATH. dtai2_bundle.sh tries to load it "
                 "from a module first; set NSYS_MODULES to the right name.")

    args = sys.argv[1:]
    outdir = flag_value(args, "--outdir", ".")
    tag = "_".join([
        flag_value(args, "--rung"),
        flag_value(args, "--bucket"),
        "k" + flag_value(args, "--k", "0"),
        "j" + os.environ.get("SLURM_JOB_ID", "none"),
    ])
    report = os.path.join(outdir, "nsys", tag)
    os.makedirs(os.path.dirname(report), exist_ok=True)

    cmd = [
        nsys, "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-graph-trace=node",
        "--trace-fork-before-exec=true",   # vLLM's engine core is a child process
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "-o", report,
        sys.executable, harness,
    ] + args
    print("nsys_v5: " + " ".join(cmd), flush=True)
    os.execv(nsys, cmd)


if __name__ == "__main__":
    main()
