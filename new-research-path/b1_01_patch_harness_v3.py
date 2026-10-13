#!/usr/bin/env python3
"""
Block 1, step 1 -- patch run_benchmark.py from harness v2 to v3.

Design constraint: every change is ADDITIVE and every v2 default is preserved
byte-for-byte, so re-running an existing cell under v3 reproduces the v2 row
exactly, with more recorded alongside it. Nothing here alters a measurement
unless a new flag is passed.

Why a patcher and not a replacement file: re-typing 338 lines into a tool call
is the silent-corruption failure this project keeps paying for. This script
locates each edit by a unique anchor string taken from the file itself, asserts
the anchor appears EXACTLY ONCE, and refuses to write anything if any anchor is
missing or ambiguous. A failed anchor is a loud stop, never a partial patch.

Writes run_benchmark.v3.py beside the original and prints a diff. It does not
touch run_benchmark.py. You move it into place after reading the diff.

Usage
-----
  python3 b1_01_patch_harness_v3.py /path/to/ladder/scripts/run_benchmark.py
  python3 b1_01_patch_harness_v3.py ... --write     # also write the .v3.py file

The eight changes
-----------------
1. HARNESS_VERSION -> v3-2026-10-13, so v2 and v3 rows are never pooled.
2. New flags: --temperature, --prompt-lookup-min/max, --loop-timing.
   All default to the v2 hardcoded values.
3. prompt_lookup_min/max read from args instead of the literals 3 and 5.
4. SamplingParams temperature read from args instead of the literal 0.0.
5. Per-request filename gains _j<SLURM_JOB_ID>. The run_id is NOT changed,
   because it is the cell key in results.jsonl and every analysis tool groups
   on it -- adding a job id there would make every run its own cell. The
   collision was only ever in the file name.
6. The speculative method is read back off the engine config and asserted
   against --rung. vLLM can silently override an explicitly requested method
   from model_type and path substrings (vllm-project/vllm#47486), which is the
   same failure shape as the V2-runner fallback.
7. The engine's RESOLVED config is recorded: max_num_seqs, prefix caching and
   the speculative method as the engine actually set them. Today every row
   reads prefix_caching: null, meaning "flag not passed", while the logs show
   the engine default was ON with 4-8% hit rates -- so the rows cannot answer a
   question the logs can.
8. --loop-timing generates one prompt at a time and times each with
   perf_counter, which is the only route to per-request latency on this build:
   lines 188-198 already request first_token_time, last_token_time and
   finished_time, and this vLLM populates none of them. Off by default, and it
   refuses to run above concurrency 1, where per-request latency is not
   separable anyway.

What it deliberately does NOT do
--------------------------------
Achieved concurrency is not captured in-process. Sampling the engine's
num_requests_running gauge needs a background thread inside the measurement
path, and a threading bug would cost queue time for data the job log already
contains. v3 records the RESOLVED max_num_seqs; achieved concurrency is
extracted from the log afterwards by a separate tool.
"""

import argparse
import difflib
import os
import sys

# ---------------------------------------------------------------- the edits
# (description, anchor, replacement). Anchor must match exactly once.
EDITS = [

    ("1. harness version",
     'HARNESS_VERSION = "v2-2026-10-05"',
     'HARNESS_VERSION = "v3-2026-10-13"'),

    ("2. new flags",
     '    p.add_argument("--no-token-ids", action="store_true",',
     '''    # ---- v3 additions. Every default below reproduces v2 behaviour.
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature; 0.0 (the v2 default) is greedy. "
                        "Above 0 the --seed becomes live, which it is not under "
                        "greedy decoding")
    p.add_argument("--prompt-lookup-min", type=int, default=3,
                   help="n-gram rung: shortest prompt match to propose from "
                        "(v2 hardcoded 3)")
    p.add_argument("--prompt-lookup-max", type=int, default=5,
                   help="n-gram rung: longest prompt match to propose from "
                        "(v2 hardcoded 5). This is the coverage knob -- measured "
                        "coverage is only 16-20%%")
    p.add_argument("--loop-timing", action="store_true",
                   help="generate one prompt at a time and time each, to get "
                        "per-request latency this build will not report. Only "
                        "valid at --max-num-seqs 1. Changes the arrival process, "
                        "so loop-mode rows are a separate population")
    p.add_argument("--no-token-ids", action="store_true",'''),

    ("3. prompt_lookup from args",
     '''                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,''',
     '''                "prompt_lookup_max": args.prompt_lookup_max,
                "prompt_lookup_min": args.prompt_lookup_min,'''),

    ("4. temperature from args",
     '    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)',
     '''    sp = SamplingParams(temperature=args.temperature,
                        max_tokens=args.max_tokens, seed=args.seed)'''),

    ("6. assert the method the engine actually chose",
     '''        **spec_kwargs(args),
    )
''',
     '''        **spec_kwargs(args),
    )

    # ---- v3: read the resolved config back off the engine.
    #
    # vLLM can silently override an explicitly requested speculative method
    # from model_type or a path substring (vllm-project/vllm#47486). A row
    # labelled with the rung we asked for, produced by a method we did not get,
    # is worse than a crash. Attribute paths move between versions, so every
    # read is guarded; a miss records null rather than raising, and we abort
    # only when a method IS visible and disagrees with --rung.
    resolved = {"max_num_seqs": None, "prefix_caching": None,
                "spec_method": None, "spec_num_tokens": None}
    try:
        cfg = llm.llm_engine.vllm_config
        sched = getattr(cfg, "scheduler_config", None)
        cache = getattr(cfg, "cache_config", None)
        spec = getattr(cfg, "speculative_config", None)
        if sched is not None:
            resolved["max_num_seqs"] = getattr(sched, "max_num_seqs", None)
        if cache is not None:
            resolved["prefix_caching"] = getattr(cache, "enable_prefix_caching", None)
        if spec is not None:
            resolved["spec_method"] = getattr(spec, "method", None)
            resolved["spec_num_tokens"] = getattr(spec, "num_speculative_tokens", None)
    except Exception as e:
        resolved["error"] = repr(e)

    print(f"[{run_id}] resolved: {resolved}", flush=True)

    want = None if args.rung == "off" else args.rung
    got = resolved.get("spec_method")
    if want is None and got:
        sys.exit(f"ABORT: rung 'off' but the engine reports speculative method "
                 f"'{got}'. The baseline would not be a baseline.")
    if want and got and str(got) != want:
        sys.exit(f"ABORT: asked for method '{want}', engine resolved '{got}'. "
                 f"See vllm-project/vllm#47486. The row would be mislabelled.")
    if want and not got:
        print(f"[{run_id}] WARNING: could not read the method off the engine "
              f"config; the assertion is unverified for this run.", flush=True)
'''),

    ("7. record the resolved config",
     '        "prefix_caching": args.prefix_caching,  # None means engine default',
     '''        "prefix_caching": args.prefix_caching,  # None means engine default
        # v3: what the engine actually used, as opposed to what was asked for.
        # Every v2 row reads prefix_caching: null, meaning only "flag not
        # passed" -- while the logs show the default was ON. The rows could not
        # answer a question the logs could.
        "resolved": resolved,
        "temperature": args.temperature,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
        "loop_timing": bool(args.loop_timing),'''),

    ("8. loop-mode per-request timing",
     '''    before = snapshot(llm)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0
    after = snapshot(llm)
''',
     '''    before = snapshot(llm)
    loop_latency = {}
    if args.loop_timing:
        # One request at a time, timed individually. This is the only route to
        # per-request latency on this build: request_timing() above asks for
        # first_token_time, last_token_time and finished_time, and this vLLM
        # populates none of them, so per_request_timing has always been 0.
        #
        # At max_num_seqs=1 the engine already serves one request at a time, so
        # this should be close to behaviour-preserving -- but it changes the
        # arrival process and it is NOT assumed equivalent. The summed loop
        # elapsed is compared against the batched elapsed for the same cell
        # before any loop-mode number is used, and loop rows carry
        # loop_timing: true so they are never pooled with batched rows.
        if (args.max_num_seqs or 0) != 1:
            sys.exit("--loop-timing requires --max-num-seqs 1; per-request "
                     "latency is not separable at higher concurrency.")
        outputs = []
        t0 = time.perf_counter()
        for rid, pr in zip(ids, prompts):
            s = time.perf_counter()
            o = llm.generate([pr], sp, use_tqdm=False)
            loop_latency[rid] = time.perf_counter() - s
            outputs.extend(o)
        elapsed = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0
    after = snapshot(llm)
'''),

    ("8b. attach the loop latency per record",
     '''        t = request_timing(o)
        if t:
            rec["timing"] = t
            if "latency_s" in t:
                timing_seen += 1''',
     '''        t = request_timing(o)
        if rid in loop_latency:
            # Measured by this harness, not reported by the engine. Named
            # distinctly so it is never confused with an engine metric.
            t = dict(t)
            t["loop_latency_s"] = loop_latency[rid]
        if t:
            rec["timing"] = t
            if "latency_s" in t or "loop_latency_s" in t:
                timing_seen += 1'''),

    ("5. job id in the per-request filename",
     '    with open(os.path.join(args.outdir, f"perreq_{run_id}.json"), "w") as f:',
     '''    # v3: the job id goes in the FILE NAME only, never in run_id. run_id is
    # the cell key in results.jsonl and every analysis tool groups on it, so a
    # job id there would make each run its own cell. The collision this fixes
    # was only ever between files: a cell run twice wrote the same name twice,
    # and the second run's token_ids replaced the first's.
    _job = os.environ.get("SLURM_JOB_ID", "nojob")
    _perreq = f"perreq_{run_id}_j{_job}.json"
    with open(os.path.join(args.outdir, _perreq), "w") as f:'''),
]

# Suffix run_id when a v3 knob leaves its v2 default, so new-knob runs cannot
# overwrite or be confused with the existing grid.
RUNID_EDIT = (
    "run_id suffix for non-default v3 knobs",
    '''    if args.prefix_caching is False:
        # Keep the determinism control in its own files rather than letting it
        # overwrite the matching normal run.
        run_id += "_nopc"''',
    '''    if args.prefix_caching is False:
        # Keep the determinism control in its own files rather than letting it
        # overwrite the matching normal run.
        run_id += "_nopc"
    # v3: a run with a knob off its v2 default is a different cell and must not
    # share a run_id with the existing grid.
    if args.temperature != 0.0:
        # "p" for the decimal point: a dot inside the file name stem makes any
        # trailing \\.json$ pattern ambiguous, and check_lossless.py has one.
        run_id += "_T" + f"{args.temperature:g}".replace(".", "p")
    if (args.prompt_lookup_min, args.prompt_lookup_max) != (3, 5):
        run_id += f"_pl{args.prompt_lookup_min}-{args.prompt_lookup_max}"
    if args.loop_timing:
        run_id += "_loop"''',
)


def apply_edits(text, edits):
    problems = []
    for desc, anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            problems.append(f"  {desc}: anchor found {n} times, expected exactly 1")
            continue
        text = text.replace(anchor, repl, 1)
    return text, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to run_benchmark.py (v2)")
    ap.add_argument("--write", action="store_true",
                    help="write <path dir>/run_benchmark.v3.py")
    a = ap.parse_args()

    src = open(a.path).read()
    if 'HARNESS_VERSION = "v2-2026-10-05"' not in src:
        sys.exit(f"{a.path} does not look like harness v2 -- refusing to patch. "
                 f"Check the version string before going further.")

    out, problems = apply_edits(src, EDITS + [RUNID_EDIT])

    print(f"patching {a.path}  ({len(src.splitlines())} lines)")
    if problems:
        print("\nANCHOR FAILURES -- nothing was written:")
        print("\n".join(problems))
        print("\nThe file is not the version this patcher was written against.")
        print("Do not hand-apply the diff; re-derive the anchors from the file.")
        sys.exit(2)
    print(f"all {len(EDITS) + 1} anchors matched exactly once")

    diff = list(difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="run_benchmark.py (v2)", tofile="run_benchmark.v3.py", n=3))
    sys.stdout.writelines(diff)
    print(f"\n{len(src.splitlines())} -> {len(out.splitlines())} lines")

    if a.write:
        dest = os.path.join(os.path.dirname(os.path.abspath(a.path)),
                            "run_benchmark.v3.py")
        with open(dest, "w") as f:
            f.write(out)
        print(f"\nwrote {dest}")
        print("The original is untouched. Before swapping it in, prove v3 is a "
              "no-op on\nan existing cell: run one 2K off arm with both files to "
              "the same outdir and\ncheck that throughput and token_ids match "
              "within the 0.2% node spread.")
    else:
        print("\n(--write to produce run_benchmark.v3.py; nothing written)")


if __name__ == "__main__":
    main()
