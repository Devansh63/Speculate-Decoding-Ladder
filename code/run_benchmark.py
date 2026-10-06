#!/usr/bin/env python3
"""Phase 0 benchmark for the Speculation Ladder.

One rung, one context bucket, one run. Records acceptance by draft position,
not just throughput, and writes the output token ids so a later run can be
checked against the no-speculation arm.

Example:
    python3 run_benchmark.py --rung off    --bucket 2K --limit 50 ...
    python3 run_benchmark.py --rung ngram  --bucket 2K --limit 50 ...
    python3 run_benchmark.py --rung dflash --bucket 2K --limit 50 ...

Every run must be launched with VLLM_USE_V2_MODEL_RUNNER=0 so that all rungs
execute on the stable model runner. The script refuses to start otherwise.

Harness v2, 5 Oct 2026. Three additions, all of them additive, none of them
changing a default, so a job already queued against v1 produces the same
measurement with more recorded alongside it:

  1. Full token ids per request, not only a SHA-1. A single flipped token in a
     256 token generation changes the hash, so hashes cannot distinguish "one
     token differs" from "completely different answer". Both occur, and only
     the first is acceptable. Roughly 60 KB per arm.
  2. Per request timing where vLLM exposes it, so a paired bootstrap over
     prompts is possible. Without it there is one aggregate number per arm and
     no way to put an interval on it.
  3. A switch for prefix caching, default unchanged (on). Caching is a known
     source of run to run non-determinism because a request that hits the
     cache and one that recomputes do not follow bitwise identical paths.
     Having the switch lets us measure how much of the noise floor it owns.

Everything added here is wrapped so that a failure records a null and the
measurement still completes. Losing a run to a bug in an instrumentation path
would cost hours of queue time for data we already know how to get.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time

from vllm import LLM, SamplingParams

HARNESS_VERSION = "v2-2026-10-05"


def parse_args():
    p = argparse.ArgumentParser(description="Speculation Ladder Phase 0 benchmark")
    p.add_argument("--dataset", required=True, help="Path to the bucket JSONL")
    p.add_argument("--model", required=True, help="Path to the target model snapshot")
    p.add_argument("--draft", default=None, help="Path to the DFlash head (rung dflash only)")
    p.add_argument("--rung", required=True, choices=["off", "ngram", "dflash"])
    p.add_argument("--bucket", required=True, help="Context bucket label, for example 2K")
    p.add_argument("--limit", type=int, default=100, help="Number of prompts to run")
    p.add_argument("--max-tokens", type=int, default=256, help="Generation cap")
    p.add_argument("--k", type=int, default=None, help="Draft length; default 5 for ngram, 15 for dflash")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-model-len", type=int, default=33792, help="32K context plus room for prompt and generation")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=0, help="concurrency cap; 0 = engine default")
    p.add_argument("--outdir", required=True, help="Directory for results")
    p.add_argument("--prompt-style", choices=["qa", "context"], default="qa")
    # Default None means "leave the engine alone", which is what every run so
    # far did. Only an explicit flag changes behaviour.
    p.add_argument("--prefix-caching", dest="prefix_caching", action="store_true", default=None,
                   help="force prefix caching on")
    p.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false",
                   help="force prefix caching off; use for determinism controls")
    p.add_argument("--no-token-ids", action="store_true",
                   help="skip storing full token ids (they are about 60 KB per arm)")
    return p.parse_args()


def build_prompts(path, limit, style):
    """Read the bucket file and build identical prompts for every rung."""
    prompts, ids = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            ctx = d.get("context", "")
            if not ctx:
                continue
            if style == "qa" and d.get("question"):
                prompt = (
                    f"{ctx}\n\nQuestion: {d['question']}\n"
                    f"A. {d.get('choice_A','')}\nB. {d.get('choice_B','')}\n"
                    f"C. {d.get('choice_C','')}\nD. {d.get('choice_D','')}\n\n"
                    "Answer the question and explain your reasoning.\n\nAnswer:"
                )
            else:
                prompt = ctx
            prompts.append(prompt)
            ids.append(d.get("_id", f"idx{len(ids)}"))
    return prompts, ids


def spec_kwargs(args):
    """Speculative settings per rung, using this build's speculative_config dict."""
    if args.rung == "off":
        return {}
    if args.rung == "ngram":
        k = args.k or 5
        return {
            "speculative_config": {
                "method": "ngram",
                "num_speculative_tokens": k,
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
            }
        }
    if args.rung == "dflash":
        if not args.draft:
            sys.exit("rung dflash needs --draft")
        k = args.k or 15
        return {
            "speculative_config": {
                "method": "dflash",
                "model": args.draft,
                "num_speculative_tokens": k,
            }
        }
    raise ValueError(args.rung)


def snapshot(llm):
    """Grab the spec decode counters. Returns a plain dict."""
    out = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0, "accepted_per_pos": []}
    try:
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            if name == "vllm:spec_decode_num_drafts":
                out["drafts"] = getattr(m, "value", 0)
            elif name == "vllm:spec_decode_num_draft_tokens":
                out["draft_tokens"] = getattr(m, "value", 0)
            elif name == "vllm:spec_decode_num_accepted_tokens":
                out["accepted_tokens"] = getattr(m, "value", 0)
            elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                out["accepted_per_pos"] = list(getattr(m, "values", []))
    except Exception as e:  # metrics are a nice to have, never fatal
        out["error"] = repr(e)
    return out


def diff(before, after):
    d = {
        "drafts": after["drafts"] - before["drafts"],
        "draft_tokens": after["draft_tokens"] - before["draft_tokens"],
        "accepted_tokens": after["accepted_tokens"] - before["accepted_tokens"],
    }
    pos_a, pos_b = after.get("accepted_per_pos", []), before.get("accepted_per_pos", [])
    if pos_a:
        if len(pos_b) == len(pos_a):
            d["accepted_per_pos"] = [a - b for a, b in zip(pos_a, pos_b)]
        else:
            d["accepted_per_pos"] = pos_a
    # Derived quantities: the numbers the milestone actually needs.
    if d["drafts"] > 0:
        d["mean_accepted_length"] = 1.0 + d["accepted_tokens"] / d["drafts"]
        if d.get("accepted_per_pos"):
            d["acceptance_by_position"] = [c / d["drafts"] for c in d["accepted_per_pos"]]
    if d["draft_tokens"] > 0:
        d["draft_acceptance_rate"] = d["accepted_tokens"] / d["draft_tokens"]
    return d


def request_timing(o):
    """Per request timing, if this vLLM build exposes it.

    The field set differs across versions and may be absent entirely on the V1
    engine, so every read is guarded and a miss records nulls rather than
    raising. Latency and time to first token are what a paired bootstrap over
    prompts needs; without them there is one aggregate number per arm and no
    way to attach an interval to it.
    """
    out = {}
    try:
        m = getattr(o, "metrics", None)
        if m is None:
            return out
        for field in ("arrival_time", "first_scheduled_time", "first_token_time",
                      "last_token_time", "finished_time", "time_in_queue"):
            v = getattr(m, field, None)
            if isinstance(v, (int, float)):
                out[field] = float(v)
        a = out.get("arrival_time")
        if a is not None:
            if out.get("finished_time") is not None:
                out["latency_s"] = out["finished_time"] - a
            if out.get("first_token_time") is not None:
                out["ttft_s"] = out["first_token_time"] - a
    except Exception as e:
        out["error"] = repr(e)
    return out


def main():
    args = parse_args()

    runner_pin = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if runner_pin != "0":
        sys.exit(
            "VLLM_USE_V2_MODEL_RUNNER is not '0'. Both must-do rungs have to run on "
            "the stable runner, or n-gram silently falls back to it while DFlash "
            "stays on V2 and the comparison is void. Export it and rerun."
        )

    os.makedirs(args.outdir, exist_ok=True)
    run_id = f"{args.rung}_{args.bucket}_k{args.k or 0}_n{args.limit}_s{args.seed}_c{args.max_num_seqs}"
    if args.prefix_caching is False:
        # Keep the determinism control in its own files rather than letting it
        # overwrite the matching normal run.
        run_id += "_nopc"

    prompts, ids = build_prompts(args.dataset, args.limit, args.prompt_style)
    if not prompts:
        sys.exit(f"no prompts read from {args.dataset}")
    print(f"[{run_id}] {len(prompts)} prompts from {args.dataset}", flush=True)
    print(f"[{run_id}] harness {HARNESS_VERSION}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
        **({"enable_prefix_caching": args.prefix_caching}
           if args.prefix_caching is not None else {}),
        trust_remote_code=True,
        disable_log_stats=False,  # required, or get_metrics returns nothing
        **spec_kwargs(args),
    )

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    print("warmup", flush=True)
    llm.generate(prompts[:2], sp, use_tqdm=False)

    before = snapshot(llm)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0
    after = snapshot(llm)

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)

    # Per request. The token ids are the point: a SHA-1 says two generations
    # differ, it cannot say whether one token moved or the whole answer did.
    # On a continuous batching server those are very different claims, and the
    # losslessness result turns on telling them apart.
    per_request = []
    timing_seen = 0
    for rid, o in zip(ids, outputs):
        tok = list(o.outputs[0].token_ids)
        rec = {
            "id": rid,
            "prompt_tokens": len(o.prompt_token_ids),
            "gen_tokens": len(tok),
            "sha1": hashlib.sha1(",".join(map(str, tok)).encode()).hexdigest(),
            "finish_reason": getattr(o.outputs[0], "finish_reason", None),
        }
        if not args.no_token_ids:
            rec["token_ids"] = tok
        t = request_timing(o)
        if t:
            rec["timing"] = t
            if "latency_s" in t:
                timing_seen += 1
        per_request.append(rec)

    import torch  # imported late so the runner check fails fast without CUDA init

    record = {
        "run_id": run_id,
        "harness": HARNESS_VERSION,
        "rung": args.rung,
        "bucket": args.bucket,
        "k": args.k or (5 if args.rung == "ngram" else 15 if args.rung == "dflash" else 0),
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "max_num_seqs": args.max_num_seqs,
        "prefix_caching": args.prefix_caching,  # None means engine default
        "prompt_style": args.prompt_style,
        "dataset": args.dataset,
        "model": args.model,
        "draft": args.draft,
        "elapsed_s": elapsed,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "throughput_tok_s": gen_tokens / elapsed if elapsed else 0.0,
        "per_request_timing": timing_seen,  # how many requests carried usable timing
        "spec": diff(before, after),
        "env": {
            "vllm": __import__("vllm").__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "runner_v2": runner_pin,
            "host": socket.gethostname(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "python": platform.python_version(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    with open(os.path.join(args.outdir, "results.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    with open(os.path.join(args.outdir, f"perreq_{run_id}.json"), "w") as f:
        json.dump(per_request, f)

    s = record["spec"]
    print("\n=== " + run_id + " ===")
    print(f"elapsed {elapsed:.1f}s   generated {gen_tokens} tok   "
          f"{record['throughput_tok_s']:.1f} tok/s")
    print(f"per request timing available on {timing_seen}/{len(per_request)} requests")
    if s.get("drafts"):
        print(f"drafts {s['drafts']}   mean accepted length {s['mean_accepted_length']:.3f}")
        print(f"draft acceptance rate {s.get('draft_acceptance_rate', 0):.3f}")
        if s.get("acceptance_by_position"):
            print("acceptance by position: " +
                  ", ".join(f"{p:.3f}" for p in s["acceptance_by_position"]))
    elif args.rung != "off":
        print("WARNING: no spec decode counters. Check disable_log_stats and the rung.")
    print("=========================\n")


if __name__ == "__main__":
    main()
