#!/usr/bin/env python3
"""Phase 0 benchmark for the Speculation Ladder.

One rung, one context bucket, one run. Records acceptance by draft position,
not just throughput, and writes the output token ids so a later run can be
checked for byte equality against the no-speculation arm.

Example:
    python3 run_benchmark.py --rung off    --bucket 2K --limit 100 ...
    python3 run_benchmark.py --rung ngram  --bucket 2K --limit 100 ...
    python3 run_benchmark.py --rung dflash --bucket 2K --limit 100 ...

Every run must be launched with VLLM_USE_V2_MODEL_RUNNER=0 so that all rungs
execute on the stable model runner. The script refuses to start otherwise.
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

    prompts, ids = build_prompts(args.dataset, args.limit, args.prompt_style)
    if not prompts:
        sys.exit(f"no prompts read from {args.dataset}")
    print(f"[{run_id}] {len(prompts)} prompts from {args.dataset}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
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

    # Per request: token ids hashed, so the no-speculation run can be compared
    # against every speculative run without storing the text twice.
    per_request = []
    for rid, o in zip(ids, outputs):
        tok = list(o.outputs[0].token_ids)
        per_request.append(
            {
                "id": rid,
                "prompt_tokens": len(o.prompt_token_ids),
                "gen_tokens": len(tok),
                "sha1": hashlib.sha1(",".join(map(str, tok)).encode()).hexdigest(),
            }
        )

    import torch  # imported late so the runner check fails fast without CUDA init

    record = {
        "run_id": run_id,
        "rung": args.rung,
        "bucket": args.bucket,
        "k": args.k or (5 if args.rung == "ngram" else 15 if args.rung == "dflash" else 0),
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "max_num_seqs": args.max_num_seqs,
        "prompt_style": args.prompt_style,
        "dataset": args.dataset,
        "model": args.model,
        "draft": args.draft,
        "elapsed_s": elapsed,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "throughput_tok_s": gen_tokens / elapsed if elapsed else 0.0,
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
