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

Harness v4, 12 Nov 2026. A variant of v3, not an edit of it (RULES section 6).
Every default reproduces v3, so a v3 submission pointed here measures the same
thing with more recorded. Four additions:

  1. --async-scheduling {default,on,off} and resolved.async_scheduling. At the
     pin (vllm/config/vllm.py, 1456 to 1505 at bf13ecc2f) async scheduling
     defaults ON for off and dflash and OFF with a warning for CPU ngram, so
     every filed n-gram ratio is across scheduling modes. The resolved state
     is now read off the engine and written in every row, and a run whose
     resolved state disagrees with an explicit request aborts.
  2. Rung ngram_gpu (NgramGPUTypes, vllm/config/speculative.py line 66), the
     n-gram proposer that is allowed to run async, so both rungs can share
     one path.
  3. --domain, a label for per-domain workload files built by make_domains.py.
     It goes into run_id so a domain cell can never be pooled with the
     all-domain cell of the same bucket, and the harness aborts if any prompt
     it read carries a different domain.
  4. --per-request-spec {none,summary,detailed}, which passes
     per_request_spec_decode_metrics to the engine and stores what comes back
     on each request. This is the per-prompt acceptance record; it is OFF by
     default because its cost on this build has not been measured.

Harness v5, 15 Nov 2026. A variant of v4 for vLLM 0.33.0 on Model Runner V2.
v4 and everything before it stay as they are: they are the record of how the
filed A40 and A100 rows were produced at bf13ecc2f on the stable runner.

One change in behaviour. v1 to v4 refuse to start unless
VLLM_USE_V2_MODEL_RUNNER=0. v5 refuses to start unless it is 1, reads
vllm_config.use_v2_model_runner back off the engine, records it under
`resolved`, and aborts if the engine fell back to V1. The principle is the
same one: every arm on one runner, stated explicitly, never left to a default.

What that buys at 0.33.0 (read from the source at the v0.33.0 tag, not yet
observed in a run):

  - `ngram` and `ngram_gpu` both resolve to NgramGPUSpeculator on V2
    (vllm/v1/worker/gpu/spec_decode/__init__.py). There is no CPU n-gram on
    this runner, so the two labels are ONE arm here. Run one of them.
  - The async-scheduling fallback for n-gram applies to the V1 runner only
    (vllm/config/vllm.py, v1_async_sched_incompat). On V2 every arm defaults
    to async on, so off, n-gram and DFlash share a scheduling mode without a
    second baseline. The resolved state is still recorded and still checked.

Rows written by v5 come from a different engine version, runner and binary
than every filed row. Never form a ratio between a v5 row and an earlier one.

Harness v5b, 16 Nov 2026. A variant of v5, not an edit of it (RULES section 6).
Written for the dtai2 job set (dtai2_bundle.sh); v5 is untouched. With no
schedule requested, every default and every recorded field of v5 is
reproduced, and v5b only adds fields. Two additions:

  1. --k-schedule, vLLM 0.33.0's num_speculative_tokens_per_batch_size
     (vllm/config/speculative.py:524), given as "start-end:k,start-end:k".
     If the flag is absent the harness reads LADDER_KSCHED from the
     environment, because ksweep.sh has no knob for it and is Sahil's file.
     The schedule is ignored for rung off.
     What it does at the tag, read from the source and not yet observed in a
     run: the scheduler looks up k by the number of scheduled requests
     (vllm/v1/core/sched/scheduler.py:1710-1714); on Model Runner V2 the
     speculator still drafts num_speculative_tokens and the runner masks the
     surplus (vllm/v1/worker/gpu/model_runner.py:2320-2325), so a smaller
     scheduled k saves verification, not DFlash's draft pass. At k=0, DFlash
     stores the context K/V to stay in sync and skips the query pass
     (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:546-552). So
     "1-1024:0" is a DFlash arm that keeps its drafter current and never
     drafts: its cost over off is DFlash's per-step upkeep alone, the k=0 arm
     the A40 plan could not build.
     The schedule goes into run_id (_ks...), is read back off the engine as
     resolved.spec_k_schedule, and the run aborts if the engine dropped it
     (vllm/config/vllm.py:1088-1127 can, with data parallelism or on V1).
  2. draft_config: the draft checkpoint's architectures, depth, KV heads and
     causal flags, read from its config.json, and the speculator class vLLM's
     dispatch rule picks for those architectures
     (vllm/v1/worker/gpu/spec_decode/__init__.py at the tag). On V2,
     method "dflash" can build any of four classes.
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

HARNESS_VERSION = "v5b-2026-11-16"


def parse_args():
    p = argparse.ArgumentParser(description="Speculation Ladder Phase 0 benchmark")
    p.add_argument("--dataset", required=True, help="Path to the bucket JSONL")
    p.add_argument("--model", required=True, help="Path to the target model snapshot")
    p.add_argument("--draft", default=None, help="Path to the DFlash head (rung dflash only)")
    p.add_argument("--rung", required=True, choices=["off", "ngram", "ngram_gpu", "dflash"])
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
    # ---- v3 additions. Every default below reproduces v2 behaviour.
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
    p.add_argument("--no-token-ids", action="store_true",
                   help="skip storing full token ids (they are about 60 KB per arm)")
    # ---- v4 additions. Every default below reproduces v3 behaviour.
    p.add_argument("--async-scheduling", choices=["default", "on", "off"],
                   default="default",
                   help="'default' leaves the engine alone (v3 behaviour: on "
                        "for off and dflash, off for CPU ngram). 'off' is the "
                        "matched baseline for CPU ngram. 'on' with CPU ngram "
                        "is rejected by the engine at this pin")
    p.add_argument("--domain", default=None,
                   help="label for a per-domain workload file (make_domains.py). "
                        "Goes into run_id; every prompt read must carry it")
    p.add_argument("--per-request-spec", choices=["none", "summary", "detailed"],
                   default="none",
                   help="per_request_spec_decode_metrics: per-prompt acceptance, "
                        "'detailed' adds the ordered per-step arrays")
    # v5b
    p.add_argument("--k-schedule", default=os.environ.get("LADDER_KSCHED") or None,
                   help="num_speculative_tokens_per_batch_size as "
                        "'start-end:k,...', e.g. '1-1024:0'. Default: "
                        "$LADDER_KSCHED. Ignored for rung off")
    return p.parse_args()


def parse_k_schedule(text, k_max):
    """'1-4:7,5-16:2' -> [(1, 4, 7), (5, 16, 2)], with vLLM's own rules.

    The rules are those of validate_and_normalize_dynamic_sd_schedule
    (vllm/v1/spec_decode/dynamic/utils.py at v0.33.0), checked here so a bad
    schedule fails before a model load. One rule is ours: no k above k_max,
    because the engine would clamp it silently and the row would carry a k
    that never ran.
    """
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rng, k = part.split(":")
            a, b = rng.split("-")
            out.append((int(a), int(b), int(k)))
        except ValueError:
            sys.exit(f"ABORT: --k-schedule entry '{part}' is not start-end:k")
    if not out:
        sys.exit("ABORT: --k-schedule is empty")
    out.sort()
    prev_end = 0
    for a, b, k in out:
        if a <= 0 or b < a:
            sys.exit(f"ABORT: --k-schedule range {a}-{b} is not a positive, ordered range")
        if a <= prev_end:
            sys.exit("ABORT: --k-schedule ranges overlap")
        if k < 0 or k > k_max:
            sys.exit(f"ABORT: --k-schedule k={k} is outside 0..{k_max} (the arm's --k)")
        prev_end = b
    if out[0][0] != 1:
        sys.exit("ABORT: --k-schedule must start at batch size 1 (vLLM requires it)")
    return out


def k_schedule_tag(sched):
    """run_id fragment. Digits, 'to', 'k' and '-' only: no dot (check_lossless
    matches a trailing .json) and no underscore (run_id fields split on it)."""
    return "_ks" + "-".join(f"{a}to{b}k{k}" for a, b, k in sched)


def speculator_for(architectures):
    """The class vLLM 0.33.0's init_speculator builds for method 'dflash',
    by the same rule (vllm/v1/worker/gpu/spec_decode/__init__.py). The xpress
    names are read from the installed vLLM when it imports, else the tag's."""
    try:
        from vllm.config.speculative import XPRESS_ARCHITECTURES as xpress
        src = "installed vllm"
    except Exception:
        xpress = {"Qwen3XPressModel", "DFlashQwen3XPressModel"}
        src = "v0.33.0 source"
    archs = set(architectures or [])
    if archs & set(xpress):
        cls = "XPressSpeculator"
    elif "LiLiCorrDraftModel" in archs:
        cls = "LiLiCorrSpeculator"
    elif archs & {"DFlash2DraftModel", "DFlash2BeamDraftModel"}:
        cls = "DFlash2Speculator"
    else:
        cls = "DFlashSpeculator"
    return cls, src


def draft_config(path):
    """What the draft checkpoint says about itself. Never fatal."""
    out = {}
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        lt = c.get("layer_types")
        dc = c.get("dflash_config") if isinstance(c.get("dflash_config"), dict) else {}
        out = {
            "architectures": c.get("architectures"),
            "model_type": c.get("model_type"),
            "num_hidden_layers": c.get("num_hidden_layers"),
            "num_attention_heads": c.get("num_attention_heads"),
            "num_key_value_heads": c.get("num_key_value_heads"),
            "head_dim": c.get("head_dim"),
            "hidden_size": c.get("hidden_size"),
            "block_size": c.get("block_size", dc.get("block_size")),
            "is_causal": c.get("is_causal"),
            "dflash_config_causal": dc.get("causal"),
            "layer_types": sorted(set(lt)) if isinstance(lt, list) else lt,
        }
        out["speculator_class"], out["speculator_rule_from"] = speculator_for(c.get("architectures"))
    except Exception as e:
        out["error"] = repr(e)
    return out


def build_prompts(path, limit, style):
    """Read the bucket file and build identical prompts for every rung."""
    prompts, ids, domains = [], [], []
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
            domains.append(d.get("domain", ""))
    return prompts, ids, domains


def domain_slug(name):
    """'Single-Document QA' -> 'single_document_qa'. Same rule as make_domains.py."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in (name or ""))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def spec_kwargs(args):
    """Speculative settings per rung, using this build's speculative_config dict."""
    if args.rung == "off":
        return {}
    kw = _spec_kwargs_v5(args)
    if args.k_schedule_parsed:  # v5b
        kw["speculative_config"]["num_speculative_tokens_per_batch_size"] = [
            tuple(e) for e in args.k_schedule_parsed]
    return kw


def _spec_kwargs_v5(args):
    """v5's spec_kwargs, unchanged."""
    if args.rung in ("ngram", "ngram_gpu"):
        k = args.k or 5
        return {
            "speculative_config": {
                "method": args.rung,
                "num_speculative_tokens": k,
                "prompt_lookup_max": args.prompt_lookup_max,
                "prompt_lookup_min": args.prompt_lookup_min,
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
    # v5b: the schedule is parsed before anything loads, and only for a
    # speculative rung. The off arm of a schedule cell is a plain off arm.
    args.k_schedule_parsed = None
    if args.k_schedule and args.rung != "off":
        k_max = args.k or (5 if args.rung in ("ngram", "ngram_gpu") else 15)
        args.k_schedule_parsed = parse_k_schedule(args.k_schedule, k_max)
    elif args.k_schedule:
        print(f"note: --k-schedule '{args.k_schedule}' ignored for rung off", flush=True)

    runner_pin = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if runner_pin != "1":
        sys.exit(
            "VLLM_USE_V2_MODEL_RUNNER is not '1'. Harness v5 runs every arm on "
            "Model Runner V2 and will not leave the choice to the engine's "
            "default. Export it and rerun. For the stable runner at the old "
            "pin, use run_benchmark.v4.py."
        )

    os.makedirs(args.outdir, exist_ok=True)
    run_id = f"{args.rung}_{args.bucket}_k{args.k or 0}_n{args.limit}_s{args.seed}_c{args.max_num_seqs}"
    if args.prefix_caching is False:
        # Keep the determinism control in its own files rather than letting it
        # overwrite the matching normal run.
        run_id += "_nopc"
    # v3: a run with a knob off its v2 default is a different cell and must not
    # share a run_id with the existing grid.
    if args.temperature != 0.0:
        # "p" for the decimal point: a dot inside the file name stem makes any
        # trailing \.json$ pattern ambiguous, and check_lossless.py has one.
        run_id += "_T" + f"{args.temperature:g}".replace(".", "p")
    if (args.prompt_lookup_min, args.prompt_lookup_max) != (3, 5):
        run_id += f"_pl{args.prompt_lookup_min}-{args.prompt_lookup_max}"
    if args.loop_timing:
        run_id += "_loop"
    # v4: same principle. An explicit scheduling mode, a domain file and the
    # per-request metric switch are each a different cell.
    if args.async_scheduling != "default":
        run_id += "_async" + ("1" if args.async_scheduling == "on" else "0")
    if args.per_request_spec != "none":
        run_id += "_prs" + args.per_request_spec[0]
    if args.domain:
        run_id += "_d" + domain_slug(args.domain)
    if args.k_schedule_parsed:  # v5b
        run_id += k_schedule_tag(args.k_schedule_parsed)

    prompts, ids, domains = build_prompts(args.dataset, args.limit, args.prompt_style)
    if not prompts:
        sys.exit(f"no prompts read from {args.dataset}")
    if args.domain:
        wrong = sorted(set(d for d in domains if domain_slug(d) != domain_slug(args.domain)))
        if wrong:
            sys.exit(f"ABORT: --domain {args.domain} but {args.dataset} holds "
                     f"prompts from {wrong}. The row would be mislabelled.")
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
        **({"async_scheduling": args.async_scheduling == "on"}
           if args.async_scheduling != "default" else {}),
        **({"per_request_spec_decode_metrics": args.per_request_spec}
           if args.per_request_spec != "none" else {}),
        trust_remote_code=True,
        disable_log_stats=False,  # required, or get_metrics returns nothing
        **spec_kwargs(args),
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
                "spec_method": None, "spec_num_tokens": None,
                "async_scheduling": None, "use_v2_model_runner": None}
    try:
        cfg = llm.llm_engine.vllm_config
        resolved["use_v2_model_runner"] = getattr(cfg, "use_v2_model_runner", None)
        sched = getattr(cfg, "scheduler_config", None)
        cache = getattr(cfg, "cache_config", None)
        spec = getattr(cfg, "speculative_config", None)
        if sched is not None:
            resolved["max_num_seqs"] = getattr(sched, "max_num_seqs", None)
            resolved["async_scheduling"] = getattr(sched, "async_scheduling", None)
        if cache is not None:
            resolved["prefix_caching"] = getattr(cache, "enable_prefix_caching", None)
        if spec is not None:
            resolved["spec_method"] = getattr(spec, "method", None)
            resolved["spec_num_tokens"] = getattr(spec, "num_speculative_tokens", None)
            # v5b
            ks = getattr(spec, "num_speculative_tokens_per_batch_size", None)
            resolved["spec_k_schedule"] = [list(map(int, e)) for e in ks] if ks else None
    except Exception as e:
        resolved["error"] = repr(e)

    print(f"[{run_id}] resolved: {resolved}", flush=True)

    # v5: the engine can still pick V1 for a feature V2 lacks. That would put
    # this arm on a different runner from the others in the job.
    if resolved.get("use_v2_model_runner") is False:
        sys.exit("ABORT: asked for Model Runner V2, the engine resolved V1 for "
                 "this configuration. The arm would be on a different runner.")
    if resolved.get("use_v2_model_runner") is None:
        print(f"[{run_id}] WARNING: could not read use_v2_model_runner off the "
              f"engine config; the runner is unverified for this run.", flush=True)

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

    # v5b: a schedule the engine dropped would leave a fixed-k arm labelled as
    # a scheduled one. vllm/config/vllm.py:1088-1127 drops it with a warning.
    if args.k_schedule_parsed:
        want_ks = [list(e) for e in args.k_schedule_parsed]
        got_ks = resolved.get("spec_k_schedule")
        if "error" in resolved or "spec_k_schedule" not in resolved:
            print(f"[{run_id}] WARNING: could not read the k schedule off the "
                  f"engine config; the schedule is unverified for this run.", flush=True)
        elif got_ks is None:
            sys.exit(f"ABORT: asked for k schedule {want_ks}, the engine dropped it. "
                     f"The arm would be a fixed-k arm with a schedule label.")
        elif sorted(got_ks) != sorted(want_ks):
            sys.exit(f"ABORT: asked for k schedule {want_ks}, engine has {got_ks}.")

    # v4: an explicit scheduling mode that the engine did not honour would
    # put the row in the wrong half of every matched comparison.
    if args.async_scheduling != "default":
        want_async = args.async_scheduling == "on"
        got_async = resolved.get("async_scheduling")
        if got_async is None:
            print(f"[{run_id}] WARNING: could not read async_scheduling off the "
                  f"engine config; the requested state is unverified.", flush=True)
        elif bool(got_async) != want_async:
            sys.exit(f"ABORT: asked for async_scheduling={want_async}, engine "
                     f"resolved {got_async}. The row would be mislabelled.")

    sp = SamplingParams(temperature=args.temperature,
                        max_tokens=args.max_tokens, seed=args.seed)

    print("warmup", flush=True)
    llm.generate(prompts[:2], sp, use_tqdm=False)

    before = snapshot(llm)
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

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)

    # Per request. The token ids are the point: a SHA-1 says two generations
    # differ, it cannot say whether one token moved or the whole answer did.
    # On a continuous batching server those are very different claims, and the
    # losslessness result turns on telling them apart.
    per_request = []
    timing_seen = 0
    spec_seen = 0
    for (rid, dom), o in zip(zip(ids, domains), outputs):
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
        if dom:
            rec["domain"] = dom
        # v4: per-prompt acceptance, when the engine was asked for it. The
        # object is a dataclass or a msgspec struct depending on the build, so
        # try both and fall back to repr rather than lose the run.
        if args.per_request_spec != "none":
            try:
                sm = getattr(o.outputs[0], "spec_decode_metrics", None)
                if sm is not None:
                    if hasattr(sm, "__dataclass_fields__"):
                        import dataclasses
                        rec["spec_decode_metrics"] = dataclasses.asdict(sm)
                    elif hasattr(sm, "__struct_fields__"):
                        rec["spec_decode_metrics"] = {
                            f: getattr(sm, f) for f in sm.__struct_fields__}
                    else:
                        rec["spec_decode_metrics"] = dict(vars(sm))
                    json.dumps(rec["spec_decode_metrics"])  # must be serialisable
                    spec_seen += 1
            except Exception as e:
                rec["spec_decode_metrics"] = {"error": repr(e)}
        t = request_timing(o)
        if rid in loop_latency:
            # Measured by this harness, not reported by the engine. Named
            # distinctly so it is never confused with an engine metric.
            t = dict(t)
            t["loop_latency_s"] = loop_latency[rid]
        if t:
            rec["timing"] = t
            if "latency_s" in t or "loop_latency_s" in t:
                timing_seen += 1
        per_request.append(rec)

    import torch  # imported late so the runner check fails fast without CUDA init

    record = {
        "run_id": run_id,
        "harness": HARNESS_VERSION,
        "rung": args.rung,
        "bucket": args.bucket,
        "k": args.k or (5 if args.rung in ("ngram", "ngram_gpu") else 15 if args.rung == "dflash" else 0),
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "max_num_seqs": args.max_num_seqs,
        "prefix_caching": args.prefix_caching,  # None means engine default
        # v3: what the engine actually used, as opposed to what was asked for.
        # Every v2 row reads prefix_caching: null, meaning only "flag not
        # passed" -- while the logs show the default was ON. The rows could not
        # answer a question the logs could.
        "resolved": resolved,
        "temperature": args.temperature,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
        "loop_timing": bool(args.loop_timing),
        # v4. "requested" is what was asked; resolved.async_scheduling above
        # is what the engine ran. Compare arms on the resolved value only.
        "async_scheduling_requested": args.async_scheduling,
        "domain": args.domain,
        "domains_read": sorted(set(d for d in domains if d)),
        "per_request_spec": args.per_request_spec,
        "per_request_spec_seen": spec_seen,
        # v5b
        "k_schedule_requested": ([list(e) for e in args.k_schedule_parsed]
                                 if args.k_schedule_parsed else None),
        "draft_config": draft_config(args.draft) if args.draft else None,
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

    # The job id goes in the FILE NAME only, never in run_id: run_id is the
    # cell key in results.jsonl and every analysis tool groups on it, so a job
    # id there would make each run its own cell.
    #
    # And the job id ALONE IS NOT ENOUGH. Two runs of the same cell inside one
    # job share it -- which is exactly what a REPEAT=2 control does, and job
    # 22875382 proved it: two 2K off runs, one surviving file. So pick the
    # first free name instead of a computed one. Nothing can overwrite anything
    # here, whatever the caller does or how often it is re-run.
    _job = os.environ.get("SLURM_JOB_ID", "nojob")
    _base = f"perreq_{run_id}_j{_job}"
    _perreq, _rep = f"{_base}.json", 1
    while os.path.exists(os.path.join(args.outdir, _perreq)):
        _rep += 1
        _perreq = f"{_base}_r{_rep}.json"
    if _rep > 1:
        print(f"[{run_id}] per-request file existed; writing {_perreq}",
              flush=True)
    # Name the file in the row, so the join from a row to its per-request
    # records is explicit instead of reconstructed from a pattern.
    record["perreq"] = _perreq

    with open(os.path.join(args.outdir, "results.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    with open(os.path.join(args.outdir, _perreq), "w") as f:
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
