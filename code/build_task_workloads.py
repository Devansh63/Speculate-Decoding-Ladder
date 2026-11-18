#!/usr/bin/env python3
"""Build task-type workloads at exact prompt lengths, for the task-<B> bundles.

Run once on a DeltaAI login node (no GPU), after `source deltaai_env.sh`:

    "$LADDER_VENV/bin/python3" build_task_workloads.py --check    # counts only, writes nothing
    "$LADDER_VENV/bin/python3" build_task_workloads.py            # build

Why: every LongBench v2 prompt is the same multiple-choice task, so its
"domains" vary the text, not the task. The Berkeley study ("Speculative
Decoding: Performance or Illusion?", MLSys 2026) found that the task decides
between n-gram and a trained drafter when the output copies the input. These
files put four different tasks at the same lengths as the rest of the data,
chosen for how their output relates to their input:

  code     code completion (LongBench v1 lcc, then repobench-p; Python, Java,
           C#): the output continues code that repeats names and lines from
           the input. n-gram's best case.
  summ     summarization (gov_report): long free-form output that paraphrases
           the input.
  fewshot  few-shot dialogue summarization (samsum): many worked examples in
           one fixed format, then a new dialogue. Structured repetition.
  mdqa     multi-document QA (hotpotqa): a short answer extracted from several
           passages.

Source: THUDM/LongBench data.zip on Hugging Face (the dataset's loading script
is not used; recent `datasets` versions refuse to run it). Prompts use
LongBench's own templates. Only the context is cut, to make the whole prompt
the target length in Qwen3-8B tokens: code and fewshot keep the END of the
context (the completion point, and the examples next to the query), summ and
mdqa keep the START. A document shorter than its budget is skipped, never
padded or repeated: repetition would manufacture n-gram hits.

Output, one directory per task, in the layout ksweep.sh reads
(LADDER_DATA=<dir>, file longbench_v2_<B>.jsonl, the name kept for that reason):

  $LADDER_ROOT/data/tasks/<task>/longbench_v2_<B>.jsonl
  $LADDER_ROOT/data/tasks/manifest.json   counts, sources, sha256

Each line: _id, domain (the task), sub_domain (the source dataset), context
(the whole prompt; run with PSTYLE=context), target_tokens, prompt_tokens.
Up to 52 lines per file: 50 measured and 2 for the warm-up, as elsewhere.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

# LongBench's dataset2prompt.json. Only {context} is cut.
TEMPLATES = {
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "gov_report": ("You are given a report by a government agency. Write a one-page summary of the "
                   "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"),
    "samsum": ("Summarize the dialogue into a few short sentences. The following are some examples."
               "\n\n{context}\n\n{input}"),
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer and do not "
                 "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
                 "question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nQuestion: {input}\nAnswer:"),
}
TASKS = {"code": ["lcc", "repobench-p"], "summ": ["gov_report"], "fewshot": ["samsum"], "mdqa": ["hotpotqa"]}
KEEP_END = {"code": True, "fewshot": True, "summ": False, "mdqa": False}
TARGETS = {"2K": 2048, "8K": 8192, "32K": 32768}
PER_FILE = 52   # 50 measured + 2 warm-up
MIN_USABLE = 27  # 25 measured + 2 warm-up, as at 32K elsewhere
SNAPSHOT = "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


def build(tok, zf, buckets):
    names = {os.path.basename(n): n for n in zf.namelist()}

    def records(ds):
        with zf.open(names[f"{ds}.jsonl"]) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    out = {}
    for task, sources in TASKS.items():
        for label in buckets:
            target = TARGETS[label]
            rows, seen, short = [], 0, 0
            for ds in sources:
                tmpl = TEMPLATES[ds]
                for d in records(ds):
                    if len(rows) >= PER_FILE:
                        break
                    if d.get("language") == "zh":
                        continue
                    seen += 1
                    inp = d.get("input", "") or ""
                    fixed = len(tok.encode(tmpl.format(context="", input=inp), add_special_tokens=False))
                    budget = target - fixed
                    ids = tok.encode(d.get("context", "") or "", add_special_tokens=False)
                    if budget <= 0 or len(ids) < budget:
                        short += 1
                        continue
                    ids = ids[-budget:] if KEEP_END[task] else ids[:budget]
                    prompt = tmpl.format(context=tok.decode(ids, clean_up_tokenization_spaces=False), input=inp)
                    rows.append({"_id": d.get("_id") or f"{ds}-{seen}", "domain": task, "sub_domain": ds,
                                 "context": prompt, "target_tokens": target,
                                 "prompt_tokens": len(tok.encode(prompt, add_special_tokens=False))})
            out[(task, label)] = (rows, seen, short)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="count what each file would hold; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--buckets", default="2K 8K 32K")
    ap.add_argument("--tokenizer", help="tokenizer directory (default: Qwen3-8B under LADDER_HF)")
    ap.add_argument("--zip", help="a local LongBench data.zip (default: download from Hugging Face)")
    args = ap.parse_args()

    root = os.environ.get("LADDER_ROOT")
    hf = os.environ.get("LADDER_HF")
    if not root or (not hf and not args.tokenizer):
        sys.exit("LADDER_ROOT and LADDER_HF are not set: source deltaai_env.sh first")
    if root.startswith("/projects/"):
        sys.exit("LADDER_ROOT is under /projects; compute nodes cannot read it")
    out_root = os.path.join(root, "data", "tasks")
    if os.path.isdir(out_root) and os.listdir(out_root) and not (args.check or args.force):
        sys.exit(f"{out_root} already has files; pass --force to rebuild (or --check)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or os.path.join(hf, SNAPSHOT), trust_remote_code=True)
    zpath = args.zip
    if not zpath:
        from huggingface_hub import hf_hub_download
        zpath = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset",
                                cache_dir=os.path.join(root, "hf", "datasets"))
    zsha = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    res = build(tok, zipfile.ZipFile(zpath), args.buckets.split())

    manifest = {"source": "THUDM/LongBench data.zip", "zip_sha256": zsha, "tokenizer": SNAPSHOT,
                "per_file": PER_FILE, "files": {}}
    usable = []
    for (task, label), (rows, seen, short) in res.items():
        n = len(rows)
        lens = [r["prompt_tokens"] for r in rows]
        tag = "ok" if n >= PER_FILE else ("usable, fewer prompts" if n >= MIN_USABLE else "NOT USABLE")
        print(f"{task:8s} {label:4s} {n:3d} prompts ({seen} documents read, {short} too short)"
              + (f", {min(lens)} to {max(lens)} tokens" if lens else "") + f"  {tag}")
        if n < MIN_USABLE:
            continue
        usable.append(f"{task}/{label}")
        if args.check:
            continue
        d = os.path.join(out_root, task)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"longbench_v2_{label}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest["files"][f"{task}/{label}"] = {
            "prompts": n, "sources": sorted({r["sub_domain"] for r in rows}),
            "prompt_tokens_min": min(lens), "prompt_tokens_max": max(lens),
            "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    if not args.check:
        os.makedirs(out_root, exist_ok=True)
        json.dump(manifest, open(os.path.join(out_root, "manifest.json"), "w"), indent=1)
        print(f"wrote {len(manifest['files'])} files under {out_root}")
    print("usable:", " ".join(usable) or "none")


if __name__ == "__main__":
    main()
