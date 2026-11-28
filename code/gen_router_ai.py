#!/usr/bin/env python3
"""Generate the AI-written parts of the router workloads with the DeepSeek API.

Run where the API is reachable (a DeltaAI login node or a laptop), after
build_router_workloads.py has written the requests:

    export DEEPSEEK_API_KEY=...          # in the shell only; never in a file in this repo
    python3 gen_router_ai.py --dry-run   # what would be asked, and a cost estimate
    python3 gen_router_ai.py             # ask; safe to rerun, it only asks what is missing
    python3 build_router_workloads.py --only ai   # assemble the AI workloads

Standard library only (urllib), so it runs with any Python 3.8+.

What the model writes, and what it never writes. DeepSeek writes INPUTS only:
instructions over real documents, edit requests over real code, the filler
for the padding test, and twins of real chat requests. It never writes the
text the drafters are measured on: that is always Qwen3-8B's own output, in
the GPU job. Acceptance depends on what the target writes, which is why
drafter training recipes regenerate every response with the target, and why
our labels are measurements, not judgments.

  synth_<B>        six requests per real LongBench v2 document, one per copy
                   level: verbatim, edit, quote, summarize, brief, create
  codeedit_<B>     four requests per real code excerpt: whole_file,
                   one_function, fix_or_feature, question
  filler_mdqa      invented encyclopedia passages (the llm_pad arm of the
                   padding test; never used for training)
  filler_fewshot   invented chat dialogues with summaries (the same)
  twins            a new request of the same kind for each real WildChat one

Requests:  <out>/_ai/requests/<kind>.jsonl   (from build_router_workloads.py)
Responses: <out>/_ai/responses/<kind>.jsonl  (appended; a request with an ok
           response is never asked again)

Model: $DEEPSEEK_MODEL, default deepseek-flash (DeepSeek's pricing page lists
deepseek-flash and deepseek-v4-pro on 25 Nov 2026). Endpoint:
https://api.deepseek.com/chat/completions, OpenAI format, JSON output mode.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
# USD per 1M tokens, deepseek-flash peak rates on 25 Nov 2026 (input cache miss, output),
# unchanged on 28 Nov; off-peak (outside 01:00-04:00 and 06:00-10:00 UTC on weekdays) is
# half. Only for the estimate printed by --dry-run.
PRICE_IN, PRICE_OUT = 0.30, 1.20

SYSTEM = ("You help build a benchmark of realistic user requests for a language model. "
          "You always answer with a single JSON object and nothing else.")

THEMES = ["small European towns", "minor 19th-century politicians", "regional sports clubs",
          "obscure scientific terms", "independent films", "family-owned companies", "rivers and lakes",
          "local festivals", "early aviation pioneers", "rare plant species", "university departments",
          "historic buildings", "folk musicians", "railway lines", "board games", "regional dishes",
          "naval vessels", "minor poets", "mountain passes", "radio stations"]
TOPICS = ["weekend plans", "a late delivery", "a birthday gift", "work schedules", "a lost phone",
          "a football match", "moving house", "a recipe", "a job interview", "a broken car",
          "concert tickets", "a school project", "a vet visit", "holiday photos", "a new flat",
          "a gym class", "a family dinner", "an exam result", "a book club", "a rainy day"]


def prompt_for(req):
    k = req["kind"]
    if k == "synth":
        return (
            "Below is a document. Write six requests that a real user might send to an AI assistant "
            "together with this document, one of each type:\n"
            "- verbatim: asks to copy out specific text exactly as written (a section, a list, a "
            "table, every sentence about a named thing)\n"
            "- edit: asks to rewrite a specific passage with one small, specific change (tense, "
            "person, names, bullet points), keeping everything else\n"
            "- quote: asks a question and wants the answer backed by direct quotes\n"
            "- summarize: asks for a summary of a named part or of the whole\n"
            "- brief: asks a factual question whose answer is a few words\n"
            "- create: asks for new writing based on the document (an email, a critique, a story, "
            "a plan)\n"
            "Each request must fit exactly this text: refer to names, sections or details that "
            "appear in it. Write each as the user would, in English, in one to three sentences. Do "
            "not answer them.\n"
            'Return JSON: {"verbatim": "...", "edit": "...", "quote": "...", "summarize": "...", '
            '"brief": "...", "create": "..."}\n\nDOCUMENT:\n<<<\n' + req["document"] + "\n>>>"), 1500
    if k == "codeedit":
        return (
            "Below is a code file. Write four requests a developer might send to an AI coding "
            "assistant together with this file, one of each kind, using real identifiers from the "
            "code:\n"
            "- whole_file: a change across the file that needs the whole updated file back (rename "
            "something used in many places, add type hints or logging throughout)\n"
            "- one_function: a change to one named function or method, asking for that function back\n"
            "- fix_or_feature: a bug fix or small feature in one named function, asking for its new "
            "version\n"
            "- question: a question about how part of the code works, to be answered in prose\n"
            "Write each as the developer would, in one to three sentences. Do not answer them.\n"
            'Return JSON: {"whole_file": "...", "one_function": "...", "fix_or_feature": "...", '
            '"question": "..."}\n\nCODE:\n<<<\n' + req["code"] + "\n>>>"), 1200
    if k == "filler_mdqa":
        rng = random.Random(req["rid"])
        th = ", ".join(rng.sample(THEMES, 3))
        return (
            f"Write {req['n']} short encyclopedia articles in the style of Wikipedia introductions, "
            f"each about a different invented but realistic subject. Subjects for this batch: {th}. "
            "Do not write about real, well-known people, places or organizations. Each article has a "
            "title line and 120 to 220 words of text with concrete names, dates and numbers.\n"
            'Return JSON: {"passages": [{"title": "...", "text": "..."}]}'), 6000
    if k == "filler_fewshot":
        rng = random.Random(req["rid"])
        tp = ", ".join(rng.sample(TOPICS, 4))
        return (
            f"Write {req['n']} short conversations of the kind people have on a messaging app, "
            "between two or three people with first names, 3 to 12 messages each, each followed by "
            "a one or two sentence summary in the third person. Topics for this batch: "
            f"{tp}. Write each message as a line 'Name: text'.\n"
            'Return JSON: {"examples": [{"dialogue": "Name: ...\\nName: ...", "summary": "..."}]}'), 5000
    if k == "twin":
        return (
            "Below is a real request that a user sent to an AI chatbot. Write a different request of "
            "the same kind: the same type of task, the same tone and language, and about the same "
            "length. If it contains pasted material (code, text, data, a document), replace that with "
            "different material of the same kind and a similar size. Do not answer it or comment on "
            'it.\nReturn JSON: {"request": "..."}\n\nREQUEST:\n<<<\n' + req["request"] + "\n>>>"), 4000
    raise ValueError(k)


def valid(req, data):
    k = req["kind"]
    if not isinstance(data, dict):
        return "not an object"
    keys = {"synth": ["verbatim", "edit", "quote", "summarize", "brief", "create"],
            "codeedit": ["whole_file", "one_function", "fix_or_feature", "question"]}.get(k)
    if keys:
        bad = [x for x in keys if not isinstance(data.get(x), str) or not 10 <= len(data[x]) <= 1500]
        return f"missing or odd keys {bad}" if bad else None
    if k == "filler_mdqa":
        ps = [p for p in data.get("passages") or [] if isinstance(p, dict)
              and isinstance(p.get("title"), str) and isinstance(p.get("text"), str) and 300 <= len(p["text"]) <= 3000]
        return None if len(ps) >= max(3, req["n"] // 2) else f"only {len(ps)} usable passages"
    if k == "filler_fewshot":
        es = [e for e in data.get("examples") or [] if isinstance(e, dict) and isinstance(e.get("dialogue"), str)
              and isinstance(e.get("summary"), str) and e["dialogue"].count(":") >= 3]
        return None if len(es) >= max(3, req["n"] // 2) else f"only {len(es)} usable examples"
    if k == "twin":
        t = data.get("request")
        if not isinstance(t, str):
            return "no request"
        r = len(t) / max(1, len(req["request"]))
        return None if 0.3 <= r <= 3.0 else f"length ratio {r:.2f}"
    return "unknown kind"


def call(key, prompt, max_tokens, timeout=300):
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}, "temperature": 1.0,
                       # deepseek-flash thinks by default (DeepSeek docs, checked 28 Nov);
                       # reasoning would spend max_tokens before the JSON is written
                       "thinking": {"type": "disabled"},
                       "max_tokens": max_tokens, "stream": False}).encode()
    rq = urllib.request.Request(URL, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask(key, req, tries=5):
    prompt, mt = prompt_for(req)
    err = None
    for a in range(tries):
        try:
            resp = call(key, prompt, mt)
            txt = resp["choices"][0]["message"]["content"]
            data = json.loads(txt)
            why = valid(req, data)
            if why is None:
                return {"rid": req["rid"], "ok": True, "data": data, "model": resp.get("model", MODEL),
                        "usage": resp.get("usage"), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            err = "invalid: " + why
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code in (401, 402, 403):   # bad key, no balance, forbidden: retrying will not help
                return {"rid": req["rid"], "ok": False, "error": err, "fatal": True}
        except Exception as e:
            err = type(e).__name__ + ": " + str(e)[:200]
        time.sleep(min(60, 2 ** a + random.random()))
    return {"rid": req["rid"], "ok": False, "error": err}


def main():
    ap = argparse.ArgumentParser(description="DeepSeek-written inputs for the router workloads")
    ap.add_argument("--out", help="the workload root (default $LADDER_ROOT/data/router)")
    ap.add_argument("--kinds", help="only these request files, e.g. 'synth_2K twins'")
    ap.add_argument("--limit", type=int, default=0, help="at most this many new requests per file (a pilot)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count, estimate the cost, ask nothing")
    args = ap.parse_args()
    root = args.out or (os.environ.get("LADDER_ROOT") and os.path.join(os.environ["LADDER_ROOT"], "data", "router"))
    if not root:
        sys.exit("pass --out or set LADDER_ROOT")
    rdir, odir = os.path.join(root, "_ai", "requests"), os.path.join(root, "_ai", "responses")
    if not os.path.isdir(rdir):
        sys.exit(f"no {rdir}: run build_router_workloads.py first")
    os.makedirs(odir, exist_ok=True)
    kinds = sorted(f[:-6] for f in os.listdir(rdir) if f.endswith(".jsonl"))
    if args.kinds:
        kinds = [k for k in kinds if k in args.kinds.split()]
    todo, tin, tout = {}, 0, 0
    for k in kinds:
        reqs = [json.loads(l) for l in open(os.path.join(rdir, k + ".jsonl")) if l.strip()]
        done = set()
        op = os.path.join(odir, k + ".jsonl")
        if os.path.exists(op):
            for l in open(op):
                if l.strip():
                    r = json.loads(l)
                    if r.get("ok"):
                        done.add(r["rid"])
        left = [r for r in reqs if r["rid"] not in done]
        if args.limit:
            left = left[:args.limit]
        todo[k] = left
        cin = sum(len(prompt_for(r)[0]) for r in left) / 4
        cout = sum(prompt_for(r)[1] for r in left) * 0.5
        tin, tout = tin + cin, tout + cout
        print(f"{k:18s} {len(reqs):4d} requests, {len(done):4d} done, {len(left):4d} to ask"
              f"  (~{cin / 1e6:.2f}M tokens in)")
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    print(f"total ~{tin / 1e6:.2f}M tokens in, ~{tout / 1e6:.2f}M out, about ${cost:.2f} at {MODEL} peak rates "
          f"(off-peak is half)")
    if args.dry_run:
        return
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set. Export it in this shell (never write it into the repo).")
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    fatal = threading.Event()
    for k, left in todo.items():
        if not left:
            continue
        op = os.path.join(odir, k + ".jsonl")

        def one(req):
            if fatal.is_set():
                return
            res = ask(key, req)
            with lock:
                with open(op, "a") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                stats["ok" if res["ok"] else "fail"] += 1
                if res.get("fatal"):
                    fatal.set()
                    print(f"FATAL: {res['error']} (check the key and the account balance)", flush=True)
                n = stats["ok"] + stats["fail"]
                if n % 10 == 0 or not res["ok"]:
                    print(f"  {k}: {stats['ok']} ok, {stats['fail']} failed"
                          + ("" if res["ok"] else f"  last error: {res.get('error')}"), flush=True)
        with cf.ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, left))
        if fatal.is_set():
            sys.exit(1)
    print(f"done: {stats['ok']} ok, {stats['fail']} failed. Rerun to retry the failures, then:")
    print("  python3 build_router_workloads.py --only ai")


if __name__ == "__main__":
    main()
