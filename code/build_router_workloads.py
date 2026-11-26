#!/usr/bin/env python3
"""Build the router-data workloads for the dtai3 job set (25 Nov 2026).

Run on a DeltaAI login node (no GPU; it downloads from Hugging Face), after
`source deltaai_env.sh`:

    "$LADDER_VENV/bin/python3" build_router_workloads.py --check   # counts only
    "$LADDER_VENV/bin/python3" build_router_workloads.py           # build
    # then, if you use the AI workloads (gen_router_ai.py, needs DEEPSEEK_API_KEY):
    "$LADDER_VENV/bin/python3" gen_router_ai.py
    "$LADDER_VENV/bin/python3" build_router_workloads.py --only ai  # assemble them

Why these files (notes/2026-11-25-router-data-jobs.md has the full argument):
every prompt measured before 25 Nov was sent as raw text, at most 32K, from
two LongBench sets whose instructions never vary inside a task. A router has
to learn from the cases where the choice between drafters is close and where
how much the output copies the prompt changes; those cases were missing. This
script builds them from real text, and pads only where the padding itself is
real text of the task's own kind.

Layout: <out>/<workload>/<bucket>.jsonl and <out>/manifest.json, where <out>
defaults to $LADDER_ROOT/data/router. router_cell.sh reads
LADDER_DATA=<out>/<workload> and the file <bucket>.jsonl. Every file holds the
measured prompts first and two warm-up prompts last, as harness v5c and v5d
expect. Every row has _id, workload, task, bucket, construction, source,
target_tokens, prompt_tokens (the user content, untemplated; the chat template
adds about 10 to 20), and either context (the whole user message, run with
PSTYLE=context) or messages (multi-turn: chats and agent traces).

Constructions, recorded per row so that no padded prompt is ever pooled with
a natural one by accident:
  natural            a real document at or above the bucket, cut to it
                     (the rule of build_workload.py and build_task_workloads.py)
  natural_length     a real conversation or trace whose own length falls in
                     the band; nothing cut, nothing added
  real_distractors   a real HotpotQA item plus passages from OTHER real HotpotQA
                     items, shuffled (how HotpotQA's own distractor setting
                     works); the item's passages are never dropped
  many_shot_real     a real SAMSum query plus more real SAMSum examples
                     (many-shot prompting)
  repo_prefix        a real code repository from LongBench v2, cut at a line
                     boundary, as code completion
  long_document      a real long document from LongBench v2 under a
                     summarization instruction
  instruction_variant  the same real document under different instructions
                     (ragquote, codeedit): the instruction is the variable
  repeat_pad         NEGATIVE CONTROL ONLY: a 2K prompt padded to 8K with one
                     repeated sentence, as SPEED-Bench pads its length buckets
                     ("Answer now please." lines). Exists to show what such
                     padding does to n-gram, never for training
  llm_pad, llm_instruction, llm_twin   AI-generated parts (gen_router_ai.py)

Never padded with repeated or invented text except in the two clearly named
test workloads (eq_*_repeatpad, eq_*_llmpad), which exist to measure whether
padding changes the drafters' behaviour against the natural 8K files.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import zipfile

SNAPSHOT = "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
TARGETS = {"nat": 0, "2K": 2048, "8K": 8192, "16K": 16384, "32K": 32768}
# measured + 2 warm-up rows per file
N_C1 = {"2K": 52, "8K": 52, "16K": 52, "32K": 27}
N_QA = {"2K": 130, "8K": 130, "16K": 66, "32K": 50}      # QA also feeds the load cells
N_TASK_LOAD = 130                                      # tasks at 2K and 8K, for load cells
AI_DIR = "_ai"

# LongBench v1 templates (dataset2prompt.json), as build_task_workloads.py
TPL_LCC = "Please complete the code given below. \n{context}Next line of code:\n"
TPL_SAMSUM = ("Summarize the dialogue into a few short sentences. The following are some examples."
              "\n\n{context}\n\n{input}")
TPL_HOTPOT = ("Answer the question based on the given passages. Only give me the answer and do not "
              "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
              "question based on the given passages. Only give me the answer and do not output any "
              "other words.\n\nQuestion: {input}\nAnswer:")
TPL_LONGDOC = ("You are given a document. Write a one-page summary of the document.\n\nDocument:\n"
               "{context}\n\nNow, write a one-page summary of the document.\n\nSummary:")
REPEAT_SENTENCE = "Answer now please."

# Same text as run_benchmark v5c/v5d build_prompts(style="qa"), so a chat QA
# prompt is the raw QA prompt of the same id inside the chat template.
def qa_prompt(ctx, d):
    return (f"{ctx}\n\nQuestion: {d['question']}\n"
            f"A. {d.get('choice_A','')}\nB. {d.get('choice_B','')}\n"
            f"C. {d.get('choice_C','')}\nD. {d.get('choice_D','')}\n\n"
            "Answer the question and explain your reasoning.\n\nAnswer:")


def choices_block(d):
    return (f"Question: {d['question']}\nA. {d.get('choice_A','')}\nB. {d.get('choice_B','')}\n"
            f"C. {d.get('choice_C','')}\nD. {d.get('choice_D','')}")


RAG_VARIANTS = [  # (variant, copy_level, instruction placed after the document)
    ("extract", "very_high",
     "Copy out, word for word, the passage of the document above that best answers the question "
     "below, then say which choice it supports."),
    ("quote", "high",
     "Answer the question below. Support your answer with direct quotes from the document above, "
     "copied word for word inside quotation marks."),
    ("explain", "low",
     "Answer the question below and explain your reasoning in your own words. Do not quote the "
     "document."),
    ("brief", "none", "Answer the question below with only the letter of the correct choice."),
]

EDIT_VARIANTS = [  # (variant, copy_level, instruction template)
    ("rename_all", "very_high",
     "Rename `{name}` to `{new}` everywhere in the code above. Output the complete updated code, "
     "with no other changes and no explanation."),
    ("comment_fn", "high",
     "Add a short comment above each block of statements inside `{name}` saying what it does. "
     "Output the complete updated `{name}` and nothing else."),
    ("refactor_fn", "mid",
     "Refactor `{name}` to make it easier to read without changing what it does. Output only the "
     "new version of `{name}`."),
    ("explain_fn", "low", "In two or three sentences, explain what `{name}` does and when it is used."),
]


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def rng_for(*parts):
    return random.Random(int(h("|".join(map(str, parts)))[:12], 16))


class Tok:
    """The target's tokenizer: content lengths and chat-templated lengths."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._memo = {}

    def enc(self, s):
        return self.t.encode(s, add_special_tokens=False)

    def dec(self, ids):
        return self.t.decode(ids, clean_up_tokenization_spaces=False)

    def n(self, s):
        if len(s) < 20000:
            k = hash(s)
            v = self._memo.get(k)
            if v is None:
                v = self._memo[k] = len(self.enc(s))
            return v
        return len(self.enc(s))

    def head(self, s, L):
        """The first tokens of s, enough to cut L from: tokenizing a prefix of
        12 characters per token gives the same first L tokens as the whole
        text (BPE merges are local), without tokenizing million-word contexts."""
        return self.enc(s[:12 * L + 1000]) if len(s) > 12 * L + 1000 else self.enc(s)

    def chat_len(self, messages):
        text = self.t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        return len(self.t(text, add_special_tokens=False)["input_ids"])


def cut_to(tok, text, budget, keep_end):
    ids = tok.enc(text)
    if len(ids) < budget:
        return None
    ids = ids[-budget:] if keep_end else ids[:budget]
    return tok.dec(ids)


def fit_prompt(tok, tpl, context, budget_total, keep_end, **kw):
    """Cut `context` so the whole templated prompt is budget_total tokens."""
    fixed = tok.n(tpl.format(context="", **kw))
    c = cut_to(tok, context, budget_total - fixed, keep_end)
    if c is None:
        return None
    p = tpl.format(context=c, **kw)
    return p


def row(**kw):
    return kw


# ---------------------------------------------------------------- sources

class Sources:
    def __init__(self, args, tok):
        self.args, self.tok = args, tok
        self._v1, self._v2 = None, None

    def _dl(self, repo, fn):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, fn, repo_type="dataset", cache_dir=self.args.cache)

    def v1(self, ds):
        if self._v1 is None:
            z = zipfile.ZipFile(self.args.zip or self._dl("THUDM/LongBench", "data.zip"))
            self._v1 = (z, {os.path.basename(n): n for n in z.namelist()})
        z, names = self._v1
        out = []
        with z.open(names[f"{ds}.jsonl"]) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("language") != "zh":
                        out.append(d)
        return out

    def v2(self):
        if self._v2 is None:
            self._v2 = json.load(open(self.args.v2 or self._dl("THUDM/LongBench-v2", "data.json")))
        return self._v2

    def parquet_rows(self, repo, total, n_shards, columns=None):
        """Rows of the first n_shards of a dataset stored as data/train-*-of-<total>.parquet."""
        import pyarrow.parquet as pq
        for i in range(min(n_shards, total)):
            fn = f"data/train-{i:05d}-of-{total:05d}.parquet"
            t = pq.read_table(self._dl(repo, fn), columns=columns)
            for r in t.to_pylist():
                yield r

    def instructcoder(self):
        return json.load(open(self._dl("likaixin/InstructCoder", "valid.json")))


# ---------------------------------------------------------------- QA (LongBench v2)

def build_qa(S, tok, buckets):
    """The rule of build_workload.py: file order, context at or above L tokens,
    cut to its first L. The first 52 (27 at 32K) rows are the raw runs' prompts."""
    out = {}
    for b in buckets:
        L = TARGETS[b]
        rows = []
        for d in S.v2():
            if len(rows) >= N_QA[b]:
                break
            ids = tok.head(d.get("context", ""), L)
            if len(ids) < L:
                continue
            ctx = tok.dec(ids[:L])
            p = qa_prompt(ctx, d)
            rows.append(row(_id=d["_id"], workload="qa", task="qa", domain=d.get("domain"), bucket=b,
                            construction="natural", source="THUDM/LongBench-v2", target_tokens=L,
                            prompt_tokens=tok.n(p), context=p))
        out[("qa", b)] = rows
    return out


def qa_ids(built, b):
    return {r["_id"] for r in built.get(("qa", b), [])}


# ---------------------------------------------------------------- tasks

def build_tasks_natural(S, tok, buckets, per_file):
    """2K, 8K (and the natural 16K cells) with build_task_workloads.build(), so
    the first 52 rows at 2K and 8K are exactly the raw task files' prompts."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_task_workloads as B

    S.v1("lcc")  # opens the zip
    z, _ = S._v1
    saved = (dict(B.TARGETS), B.PER_FILE)
    B.TARGETS.update({"16K": 16384})
    B.PER_FILE = per_file
    try:
        res = B.build(tok.t, z, buckets)
    finally:
        B.TARGETS.clear(); B.TARGETS.update(saved[0]); B.PER_FILE = saved[1]
    out = {}
    for (task, b), (rows, seen, short) in res.items():
        out[(task, b)] = [row(_id=r["_id"], workload=task, task=task, domain=task, bucket=b,
                              construction="natural", source="THUDM/LongBench:" + r["sub_domain"],
                              target_tokens=r["target_tokens"], prompt_tokens=r["prompt_tokens"],
                              context=r["context"]) for r in rows]
    return out


def samsum_pool(S):
    """Every real (dialogue, summary) example inside LongBench's SAMSum contexts."""
    pool, seen = [], set()
    for d in S.v1("samsum"):
        for part in re.split(r"(?=^Dialogue: )", d["context"], flags=re.M):
            part = part.strip()
            if not part.startswith("Dialogue:") or "\nSummary:" not in part:
                continue
            k = h(part)
            if k not in seen:
                seen.add(k)
                pool.append(part)
    return pool


def many_shot(S, tok, base_items, L, pool, n_rows, tag, start_ctx=None):
    """A SAMSum query with real examples before it: the item's own examples
    nearest the query, more real examples from other items before those, the
    whole cut from the start to exactly L tokens (as the natural builder cuts
    few-shot contexts)."""
    rows = []
    for d in base_items:
        if len(rows) >= n_rows:
            break
        own = start_ctx(d) if start_ctx else d["context"]
        fixed = tok.n(TPL_SAMSUM.format(context="", input=d["input"]))
        need = L - fixed
        rng = rng_for(tag, d["_id"])
        extra, have = [], tok.n(own)
        cand = [e for e in pool if e[:60] not in own and d["input"].strip()[:60] not in e]
        rng.shuffle(cand)
        for e in cand:
            if have >= need + 400:  # a margin: joined text tokenizes a little shorter than its parts
                break
            extra.append(e)
            have += tok.n(e) + 1
        ctx = "\n".join(extra + [own])
        ids = tok.enc(ctx)
        if len(ids) < need:
            continue
        p = TPL_SAMSUM.format(context=tok.dec(ids[-need:]), input=d["input"])
        rows.append(row(_id=f"{tag}:{d['_id']}", workload=None, task="fewshot", domain="fewshot",
                        bucket=None, construction="many_shot_real", source="THUDM/LongBench:samsum",
                        target_tokens=L, prompt_tokens=tok.n(p), base_id=d["_id"],
                        added_examples=len(extra), context=p))
    return rows


PASSAGE_RE = re.compile(r"Passage \d+:\n")


def split_passages(ctx):
    parts = PASSAGE_RE.split(ctx)
    return [p.strip("\n") for p in parts if p.strip()]


def hotpot_pool(S):
    pool, seen = [], set()
    for d in S.v1("hotpotqa"):
        for p in split_passages(d["context"]):
            title = p.split("\n", 1)[0].strip()
            if title and title not in seen and "\n" in p:
                seen.add(title)
                pool.append(p)
    return pool


def distractors(S, tok, base_items, L, pool, n_rows, tag, own_fn=None):
    """A real HotpotQA item's passages among passages from other real items,
    shuffled and renumbered; the item's own passages are always kept whole."""
    rows = []
    for d in base_items:
        if len(rows) >= n_rows:
            break
        own = split_passages(own_fn(d) if own_fn else d["context"])
        titles = {p.split("\n", 1)[0].strip() for p in own}
        fixed = tok.n(TPL_HOTPOT.format(context="", input=d["input"]))
        need = L - fixed
        have = sum(tok.n(p) + 6 for p in own)
        rng = rng_for(tag, d["_id"])
        cand = [p for p in pool if p.split("\n", 1)[0].strip() not in titles]
        rng.shuffle(cand)
        extra = []
        for p in cand:
            if have >= need + 400:
                break
            extra.append(p)
            have += tok.n(p) + 6
        # The cut at `need` tokens falls in the last passage or two. Reshuffle
        # until every one of the item's own passages lies wholly before it, so
        # the question stays answerable; give up on the item after 50 tries.
        kept = None
        for _ in range(50):
            allp = own + extra
            rng.shuffle(allp)
            ctx = "\n".join(f"Passage {i + 1}:\n{p}" for i, p in enumerate(allp))
            ids = tok.enc(ctx)
            if len(ids) < need:
                break
            cand_kept = tok.dec(ids[:need])
            if all(p in cand_kept for p in own):
                kept = cand_kept
                break
        if kept is None:
            continue
        p = TPL_HOTPOT.format(context=kept, input=d["input"])
        rows.append(row(_id=f"{tag}:{d['_id']}", workload=None, task="mdqa", domain="mdqa", bucket=None,
                        construction="real_distractors", source="THUDM/LongBench:hotpotqa",
                        target_tokens=L, prompt_tokens=tok.n(p), base_id=d["_id"],
                        own_passages=len(own), added_passages=len(extra), context=p))
    return rows


def repo_prefix(S, tok, L, n_rows, exclude):
    rows = []
    fixed = tok.n(TPL_LCC.format(context=""))
    for d in S.v2():
        if len(rows) >= n_rows:
            break
        if d.get("domain") != "Code Repository Understanding" or d["_id"] in exclude:
            continue
        ids = tok.head(d["context"], L)
        if len(ids) < L:
            continue
        pre = tok.dec(ids[:L - fixed])
        pre = pre[:pre.rfind("\n") + 1]   # end at a line boundary, inside the repository text
        p = TPL_LCC.format(context=pre)
        rows.append(row(_id=f"repo{L // 1024}K:{d['_id']}", workload="code", task="code", domain="code",
                        bucket=None, construction="repo_prefix", source="THUDM/LongBench-v2:code_repo",
                        target_tokens=L, prompt_tokens=tok.n(p), base_id=d["_id"], context=p))
    return rows


def long_docs(S, tok, L, n_rows, exclude):
    rows = []
    for d in S.v2():
        if len(rows) >= n_rows:
            break
        if d.get("domain") not in ("Single-Document QA",) or d["_id"] in exclude:
            continue
        fixed = tok.n(TPL_LONGDOC.format(context=""))
        ids = tok.head(d["context"], L)
        if len(ids) < L - fixed:
            continue
        p = TPL_LONGDOC.format(context=tok.dec(ids[:L - fixed]))
        rows.append(row(_id=f"doc{L // 1024}K:{d['_id']}", workload="summ", task="summ", domain="summ",
                        bucket=None, construction="long_document",
                        source="THUDM/LongBench-v2:" + str(d.get("sub_domain")), target_tokens=L,
                        prompt_tokens=tok.n(p), base_id=d["_id"], context=p))
    return rows


def build_tasks(S, tok, buckets, built):
    out = {}
    nat_b = [b for b in buckets if b in ("2K", "8K")]
    if nat_b:
        out.update(build_tasks_natural(S, tok, nat_b, N_TASK_LOAD + 2))
    if "16K" in buckets:
        nat16 = build_tasks_natural(S, tok, ["16K"], N_C1["16K"])
        for t in ("code", "summ", "mdqa"):
            out[(t, "16K")] = nat16[(t, "16K")]
    pool_s, pool_h = None, None
    for b in [x for x in buckets if x in ("16K", "32K")]:
        L, n = TARGETS[b], N_C1[b]
        if pool_s is None:
            pool_s, pool_h = samsum_pool(S), hotpot_pool(S)
        fs = many_shot(S, tok, S.v1("samsum"), L, pool_s, n, f"fs{b}")
        out[("fewshot", b)] = fs
        if b == "32K":
            out[("mdqa", b)] = distractors(S, tok, S.v1("hotpotqa"), L, pool_h, n, f"mdqa{b}")
            out[("code", b)] = repo_prefix(S, tok, L, n, qa_ids(built, b))
            out[("summ", b)] = long_docs(S, tok, L, n, qa_ids(built, b))
    for (t, b), rows in out.items():
        for r in rows:
            r["workload"], r["bucket"] = t, b
    return out


# ---------------------------------------------------------------- instruction variants

def build_ragquote(S, tok, buckets, built):
    out = {}
    for b in buckets:
        L = TARGETS[b]
        n_docs = 12 if b == "32K" else 16
        used = qa_ids(built, b)
        rows = []
        docs = [d for d in reversed(S.v2()) if d.get("domain") in ("Single-Document QA", "Multi-Document QA")
                and d["_id"] not in used]
        for d in docs:
            if len(rows) >= 4 * n_docs:
                break
            q = choices_block(d)
            longest = max(tok.n(f"{{context}}\n\n{v[2]}\n\n{q}") for v in RAG_VARIANTS)
            ids = tok.head(d["context"], L)
            budget = L - longest
            if len(ids) < budget:
                continue
            ctx = tok.dec(ids[:budget])
            for name, level, instr in RAG_VARIANTS:
                p = f"{ctx}\n\n{instr}\n\n{q}"
                rows.append(row(_id=f"rq{b}:{d['_id']}:{name}", workload="ragquote", task="ragquote",
                                domain=d.get("domain"), bucket=b, construction="instruction_variant",
                                source="THUDM/LongBench-v2", target_tokens=L, prompt_tokens=tok.n(p),
                                base_id=d["_id"], variant=name, copy_level=level, context=p))
        out[("ragquote", b)] = rows
    return out


DEF_RE = [re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M),
          re.compile(r"^[ \t]*(?:(?:public|private|protected|internal|static|final|override|virtual|"
                     r"async|synchronized|abstract)[ \t]+)+[\w<>\[\],.?]+[ \t]+([A-Za-z_]\w*)[ \t]*\(", re.M)]
SKIP_NAMES = {"if", "for", "while", "switch", "catch", "return", "new", "main", "__init__", "get", "set"}


def pick_function(code, rng):
    names = []
    for rx in DEF_RE:
        for m in rx.finditer(code):
            nm = m.group(1)
            if nm not in SKIP_NAMES and len(nm) > 2 and code.count(nm) >= 2:
                names.append((m.start(), nm))
    if not names:
        return None
    names.sort()
    mid = [n for pos, n in names if len(code) * 0.2 < pos < len(code) * 0.8] or [n for _, n in names]
    return rng.choice(mid)


def code_excerpt(tok, code, budget):
    ids = tok.enc(code)
    if len(ids) < budget:
        return None
    t = tok.dec(ids[:budget])
    return t[:t.rfind("\n") + 1]


def build_codeedit(S, tok, buckets):
    out = {}
    if "nat" in buckets:
        rows = []
        items = [x for x in S.instructcoder() if x.get("input") and x.get("instruction")]
        items.sort(key=lambda x: h(x["instruction"] + x["input"]))
        for x in items:
            if len(rows) >= 52:
                break
            if tok.n(x["input"]) < 150:
                continue
            p = (f"{x['instruction']}\n\n```\n{x['input']}\n```\n\n"
                 "Return the complete updated code, and nothing else.")
            rows.append(row(_id="ic:" + h(x["instruction"] + x["input"])[:16], workload="codeedit",
                            task="codeedit", domain="codeedit", bucket="nat", construction="natural_length",
                            source="likaixin/InstructCoder:valid", target_tokens=0, prompt_tokens=tok.n(p),
                            variant="instructcoder", copy_level="high", context=p))
        out[("codeedit", "nat")] = rows
    srcs = {"2K": ["lcc", "repobench-p"], "8K": ["lcc", "repobench-p"], "16K": ["repobench-p"]}
    longest = max(tok.n("Here is some code:\n\n```\n\n```\n\n" + v[2].format(name="x" * 20, new="y" * 28))
                  for v in EDIT_VARIANTS)
    for b in [x for x in buckets if x in srcs]:
        L = TARGETS[b]
        rows, i = [], 0
        for ds in srcs[b]:
            for d in S.v1(ds):
                if len(rows) >= N_C1[b]:
                    break
                var = EDIT_VARIANTS[i % len(EDIT_VARIANTS)]
                ex = code_excerpt(tok, d["context"], L - longest)
                if ex is None:
                    continue
                rng = rng_for("edit", b, d["_id"])
                name = pick_function(ex, rng)
                if not name:
                    continue
                instr = var[2].format(name=name, new=name + "_renamed")
                p = f"Here is some code:\n\n```\n{ex}```\n\n{instr}"
                rows.append(row(_id=f"ce{b}:{d['_id']}", workload="codeedit", task="codeedit",
                                domain="codeedit", bucket=b, construction="instruction_variant",
                                source="THUDM/LongBench:" + ds, target_tokens=L, prompt_tokens=tok.n(p),
                                base_id=d["_id"], variant=var[0], copy_level=var[1], target_name=name,
                                excerpt_sha=h(ex)[:16], context=p))
                i += 1
        out[("codeedit", b)] = rows
    return out


# ---------------------------------------------------------------- natural-length chats and traces

def band_of(n):
    for b, lo, hi in (("2K", 1536, 3072), ("8K", 6144, 10752), ("16K", 13312, 19456), ("32K", 26624, 34816)):
        if lo <= n <= hi:
            return b
    return None


def build_chat(S, tok, buckets, shards):
    want = {b: N_C1[b] for b in buckets}
    got = {b: [] for b in buckets}
    cols = ["conversation_hash", "conversation", "turn", "language", "toxic", "redacted"]
    cands = []
    for r in S.parquet_rows("allenai/WildChat-1M", 14, shards, cols):
        if r["language"] != "English" or r["toxic"] or r["redacted"] or (r["turn"] or 0) < 2:
            continue
        conv = r["conversation"]
        last = max((i for i, m in enumerate(conv) if m["role"] == "user"), default=None)
        if last is None or last < 2:
            continue
        msgs = [{"role": m["role"], "content": m["content"]} for m in conv[:last + 1]]
        if any(not (m["content"] or "").strip() for m in msgs):
            continue
        est = sum(len(m["content"]) for m in msgs) // 4
        if est < 1000:
            continue
        cands.append((h(r["conversation_hash"]), r["conversation_hash"], msgs))
    cands.sort()
    near = {"2K": (1100, 4300), "8K": (4400, 15000), "16K": (9500, 27000), "32K": (19000, 48000)}
    for _, ch, msgs in cands:
        if all(len(got[b]) >= want[b] for b in got):
            break
        est = sum(len(m["content"]) for m in msgs) // 4
        if not any(near[b][0] <= est <= near[b][1] and len(got[b]) < want[b] for b in got):
            continue
        n = tok.chat_len(msgs)
        b = band_of(n)
        if b in got and len(got[b]) < want[b]:
            got[b].append(row(_id="wc:" + ch, workload="chat", task="chat", domain="chat", bucket=b,
                              construction="natural_length", source="allenai/WildChat-1M",
                              target_tokens=TARGETS[b], prompt_tokens=n, turns=sum(m["role"] == "user" for m in msgs),
                              messages=msgs))
    return {("chat", b): v for b, v in got.items()}


def build_agent(S, tok, buckets, shards):
    want = {b: N_C1[b] for b in buckets}
    got = {b: [] for b in buckets}
    seen = set()
    cands = []
    for r in S.parquet_rows("nebius/SWE-agent-trajectories", 12, shards,
                            ["instance_id", "model_name", "trajectory"]):
        cands.append((h(r["instance_id"] + r["model_name"]), r))
    cands.sort(key=lambda x: x[0])
    for _, r in cands:
        if r["instance_id"] in seen or all(len(got[b]) >= want[b] for b in got):
            continue
        msgs = []
        for m in r["trajectory"]:
            role = {"system": "system", "user": "user", "ai": "assistant"}.get(m["role"])
            text = m["system_prompt"] if m["role"] == "system" else m["text"]
            if role and text and text != "None":
                msgs.append({"role": role, "content": text})
        # candidate cuts: just before an assistant turn
        best = {}
        approx = 0
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and i > 1:
                est = approx // 4
                for b in got:
                    L = TARGETS[b]
                    if 0.8 * L <= est <= 1.2 * L and len(got[b]) < want[b]:
                        best.setdefault(b, []).append(i)
            approx += len(m["content"])
        for b, cuts in best.items():
            if len(got[b]) >= want[b]:
                continue
            L = TARGETS[b]
            scored = []
            for i in cuts[:: max(1, len(cuts) // 6)]:
                n = tok.chat_len(msgs[:i])
                if band_of(n) == b:
                    scored.append((abs(n - L), i, n))
            if scored:
                _, i, n = min(scored)
                got[b].append(row(_id=f"swe:{r['instance_id']}:{i}", workload="agent", task="agent",
                                  domain="agent", bucket=b, construction="natural_length",
                                  source="nebius/SWE-agent-trajectories:" + r["model_name"],
                                  target_tokens=L, prompt_tokens=n, turns=sum(x["role"] == "assistant" for x in msgs[:i]),
                                  messages=msgs[:i]))
                seen.add(r["instance_id"])
                break
    return {("agent", b): v for b, v in got.items()}


def build_twins_real(S, tok, shards, n=52):
    rows, cands = [], []
    cols = ["conversation_hash", "conversation", "turn", "language", "toxic", "redacted"]
    for r in S.parquet_rows("allenai/WildChat-1M", 14, shards, cols):
        if r["language"] != "English" or r["toxic"] or r["redacted"] or r["turn"] != 1:
            continue
        u = r["conversation"][0]["content"] or ""
        if 600 <= len(u) <= 6000:
            cands.append((h(r["conversation_hash"]), r["conversation_hash"], u))
    cands.sort()
    for _, ch, u in cands:
        if len(rows) >= n:
            break
        k = tok.n(u)
        if 150 <= k <= 1500:
            rows.append(row(_id="wc1:" + ch, workload="twins_real", task="chat", domain="chat", bucket="nat",
                            construction="natural_length", source="allenai/WildChat-1M", target_tokens=0,
                            prompt_tokens=k, context=u))
    return {("twins_real", "nat"): rows}


# ---------------------------------------------------------------- padding test (8K)

def eq_bases(S, built):
    """The items behind the 2K mdqa and fewshot prompts, and the part of each 2K
    prompt that is the item's own context, so 8K padding starts from exactly the
    text the 2K cell measured."""
    base_md = {r["_id"] for r in built[("mdqa", "2K")][:N_C1["2K"]]}
    base_fs = {r["_id"] for r in built[("fewshot", "2K")][:N_C1["2K"]]}
    order_md = [r["_id"] for r in built[("mdqa", "2K")][:N_C1["2K"]]]
    order_fs = [r["_id"] for r in built[("fewshot", "2K")][:N_C1["2K"]]]
    by_md = {d["_id"]: d for d in S.v1("hotpotqa") if d["_id"] in base_md}
    by_fs = {d["_id"]: d for d in S.v1("samsum") if d["_id"] in base_fs}
    items_md = [by_md[i] for i in order_md if i in by_md]
    items_fs = [by_fs[i] for i in order_fs if i in by_fs]
    two = {r["_id"]: r["context"] for r in built[("mdqa", "2K")]}
    two_fs = {r["_id"]: r["context"] for r in built[("fewshot", "2K")]}

    def own_md(d):  # the passages the 2K prompt kept
        p = two[d["_id"]]
        a = p.index("The following are given passages.\n") + len("The following are given passages.\n")
        return p[a:p.index("\n\nAnswer the question based on the given passages.", a)]

    def own_fs(d):
        p = two_fs[d["_id"]]
        a = p.index("The following are some examples.\n\n") + len("The following are some examples.\n\n")
        return p[a:p.rindex("\n\n" + d["input"])]
    return items_md, items_fs, own_md, own_fs


def build_eqtest(S, tok, built):
    """mdqa and fewshot at 8K, built from the 2K task prompts' items: real padding,
    and the repeated-sentence negative control. Natural 8K is the tasks' own 8K file."""
    out = {}
    pool_h, pool_s = hotpot_pool(S), samsum_pool(S)
    items_md, items_fs, own_md, own_fs = eq_bases(S, built)
    md = distractors(S, tok, items_md, 8192, pool_h, 52, "eqmd", own_fn=own_md)
    fs = many_shot(S, tok, items_fs, 8192, pool_s, 52, "eqfs", start_ctx=own_fs)
    for rows, wl in ((md, "eq_mdqa_realpad"), (fs, "eq_fewshot_realpad")):
        for r in rows:
            r["workload"], r["bucket"] = wl, "8K"
        out[(wl, "8K")] = rows
    for t, src in (("mdqa", built[("mdqa", "2K")]), ("fewshot", built[("fewshot", "2K")])):
        rows = []
        for r0 in src[:N_C1["2K"]]:
            base = r0["context"]
            line = "\n" + REPEAT_SENTENCE
            per = tok.n(line * 200) / 200.0   # repeated, the line tokenizes shorter than alone
            k = max(1, int((8192 - tok.n(base)) / per))
            p = base + line * k
            while tok.n(p) < 8192 - 2:
                p += line
            rows.append(row(_id=f"eqrep{t}:{r0['_id']}", workload=f"eq_{t}_repeatpad", task=t, domain=t,
                            bucket="8K", construction="repeat_pad", source=r0["source"], target_tokens=8192,
                            prompt_tokens=tok.n(p), base_id=r0["_id"], context=p))
        out[(f"eq_{t}_repeatpad", "8K")] = rows
    return out


# ---------------------------------------------------------------- mixed lengths

def build_mix(built):
    a, b = built[("qa", "2K")], built[("qa", "32K")]
    rows = []
    # The same LongBench v2 document can be in both buckets, so ids get the
    # bucket in front: per-request records are joined on _id.
    tag = lambda r, k: dict(r, _id=f"{k}:{r['_id']}", base_id=r["_id"], workload="mix", bucket="mix", length=k)
    for x, y in zip(a[:32], b[:32]):
        rows += [tag(x, "2K"), tag(y, "32K")]
    rows += [tag(a[32], "2K"), tag(b[32], "32K")]
    return {("mix", "mix"): rows}


# ---------------------------------------------------------------- AI requests and assembly

def ai_requests(tok, built, S, out_root):
    """Write what gen_router_ai.py should ask DeepSeek for. Plain JSON, no key."""
    d = os.path.join(out_root, AI_DIR, "requests")
    os.makedirs(d, exist_ok=True)
    reqs = {}
    # synth: diverse instructions over real documents not used by qa or ragquote
    for b, n_docs in (("2K", 30), ("8K", 30), ("16K", 30), ("32K", 16)):
        used = qa_ids(built, b) | {r.get("base_id") for r in built.get(("ragquote", b), [])}
        L, rows = TARGETS[b], []
        doms = {}
        for x in S.v2():
            if x["_id"] in used:
                continue
            doms.setdefault(x.get("domain"), []).append(x)
        order = []
        for i in range(400):  # round-robin over domains, so every domain is in the mix
            for dm in sorted(doms):
                if i < len(doms[dm]):
                    order.append(doms[dm][i])
        for x in order:
            if len(rows) >= n_docs:
                break
            ids = tok.head(x["context"], L)
            if len(ids) < L - 400:
                continue
            ex = tok.dec(ids[:L - 400])
            rows.append({"rid": f"syn{b}:{x['_id']}", "kind": "synth", "bucket": b, "domain": x.get("domain"),
                         "base_id": x["_id"], "document": ex})
        reqs[f"synth_{b}"] = rows
    # code edits written by the model, over the same excerpts as codeedit
    for b in ("2K", "8K", "16K"):
        rows = []
        for r in built.get(("codeedit", b), []):
            ex = r["context"].split("```\n", 1)[1].rsplit("```", 1)[0]
            rows.append({"rid": r["_id"], "kind": "codeedit", "bucket": b, "base_id": r.get("base_id"),
                         "code": ex})
        reqs[f"codeedit_{b}"] = rows
    reqs["filler_mdqa"] = [{"rid": f"fmd{i}", "kind": "filler_mdqa", "n": 10} for i in range(45)]
    reqs["filler_fewshot"] = [{"rid": f"ffs{i}", "kind": "filler_fewshot", "n": 10} for i in range(45)]
    reqs["twins"] = [{"rid": r["_id"], "kind": "twin", "request": r["context"]}
                     for r in built.get(("twins_real", "nat"), [])]
    for k, rows in reqs.items():
        with open(os.path.join(d, k + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {k: len(v) for k, v in reqs.items()}


def load_responses(out_root, kind):
    p = os.path.join(out_root, AI_DIR, "responses", kind + ".jsonl")
    if not os.path.exists(p):
        return None
    res = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                res[r["rid"]] = r
    return res


SYNTH_LEVELS = [("verbatim", "very_high"), ("edit", "high"), ("quote", "high"), ("summarize", "mid"),
                ("brief", "none"), ("create", "low")]
CODE_LEVELS = [("whole_file", "very_high"), ("one_function", "high"), ("fix_or_feature", "mid"),
               ("question", "low")]


def assemble_ai(S, tok, built, out_root):
    out, notes = {}, []
    reqdir = os.path.join(out_root, AI_DIR, "requests")

    def reqs(kind):
        p = os.path.join(reqdir, kind + ".jsonl")
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    for b in ("2K", "8K", "16K", "32K"):
        res = load_responses(out_root, f"synth_{b}")
        if not res:
            notes.append(f"synth {b}: no responses yet"); continue
        rows, k = [], 0
        for q in reqs(f"synth_{b}"):
            r = res.get(q["rid"])
            if not r:
                continue
            for j in range(2):  # two levels per document, rotating so each level is used equally
                lvl, cl = SYNTH_LEVELS[(k + 3 * j) % len(SYNTH_LEVELS)]
                instr = (r["data"].get(lvl) or "").strip()
                if not instr:
                    continue
                p = f"{q['document']}\n\n{instr}"
                rows.append(row(_id=f"{q['rid']}:{lvl}", workload="synth", task="synth", domain=q.get("domain"),
                                bucket=b, construction="llm_instruction", source="THUDM/LongBench-v2 + " + r.get("model", "deepseek"),
                                target_tokens=TARGETS[b], prompt_tokens=tok.n(p), base_id=q["base_id"],
                                variant=lvl, copy_level=cl, context=p))
            k += 1
        out[("synth", b)] = rows
    for b in ("2K", "8K", "16K"):
        res = load_responses(out_root, f"codeedit_{b}")
        if not res:
            notes.append(f"codeedit_ai {b}: no responses yet"); continue
        rows = []
        for i, q in enumerate(reqs(f"codeedit_{b}")):
            r = res.get(q["rid"])
            if not r:
                continue
            lvl, cl = CODE_LEVELS[i % len(CODE_LEVELS)]
            instr = (r["data"].get(lvl) or "").strip()
            if not instr:
                continue
            p = f"Here is some code:\n\n```\n{q['code']}```\n\n{instr}"
            rows.append(row(_id=f"{q['rid']}:ai:{lvl}", workload="codeedit_ai", task="codeedit", domain="codeedit",
                            bucket=b, construction="llm_instruction", source="THUDM/LongBench + " + r.get("model", "deepseek"),
                            target_tokens=TARGETS[b], prompt_tokens=tok.n(p), base_id=q.get("base_id"),
                            variant=lvl, copy_level=cl, context=p))
        out[("codeedit_ai", b)] = rows
    fm, ff = load_responses(out_root, "filler_mdqa"), load_responses(out_root, "filler_fewshot")
    if fm and ff:
        pool_p = [f"{p['title'].strip()}\n{p['text'].strip()}" for r in fm.values()
                  for p in r["data"].get("passages", []) if p.get("title") and p.get("text")]
        pool_e = [f"Dialogue: {e['dialogue'].strip()}\nSummary: {e['summary'].strip()}" for r in ff.values()
                  for e in r["data"].get("examples", []) if e.get("dialogue") and e.get("summary")]
        items_md, items_fs, own_md, own_fs = eq_bases(S, built)
        md = distractors(S, tok, items_md, 8192, pool_p, 52, "eqllmmd", own_fn=own_md)
        fs = many_shot(S, tok, items_fs, 8192, pool_e, 52, "eqllmfs", start_ctx=own_fs)
        for rows, wl in ((md, "eq_mdqa_llmpad"), (fs, "eq_fewshot_llmpad")):
            for r in rows:
                r["workload"], r["bucket"], r["construction"] = wl, "8K", "llm_pad"
            out[(wl, "8K")] = rows
    else:
        notes.append("eq_*_llmpad: no filler responses yet")
    tw = load_responses(out_root, "twins")
    if tw:
        rows = []
        for q in reqs("twins"):
            r = tw.get(q["rid"])
            t = (r or {}).get("data", {}).get("request", "").strip() if r else ""
            if t:
                rows.append(row(_id=q["rid"] + ":twin", workload="twins_ai", task="chat", domain="chat", bucket="nat",
                                construction="llm_twin", source="deepseek twin of " + q["rid"], target_tokens=0,
                                prompt_tokens=tok.n(t), base_id=q["rid"], context=t))
        out[("twins_ai", "nat")] = rows
    else:
        notes.append("twins_ai: no responses yet")
    return out, notes


# ---------------------------------------------------------------- main

GROUPS = {"qa": ["qa"], "tasks": ["code", "summ", "fewshot", "mdqa"], "ragquote": ["ragquote"],
          "codeedit": ["codeedit"], "chat": ["chat"], "agent": ["agent"], "eqtest": ["eq"],
          "twins": ["twins_real"], "mix": ["mix"], "ai": ["synth", "codeedit_ai", "eq_llm", "twins_ai"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="default $LADDER_ROOT/data/router")
    ap.add_argument("--tokenizer", help="default the pinned Qwen3-8B snapshot under $LADDER_HF")
    ap.add_argument("--zip", help="a local LongBench data.zip")
    ap.add_argument("--v2", help="a local LongBench-v2 data.json")
    ap.add_argument("--cache", help="Hugging Face dataset cache (default $LADDER_ROOT/hf/datasets)")
    ap.add_argument("--only", default="qa tasks ragquote codeedit chat agent eqtest twins mix ai",
                    help="groups: " + " ".join(GROUPS))
    ap.add_argument("--wildchat-shards", type=int, default=2)
    ap.add_argument("--swe-shards", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="count and print; write no workload files")
    ap.add_argument("--force", action="store_true", help="overwrite existing workload files")
    args = ap.parse_args()

    root, hf = os.environ.get("LADDER_ROOT"), os.environ.get("LADDER_HF")
    out_root = args.out or (root and os.path.join(root, "data", "router"))
    if not out_root:
        sys.exit("set LADDER_ROOT (source deltaai_env.sh) or pass --out")
    if out_root.startswith("/projects/"):
        sys.exit("the output is under /projects, which ghx4 jobs may not read")
    args.cache = args.cache or os.path.join(root or out_root, "hf", "datasets")
    tok = Tok(args.tokenizer or os.path.join(hf or "", SNAPSHOT))
    S = Sources(args, tok)
    groups = args.only.split()
    for g in groups:
        if g not in GROUPS:
            sys.exit(f"unknown group {g}; groups: {' '.join(GROUPS)}")

    built = {}
    # QA and the 2K tasks are needed by several groups, so they are always built in memory.
    print("building qa ...", flush=True)
    built.update(build_qa(S, tok, ["2K", "8K", "16K", "32K"]))
    print("building tasks ...", flush=True)
    built.update(build_tasks(S, tok, ["2K", "8K", "16K", "32K"] if "tasks" in groups else ["2K"], built))
    if "ragquote" in groups or "ai" in groups:
        print("building ragquote ...", flush=True)
        built.update(build_ragquote(S, tok, ["2K", "8K", "16K", "32K"], built))
    if "codeedit" in groups or "ai" in groups:
        print("building codeedit ...", flush=True)
        built.update(build_codeedit(S, tok, ["nat", "2K", "8K", "16K"]))
    if "chat" in groups:
        print("building chat (WildChat) ...", flush=True)
        built.update(build_chat(S, tok, ["2K", "8K"], args.wildchat_shards))
    if "twins" in groups or "ai" in groups:
        print("building twins (WildChat single turn) ...", flush=True)
        built.update(build_twins_real(S, tok, args.wildchat_shards))
    if "agent" in groups:
        print("building agent (SWE-agent) ...", flush=True)
        built.update(build_agent(S, tok, ["8K", "16K", "32K"], args.swe_shards))
    if "eqtest" in groups:
        print("building eqtest ...", flush=True)
        built.update(build_eqtest(S, tok, built))
    if "mix" in groups:
        built.update(build_mix(built))
    notes = []
    if "ai" in groups:
        n = ai_requests(tok, built, S, out_root) if not args.check else {}
        print("AI requests written:", n)
        ai, notes = assemble_ai(S, tok, built, out_root)
        built.update(ai)

    keep = set()
    for g in groups:
        for w in GROUPS[g]:
            keep.add(w)
    def wanted(wl):
        if wl in keep:
            return True
        if wl.startswith("eq_") and wl.endswith(("realpad", "repeatpad")) and "eq" in keep:
            return True
        if wl.startswith("eq_") and wl.endswith("llmpad") and "eq_llm" in keep:
            return True
        return False

    man_path = os.path.join(out_root, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"files": {}}
    manifest["tokenizer"] = SNAPSHOT
    print(f"\n{'workload':22s} {'bucket':6s} {'rows':>5s} {'measured':>8s}  tokens (content)   constructions")
    for (wl, b), rows in sorted(built.items()):
        if not wanted(wl):
            continue
        lens = [r["prompt_tokens"] for r in rows]
        cons = sorted({r["construction"] for r in rows})
        print(f"{wl:22s} {b:6s} {len(rows):5d} {max(0, len(rows) - 2):8d}  "
              + (f"{min(lens):6d} to {max(lens):6d}" if lens else "     none        ") + "   " + ",".join(cons))
        if args.check or len(rows) < 7:
            continue
        d = os.path.join(out_root, wl)
        p = os.path.join(d, f"{b}.jsonl")
        if os.path.exists(p) and not args.force:
            print(f"   exists, kept (pass --force to rebuild): {p}")
            continue
        os.makedirs(d, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["files"][f"{wl}/{b}"] = {
            "rows": len(rows), "measured": len(rows) - 2, "constructions": cons,
            "prompt_tokens_min": min(lens), "prompt_tokens_max": max(lens),
            "sources": sorted({r["source"] for r in rows}),
            "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    for n in notes:
        print("note:", n)
    if not args.check:
        os.makedirs(out_root, exist_ok=True)
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"\nmanifest: {man_path}")


if __name__ == "__main__":
    main()
