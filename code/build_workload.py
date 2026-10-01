"""Build LongBench-v2 context buckets at exact token lengths.

Run on the Delta login node (free). Documents shorter than the target are
discarded, longer ones are truncated. Context is NEVER built by repeating
text: that would manufacture n-gram hits and fabricate the crossover the
project is testing for.

Yield on 30 Sep 2026: 503 samples at 2K, 503 at 8K, 387 at 32K.
"""

import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer

SNAPSHOT_DIR = ("/work/nvme/bikd/" + os.environ["USER"] +
                "/Speculative_Ladder/hf/hub/models--Qwen--Qwen3-8B/snapshots/"
                "b968826d9c46dd6066d109eabc6255188de91218")
OUTPUT_DIR = ("/projects/bikd/" + os.environ["USER"] +
              "/Speculative_Ladder/ladder/data/longbench_v2")
TARGET_LENGTHS = {"2K": 2048, "8K": 8192, "32K": 32768}

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_DIR, trust_remote_code=True)
ds = load_dataset("THUDM/LongBench-v2", split="train")

for label, length in TARGET_LENGTHS.items():
    out = os.path.join(OUTPUT_DIR, f"longbench_v2_{label}.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for item in ds:
            ids = tokenizer.encode(item.get("context", ""), add_special_tokens=False)
            if len(ids) < length:
                continue
            rec = {k: item.get(k, "") for k in
                   ("_id", "domain", "sub_domain", "question",
                    "choice_A", "choice_B", "choice_C", "choice_D", "answer")}
            rec["target_tokens"] = length
            rec["context"] = tokenizer.decode(ids[:length],
                                              clean_up_tokenization_spaces=False)
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[{label}] {n} samples -> {out}")
