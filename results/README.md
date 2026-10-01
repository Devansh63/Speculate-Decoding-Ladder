# Results

One dated markdown record per run, and the raw `results.jsonl` under `raw/`.

The jsonl is the source of truth. Every figure must be generated from it by a
script; a figure that cannot be regenerated does not go in the report. The
markdown records are for humans: what the run was, what it shows, and, just as
importantly, what it does not show.

Pulling results back from Delta:

```bash
scp dagraw2@login.delta.ncsa.illinois.edu:/projects/bikd/dagraw2/Speculative_Ladder/results/a40/results.jsonl \
    results/raw/
```

Each record states platform, engine commit, model revisions, workload, decoding
settings and concurrency, because a number without that context cannot be
compared against anything later.
