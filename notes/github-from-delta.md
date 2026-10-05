# Pushing results to GitHub from Delta

One-time setup so every job's output lands in the repo without going through
a chat window. Run all of this on a **login node**; it needs no GPU.

## 1. Make a token

On github.com: Settings, Developer settings, Personal access tokens,
**Fine-grained tokens**, Generate new token.

- Repository access: Only select repositories, `Devansh63/Speculate-Decoding-Ladder`
- Permissions: Repository permissions, **Contents: Read and write**
- Expiration: through the end of the semester

Nothing else. A fine-grained token scoped to this one repository cannot touch
anything else in the account, which is the point.

## 2. Clone and configure

```bash
ROOT=/projects/bikd/$USER/Speculative_Ladder
git clone https://github.com/Devansh63/Speculate-Decoding-Ladder.git $ROOT/repo
cd $ROOT/repo
git config user.name  "Devansh Agrawal"
git config user.email "devanshagrawal63@gmail.com"
git config credential.helper "store --file $HOME/.git-credentials-ladder"
```

The author name and email must match the GitHub account or the commits will
not be attributed on the contribution graph.

## 3. First push stores the token

```bash
cd $ROOT/repo
git commit --allow-empty -m "Verify push from Delta login node"
git push
```

It asks for a username (`Devansh63`) and a password (**paste the token**, not
the account password). After that it is stored and never asked again.

Lock the file down immediately:

```bash
chmod 600 $HOME/.git-credentials-ladder
```

It sits in `/u/$USER`, which is per-user and not in the shared project space,
so a teammate with `/projects/bikd` access cannot read it. Each of you sets up
your own token so commits are attributed correctly.

## 4. After every job

```bash
bash $ROOT/ladder/scripts/push_results.sh "A100 2K sweep, job <id>"
```

That copies `results.jsonl`, the per-request hash files and the compressed
Slurm logs into the clone, regenerates `summary.txt` from the raw rows, commits
and pushes.

## Why the Slurm logs are committed

They are the provenance for every row: the node, the GPU and its compute
capability, vLLM's reported KV cache size and achieved concurrency, and the
exact timings of each arm. Without them a number in `results.jsonl` cannot be
defended six weeks later when the paper is being written. Compressed they are
small.

## If the token stops working

Fine-grained tokens expire. A push that suddenly fails with a 403 usually
means expiry, not a permissions problem. Regenerate it, then:

```bash
rm $HOME/.git-credentials-ladder
cd $ROOT/repo && git push   # asks again, stores the new one
```
