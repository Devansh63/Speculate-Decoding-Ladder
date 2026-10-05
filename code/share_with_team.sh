#!/bin/bash
# Make the project runnable by anyone in the delta_bikd group.
#
# Run ONCE, by dagraw2, on a login node. Safe to re-run; everything is
# idempotent.
#
#   bash share_with_team.sh
#
# What it does, and why each part is needed:
#
#   1. Opens group read on the model snapshots, the workload files and the
#      scripts. Without this a teammate's job dies at model load.
#   2. Opens group write on results, logs and the compile cache, and sets the
#      setgid bit on those directories so files a teammate creates stay
#      group-writable instead of reverting to private.
#   3. Copies the virtualenv into the shared project space and rewrites the
#      absolute paths inside it, so a teammate does not have to spend an hour
#      building vLLM. Skipped if a shared copy already exists.
#
# The venv copy is the only fragile step. A Python venv records its own
# location in bin/activate and in the shebang of every console script, so a
# plain cp produces something that half works. Both are rewritten here, and
# the copy is verified by importing vllm at the end.

set -uo pipefail

SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
HF=${LADDER_HF:-/work/nvme/bikd/dagraw2/Speculative_Ladder/hf}
SRC_VENV=${SRC_VENV:-$HOME/vllm_env}
DST_VENV=$SHARED/vllm_env

echo "shared root : $SHARED"
echo "model cache : $HF"
echo "venv source : $SRC_VENV"
echo "venv target : $DST_VENV"
echo

# ---------------------------------------------------------------- 1. read access
echo "== granting group read on models, data and scripts =="
for p in "$HF" "$SHARED/ladder"; do
  [ -e "$p" ] || { echo "  skip, not found: $p"; continue; }
  chmod -R g+rX "$p" && echo "  ok: $p"
done

# ------------------------------------------------- 2. write access + inheritance
echo
echo "== granting group write on results, logs and cache =="
for p in "$SHARED/results" "$SHARED/logs" "$SHARED/.vllm_cache"; do
  mkdir -p "$p"
  chmod -R g+rwX "$p"
  # setgid on directories so new files inherit delta_bikd rather than the
  # creator's default group. Without this, Sahil's result files land in a
  # group dagraw2 cannot write, and the next sync fails.
  find "$p" -type d -exec chmod g+s {} + 2>/dev/null
  echo "  ok: $p"
done

# Also setgid the parents, so newly created subdirectories behave.
chmod g+rwxs "$SHARED" 2>/dev/null || true

# ------------------------------------------------------------------ 3. the venv
echo
if [ -x "$DST_VENV/bin/python3" ]; then
  echo "== shared venv already present at $DST_VENV, skipping copy =="
else
  if [ ! -x "$SRC_VENV/bin/python3" ]; then
    echo "== no source venv at $SRC_VENV, skipping =="
  else
    echo "== copying venv, this moves roughly 10 GB and takes a few minutes =="
    cp -a "$SRC_VENV" "$DST_VENV"

    echo "== rewriting absolute paths inside the copy =="
    # bin/activate and friends hardcode VIRTUAL_ENV.
    for f in "$DST_VENV"/bin/activate "$DST_VENV"/bin/activate.*; do
      [ -f "$f" ] || continue
      sed -i "s|$SRC_VENV|$DST_VENV|g" "$f"
    done
    # Console script shebangs point at the old interpreter path. Rewrite only
    # the first line, and only where it actually matches, so no binary or
    # unrelated file is touched.
    python3 - "$SRC_VENV" "$DST_VENV" <<'PYEOF'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
old = ("#!" + src).encode()
new = ("#!" + dst).encode()
changed = 0
bindir = os.path.join(dst, "bin")
for name in sorted(os.listdir(bindir)):
    p = os.path.join(bindir, name)
    if not os.path.isfile(p) or os.path.islink(p):
        continue
    try:
        with open(p, "rb") as fh:
            head = fh.read(len(old))
            if head != old:
                continue
            rest = fh.read()
    except OSError:
        continue
    with open(p, "wb") as fh:
        fh.write(new + rest)
    changed += 1
print(f"  rewrote {changed} shebangs")
PYEOF

    chmod -R g+rX "$DST_VENV"
    echo "  ok: $DST_VENV"
  fi
fi

# ------------------------------------------------------------------- 4. verify
echo
echo "== verifying the shared venv =="
if [ -x "$DST_VENV/bin/python3" ]; then
  module load cray-python/3.12.12 >/dev/null 2>&1 || true
  if "$DST_VENV/bin/python3" -c "import vllm, torch; print('  vllm', vllm.__version__); print('  torch', torch.__version__)"; then
    echo "  shared venv imports cleanly"
  else
    echo "  WARNING: the shared venv does not import. Do not hand it to anyone."
    echo "  Fall back to each person building their own, see"
    echo "  notes/teammate-quickstart.md."
  fi
fi

echo
echo "== group and permission summary =="
ls -ld "$SHARED" "$SHARED/results" "$SHARED/logs" "$SHARED/.vllm_cache" 2>/dev/null
echo
echo "Hand your teammate notes/teammate-quickstart.md. Nothing else is needed."
