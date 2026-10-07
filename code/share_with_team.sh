#!/bin/bash
# Make the project runnable by anyone in the project group.
#
# Run ONCE, by the project tree's owner, on a login node. Safe to re-run;
# everything is idempotent, and re-running repairs a previously bad copy.
#
#   bash share_with_team.sh
#
# What it does:
#
#   1. Opens group read on the model snapshots, the workload files and the
#      scripts. Without this a teammate's job dies at model load.
#   2. Opens group write on results, logs and the compile cache, and sets the
#      setgid bit on those directories so files a teammate creates stay
#      group-writable instead of reverting to private.
#   3. Copies the virtualenv into the shared project space and rewrites the
#      absolute paths inside it, so a teammate does not have to spend an hour
#      building vLLM.
#   4. Forces the project group on everything.
#
# Step 4 is not optional, and leaving it out broke this once already. cp -a
# preserves the SOURCE's group, which for a venv copied out of a home
# directory is the creator's personal group. That explicit group-set overrides
# the setgid inheritance on the destination's parent, so the copy lands owned
# by a group no teammate belongs to. A teammate then cannot traverse the
# directory at all, phase0.sh's [ -x "$VENV/bin/python3" ] test fails, and the
# job aborts within seconds. The chmod before it makes things no better: it
# grants read to the wrong group.
#
# The verification at the end reports group and mode, not just whether the
# import worked. An import test run by the owner proves nothing about anyone
# else, because the owner reads their own files regardless of group.

set -uo pipefail

SHARED=${LADDER_SHARED:-/projects/bikd/dagraw2/Speculative_Ladder}
HF=${LADDER_HF:-/work/nvme/bikd/dagraw2/Speculative_Ladder/hf}
SRC_VENV=${SRC_VENV:-$HOME/vllm_env}
DST_VENV=$SHARED/vllm_env
GROUP=${LADDER_GROUP:-delta_bikd}

echo "shared root  : $SHARED"
echo "model cache  : $HF"
echo "venv source  : $SRC_VENV"
echo "venv target  : $DST_VENV"
echo "project group: $GROUP"
echo

if ! id -nG | tr ' ' '\n' | grep -qx "$GROUP"; then
  echo "ABORT: you are not a member of group '$GROUP'."
  echo "       Your groups: $(id -nG)"
  echo "       Set LADDER_GROUP to the right one and re-run."
  exit 1
fi

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
  find "$p" -type d -exec chmod g+s {} + 2>/dev/null
  echo "  ok: $p"
done
chmod g+rwxs "$SHARED" 2>/dev/null || true

# ------------------------------------------------------------------ 3. the venv
echo
if [ -x "$DST_VENV/bin/python3" ]; then
  echo "== shared venv already present at $DST_VENV, skipping the copy =="
  echo "   (group and permissions are still repaired below)"
else
  if [ ! -x "$SRC_VENV/bin/python3" ]; then
    echo "== no source venv at $SRC_VENV, skipping =="
  else
    echo "== copying venv, this moves several GB and takes a few minutes =="
    cp -a "$SRC_VENV" "$DST_VENV"

    echo "== rewriting absolute paths inside the copy =="
    for f in "$DST_VENV"/bin/activate "$DST_VENV"/bin/activate.*; do
      [ -f "$f" ] || continue
      sed -i "s|$SRC_VENV|$DST_VENV|g" "$f"
    done
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
  fi
fi

# ------------------------------------------------- 4. the group, every time
# Runs whether or not the copy happened, so re-running repairs a bad copy.
if [ -d "$DST_VENV" ]; then
  echo
  echo "== forcing group '$GROUP' on the shared venv =="
  chgrp -R "$GROUP" "$DST_VENV" && echo "  chgrp ok"
  chmod -R g+rX "$DST_VENV"
  find "$DST_VENV" -type d -exec chmod g+s {} + 2>/dev/null
  echo "  ok: $DST_VENV"
fi

echo
echo "== sweeping the whole tree for files not owned by '$GROUP' =="
bad=$(find "$SHARED" "$HF" ! -group "$GROUP" -printf '%g %M %p\n' 2>/dev/null | head -20)
if [ -z "$bad" ]; then
  echo "  clean, everything is group $GROUP"
else
  echo "$bad"
  echo "  ^ these are unreadable by teammates. Fixing:"
  chgrp -R "$GROUP" "$SHARED" "$HF" 2>/dev/null
  chmod -R g+rX "$SHARED" "$HF" 2>/dev/null
  echo "  re-run this script to confirm the sweep comes back clean"
fi

# ------------------------------------------------------------------- 5. verify
echo
echo "== verification =="
echo "group and mode, which is what actually decides whether a teammate can read:"
ls -ld "$SHARED" "$SHARED/ladder" "$SHARED/results" "$SHARED/logs" \
       "$SHARED/.vllm_cache" "$DST_VENV" "$DST_VENV/bin" 2>/dev/null
echo
if [ -x "$DST_VENV/bin/python3" ]; then
  module load cray-python/3.12.12 >/dev/null 2>&1 || true
  if "$DST_VENV/bin/python3" -c "import vllm, torch; print('  vllm', vllm.__version__); print('  torch', torch.__version__)"; then
    echo "  shared venv imports cleanly (as the owner)"
  else
    echo "  WARNING: the shared venv does not import. Do not hand it to anyone."
  fi
fi

echo
echo "NOTE: the import test above ran as the owner, who can read these files"
echo "regardless of group, so it does not prove a teammate can use them. The"
echo "group and mode listing is the real check. Ask your teammate to run:"
echo
echo "  ls -ld $DST_VENV && $DST_VENV/bin/python3 -c 'import vllm; print(vllm.__version__)'"
echo
echo "Only a success from their account confirms the setup."
