#!/bin/bash
# Feed this user's waiting dtai3 pieces through ghx4-interactive one at a time.
# A copy of dtai2_feeder.sh for the dtai3 job names (25 Nov 2026); dtai2_feeder.sh is unchanged.
# ONLY_BUNDLES here takes <workload>_<bucket> names, e.g. ONLY_BUNDLES="qa_2K qa_8K".
# Written 18 Nov 2026: after 12 hours on ghx4, 2 of 43 pieces had started
# (ours ranked about 1760th of 1900), while ghx4-interactive gives a partition
# factor of 10000 against ghx4's 100 but allows one job per user
# (qos_ghx4int: MaxJobsPU=1, MaxSubmitPU=1), at a charge factor of 2.0.
#
# How it works. Every INTERVAL seconds: if this user has no job in
# ghx4-interactive (pending, running or completing), it moves the waiting
# piece with the lowest job id from ghx4 to ghx4-interactive with
#   scontrol update JobId=<id> Partition=ghx4-interactive
# The job keeps its id, its script and its results directory, so nothing is
# resubmitted and nothing can run twice. Pieces that ghx4 starts on its own
# simply stop being candidates. Only jobs named dtai3-<workload>_<bucket>-p<piece> are
# moved, never the whole-bundle jobs, and only with a time limit of 2 h or
# less (the partition's MaxTime).
#
# Run it on a login node inside tmux, so it outlives the ssh session:
#   tmux new -s feeder "bash dtai3_feeder.sh"        (detach: Ctrl-b then d)
#   tmux attach -t feeder                             (look at it again)
# Stop it: touch ~/STOP_FEEDER (it exits at its next check), or Ctrl-c inside tmux.
#
#   INTERVAL=60           seconds between checks
#   ONLY_BUNDLES="c1-2K c1-8K"   move pieces of these bundles only
#   MAX_MOVES=0           stop after this many moves (0: no limit)
#   DRY=1                 print what it would move, move nothing, then exit
# Log: $LADDER_ROOT/logs/feeder-<start time>.log, or ~/feeder-<start time>.log.

set -uo pipefail
INTERVAL=${INTERVAL:-60}
ONLY_BUNDLES=${ONLY_BUNDLES:-}
MAX_MOVES=${MAX_MOVES:-0}
DRY=${DRY:-0}
DEST=ghx4-interactive
STOP=${STOP_FILE:-$HOME/STOP_FEEDER}
ME=${USER:-$(id -un)}
LOGDIR=${LADDER_ROOT:+$LADDER_ROOT/logs}; LOGDIR=${LOGDIR:-$HOME}
mkdir -p "$LOGDIR"
LOG="$LOGDIR/feeder-$(date +%Y%m%d-%H%M%S).log"
rm -f "$STOP"

log () { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

to_min () {  # Slurm time limit ([D-]HH:MM:SS or MM:SS) -> minutes; 99999 if unreadable
  awk -v t="$1" 'BEGIN { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
    n = split(t, p, ":"); if (n == 3) m = p[1]*60 + p[2]; else if (n == 2) m = p[1]; else { print 99999; exit }
    print d*1440 + m }'
}

candidates () {  # waiting pieces in ghx4, lowest id first: "<id> <name> <limit>"
  squeue -h -u "$ME" -p ghx4 -t PD -o "%i %j %l" 2>/dev/null | sort -n | while read -r id name lim; do
    [[ "$name" =~ ^dtai3-(.+)-p[A-Za-z0-9]+$ ]] || continue
    if [ -n "$ONLY_BUNDLES" ]; then
      case " $ONLY_BUNDLES " in *" ${BASH_REMATCH[1]} "*) ;; *) continue ;; esac
    fi
    [ "$(to_min "$lim")" -le 120 ] || continue
    echo "$id $name $lim"
  done
}

busy () { squeue -h -u "$ME" -p "$DEST" -t PD,R,CG,CF -o "%i %j %T" 2>/dev/null; }

if [ "$DRY" = "1" ]; then
  log "DRY: in $DEST now: $(busy | tr '\n' ';')"
  log "DRY: candidates, in the order they would move:"
  candidates | while read -r l; do log "  $l"; done
  exit 0
fi

log "feeder started: user $ME, every ${INTERVAL}s, bundles '${ONLY_BUNDLES:-all}', max moves ${MAX_MOVES}, log $LOG"
moves=0; fails=0; last=""
while :; do
  [ -e "$STOP" ] && { log "stop file $STOP found; exiting after $moves moves"; rm -f "$STOP"; exit 0; }
  b=$(busy)
  if [ -n "$b" ]; then
    [ "$b" != "$last" ] && log "in $DEST: $(echo "$b" | tr '\n' ';')"
    last=$b
  else
    last=""
    next=$(candidates | head -1)
    if [ -z "$next" ]; then
      log "no waiting pieces left in ghx4; exiting after $moves moves"
      exit 0
    fi
    read -r id name lim <<< "$next"
    if out=$(scontrol update JobId="$id" Partition="$DEST" 2>&1); then
      moves=$((moves + 1)); fails=0
      log "moved $id $name (limit $lim) to $DEST  [move $moves]"
      if [ "$MAX_MOVES" -gt 0 ] && [ "$moves" -ge "$MAX_MOVES" ]; then
        log "reached MAX_MOVES=$MAX_MOVES; exiting (the moved job still runs)"
        exit 0
      fi
    else
      fails=$((fails + 1))
      log "could not move $id $name: $out  [failure $fails]"
      if [ "$fails" -ge 5 ]; then log "5 failures in a row; exiting"; exit 1; fi
    fi
  fi
  sleep "$INTERVAL"
done
