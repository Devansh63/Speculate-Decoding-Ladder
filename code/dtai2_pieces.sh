#!/bin/bash
# The dtai2 c1 and conc bundles cut into pieces that each fit a 1 h walltime,
# submitted through submit_dtai2.sh one piece at a time. Written 16 Nov 2026
# evening: after 13 hours on ghx4 (2747 jobs waiting, ours 1057th of 1857 by
# priority) nothing asking for 2 h or more had started, while both 1 h jobs
# started within 20 minutes. See "pieces" in dtai2_bundle.sh.
#
#   bash dtai2_pieces.sh SMOKE=<dir> SMOKE_LOG=<log>                 plan only
#   bash dtai2_pieces.sh SMOKE=<dir> SMOKE_LOG=<log> GO=1            submit
#   PIECESET=task bash dtai2_pieces.sh ...                            the task-type pieces (below)
#   PIECES_FOR="c1-8K conc-2K" bash dtai2_pieces.sh ...              some bundles only
#   ONLY_PIECES="c1-32K:prek0 c1-2K:coreD2" bash dtai2_pieces.sh ...  these pieces only
#   SKIP_PIECES="c1-32K:prek0 c1-2K:coreD2" bash dtai2_pieces.sh ...  all but these
#   (a piece is <bundle>:<tag>; a name not in the list below is refused)
#
# Cancel the whole-bundle jobs these replace first, or the same cells run
# twice (into different directories, so nothing is overwritten, but the hours
# are spent twice).
#
# What a piece changes. Each piece runs its own off arm, so every speedup
# still divides by an off arm timed in the same job on the same GPU; the
# extra off arms cost about 1.5 GPU-hours over the whole list. The rep cell
# no longer shares a job with core, so rep against core becomes a repeat
# across jobs and nodes, not within one job. Results go to gh200v2_*_p<tag>.
#
# Sizes (GH200 estimates at the 2K smoke factor 0.199 and 3 min of engine
# start per arm): every piece below is under 40 minutes at that factor. The
# 32K factor is not measured yet; at 0.35 the largest 32K piece is about
# 50 minutes. submit_dtai2.sh warns when an estimate passes 75% of the hour.

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# bundle | cells | piece tag | ONLY_NGRAM_GPU | ONLY_DFLASH   ("-" = keep the cell's list)
PIECES='
c1-2K|core|coreN|-|none
c1-2K|core|coreD1|none|1 2 4
c1-2K|core|coreD2|none|7 11 15
c1-2K|loopA|loopN|-|none
c1-2K|loopA|loopD1|none|1 2 4
c1-2K|loopA|loopD2|none|7 11 15
c1-2K|acc|acc|-|-
c1-2K|rep|rep|-|-
c1-8K|pre k0|prek0|-|-
c1-8K|core|coreN|-|none
c1-8K|core|coreD1|none|1 2 4
c1-8K|core|coreD2|none|7 11 15
c1-8K|loopA|loopN|-|none
c1-8K|loopA|loopD1|none|1 2 4
c1-8K|loopA|loopD2|none|7 11 15
c1-8K|acc|acc|-|-
c1-8K|rep|rep|-|-
c1-32K|pre k0|prek0|-|-
c1-32K|core|coreN|-|none
c1-32K|core|coreD1|none|1 2 4
c1-32K|core|coreD2|none|7 11 15
c1-32K|loopA|loopN|-|none
c1-32K|loopA|loopD1|none|1 2 4
c1-32K|loopA|loopD2|none|7 11 15
c1-32K|acc|acc|-|-
c1-32K|nopc|nopc|-|-
c1-32K|rep|rep|-|-
conc-2K|pre1 preTop|pre|-|-
conc-2K|c1|c1N|-|none
conc-2K|c1|c1D|none|-
conc-2K|c4|c4|-|-
conc-2K|c8|c8|-|-
conc-2K|c16|c16|-|-
conc-2K|c32|c32|-|-
conc-8K|pre1 preTop|pre|-|-
conc-8K|c1|c1N|-|none
conc-8K|c1|c1D|none|-
conc-8K|c4|c4|-|-
conc-8K|c8|c8|-|-
conc-8K|c16|c16|-|-
conc-32K|pre1 preTop|pre|-|-
conc-32K|c1|c1N|-|none
conc-32K|c1|c1D|none|-
conc-32K|c2|c2|-|-
conc-32K|c4|c4|-|-
conc-32K|c8|c8|-|-
'

# Task-type pieces (added 18 Nov): one per task and length, 2K first so the
# feeder, which moves the lowest job id first, runs them in this order.
# Needs build_task_workloads.py's files; about 10 to 15 minutes each.
PIECES_TASK='
task-2K|code|code|-|-
task-2K|summ|summ|-|-
task-2K|fewshot|fewshot|-|-
task-2K|mdqa|mdqa|-|-
task-8K|code|code|-|-
task-8K|summ|summ|-|-
task-8K|fewshot|fewshot|-|-
task-8K|mdqa|mdqa|-|-
'
case "${PIECESET:-dtai2}" in
  dtai2) ;;
  task) PIECES=$PIECES_TASK ;;
  *) echo "PIECESET='${PIECESET}': dtai2 or task" >&2; exit 1 ;;
esac

KNOWN=" $(printf '%s\n' "$PIECES" | awk -F'|' 'NF { printf "%s:%s ", $1, $3 }')"
for want in ${ONLY_PIECES:-} ${SKIP_PIECES:-}; do
  case "$KNOWN" in *" $want "*) ;; *) echo "unknown piece '$want' (use <bundle>:<tag>, for example c1-2K:coreN)" >&2; exit 1 ;; esac
done
n=0; fails=0
while IFS='|' read -r bn cells tag ng df; do
  [ -z "$bn" ] && continue
  if [ -n "${PIECES_FOR:-}" ]; then
    case " $PIECES_FOR " in *" $bn "*) ;; *) continue ;; esac
  fi
  if [ -n "${ONLY_PIECES:-}" ]; then
    case " $ONLY_PIECES " in *" $bn:$tag "*) ;; *) continue ;; esac
  fi
  case " ${SKIP_PIECES:-} " in *" $bn:$tag "*) continue ;; esac
  args=(BUNDLES="$bn" CELLS="$cells" PIECE="$tag" WALL_H=1)
  [ "$ng" != "-" ] && args+=(ONLY_NGRAM_GPU="$ng")
  [ "$df" != "-" ] && args+=(ONLY_DFLASH="$df")
  n=$((n + 1))
  echo "---- piece $n: $bn cells='$cells' piece=$tag ngram_gpu=${ng/-/all} dflash=${df/-/all}"
  out=$(bash "$HERE/submit_dtai2.sh" "$@" "${args[@]}" 2>&1); rc=$?
  # The plan line, any warning, and the submitted line; the rest repeats every time.
  printf '%s\n' "$out" | grep -E "^$bn |^WARNING|^submitted|^REFUSED|^NOT SUBMITTED|SMOKE: (FAIL|did not)" || true
  if [ "$rc" -ne 0 ]; then
    fails=$((fails + 1)); printf '%s\n' "$out" | tail -5
    echo "STOPPED at piece $n: submit_dtai2.sh exited $rc. Nothing after it was submitted."
    exit 1
  fi
done <<< "$PIECES"
echo
echo "$n pieces$( case " $* " in *" GO=1 "*) echo " submitted" ;; *) echo " planned, nothing submitted (add GO=1)" ;; esac )."
echo "Check:  squeue -u \$USER -o \"%.10i %.24j %.2t %.10l %.12R\""
