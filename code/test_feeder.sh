#!/bin/bash
# Offline test of dtai2_feeder.sh with stand-in squeue, scontrol and sleep.
# Run: bash test_feeder.sh dtai2_feeder.sh
F=${1:?path to dtai2_feeder.sh}
W=$(mktemp -d); mkdir -p "$W/bin" "$W/home"
PASS=0; FAIL=0
check () { if eval "$2"; then PASS=$((PASS+1)); echo "  ok   $1"; else FAIL=$((FAIL+1)); echo "  FAIL $1"; [ -n "${3:-}" ] && echo "       $3"; fi; }
# state: id name partition state limit
cat > "$W/bin/squeue" <<'S'
#!/bin/bash
part=""; states=""; fmt=""
while [ $# -gt 0 ]; do case "$1" in -p) part=$2; shift;; -t) states=$2; shift;; -o) fmt=$2; shift;; esac; shift; done
while read -r id name p st lim; do
  [ -n "$part" ] && [ "$p" != "$part" ] && continue
  if [ -n "$states" ]; then case ",$states," in *",$st,"*) ;; *) continue;; esac; fi
  case "$fmt" in "%i %j %l") echo "$id $name $lim";; *) echo "$id $name $st";; esac
done < "$STATE"
S
cat > "$W/bin/scontrol" <<'S'
#!/bin/bash
[ -n "${SCONTROL_FAIL:-}" ] && { echo "scontrol_update error: Access/permission denied"; exit 1; }
id=${2#JobId=}; p=${3#Partition=}
grep -q "^$id .* PD " "$STATE" || { echo "Job is no longer pending execution for job $id"; exit 1; }
awk -v id="$id" -v p="$p" '$1==id {$3=p} {print}' "$STATE" > "$STATE.n" && mv "$STATE.n" "$STATE"
echo "$id" >> "$MOVED"
S
# sleep: whatever sits in ghx4-interactive finishes
cat > "$W/bin/sleep" <<'S'
#!/bin/bash
awk '$3 != "ghx4-interactive"' "$STATE" > "$STATE.n" && mv "$STATE.n" "$STATE"
echo tick >> "$TICKS"
S
chmod +x "$W/bin/"*
fresh () { cat > "$W/state" <<'X'
3612438 dtai2-c1-2K ghx4 PD 6:00:00
3618566 dtai2-c1-2K-ploopD2 ghx4 PD 45:00
3618564 dtai2-c1-2K-ploopN ghx4 PD 45:00
3618565 dtai2-c1-2K-ploopD1 ghx4 R 45:00
3618600 dtai2-conc-32K-ppre ghx4 PD 45:00
3618700 dtai2-c1-8K-plong ghx4 PD 3:00:00
3618701 someone-elses-job ghx4 PD 30:00
X
: > "$W/moved"; : > "$W/ticks"; }
run () { (export PATH="$W/bin:$PATH" HOME="$W/home" STATE="$W/state" MOVED="$W/moved" TICKS="$W/ticks" USER=t INTERVAL=1 LADDER_ROOT="$W/root"; timeout 20 env "$@" bash "$F" > "$W/out" 2>&1); echo $?; }

echo "T11 feeder"
fresh; rc=$(run)
check "moves the waiting pieces one at a time, lowest id first, then exits" "[ $rc -eq 0 ] && [ \"\$(tr '\n' ' ' < '$W/moved')\" = '3618564 3618566 3618600 ' ] && grep -q 'no waiting pieces left' '$W/out'" "rc=$rc moved: $(tr '\n' ' ' < "$W/moved")"
check "never moves a whole-bundle job, a running piece, a piece over 2 h, or a job not named like a piece" "! grep -qE '^(3612438|3618565|3618700|3618701)$' '$W/moved'"
fresh; echo "3618999 dtai2-c1-32K-pacc ghx4-interactive R 45:00" >> "$W/state"
cat > "$W/bin/sleep" <<'S'
#!/bin/bash
n=$(grep -c . "$TICKS"); echo tick >> "$TICKS"
[ "$n" -ge 2 ] && awk '$3 != "ghx4-interactive"' "$STATE" > "$STATE.n" && mv "$STATE.n" "$STATE"
true
S
chmod +x "$W/bin/sleep"; rc=$(run)
check "waits while a job is already in ghx4-interactive, then moves" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/ticks') -ge 3 ] && head -1 '$W/moved' | grep -q 3618564 && grep -q 'in ghx4-interactive: 3618999' '$W/out'"
cat > "$W/bin/sleep" <<'S'
#!/bin/bash
awk '$3 != "ghx4-interactive"' "$STATE" > "$STATE.n" && mv "$STATE.n" "$STATE"
echo tick >> "$TICKS"
S
chmod +x "$W/bin/sleep"
fresh; rc=$(run ONLY_BUNDLES=conc-32K)
check "ONLY_BUNDLES moves that bundle's pieces only" "[ $rc -eq 0 ] && [ \"\$(cat '$W/moved')\" = '3618600' ]"
fresh; rc=$(run MAX_MOVES=1)
check "MAX_MOVES=1 moves one piece and exits" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/moved') -eq 1 ] && grep -q 'reached MAX_MOVES=1' '$W/out'"
fresh; rc=$(run DRY=1)
check "DRY=1 lists the candidates in order and moves nothing" "[ $rc -eq 0 ] && [ ! -s '$W/moved' ] && grep -A3 'candidates' '$W/out' | grep -q 3618564 && ! grep -q 3612438 '$W/out'"
fresh; rc=$(run SCONTROL_FAIL=1)
check "gives up after 5 failed moves in a row, moving nothing" "[ $rc -eq 1 ] && [ ! -s '$W/moved' ] && grep -q '5 failures in a row' '$W/out'"
fresh; cat > "$W/bin/sleep" <<'S'
#!/bin/bash
awk '$3 != "ghx4-interactive"' "$STATE" > "$STATE.n" && mv "$STATE.n" "$STATE"
touch "$HOME/STOP_FEEDER"
S
chmod +x "$W/bin/sleep"; rc=$(run)
check "a stop file ends it at the next check" "[ $rc -eq 0 ] && [ \$(grep -c . '$W/moved') -eq 1 ] && grep -q 'stop file' '$W/out' && [ ! -e '$W/home/STOP_FEEDER' ]"
check "writes its log under LADDER_ROOT/logs" "ls '$W/root/logs/'feeder-*.log >/dev/null 2>&1"
echo "passed $PASS, failed $FAIL"; rm -rf "$W"; [ "$FAIL" -eq 0 ]
