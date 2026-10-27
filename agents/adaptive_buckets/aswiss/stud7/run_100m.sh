#!/bin/bash
# Long 7-stud ASWISS comparison (default 100M roots), resumable. Linux / WSL counterpart of run_100m.ps1.
# Run from the repository root:
#   bash agents/adaptive_buckets/aswiss/stud7/run_100m.sh [run|status|stop|eval|final|plot]
# Environment overrides: ROOTS=100000000 SEED=1 ARMS="aswiss k256 k341 k512 k1024" OUT=... WORKERS=4 PYTHON=python3
# "run" again after a stop, crash or reboot resumes every unfinished arm from its checkpoint (bit-exact).
set -euo pipefail
ACTION=${1:-run}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOTS=${ROOTS:-100000000}; SEED=${SEED:-1}; ARMS=${ARMS:-"aswiss k256 k341 k512 k1024"}
OUT=${OUT:-agents/adaptive_buckets/aswiss/stud7/data/runs_100m}; WORKERS=${WORKERS:-4}; PYTHON=${PYTHON:-python3}
EXE=$HERE/aswiss_stud
declare -A NAME=([aswiss]=aswiss_k1024_z2 [k256]=atom_k256 [k341]=atom_k341 [k512]=atom_k512 [k1024]=atom_k1024)
declare -A PART=([aswiss]=aswiss [k256]=atom [k341]=atom [k512]=atom [k1024]=atom)
declare -A K=([aswiss]=1024 [k256]=256 [k341]=341 [k512]=512 [k1024]=1024)
declare -A LABEL=([aswiss]="ASWISS (context start, K=1024 atoms, z=2)" [k256]="k-means 256"
                  [k341]="k-means 341 (matched to ASWISS at 10M)" [k512]="k-means 512" [k1024]="k-means 1024 = atoms trained directly")
atlas() { echo "$HERE/atlas/atlas_power$1_heur_eps20_seed37301.bin"; }
mkdir -p "$OUT"
case $ACTION in
run)
    [ -x "$EXE" ] || g++ -O3 -std=c++17 "$HERE/aswiss_stud.cpp" -o "$EXE"
    for a in $ARMS; do
        name=${NAME[$a]}_s$SEED
        if [ -f "$OUT/$name.pid" ] && kill -0 "$(cat "$OUT/$name.pid")" 2>/dev/null; then echo "$name already running"; continue; fi
        nohup "$EXE" train --partition "${PART[$a]}" --atlas "$(atlas "${K[$a]}")" --z 2 --seed "$SEED" --roots "$ROOTS" \
            --epoch0 10000 --checkpoint-every 1000000 --snapshot-every 5000000 --curve-every 100000 --out "$OUT/$name" \
            >> "$OUT/$name.stderr.log" 2>&1 &
        echo $! > "$OUT/$name.pid"; echo "started $name (pid $!)"
    done ;;
status)
    for a in $ARMS; do
        name=${NAME[$a]}_s$SEED; alive=no
        [ -f "$OUT/$name.pid" ] && kill -0 "$(cat "$OUT/$name.pid")" 2>/dev/null && alive=yes
        echo "$name running=$alive last curve row: $(tail -n 1 "$OUT/$name/curve.csv" 2>/dev/null || echo -)"
    done ;;
stop)
    for a in $ARMS; do touch "$OUT/${NAME[$a]}_s$SEED/STOP"; done
    echo "STOP files written; each arm checkpoints within ~1000 roots and exits" ;;
eval|final)
    specs=(); for a in $ARMS; do specs+=("${NAME[$a]}_s$SEED:${PART[$a]}:$(atlas "${K[$a]}")"); done
    extra=(--min-roots 100000); [ "$ACTION" = final ] && extra=(--latest --pairs 20000 --tag final)
    "$PYTHON" "$HERE/evaluate.py" "$EXE" "$OUT" --runs "${specs[@]}" --workers "$WORKERS" "${extra[@]}" ;;
plot)
    specs=(); for a in $ARMS; do specs+=("${NAME[$a]}_s$SEED:${LABEL[$a]}"); done
    "$PYTHON" "$HERE/plot_stud.py" "$OUT" "$OUT/results_100m_s$SEED" --runs "${specs[@]}" --baseline "atom_k256_s$SEED" ;;
*) echo "usage: $0 run|status|stop|eval|final|plot"; exit 2 ;;
esac
