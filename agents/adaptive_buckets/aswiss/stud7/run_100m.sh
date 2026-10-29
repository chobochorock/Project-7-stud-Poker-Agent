#!/bin/bash
# Long 7-stud ASWISS comparison (default 100M roots), resumable. Linux / WSL counterpart of run_100m.ps1.
# Run from the repository root:
#   bash agents/adaptive_buckets/aswiss/stud7/run_100m.sh [run|status|stop|eval|final|plot|pack]
# Environment overrides: ROOTS=100000000 SEED=1 ARMS="aswiss k256 k341 k512 k1024" OUT=... WORKERS=4 PYTHON=python3
#   H4=learn (learn the 12-action 4th-street discard/reveal; run names get _h4), SNAPSHOT_EVERY=5000000,
#   CHECKPOINT_EVERY=1000000, CURVE_EVERY=100000
# "run" again after a stop, crash or reboot resumes every unfinished arm from its checkpoint (bit-exact).
set -euo pipefail
ACTION=${1:-run}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOTS=${ROOTS:-100000000}; SEED=${SEED:-1}; ARMS=${ARMS:-"aswiss k256 k341 k512 k1024"}
OUT=${OUT:-agents/adaptive_buckets/aswiss/stud7/data/runs_100m}; WORKERS=${WORKERS:-4}; PYTHON=${PYTHON:-python3}
H4=${H4:-heuristic}; SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-5000000}; CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000000}
CURVE_EVERY=${CURVE_EVERY:-100000}; SUFFIX=_s$SEED; [ "$H4" = learn ] && SUFFIX=_h4_s$SEED
EXE=$HERE/aswiss_stud
declare -A NAME=([aswiss]=aswiss_k1024_z2 [aswiss_b]=aswiss_b_k1024_z2 [aswiss_wb]=aswiss_wb_k1024_z2 [k256]=atom_k256
                 [hist]=history_street [hist_b]=history_b_street [hist_wb]=history_wb_street [hist_every]=history_every
                 [k341]=atom_k341 [k512]=atom_k512 [k1024]=atom_k1024)
declare -A PART=([aswiss]=aswiss [aswiss_b]=aswiss [aswiss_wb]=aswiss [k256]=atom [k341]=atom [k512]=atom [k1024]=atom
                 [hist]=history [hist_b]=history [hist_wb]=history [hist_every]=history)
declare -A K=([aswiss]=1024 [aswiss_b]=1024 [aswiss_wb]=1024 [k256]=256 [k341]=341 [k512]=512 [k1024]=1024 [hist]=1024
              [hist_b]=1024 [hist_wb]=1024 [hist_every]=1024)
# history arms: label samples capped at 2.56M roots per epoch (history keys do not saturate; bounds the memory)
declare -A EXTRA=([aswiss_b]="--label avg" [aswiss_wb]="--label window+avg" [hist]="--history-cards street --label-cap 2560000"
                  [hist_b]="--history-cards street --label avg --label-cap 2560000"
                  [hist_wb]="--history-cards street --label window+avg --label-cap 2560000"
                  [hist_every]="--history-cards every --label-cap 2560000")
declare -A LABEL=([aswiss]="ASWISS (context start, K=1024 atoms, z=2)" [aswiss_b]="ASWISS, method B labels (average policy, sampled)"
                  [aswiss_wb]="ASWISS, window + method B labels" [k256]="k-means 256"
                  [k341]="k-means 341 (matched to ASWISS at 10M)" [k512]="k-means 512" [k1024]="k-means 1024 = atoms trained directly"
                  [hist]="bucket-history keys, window labels"
                  [hist_b]="bucket-history keys, method B labels (average policy)"
                  [hist_wb]="bucket-history keys, window + method B labels"
                  [hist_every]="bucket-history keys, cards at every decision")
atlas() { echo "$HERE/atlas/atlas_power$1_heur_eps20_seed37301.bin"; }
mkdir -p "$OUT"
case $ACTION in
run)
    # rebuild when the executable is missing or older than its sources (for example after git pull)
    if [ ! -x "$EXE" ] || [ "$HERE/aswiss_stud.cpp" -nt "$EXE" ] || [ "$HERE/h4_lbr_generated.hpp" -nt "$EXE" ] ||
       [ "$HERE/../../../cpp_mccfr/stud_mccfr.cpp" -nt "$EXE" ]; then
        g++ -O3 -std=c++17 "$HERE/aswiss_stud.cpp" -o "$EXE" ||
            { echo "build failed; if arms are still running, stop them first (stop, then status)"; exit 1; }
    fi
    for a in $ARMS; do
        name=${NAME[$a]}$SUFFIX
        if [ -f "$OUT/$name.pid" ] && kill -0 "$(cat "$OUT/$name.pid")" 2>/dev/null; then echo "$name already running"; continue; fi
        nohup "$EXE" train --partition "${PART[$a]}" --atlas "$(atlas "${K[$a]}")" --z 2 --seed "$SEED" --roots "$ROOTS" \
            --epoch0 10000 --checkpoint-every "$CHECKPOINT_EVERY" --snapshot-every "$SNAPSHOT_EVERY" --curve-every "$CURVE_EVERY" \
            --h4 "$H4" ${EXTRA[$a]:-} --out "$OUT/$name" \
            >> "$OUT/$name.stderr.log" 2>&1 &
        echo $! > "$OUT/$name.pid"; echo "started $name (pid $!)"
    done ;;
status)
    for a in $ARMS; do
        name=${NAME[$a]}$SUFFIX; alive=no
        [ -f "$OUT/$name.pid" ] && kill -0 "$(cat "$OUT/$name.pid")" 2>/dev/null && alive=yes
        echo "$name running=$alive last curve row: $(tail -n 1 "$OUT/$name/curve.csv" 2>/dev/null || echo -)"
    done ;;
stop)
    for a in $ARMS; do touch "$OUT/${NAME[$a]}$SUFFIX/STOP"; done
    echo "STOP files written; each arm checkpoints within ~1000 roots and exits" ;;
eval|final)
    specs=(); for a in $ARMS; do specs+=("${NAME[$a]}$SUFFIX:${PART[$a]}:$(atlas "${K[$a]}")"); done
    extra=(--min-roots 100000); [ "$ACTION" = final ] && extra=(--latest --pairs 20000 --tag final)
    "$PYTHON" "$HERE/evaluate.py" "$EXE" "$OUT" --runs "${specs[@]}" --workers "$WORKERS" "${extra[@]}" ;;
plot)
    specs=(); for a in $ARMS; do specs+=("${NAME[$a]}$SUFFIX:${LABEL[$a]}"); done
    base=k256; for k in k256 k341 k512 k1024; do case " $ARMS " in *" $k "*) base=$k; break ;; esac; done  # first k-means arm run
    "$PYTHON" "$HERE/plot_stud.py" "$OUT" "$OUT/results$SUFFIX" --runs "${specs[@]}" --baseline "${NAME[$base]}$SUFFIX" --epoch-only ;;
pack)  # small result files of every run in OUT (no snapshots) -> results_data, for git or sharing
    "$PYTHON" "$HERE/pack_summaries.py" "$OUT" "$HERE/results_data/runs_100m${SUFFIX}_summaries.tar.gz" ;;
*) echo "usage: $0 run|status|stop|eval|final|plot|pack"; exit 2 ;;
esac
