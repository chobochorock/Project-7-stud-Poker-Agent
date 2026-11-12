#!/bin/bash
# Long 7-stud ASWISS comparison (default 100M roots), resumable. Linux / WSL counterpart of run_100m.ps1.
# Run from the repository root:
#   bash agents/adaptive_buckets/aswiss/stud7/run_100m.sh [run|status|stop|eval|final|plot|pack]
# Environment overrides: ROOTS=100000000 SEED=1 ARMS="aswiss k256 k341 k512 k1024" OUT=... WORKERS=4 PYTHON=python3
#   H4=learn (learn the 12-action 4th-street discard/reveal; run names get _h4), SNAPSHOT_EVERY=5000000,
#   CHECKPOINT_EVERY=1000000, CURVE_EVERY=100000, THREADS=1 (K > 1: the epoch-end average-policy label pass of the
#   --label avg arms runs on K threads; any K > 1 gives the same results, not those of 1, so resume with the same choice)
#   ONLINE_LBR_EVERY=0 (N > 0: online test-then-train Policy-LBR every N roots; ReBeL every ONLINE_LBR_REBEL_EVERY=10
#   hands), ONLINE_LBR_PARTICLES=16. Part of a run's settings: a run started without it cannot resume with it.
# "run" again after a stop, crash or reboot resumes every unfinished arm from its checkpoint (bit-exact).
set -euo pipefail
ACTION=${1:-run}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOTS=${ROOTS:-100000000}; SEED=${SEED:-1}; ARMS=${ARMS:-"aswiss k256 k341 k512 k1024"}
OUT=${OUT:-agents/adaptive_buckets/aswiss/stud7/data/runs_100m}; WORKERS=${WORKERS:-4}; PYTHON=${PYTHON:-python3}
H4=${H4:-heuristic}; SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-5000000}; CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000000}
CURVE_EVERY=${CURVE_EVERY:-100000}; THREADS=${THREADS:-1}; SUFFIX=_s$SEED; [ "$H4" = learn ] && SUFFIX=_h4_s$SEED
ONLINE_LBR_EVERY=${ONLINE_LBR_EVERY:-0}; ONLINE_LBR_REBEL_EVERY=${ONLINE_LBR_REBEL_EVERY:-10}; ONLINE_LBR_PARTICLES=${ONLINE_LBR_PARTICLES:-16}
# LOG_REGRET=1 (history/aswiss arms) / LOG_REGRET_KMEANS=1 (atom arms): regret.csv; LOG_SPLITS=1 (history arms): splits.csv;
# AVERAGE_ONLY=1: eval/final average policies only; presplit: epoch snapshots with PRESPLIT_PARTICLES=256 particles and PRESPLIT_PAIRS=2000 pairs
LOG_REGRET=${LOG_REGRET:-0}; LOG_REGRET_KMEANS=${LOG_REGRET_KMEANS:-0}; LOG_SPLITS=${LOG_SPLITS:-0}; AVERAGE_ONLY=${AVERAGE_ONLY:-0}
PRESPLIT_PARTICLES=${PRESPLIT_PARTICLES:-256}; PRESPLIT_PAIRS=${PRESPLIT_PAIRS:-2000}
# BUDGET_CPU_SECONDS=S: the ARMS stop when their training CPU time reaches S (rerun without it to continue)
# FREEZE_SPLITS_AT=R: tabular split arms stop judging (no label pass, no split) at epochs ending at root R or later
# JUDGE_AT=R: the epoch containing root R ends there and is judged, frozen or not (one forced split)
# pp arms (pp, pp_poker, pp_fill, pp_root): --partition pp, no atlas, judgments from 1000 * 2^k roots, online LBR on the
#   split schedule (needs ONLINE_LBR_EVERY > 0 to switch it on); PP_EXTRA="..." adds pp options (e.g. --mem-budget-mb 160)
# EXE=PATH: executable to build and use (a second name lets new arms use new code while older arms keep running)
BUDGET_CPU_SECONDS=${BUDGET_CPU_SECONDS:-0}
EXE=${EXE:-$HERE/aswiss_stud}
declare -A NAME=([aswiss]=aswiss_k1024_z2 [aswiss_b]=aswiss_b_k1024_z2 [aswiss_wb]=aswiss_wb_k1024_z2 [k256]=atom_k256
                 [hist]=history_street [hist_b]=history_b_street [hist_wb]=history_wb_street [hist_every]=history_every
                 [hist_bb]=history_bb_street [hist_bbf]=history_bbf_street
                 [k341]=atom_k341 [k512]=atom_k512 [k1024]=atom_k1024 [pp]=pp_particle [pp_poker]=pp_particle_poker
                 [pp_fill]=pp_particle_fill [pp_root]=pp_particle_root [mmd]=net_mmd [escher]=net_escher [escher_m1m]=net_escher_m1m [escher_gpu]=net_escher_gpu [rebel]=rebel)
declare -A PART=([aswiss]=aswiss [aswiss_b]=aswiss [aswiss_wb]=aswiss [k256]=atom [k341]=atom [k512]=atom [k1024]=atom
                 [hist]=history [hist_b]=history [hist_wb]=history [hist_every]=history [hist_bb]=history [hist_bbf]=history
                 [pp]=pp [pp_poker]=pp [pp_fill]=pp [pp_root]=pp [mmd]=net [escher]=net [escher_m1m]=net [escher_gpu]=net [rebel]=rebel)
declare -A K=([aswiss]=1024 [aswiss_b]=1024 [aswiss_wb]=1024 [k256]=256 [k341]=341 [k512]=512 [k1024]=1024 [hist]=1024
              [hist_b]=1024 [hist_wb]=1024 [hist_every]=1024 [hist_bb]=1024 [hist_bbf]=1024 [pp]=0 [pp_poker]=0 [pp_fill]=0 [pp_root]=0
              [mmd]=1024 [escher]=1024 [escher_m1m]=1024 [escher_gpu]=1024 [rebel]=1024)
# history arms: label samples capped at 2.56M roots per epoch (history keys do not saturate; bounds the memory)
declare -A EXTRA=([aswiss_b]="--label avg" [aswiss_wb]="--label window+avg" [hist]="--history-cards street --label-cap 2560000"
                  [hist_b]="--history-cards street --label avg --label-cap 2560000"
                  [hist_wb]="--history-cards street --label window+avg --label-cap 2560000"
                  [hist_every]="--history-cards every --label-cap 2560000"
                  [hist_bb]="--history-cards street --label avg --label-cap 2560000 --backward 1"
                  [hist_bbf]="--history-cards street --label avg --label-cap 2560000 --backward 1 --avg-fill plain"
                  [pp]="--epoch0 1000 --online-lbr-schedule split" [pp_poker]="--epoch0 1000 --online-lbr-schedule split --features poker"
                  [pp_fill]="--epoch0 1000 --online-lbr-schedule split --pp-assign fill"
                  [pp_root]="--epoch0 1000 --online-lbr-schedule split --leaf-start root"
                  [mmd]="--algo mmd" [escher]="--algo escher" [escher_m1m]="--algo escher --escher-capacity 1000000" [escher_gpu]="--algo escher"
                  [rebel]="--roots 1000000 --epoch0 64 --checkpoint-every 500 --snapshot-every 0 --curve-every 100")  # rebel: own budget (hands)
declare -A LABEL=([aswiss]="ASWISS (context start, K=1024 atoms, z=2)" [aswiss_b]="ASWISS, method B labels (average policy, sampled)"
                  [aswiss_wb]="ASWISS, window + method B labels" [k256]="k-means 256"
                  [k341]="k-means 341 (matched to ASWISS at 10M)" [k512]="k-means 512" [k1024]="k-means 1024 = atoms trained directly"
                  [hist]="bucket-history keys, window labels"
                  [hist_b]="bucket-history keys, method B labels (average policy)"
                  [hist_wb]="bucket-history keys, window + method B labels"
                  [hist_every]="bucket-history keys, cards at every decision"
                  [hist_bb]="ASWISS: bucket-history keys, method B, backward split"
                  [hist_bbf]="ASWISS + hybrid average for labels (--avg-fill plain)"
                  [pp]="prior-posterior ASWISS (particle directions, general features, no atlas)"
                  [pp_poker]="prior-posterior ASWISS, poker features" [pp_fill]="prior-posterior ASWISS, assign by filling m visits"
                  [pp_root]="prior-posterior ASWISS, new leaves from the root"
                  [mmd]="MMD self-play (policy network)" [escher]="ESCHER (neural CFR)" [escher_m1m]="ESCHER, reservoirs of 1e6" [escher_gpu]="ESCHER, networks trained on the GPU" [rebel]="ReBeL (scaled: search + value networks)")
atlas() { [ "$1" = 0 ] && return 0; echo "$HERE/atlas/atlas_power$1_heur_eps20_seed37301.bin"; }  # K = 0: no atlas (pp arms)
mkdir -p "$OUT"
case $ACTION in
run)
    # rebuild when the executable is missing or older than its sources (for example after git pull)
    if [ ! -x "$EXE" ] || [ "$HERE/aswiss_stud.cpp" -nt "$EXE" ] || [ "$HERE/h4_lbr_generated.hpp" -nt "$EXE" ] ||
       [ -n "$(find "$HERE" -maxdepth 1 -name 'pp_*.hpp' -newer "$EXE" 2>/dev/null)" ] ||
       [ "$HERE/../../../cpp_mccfr/stud_mccfr.cpp" -nt "$EXE" ] ||
       [ -n "$(find "$HERE/sota" -name '*.hpp' -newer "$EXE" 2>/dev/null)" ]; then
        g++ -O3 -std=c++17 -pthread "$HERE/aswiss_stud.cpp" -o "$EXE" ||
            { echo "build failed; if arms are still running, stop them first (stop, then status)"; exit 1; }
    fi
    for a in $ARMS; do
        name=${NAME[$a]}$SUFFIX
        if [ -f "$OUT/$name.pid" ] && kill -0 "$(cat "$OUT/$name.pid")" 2>/dev/null; then echo "$name already running"; continue; fi
        online=""
        if [ "$ONLINE_LBR_EVERY" -gt 0 ]; then
            every=$ONLINE_LBR_EVERY; [ "${PART[$a]}" = rebel ] && every=$ONLINE_LBR_REBEL_EVERY
            online="--online-lbr-every $every --online-lbr-particles $ONLINE_LBR_PARTICLES"
        fi
        [ "${PART[$a]}" = history ] && [ "$LOG_SPLITS" = 1 ] && online="$online --log-splits 1"
        gpu=""; case $a in *_gpu) gpu="${PYTHON:-python3} $HERE/gpu_trainer.py --backend ${GPU_BACKEND:-torch} --device ${GPU_DEVICE:-cuda} --log $OUT/$name/gpu_log.csv --" ;; esac
        # GPU arms: the training runs in a second process, so their budget is wall-clock (--budget-seconds)
        [ "$BUDGET_CPU_SECONDS" != 0 ] && online="$online $( [ -n "$gpu" ] && echo --budget-seconds || echo --budget-cpu-seconds ) $BUDGET_CPU_SECONDS"
        [ "${FREEZE_SPLITS_AT:-0}" != 0 ] && { [ "${PART[$a]}" = history ] || [ "${PART[$a]}" = aswiss ]; } && online="$online --freeze-splits-at $FREEZE_SPLITS_AT"
        [ "${JUDGE_AT:-0}" != 0 ] && { [ "${PART[$a]}" = history ] || [ "${PART[$a]}" = aswiss ]; } && online="$online --judge-at $JUDGE_AT"
        [ "${PART[$a]}" = pp ] && [ -n "${PP_EXTRA:-}" ] && online="$online $PP_EXTRA"
        case ${PART[$a]} in history|aswiss) [ "$LOG_REGRET" = 1 ] && online="$online --log-regret 1" ;;
                            atom) [ "$LOG_REGRET_KMEANS" = 1 ] && online="$online --log-regret 1" ;; esac
        nohup $gpu "$EXE" train --partition "${PART[$a]}" --atlas "$(atlas "${K[$a]}")" --z 2 --seed "$SEED" --roots "$ROOTS" \
            --epoch0 10000 --checkpoint-every "$CHECKPOINT_EVERY" --snapshot-every "$SNAPSHOT_EVERY" --curve-every "$CURVE_EVERY" \
            --h4 "$H4" ${EXTRA[$a]:-} $online $( [ "$THREADS" -gt 1 ] && echo --threads "$THREADS" ) --out "$OUT/$name" \
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
eval|final|presplit)
    specs=(); for a in $ARMS; do specs+=("${NAME[$a]}$SUFFIX:${PART[$a]}:$(atlas "${K[$a]}")"); done
    extra=(--min-roots 100000); [ "$ACTION" = final ] && extra=(--latest --pairs 20000 --tag final)
    [ "$ACTION" = presplit ] && extra=(--kinds epoch --min-roots 0 --particles "$PRESPLIT_PARTICLES" --pairs "$PRESPLIT_PAIRS"
                                       --rebel-pairs "$PRESPLIT_PAIRS" --gap-roots 0 --only-average --tag "presplit$PRESPLIT_PARTICLES")
    [ "$AVERAGE_ONLY" = 1 ] && [ "$ACTION" != presplit ] && extra+=(--only-average)
    "$PYTHON" "$HERE/evaluate.py" "$EXE" "$OUT" --runs "${specs[@]}" --workers "$WORKERS" --rebel-threads "$WORKERS" "${extra[@]}" ;;
plot)
    specs=(); for a in $ARMS; do specs+=("${NAME[$a]}$SUFFIX:${LABEL[$a]}"); done
    base=k256; for k in k256 k341 k512 k1024; do case " $ARMS " in *" $k "*) base=$k; break ;; esac; done  # first k-means arm run
    # the baseline is drawn even when ARMS leaves it out (e.g. ARMS=hist_bbf on a folder that also holds k256)
    case " $ARMS " in *" $base "*) ;; *) [ -d "$OUT/${NAME[$base]}$SUFFIX" ] && specs+=("${NAME[$base]}$SUFFIX:${LABEL[$base]}") ;; esac
    "$PYTHON" "$HERE/plot_stud.py" "$OUT" "$OUT/results$SUFFIX" --runs "${specs[@]}" --baseline "${NAME[$base]}$SUFFIX" --epoch-only
    "$PYTHON" "$HERE/plot_stud_time.py" "$OUT" "$OUT/results${SUFFIX}_cputime.png" --runs "${specs[@]}" --clock cpu  # equal CPU time
    pre=eval_presplit$PRESPLIT_PARTICLES.jsonl
    if ls "$OUT"/*/"$pre" > /dev/null 2>&1; then  # pre-split evaluations (presplit action)
        "$PYTHON" "$HERE/plot_stud.py" "$OUT" "$OUT/results${SUFFIX}_presplit$PRESPLIT_PARTICLES" --runs "${specs[@]}" --baseline "${NAME[$base]}$SUFFIX" --epoch-only --eval-file "$pre"
        "$PYTHON" "$HERE/plot_stud_time.py" "$OUT" "$OUT/results${SUFFIX}_presplit${PRESPLIT_PARTICLES}_time.png" --runs "${specs[@]}" --eval-file "$pre"
        "$PYTHON" "$HERE/plot_stud_time.py" "$OUT" "$OUT/results${SUFFIX}_presplit${PRESPLIT_PARTICLES}_cputime.png" --runs "${specs[@]}" --eval-file "$pre" --clock cpu
    fi
    "$PYTHON" "$HERE/plot_regret_memory.py" "$OUT" "$OUT/results${SUFFIX}_regret_memory.png" --runs "${specs[@]}" --baseline "${NAME[$base]}$SUFFIX"
    ospecs=(); for a in $ARMS; do [ -f "$OUT/${NAME[$a]}$SUFFIX/online_lbr.csv" ] && ospecs+=("${NAME[$a]}$SUFFIX:${LABEL[$a]}"); done
    if [ ${#ospecs[@]} -gt 0 ]; then  # online LBR: moving average of the last 100 and of the last 2000 deals
        obase=(); [ -f "$OUT/${NAME[$base]}$SUFFIX/online_lbr.csv" ] && obase=(--baseline "${NAME[$base]}$SUFFIX")
        for w in 100 2000; do
            "$PYTHON" "$HERE/plot_online_lbr.py" "$OUT" "$OUT/results${SUFFIX}_online_w$w.png" --runs "${ospecs[@]}" --window $w "${obase[@]}"
            "$PYTHON" "$HERE/plot_online_lbr.py" "$OUT" "$OUT/results${SUFFIX}_online_cpu_w$w.png" --runs "${ospecs[@]}" --window $w "${obase[@]}" --clock cpu
        done
    fi ;;
pack)  # small result files of every run in OUT (no snapshots) -> results_data, for git or sharing
    "$PYTHON" "$HERE/pack_summaries.py" "$OUT" "$HERE/results_data/$(basename "$OUT")${SUFFIX}_summaries.tar.gz" ;;  # named after OUT
*) echo "usage: $0 run|status|stop|eval|final|presplit|plot|pack"; exit 2 ;;
esac
