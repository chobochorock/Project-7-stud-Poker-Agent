#!/bin/bash
# Prior-posterior with accumulated regrets on mini-stud small (--mode ppacc, PP_STUD7_DESIGN_KO.md section 19; user,
# 2026-11-14). Same tree, budget, seeds, row grid and evaluator as run_tree_acc.sh and the reference runs.
#   bash run_ppacc.sh OUT SECONDS [SEEDS] [PARALLEL]
# Arms (full inheritance, the user's base case):
#   ppacc_dir_inh   divergence and prior tests by the normalised (direction) criterion, --acc-metric dir
#   ppacc_mse_inh   the same with the squared distance of the mean regret vectors (--acc-metric mse)
#   ppacc_gate      MSE, tables restart, every judgment waits until the window's abstract-game average regret is <= 0.1
#                   (--schedule regret --regret-gate 0.1; user, 2026-11-14: the same gate as the tree)
#   ppacc_dir_gate  the same with the direction criterion
#   ppacc_rt        the user's alternative (2026-11-14): posterior splits in real time (checked every 10,000 iterations)
#                   with full inheritance; prior splits wait until the window's average regret is <= 0.1; MSE
#   ppacc_rt_restart  the same with the new posterior tables restarting (a comparison)
# The user's specification as written (2026-11-14; design section 19.6): prior candidates by MSE, the divergence of a
# leaf by the distance of normalised regret vectors against its whole posterior bucket, and at a posterior split only the
# leaves whose pointer is reassigned restart their records; every judgment waits for the window's average regret <= 0.1.
#   ppspec_gate     alternating prior and posterior judgments, the new posterior tables restart
#   ppspec_rt       the user's first alternative: posterior splits in real time with full inheritance
#   ppspec_leaves   the user's second alternative: a bucket's tables are the sums of its leaves' records
set -euo pipefail
OUT=$(cd "$1" && pwd); SECS=$2; SEEDS=${3:-"1 2 3"}; P=${4:-4}
A=$(cd "$(dirname "$0")" && pwd)
D=$OUT/tree
[ -f "$D/obs.bin" ] || { echo "run run_pp.sh first (tree)"; exit 1; }
EXE=$OUT/pp_aswiss_ppacc
[ "$EXE" -nt "$A/pp_aswiss.cpp" ] || g++ -O3 -std=c++17 "$A/pp_aswiss.cpp" -o "$EXE"
run() {  # name seed args...
  local name=$1 seed=$2; shift 2; local d=$OUT/runs/${name}_s$seed
  [ -f "$d/exploitability.csv" ] && return 0
  mkdir -p "$d"
  "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" "$@" 2>> "$d/stderr.log"
  python3 "$A/eval.py" "$D" "$d" > /dev/null
  local last=$(( $(grep -c '^[0-9]' "$d/log.csv") - 1 ))
  find "$d" -name 'policy_*.bin' ! -name "policy_*_$last.bin" -delete
  rm -f "$d/state.bin" "$d/state.bin.tmp"
}
export -f run; export OUT D EXE SECS A
jobs=$OUT/jobs_ppacc.txt; : > "$jobs"
for s in $SEEDS; do
  echo "ppacc_dir_inh $s --mode ppacc --acc-metric dir --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
  echo "ppacc_mse_inh $s --mode ppacc --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
  echo "ppacc_gate $s --mode ppacc --schedule regret --regret-gate 0.1 --measure 1" >> "$jobs"
  echo "ppacc_dir_gate $s --mode ppacc --acc-metric dir --schedule regret --regret-gate 0.1 --measure 1" >> "$jobs"
  echo "ppacc_rt $s --mode ppacc --schedule regret --regret-gate 0.1 --post-realtime 1 --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
  echo "ppacc_rt_restart $s --mode ppacc --schedule regret --regret-gate 0.1 --post-realtime 1 --measure 1" >> "$jobs"
  SPEC="--mode ppacc --acc-metric mse --div-metric dir --div-ref bucket --reset-moved-only 1 --schedule regret --regret-gate 0.1 --measure 1"
  echo "ppspec_gate $s $SPEC" >> "$jobs"
  echo "ppspec_rt $s $SPEC --post-realtime 1 --inherit-regret 1 --inherit-avg 1" >> "$jobs"
  echo "ppspec_leaves $s $SPEC --post-from-leaves 1" >> "$jobs"
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
echo "done: ppacc runs in $OUT/runs"
