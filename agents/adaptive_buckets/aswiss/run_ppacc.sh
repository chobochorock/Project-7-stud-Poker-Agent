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
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
echo "done: ppacc runs in $OUT/runs"
