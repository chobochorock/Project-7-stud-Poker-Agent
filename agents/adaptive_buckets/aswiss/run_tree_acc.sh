#!/bin/bash
# Single tree split by accumulated regret on mini-stud small (PP_STUD7_DESIGN_KO.md sections 18-19; user, 2026-11-14:
# "experiments on mini-stud first"). Reuses the exported tree and the reference runs of run_pp.sh / run_pp_dir.sh
# (pure_a, ppeq_a_r4_rnlab, ppmix_avg_root, kmeans_a, mccfr: the same SECONDS, seeds, row grid and evaluator).
#   bash run_tree_acc.sh OUT SECONDS [SEEDS] [PARALLEL]
# Arms (--mode tree, every leaf its own table; --measure 1 as the reference arms):
#   tree_mse        accumulated-regret split, squared distance of the sides' mean regret vectors, tables restart
#   tree_mse_inh    the same, new leaves start from the leaf's regrets and average sums (--inherit-regret 1 --inherit-avg 1)
#   tree_dir        --acc-metric dir: the sides must also differ in the direction of their (unit) mean regret vectors
#   tree_dir_inh    the same with full inheritance
#   tree_mse_g4, tree_mse_g10, tree_mse_inh_g10   splits every 4x / 10x of iterations (--epoch-growth): more training
#                   between splits (user, 2026-11-14: "restart is the right method; the tables were not given enough time")
#   tree_mse_frz, tree_mse_inh_frz   no split after the judgment at 16,384,000 iterations (--freeze-at): the same
#                   kind of partition, trained on for the rest of the budget, restart against full inheritance
set -euo pipefail
OUT=$(cd "$1" && pwd); SECS=$2; SEEDS=${3:-"1 2 3"}; P=${4:-4}
A=$(cd "$(dirname "$0")" && pwd)
D=$OUT/tree
[ -f "$D/obs.bin" ] || { echo "run run_pp.sh first (tree)"; exit 1; }
[ "$OUT/pp_aswiss" -nt "$A/pp_aswiss.cpp" ] || g++ -O3 -std=c++17 "$A/pp_aswiss.cpp" -o "$OUT/pp_aswiss"
EXE=$OUT/pp_aswiss
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
jobs=$OUT/jobs_tree_acc.txt; : > "$jobs"
for s in $SEEDS; do
  echo "tree_mse $s --mode tree --measure 1" >> "$jobs"
  echo "tree_mse_inh $s --mode tree --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
  echo "tree_dir $s --mode tree --acc-metric dir --measure 1" >> "$jobs"
  echo "tree_dir_inh $s --mode tree --acc-metric dir --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
  echo "tree_mse_g4 $s --mode tree --epoch-growth 4 --measure 1" >> "$jobs"
  echo "tree_mse_g10 $s --mode tree --epoch-growth 10 --measure 1" >> "$jobs"
  echo "tree_mse_inh_g10 $s --mode tree --epoch-growth 10 --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
  echo "tree_mse_frz $s --mode tree --freeze-at 16384001 --measure 1" >> "$jobs"
  echo "tree_mse_inh_frz $s --mode tree --freeze-at 16384001 --inherit-regret 1 --inherit-avg 1 --measure 1" >> "$jobs"
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
arm() { local n=$1; local l=""; for s in $SEEDS; do l="$l,$OUT/runs/${n}_s$s"; done; echo "${l#,}"; }
python3 "$A/plot_pp.py" "$OUT/pp_small_tree_acc" --arm "tree, accumulated MSE, restart=$(arm tree_mse)" \
  --arm "tree, accumulated MSE, full inheritance=$(arm tree_mse_inh)" --arm "tree, accumulated direction, restart=$(arm tree_dir)" \
  --arm "tree, accumulated direction, full inheritance=$(arm tree_dir_inh)" --arm "tree, accumulated MSE, restart, split every 4x=$(arm tree_mse_g4)" \
  --arm "tree, accumulated MSE, restart, split every 10x=$(arm tree_mse_g10)" \
  --arm "tree, accumulated MSE, full inheritance, split every 10x=$(arm tree_mse_inh_g10)" \
  --arm "tree, accumulated MSE, restart, frozen after 16.4M=$(arm tree_mse_frz)" \
  --arm "tree, accumulated MSE, full inheritance, frozen after 16.4M=$(arm tree_mse_inh_frz)" --arm "pure a, iterates=$(arm pure_a)" \
  --arm "pp mix, avg, root=$(arm ppmix_avg_root)" \
  --ref "k-means (rule a counts)=$(arm kmeans_a)" --ref "MCCFR (no abstraction)=$(arm mccfr)" --kmeans "k-means (rule a counts)"
echo "done: $OUT/pp_small_tree_acc.md"
