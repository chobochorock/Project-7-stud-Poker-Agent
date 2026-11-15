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
# --judgment-rows split: a judgment that splits nothing writes no row (with few tables the waiting condition is met
# every 10,000 iterations, and each row is ~3 MB of policy files).
#   ppspec_gate     alternating prior and posterior judgments, the new posterior tables restart
#   ppspec_rt       the user's first alternative: posterior splits in real time with full inheritance
#   ppspec_leaves   the user's second alternative: a bucket's tables are the sums of its leaves' records
# Two changes to ppspec_gate (design section 19.7; user, 2026-11-15: "둘 다"):
#   ppspec_dist       (가) the divergence test is the normalised distance alone (--div-test dist; no MSE z test)
#   ppspec_keep       (나) prior candidates are kept when they fail or their leaf is not eligible (--keep-cands 1)
#   ppspec_dist_keep  both
# Prior leaves fast and fine, the posterior slowly (design section 19.8; user, 2026-11-15):
#   ppfast_gate       prior judgments at every doubling for every leaf (as tree_mse), posterior judgments when the
#                     window's average regret is <= 0.1, by the leaves' accumulated-regret directions
#   ppfast_dist_keep  the same with (가) and (나)
#   ppfast_rt         ppfast_gate with real-time prior splits (design section 19.9; user, 2026-11-15): no prior judgments;
#                     a candidate is tested once when its two sides together reach 1,000 visits, splits the leaf if it
#                     passes, else it is replaced (--prior-visits 1000)
# The hidden-regret criterion (design section 22; user, 2026-11-15). ppfast_rt's real-time prior with each candidate
# tested once at 1,000 visits by D = M(B1) + M(B2) - M(B), M(X) = max(0, max_a R(X, a)): split when
# D / t >= eps + z_d (s1 sqrt(n1) + s2 sqrt(n2)) / t (normal, one action, d = 0.1, eps = 0). The posterior is judged only
# when the window's average regret, counting per bucket only what exceeds z_d x the window's noise, is <= 0.1; then in
# each bucket the leaves are grouped by their stored window direction (argmax action of the regret sum since the last
# judgment, if positive and the leaf has >= 100 window visits) and each group B_a is tested against the rest of its
# bucket by the same D test; every passing group moves to a new bucket (the moved leaves restart).
#   ppfast_rt_hid      as above, tables restart
#   ppfast_rt_hid_inh  the new tables start from their leaves' window sums (--inherit-candidate 1)
#   ppfast_rt_hid_ah   the noise term by Azuma-Hoeffding (eps = 0, L = 64): the guaranteed bound
# (--prior-leaf-ratio 100 is not run here: on mini-stud live prior leaves stay at ~3x the tables in use, so a cap of
# 100x never binds and the arm would equal ppfast_rt_hid.)
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
  python3 "$A/eval.py" "$D" "$d" > /dev/null || { echo "eval failed: $d (policy files kept)"; return 1; }
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
  SPEC="--mode ppacc --acc-metric mse --div-metric dir --div-ref bucket --reset-moved-only 1 --schedule regret --regret-gate 0.1 --measure 1 --judgment-rows split"
  echo "ppspec_gate $s $SPEC" >> "$jobs"
  echo "ppspec_rt $s $SPEC --post-realtime 1 --inherit-regret 1 --inherit-avg 1" >> "$jobs"
  echo "ppspec_leaves $s $SPEC --post-from-leaves 1" >> "$jobs"
  echo "ppspec_dist $s $SPEC --div-test dist" >> "$jobs"
  echo "ppspec_keep $s $SPEC --keep-cands 1" >> "$jobs"
  echo "ppspec_dist_keep $s $SPEC --div-test dist --keep-cands 1" >> "$jobs"
  echo "ppfast_gate $s $SPEC --split-plan separate --prior-eligible all" >> "$jobs"
  echo "ppfast_dist_keep $s $SPEC --split-plan separate --prior-eligible all --div-test dist --keep-cands 1" >> "$jobs"
  echo "ppfast_rt $s $SPEC --split-plan separate --prior-eligible all --prior-visits 1000" >> "$jobs"
  HID="--mode ppacc --acc-metric hidden --div-metric hidden --schedule regret --regret-gate 0.1 --gate-noise normal --split-plan separate --prior-eligible all --prior-visits 1000 --measure 1 --judgment-rows split"
  echo "ppfast_rt_hid $s $HID" >> "$jobs"
  echo "ppfast_rt_hid_inh $s $HID --inherit-candidate 1" >> "$jobs"
  echo "ppfast_rt_hid_ah $s $HID --hid-bound azuma" >> "$jobs"
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
arm() { local n=$1; local l=""; for s in $SEEDS; do l="$l,$OUT/runs/${n}_s$s"; done; echo "${l#,}"; }
# figure: the waiting-condition pp arms with the single tree and pure ASWISS (run_tree_acc.sh, run_pp.sh) as references
python3 "$A/plot_pp.py" "$OUT/pp_small_ppspec" --arm "ppacc_gate=$(arm ppacc_gate)" --arm "ppacc_rt=$(arm ppacc_rt)" \
  --arm "ppacc_rt_restart=$(arm ppacc_rt_restart)" --arm "ppspec_gate=$(arm ppspec_gate)" --arm "ppspec_rt=$(arm ppspec_rt)" \
  --arm "ppspec_leaves=$(arm ppspec_leaves)" --arm "tree_mse=$(arm tree_mse)" --arm "pure_a=$(arm pure_a)" \
  --ref "kmeans_a (rule a counts)=$(arm kmeans_a)" --ref "mccfr (no abstraction)=$(arm mccfr)" --kmeans "kmeans_a (rule a counts)"
python3 "$A/plot_pp.py" "$OUT/pp_small_ppspec2" --arm "ppacc_gate=$(arm ppacc_gate)" --arm "ppacc_rt_restart=$(arm ppacc_rt_restart)" \
  --arm "ppspec_gate=$(arm ppspec_gate)" --arm "ppspec_dist=$(arm ppspec_dist)" --arm "ppspec_keep=$(arm ppspec_keep)" \
  --arm "ppspec_dist_keep=$(arm ppspec_dist_keep)" --arm "tree_mse=$(arm tree_mse)" --arm "pure_a=$(arm pure_a)" \
  --ref "kmeans_a (rule a counts)=$(arm kmeans_a)" --ref "mccfr (no abstraction)=$(arm mccfr)" --kmeans "kmeans_a (rule a counts)"
python3 "$A/plot_pp.py" "$OUT/pp_small_ppfast" --arm "ppspec_gate=$(arm ppspec_gate)" --arm "ppspec_dist_keep=$(arm ppspec_dist_keep)" \
  --arm "ppfast_gate=$(arm ppfast_gate)" --arm "ppfast_dist_keep=$(arm ppfast_dist_keep)" --arm "ppacc_rt_restart=$(arm ppacc_rt_restart)" \
  --arm "tree_mse=$(arm tree_mse)" --arm "pure_a=$(arm pure_a)" \
  --ref "kmeans_a (rule a counts)=$(arm kmeans_a)" --ref "mccfr (no abstraction)=$(arm mccfr)" --kmeans "kmeans_a (rule a counts)"
python3 "$A/plot_pp.py" "$OUT/pp_small_ppfast_rt" --arm "ppfast_gate=$(arm ppfast_gate)" --arm "ppfast_rt=$(arm ppfast_rt)" \
  --arm "ppacc_rt_restart=$(arm ppacc_rt_restart)" --arm "tree_mse=$(arm tree_mse)" --arm "pure_a=$(arm pure_a)" \
  --ref "kmeans_a (rule a counts)=$(arm kmeans_a)" --ref "mccfr (no abstraction)=$(arm mccfr)" --kmeans "kmeans_a (rule a counts)"
python3 "$A/plot_pp.py" "$OUT/pp_small_ppfast_hid" --arm "ppfast_rt=$(arm ppfast_rt)" --arm "ppfast_rt_hid=$(arm ppfast_rt_hid)" \
  --arm "ppfast_rt_hid_inh=$(arm ppfast_rt_hid_inh)" --arm "ppfast_rt_hid_ah=$(arm ppfast_rt_hid_ah)" --arm "tree_mse=$(arm tree_mse)" \
  --ref "kmeans_a (rule a counts)=$(arm kmeans_a)" --ref "mccfr (no abstraction)=$(arm mccfr)" --kmeans "kmeans_a (rule a counts)"
python3 "$A/acc_table.py" "$OUT/runs" > "$OUT/acc_final_table.md"
echo "done: ppacc runs in $OUT/runs"
