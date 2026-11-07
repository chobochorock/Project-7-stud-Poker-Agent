#!/bin/bash
# Equal wall-clock comparison on the small games with exact exploitability (2026-11-07).
#   bash run_walltime.sh GAME OUT SECONDS [SEED] [PARALLEL]      GAME = kuhn | leduc | small (mini-stud)
# Every method trains single-threaded for the same SECONDS of training time (checkpoint saving, diagnostics and
# ESCHER's checkpoint average network are kept out of that clock; column train_seconds), then ../eval.py computes the
# exact exploitability (NashConv/2) of every checkpoint. Methods:
#   aswiss  ASWISS on pure external-sampling MCCFR (--plain 1: regret matching without the RM+ floor, uniform
#           averaging), from the coarsest abstraction P0 (one bucket per time step), window labels with z = 2,
#           every label computed on the information sets themselves (no card clusters, no prior)
#   mccfr   the same pure MCCFR on the unabstracted game (one bucket per information set)
#   kmeans  the same pure MCCFR on a fixed card-feature k-means abstraction with ASWISS's final bucket count
#   mmd     MMD with the paper's deep setting (schedule 0.05 sqrt(1e7/t), clip 0.3, gamma 0.99, 30 epochs x 128,
#           lr 5e-5, ~4,000 steps per update), 128 x 128 networks (the size validated on Leduc)
#   escher  ESCHER with the paper's Table 4 budget (1,000 + 1,000 traversals, batch 2048, 500 / 500 / 10,000 steps),
#           value exploration 0.01, value network re-initialized each iteration, argmax fallback, 256 x 128 networks
#           (official code), reservoirs of 1e5 (official code)
# ReBeL has no implementation on these exported trees (ours is specific to 7-card stud); it is listed as missing.
# Jobs whose exploitability.csv exists are skipped, so the script can be rerun after an interruption.
set -euo pipefail
GAME=$1; OUT=$2; SECS=$3; SEED=${4:-1}; P=${5:-4}
HERE=$(cd "$(dirname "$0")" && pwd); A=$(dirname "$HERE")
D=$OUT/tree; R=$OUT/runs_s$SEED
mkdir -p "$R"
if [ ! -f "$D/meta.json" ]; then
  case $GAME in kuhn|leduc) python3 "$A/small/export_small.py" "$GAME" "$D" ;; *) python3 "$A/export.py" "$GAME" "$D" ;; esac
fi
[ -f "$D/bucket_identity.bin" ] || python3 -c "import numpy as np,sys; n=len(np.fromfile(sys.argv[1]+'/counts.bin',np.int32)); np.arange(n,dtype=np.int32).tofile(sys.argv[1]+'/bucket_identity.bin')" "$D"
[ "$OUT/aswiss" -nt "$A/aswiss.cpp" ] || g++ -O3 -std=c++17 "$A/aswiss.cpp" -o "$OUT/aswiss"
[ "$OUT/nn_tree" -nt "$HERE/nn_tree.cpp" ] && [ "$OUT/nn_tree" -nt "$A/stud7/sota/mmd.hpp" ] && [ "$OUT/nn_tree" -nt "$A/stud7/sota/escher.hpp" ] ||
  g++ -O3 -std=c++17 "$HERE/nn_tree.cpp" -o "$OUT/nn_tree"
case $GAME in kuhn) EPISODES=2000 ;; leduc) EPISODES=1300 ;; *) EPISODES=1000 ;; esac
TAB="--mode es --plain 1 --iters 1000000000000 --epoch0 1000 --z 2 --seed $SEED --seconds $SECS"
MMD="--algo mmd --seed $SEED --seconds $SECS --episodes-total 1000000000000 --episodes $EPISODES --epochs 30 --minibatch 128"
MMD+=" --lr 5e-5 --value-lr 5e-5 --schedule sqrt --alpha 0.05 --eta 0.05 --horizon 10000000 --clip 0.3 --gamma 0.99"
MMD+=" --hidden1 128 --hidden2 128 --reward-scale 1"
ESC="--algo escher --seed $SEED --seconds $SECS --iterations 1000000000 --value-hands 1000 --traversals 1000 --batch 2048"
ESC+=" --regret-steps 500 --value-steps 500 --avg-steps 10000 --hidden1 256 --hidden2 128 --capacity 100000 --lr 1e-3"
ESC+=" --value-lr 1e-3 --reward-scale 1 --reinit 1 --value-reinit 1 --value-explore 0.01 --fallback argmax"
run() {  # name, program, args...
  local name=$1 prog=$2; shift 2; local d=$R/$name
  [ -f "$d/exploitability.csv" ] && return 0
  rm -rf "$d"; mkdir -p "$d"
  if [ "$prog" = aswiss ]; then "$OUT/aswiss" --tree "$D" --out "$d" "$@" 2> "$d/stderr.log"
  else "$OUT/nn_tree" --tree "$D" --out "$d" "$@" 2> "$d/stderr.log"; fi
  python3 "$A/eval.py" "$D" "$d" > "$d/eval.log" 2>&1
  echo "$GAME $name done"
}
export -f run; export OUT D R A GAME
{
  echo "aswiss aswiss --buckets $D/bucket_timestep.bin --split cum --label window $TAB"
  echo "mccfr aswiss --buckets $D/bucket_identity.bin --split none $TAB"
  echo "mmd nn_tree $MMD"
  echo "escher nn_tree $ESC"
} > "$R/jobs.txt"
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$R/jobs.txt"
# k-means with ASWISS's final bucket count (card features of the exported tree), then the same pure MCCFR
mkdir -p "$R/kmeans_map"
[ -f "$R/kmeans_map/km_map.bin" ] || python3 "$A/kmeans.py" "$D" "$R/aswiss/final_bucket_map.bin" "$R/kmeans_map/km_map.bin" "$SEED" > "$R/kmeans_map/kmeans.log"
run kmeans aswiss --buckets "$R/kmeans_map/km_map.bin" --split none $TAB
echo "$GAME seed $SEED: all runs in $R"
