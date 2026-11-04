#!/bin/bash
# Exact comparison of abstraction refinement against CFR+IRA and CFR variants (README_KO.md in this folder).
# usage (from the repository root):
#   bash agents/adaptive_buckets/aswiss/compare/run_compare.sh GAME OUT [ITERS] [SEEDS] [WORKERS]
#   GAME: kuhn | leduc (small/export_small.py) or tiny | small (export.py, mini-stud)
#   defaults: ITERS 655350 for kuhn/leduc and 81910 for mini-stud, SEEDS 10 (CFR+IRA seeds), WORKERS 4
# Arms: CFR+, DCFR, PCFR+ without abstraction; ours (window labels) from P0 and from CFR+IRA's initial abstraction;
# CFR+IRA B10H90 and B100H900 per seed; P0 fixed; k-means fixed with the bucket counts of ours and of CFR+IRA B10H90
# seed 1 per P0 group. Every arm: alternating updates, same budget, checkpoints at 10, 30, 70, ... (epoch ends).
set -euo pipefail
GAME=$1; OUT=$2
case $GAME in kuhn|leduc) ITERS=${3:-655350} ;; *) ITERS=${3:-81910} ;; esac
SEEDS=${4:-10}; P=${5:-4}
HERE=$(cd "$(dirname "$0")" && pwd); A=$(dirname "$HERE")
mkdir -p "$OUT/runs"; D=$OUT/tree
if [ ! -f "$D/meta.json" ]; then
  case $GAME in kuhn|leduc) python3 "$A/small/export_small.py" "$GAME" "$D" ;; *) python3 "$A/export.py" "$GAME" "$D" ;; esac
fi
python3 -c "import numpy as np,sys; n=len(np.fromfile(sys.argv[1]+'/counts.bin',np.int32)); np.arange(n,dtype=np.int32).tofile(sys.argv[1]+'/bucket_identity.bin')" "$D"
[ "$OUT/refine" -nt "$HERE/refine.cpp" ] || g++ -O3 -std=c++17 "$HERE/refine.cpp" -o "$OUT/refine"
jobs=$OUT/jobs.txt
{
  echo "label_p0_alt --buckets $D/bucket_timestep.bin --refine label"
  echo "label_ira0_alt --init ira --refine label"
  echo "full_cfrp_alt --buckets $D/bucket_identity.bin"
  echo "full_dcfr_alt --buckets $D/bucket_identity.bin --solver dcfr"
  echo "full_pcfrp_alt --buckets $D/bucket_identity.bin --solver pcfr+"
  echo "fixed_p0_alt --buckets $D/bucket_timestep.bin"
  for s in $(seq 1 "$SEEDS"); do
    echo "ira_B10H90_s$s --init ira --refine ira --kb 10 --kh 90 --seed $s"
    echo "ira_B100H900_s$s --init ira --refine ira --kb 100 --kh 900 --seed $s"
  done
} > "$jobs"
run() {
  name=$1; shift; d=$OUT/runs/$name
  [ -f "$d/exploitability.csv" ] && return
  mkdir -p "$d"
  "$OUT/refine" --tree "$D" --iters "$ITERS" --epoch0 10 --out "$d" "$@" 2> "$d/stderr.log"
  python3 "$A/eval.py" "$D" "$d" > "$d/eval.log" 2>&1
  echo "$name done"
}
export -f run; export OUT D ITERS A
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
for pair in "kmeans_ours label_p0_alt" "kmeans_ira ira_B10H90_s1"; do
  set -- $pair; mkdir -p "$OUT/runs/$1"
  [ -f "$OUT/runs/$1/km_map.bin" ] || python3 "$A/kmeans.py" "$D" "$OUT/runs/$2/final_bucket_map.bin" "$OUT/runs/$1/km_map.bin" 1 > "$OUT/runs/$1/kmeans.log"
done
printf "kmeans_ours --buckets $OUT/runs/kmeans_ours/km_map.bin\nkmeans_ira --buckets $OUT/runs/kmeans_ira/km_map.bin\n" | xargs -P 2 -L 1 bash -c 'run "$@"' _
python3 "$HERE/plot_compare.py" "$OUT/runs" "$OUT/compare_$GAME" "$GAME" > /dev/null
echo "figure $OUT/compare_$GAME.png, tables $OUT/compare_$GAME.md"
