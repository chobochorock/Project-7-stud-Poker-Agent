#!/bin/bash
# Kuhn and Leduc: CFR+ on fixed random abstractions (tree vs free) and ASWISS from random starting abstractions.
# usage: run_small.sh OUT [ITERS] [SEEDS] [WORKERS]     (from this folder; needs g++ and numpy)
# OUT/<game>/ holds the exported tree, OUT/abs/ the abstractions, OUT/runs/<game>/<arm>/ one aswiss run each
# (log.csv, judgments.csv, closure.csv, policies, exploitability.csv from ../eval.py).
set -e
OUT=$1; T=${2:-655350}; SEEDS=${3:-10}; P=${4:-4}
HERE=$(cd "$(dirname "$0")" && pwd); A=$HERE/..
mkdir -p $OUT/abs $OUT/runs
g++ -O3 -std=c++17 $A/aswiss.cpp -o $OUT/aswiss
jobs=$OUT/jobs.txt; : > $jobs
for g in kuhn leduc; do
  python3 $HERE/export_small.py $g $OUT/$g > /dev/null
  python3 -c "import numpy as np; np.arange($( [ $g = kuhn ] && echo 12 || echo 288 ), dtype=np.int32).tofile('$OUT/$g/bucket_identity.bin')"
  fracs="0.3 0.5"; [ $g = kuhn ] && fracs="0.5"
  for f in $fracs; do for s in $(seq 1 $SEEDS); do
    python3 $HERE/random_abstraction.py $OUT/$g $f $s $OUT/abs/${g}_f${f}_s$s > /dev/null
    python3 $A/kmeans.py $OUT/$g $OUT/abs/${g}_f${f}_s${s}_tree.bin $OUT/abs/${g}_f${f}_s${s}_km.bin $s > /dev/null
  done; done
  base="--tree $OUT/$g --mode full --iters $T --epoch0 10"
  echo "$g ref $base --buckets $OUT/$g/bucket_identity.bin --split none" >> $jobs
  echo "$g fixed_P0 $base --buckets $OUT/$g/bucket_timestep.bin --split none" >> $jobs
  for f in $fracs; do for s in $(seq 1 $SEEDS); do for kind in tree free km; do
    echo "$g fixed_${kind}_f${f}_s$s $base --buckets $OUT/abs/${g}_f${f}_s${s}_$kind.bin --split none" >> $jobs
  done; done; done
  fs=$(echo $fracs | cut -d' ' -f1)
  for v in fwd bwd; do for c in 0 1; do
    flags="--split cum --label avg --unit history --backward $([ $v = bwd ] && echo 1 || echo 0) --tree-closure $c"
    echo "$g aswiss_${v}_c${c}_P0 $base --buckets $OUT/$g/bucket_timestep.bin $flags" >> $jobs
    for s in $(seq 1 $SEEDS); do for kind in tree free; do
      echo "$g aswiss_${v}_c${c}_${kind}_f${fs}_s$s $base --buckets $OUT/abs/${g}_f${fs}_s${s}_$kind.bin $flags" >> $jobs
    done; done
  done; done
done
run() {  # game arm args...
  g=$1; arm=$2; shift 2; d=$OUT/runs/$g/$arm; mkdir -p $d
  [ -f $d/exploitability.csv ] && return
  $OUT/aswiss "$@" --out $d 2> $d/stderr.log
  python3 $A/eval.py $OUT/$g $d > /dev/null
}
export -f run; export OUT A
cat $jobs | xargs -P $P -L 1 bash -c 'run "$@"' _
echo "done: $(wc -l < $jobs) runs"
