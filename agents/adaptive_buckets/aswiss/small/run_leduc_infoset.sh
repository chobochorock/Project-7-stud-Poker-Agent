#!/bin/bash
# Leduc (and Kuhn): ASWISS at the information-set level, the main method of the thesis
# (labels = argmax of each information set's window-cumulative regret of the CFR+ iterates, ASWISS_CONVERGENCE_KO.md
# theorem 1), from P0 and from the same random starting abstractions as run_small.sh. Also the split-time average
# strategy labels (method B, --label avg) at the information-set level for comparison.
# usage: run_leduc_infoset.sh OUT [ITERS] [SEEDS] [WORKERS]
set -e
OUT=$1; T=${2:-655350}; SEEDS=${3:-10}; P=${4:-4}
HERE=$(cd "$(dirname "$0")" && pwd); A=$HERE/..
mkdir -p $OUT/abs $OUT/runs
g++ -O3 -std=c++17 $A/aswiss.cpp -o $OUT/aswiss
jobs=$OUT/jobs.txt; : > $jobs
for g in kuhn leduc; do
  python3 $HERE/export_small.py $g $OUT/$g > /dev/null
  python3 -c "import numpy as np; np.arange($( [ $g = kuhn ] && echo 12 || echo 288 ), dtype=np.int32).tofile('$OUT/$g/bucket_identity.bin')"
  f=0.3; [ $g = kuhn ] && f=0.5
  for s in $(seq 1 $SEEDS); do python3 $HERE/random_abstraction.py $OUT/$g $f $s $OUT/abs/${g}_f${f}_s$s > /dev/null; done
  base="--tree $OUT/$g --mode full --iters $T --epoch0 10"
  echo "$g ref $base --buckets $OUT/$g/bucket_identity.bin --split none" >> $jobs
  for lab in window avg; do
    flags="--split cum --label $lab"
    echo "$g ${lab}_P0 $base --buckets $OUT/$g/bucket_timestep.bin $flags" >> $jobs
    for s in $(seq 1 $SEEDS); do for kind in tree free; do
      echo "$g ${lab}_${kind}_f${f}_s$s $base --buckets $OUT/abs/${g}_f${f}_s${s}_$kind.bin $flags" >> $jobs
    done; done
  done
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
