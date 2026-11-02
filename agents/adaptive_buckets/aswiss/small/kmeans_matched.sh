#!/bin/bash
# The k-means baseline of every ASWISS run of run_small.sh: card-feature k-means (../kmeans.py) with the run's final
# bucket count in each P0 group, trained by the same CFR+ for the same iterations on that fixed partition.
# usage: kmeans_matched.sh OUT [ITERS] [WORKERS]   (after run_small.sh OUT); writes OUT/runs_km/<game>/<aswiss arm>/
OUT=$1; T=${2:-655350}; P=${3:-4}
A=$(cd "$(dirname "$0")/.." && pwd)
jobs=$OUT/jobs_km.txt; : > $jobs
for d in $OUT/runs/*/aswiss_*; do
  g=$(basename $(dirname $d)); arm=$(basename $d)
  echo "$g $arm" >> $jobs
done
run() {
  g=$1; arm=$2; d=$OUT/runs_km/$g/$arm; mkdir -p $d
  [ -f $d/exploitability.csv ] && return
  python3 $A/kmeans.py $OUT/$g $OUT/runs/$g/$arm/final_bucket_map.bin $d/km_map.bin 1 > /dev/null
  $OUT/aswiss --tree $OUT/$g --mode full --iters $T --epoch0 10 --buckets $d/km_map.bin --split none --out $d 2> $d/stderr.log
  python3 $A/eval.py $OUT/$g $d > /dev/null
}
export -f run; export OUT A T
cat $jobs | xargs -P $P -L 1 bash -c 'run "$@"' _
echo "done: $(wc -l < $jobs) k-means runs"
