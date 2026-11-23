#!/bin/bash
# Re-run the uploaded experiments (original/) and the split-rule comparison (rule_compare/) into OUT.
# Usage: bash run_repro.sh OUT [WORKERS]     (numpy and scipy; about 10 minutes on 4 cores)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ORIG="$HERE/../original"; OUT=$(mkdir -p "$1" && cd "$1" && pwd); W=${2:-4}
mkdir -p "$OUT/run/out" && cp "$ORIG"/*.py "$OUT/run/" && tar xzf "$ORIG/ds_ab.tar.gz" -C "$OUT/run"
cd "$OUT/run"
sed 's#\.\./upload_aswiss_exp/##g' "$HERE/jobs.txt" | xargs -P "$W" -I{} bash -c 'line="{}"; name="${line%%|*}"; cmd="${line#*|}"; s=$(date +%s); timeout 3000 python3 $cmd > out/$name.txt 2>&1; echo "$name exit=$? $(( $(date +%s)-s ))s" >> out/timings.txt'
mkdir -p "$OUT/rule_compare" && cd "$OUT/rule_compare"
for s in 1 2 3; do python3 "$HERE/rule_compare/rule_compare.py" "$OUT/run" $s 600 rc_s$s & done
python3 "$HERE/rule_compare/unique_rate.py" "$OUT/run" 9 400 > unique_rate.txt; wait
python3 "$HERE/rule_compare/rule_summary.py" rc_s1 rc_s2 rc_s3 > rule_summary.txt
echo "outputs: $OUT/run/out and $OUT/rule_compare"
