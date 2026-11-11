#!/bin/bash
# Prior-posterior ASWISS on mini-stud small: the experiment of PRIOR_POSTERIOR_DESIGN_KO.md section 11.
#   bash run_pp.sh OUT SECONDS [SEEDS] [PARALLEL]        e.g. bash run_pp.sh ~/pp_runs 1800 "1 2 3" 4
# Every arm trains single-threaded for SECONDS of CPU time (pp_aswiss --seconds) with the same pure external-sampling
# MCCFR, the same row grid and the same evaluator (eval.py, exact exploitability):
#   ppeq     --mode pp --prior-start equal  (--measure 1)               the spec's defaults (generalized ESD)
#   pplen    --mode pp --prior-start length
#   pure     --mode pure
#   ppeq_fc  --mode pp --scale fact-counterfact
#   ppeq_a   --mode pp --outlier label      (rule (a): significance-gated argmax label, z = 2)
#   pure_a   --mode pure --outlier label
#   mccfr    --mode pure --p0 identity --split 0          (the unabstracted game)
#   kmeans   --mode pure --p0 <k-means map> --split 0     (kmeans.py on ppeq's final bucket map, per time step; same seed)
#   ppeq_a_r4 --mode pp --outlier label --prior-ratio 4 --crossfit 0   (prior >= 4 x posterior buckets, no cross-fit)
#   ppeq_r4  --mode pp --prior-ratio 4 --crossfit 0
#   ppeq_a_r4_rn, ppeq_a_r4_rnlab, ppeq_a_r4_lab: ppeq_a_r4 with --resolve-now 1, with both --resolve-now 1 and
#            --prior-criterion label, and with --prior-criterion label
#   kmeans_a, kmeans_a_r4 the same on ppeq_a's / ppeq_a_r4's final bucket maps
# Runs whose exploitability.csv exists are skipped; a run stopped midway resumes from its state.bin when rerun. After its
# evaluation (and, for pp runs, --analyze 3000 -> analysis.json) a run keeps only its last row's policy files and no
# state.bin, to save disk.
# Then plot_pp.py writes OUT/pp_small_esd_*.png / .md (spec defaults) and OUT/pp_small_a_*.png / .md (rule (a)).
set -euo pipefail
OUT=$(mkdir -p "$1" && cd "$1" && pwd); SECS=$2; SEEDS=${3:-"1 2 3"}; P=${4:-4}
A=$(cd "$(dirname "$0")" && pwd)
D=$OUT/tree
if [ ! -f "$D/obs.bin" ]; then
  python3 "$A/export.py" small "$D" > /dev/null
  python3 "$A/export_obs.py" small "$D"
fi
[ -f "$D/bucket_identity.bin" ] || python3 -c "import numpy as np,sys; n=len(np.fromfile(sys.argv[1]+'/counts.bin',np.int32)); np.arange(n,dtype=np.int32).tofile(sys.argv[1]+'/bucket_identity.bin')" "$D"
[ "$OUT/pp_aswiss" -nt "$A/pp_aswiss.cpp" ] || g++ -O3 -std=c++17 "$A/pp_aswiss.cpp" -o "$OUT/pp_aswiss"
EXE=$OUT/pp_aswiss; ESD=$A/esd_table.csv
run() {  # name seed args...
  local name=$1 seed=$2; shift 2; local d=$OUT/runs/${name}_s$seed
  [ -f "$d/exploitability.csv" ] && return 0
  mkdir -p "$d"
  "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" --esd-table "$ESD" "$@" 2>> "$d/stderr.log"
  python3 "$A/eval.py" "$D" "$d" > /dev/null
  # pp runs: the share of information sets whose prior pointer differs from their own route (needs state.bin)
  case " $* " in *" --mode pp "*) "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" --esd-table "$ESD" "$@" --analyze 3000 2>> "$d/stderr.log" || true ;; esac
  # disk: keep the last row's policies and drop the checkpoint (a finished run is never resumed)
  local last=$(( $(grep -c '^[0-9]' "$d/log.csv") - 1 ))
  find "$d" -name 'policy_*.bin' ! -name "policy_*_$last.bin" -delete
  rm -f "$d/state.bin" "$d/state.bin.tmp"
}
export -f run; export OUT D EXE ESD SECS A
jobs=$OUT/jobs.txt; : > "$jobs"
for s in $SEEDS; do
  echo "ppeq $s --mode pp --prior-start equal --measure 1" >> "$jobs"
  echo "ppeq_a $s --mode pp --prior-start equal --outlier label --measure 1" >> "$jobs"
  echo "pure $s --mode pure --measure 1" >> "$jobs"
  echo "pure_a $s --mode pure --outlier label --measure 1" >> "$jobs"
  echo "pplen $s --mode pp --prior-start length --measure 1" >> "$jobs"
  echo "ppeq_fc $s --mode pp --prior-start equal --scale fact-counterfact --measure 1" >> "$jobs"
  echo "ppeq_a_r4 $s --mode pp --prior-start equal --outlier label --prior-ratio 4 --crossfit 0 --measure 1" >> "$jobs"
  echo "ppeq_r4 $s --mode pp --prior-start equal --prior-ratio 4 --crossfit 0 --measure 1" >> "$jobs"
  echo "ppeq_a_r4_rn $s --mode pp --prior-start equal --outlier label --prior-ratio 4 --crossfit 0 --resolve-now 1 --measure 1" >> "$jobs"
  echo "ppeq_a_r4_rnlab $s --mode pp --prior-start equal --outlier label --prior-ratio 4 --crossfit 0 --resolve-now 1 --prior-criterion label --measure 1" >> "$jobs"
  echo "ppeq_a_r4_lab $s --mode pp --prior-start equal --outlier label --prior-ratio 4 --crossfit 0 --prior-criterion label --measure 1" >> "$jobs"
  echo "mccfr $s --mode pure --p0 $D/bucket_identity.bin --split 0" >> "$jobs"
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
# k-means with the pp-equal run's final bucket counts per time step (same seed)
: > "$jobs"
for s in $SEEDS; do
  for src in ppeq ppeq_a ppeq_a_r4; do
    k=kmeans${src#ppeq}; m=$OUT/runs/${k}_s$s; mkdir -p "$m"
    [ -f "$m/km_map.bin" ] || python3 "$A/kmeans.py" "$D" "$OUT/runs/${src}_s$s/final_bucket_map.bin" "$m/km_map.bin" "$s" > "$m/kmeans.log"
    echo "$k $s --mode pure --p0 $m/km_map.bin --split 0" >> "$jobs"
  done
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
arm() { local n=$1; local l=""; for s in $SEEDS; do l="$l,$OUT/runs/${n}_s$s"; done; echo "${l#,}"; }
# the spec's default split rule (generalized ESD on counterfactual regrets) and rule (a) (significance-gated argmax label)
python3 "$A/plot_pp.py" "$OUT/pp_small_esd" --arm "pp-equal=$(arm ppeq)" --arm "pp-length=$(arm pplen)" --arm "pure=$(arm pure)" \
  --arm "pp-equal fact-counterfact=$(arm ppeq_fc)" --arm "pp-equal, prior x4=$(arm ppeq_r4)" --ref "k-means=$(arm kmeans)" \
  --ref "MCCFR (no abstraction)=$(arm mccfr)" --kmeans k-means
python3 "$A/plot_pp.py" "$OUT/pp_small_a" --arm "pp-equal, rule a=$(arm ppeq_a)" --arm "pp-equal, rule a, prior x4=$(arm ppeq_a_r4)" \
  --arm "pure, rule a=$(arm pure_a)" --arm "x4 + resolve now=$(arm ppeq_a_r4_rn)" --arm "x4 + resolve now + label=$(arm ppeq_a_r4_rnlab)" \
  --arm "x4 + label=$(arm ppeq_a_r4_lab)" --ref "k-means (rule a counts)=$(arm kmeans_a)" \
  --ref "k-means (rule a x4 counts)=$(arm kmeans_a_r4)" --ref "MCCFR (no abstraction)=$(arm mccfr)" --kmeans "k-means (rule a counts)"
echo "done: $OUT/pp_small.md"
