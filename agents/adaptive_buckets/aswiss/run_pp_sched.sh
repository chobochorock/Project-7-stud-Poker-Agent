#!/bin/bash
# The split schedule on mini-stud small: judgments from iteration 10,000 instead of 1,000 (user request 2026-11-12).
#   bash run_pp_sched.sh OUT SECONDS [SEEDS] [PARALLEL]     (OUT = the run_pp.sh output, whose tree and runs it reuses)
#   (i)  static schedule from 10,000: judgments at 10,000 * 2^k              (--epoch0 10000)
#   (ii) eps schedule, eps = the statistic at iteration 10,000, where the first judgment is (--schedule eps --eps-at 10000)
# Arms (3 seeds each): pure with rule (a); pp-equal with rule (a), prior x4, no cross-fit, resolve-now and the label
# criterion (the best pp of run_pp.sh); pp-equal with the spec's defaults (ESD). The runs of run_pp.sh (judgments from
# 1,000) are the reference. Writes OUT/pp_small_sched_a_* and OUT/pp_small_sched_esd_*.
set -euo pipefail
OUT=$(cd "$1" && pwd); SECS=$2; SEEDS=${3:-"1 2 3"}; P=${4:-4}
A=$(cd "$(dirname "$0")" && pwd)
D=$OUT/tree
[ -f "$D/obs.bin" ] || { echo "run run_pp.sh first (tree)"; exit 1; }
[ "$OUT/pp_aswiss" -nt "$A/pp_aswiss.cpp" ] || g++ -O3 -std=c++17 "$A/pp_aswiss.cpp" -o "$OUT/pp_aswiss"
EXE=$OUT/pp_aswiss; ESD=$A/esd_table.csv
run() {  # name seed args... (as run_pp.sh)
  local name=$1 seed=$2; shift 2; local d=$OUT/runs/${name}_s$seed
  [ -f "$d/exploitability.csv" ] && return 0
  mkdir -p "$d"
  "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" --esd-table "$ESD" "$@" 2>> "$d/stderr.log"
  python3 "$A/eval.py" "$D" "$d" > /dev/null
  case " $* " in *" --mode pp "*) "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" --esd-table "$ESD" "$@" --analyze 3000 2>> "$d/stderr.log" || true ;; esac
  local last=$(( $(grep -c '^[0-9]' "$d/log.csv") - 1 ))
  find "$d" -name 'policy_*.bin' ! -name "policy_*_$last.bin" -delete
  rm -f "$d/state.bin" "$d/state.bin.tmp"
}
export -f run; export OUT D EXE ESD SECS A
MIX="--mode pp --prior-start equal --outlier label --prior-ratio 4 --crossfit 0 --resolve-now 1 --prior-criterion label --measure 1"
jobs=$OUT/jobs_sched.txt; : > "$jobs"
for s in $SEEDS; do
  for sch in "e10k:--epoch0 10000" "eps10k:--schedule eps --eps-at 10000"; do
    tag=${sch%%:*}; args=${sch#*:}
    echo "pure_a_$tag $s --mode pure --outlier label --measure 1 $args" >> "$jobs"
    echo "ppmix_$tag $s $MIX $args" >> "$jobs"
    echo "ppeq_$tag $s --mode pp --prior-start equal --measure 1 $args" >> "$jobs"
  done
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
arm() { local n=$1; local l=""; for s in $SEEDS; do l="$l,$OUT/runs/${n}_s$s"; done; echo "${l#,}"; }
python3 "$A/plot_pp.py" "$OUT/pp_small_sched_a" --arm "pure a, from 1k=$(arm pure_a)" --arm "pure a, static from 10k=$(arm pure_a_e10k)" \
  --arm "pure a, eps at 10k=$(arm pure_a_eps10k)" --arm "pp mix, from 1k=$(arm ppeq_a_r4_rnlab)" \
  --arm "pp mix, static from 10k=$(arm ppmix_e10k)" --arm "pp mix, eps at 10k=$(arm ppmix_eps10k)" \
  --ref "k-means (rule a counts)=$(arm kmeans_a)" --ref "MCCFR (no abstraction)=$(arm mccfr)" --kmeans "k-means (rule a counts)"
python3 "$A/plot_pp.py" "$OUT/pp_small_sched_esd" --arm "pp-equal ESD, from 1k=$(arm ppeq)" \
  --arm "pp-equal ESD, static from 10k=$(arm ppeq_e10k)" --arm "pp-equal ESD, eps at 10k=$(arm ppeq_eps10k)" \
  --ref "k-means=$(arm kmeans)" --ref "MCCFR (no abstraction)=$(arm mccfr)" --kmeans k-means
echo "done: $OUT/pp_small_sched_a.md"
