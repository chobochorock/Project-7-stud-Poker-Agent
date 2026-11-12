#!/bin/bash
# Direction and leaf start on mini-stud small (user request 2026-11-12): the best pp setting (rule (a), prior x4, no
# cross-fit, resolve-now, label criterion) with --direction iterates|avg and --leaf-start root|parent, and pure rule (a)
# with --direction avg. Reuses the tree and runs of run_pp.sh (the iterates/root pp arm is ppeq_a_r4_rnlab).
#   bash run_pp_dir.sh OUT SECONDS [SEEDS] [PARALLEL]
set -euo pipefail
OUT=$(cd "$1" && pwd); SECS=$2; SEEDS=${3:-"1 2 3"}; P=${4:-4}
A=$(cd "$(dirname "$0")" && pwd)
D=$OUT/tree
[ -f "$D/obs.bin" ] || { echo "run run_pp.sh first (tree)"; exit 1; }
[ "$OUT/pp_aswiss" -nt "$A/pp_aswiss.cpp" ] || g++ -O3 -std=c++17 "$A/pp_aswiss.cpp" -o "$OUT/pp_aswiss"
EXE=$OUT/pp_aswiss; ESD=$A/esd_table.csv
run() {  # name seed args... (as run_pp.sh; --analyze only where the genealogy is kept)
  local name=$1 seed=$2; shift 2; local d=$OUT/runs/${name}_s$seed
  [ -f "$d/exploitability.csv" ] && return 0
  mkdir -p "$d"
  "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" --esd-table "$ESD" "$@" 2>> "$d/stderr.log"
  python3 "$A/eval.py" "$D" "$d" > /dev/null
  case " $* " in *" --leaf-start parent "*) ;; *" --mode pp "*) "$EXE" --tree "$D" --out "$d" --seconds "$SECS" --seed "$seed" --esd-table "$ESD" "$@" --analyze 3000 2>> "$d/stderr.log" || true ;; esac
  local last=$(( $(grep -c '^[0-9]' "$d/log.csv") - 1 ))
  find "$d" -name 'policy_*.bin' ! -name "policy_*_$last.bin" -delete
  rm -f "$d/state.bin" "$d/state.bin.tmp"
}
export -f run; export OUT D EXE ESD SECS A
MIX="--mode pp --prior-start equal --outlier label --prior-ratio 4 --crossfit 0 --resolve-now 1 --prior-criterion label --measure 1"
jobs=$OUT/jobs_dir.txt; : > "$jobs"
for s in $SEEDS; do
  echo "ppmix_avg_root $s $MIX --direction avg" >> "$jobs"
  echo "ppmix_it_parent $s $MIX --leaf-start parent" >> "$jobs"
  echo "ppmix_avg_parent $s $MIX --direction avg --leaf-start parent" >> "$jobs"
  echo "pure_a_avg $s --mode pure --outlier label --direction avg --measure 1" >> "$jobs"
done
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$jobs"
arm() { local n=$1; local l=""; for s in $SEEDS; do l="$l,$OUT/runs/${n}_s$s"; done; echo "${l#,}"; }
python3 "$A/plot_pp.py" "$OUT/pp_small_dir" --arm "pp mix, iterates, root=$(arm ppeq_a_r4_rnlab)" \
  --arm "pp mix, iterates, parent=$(arm ppmix_it_parent)" --arm "pp mix, avg, root=$(arm ppmix_avg_root)" \
  --arm "pp mix, avg, parent=$(arm ppmix_avg_parent)" --arm "pure a, iterates=$(arm pure_a)" --arm "pure a, avg=$(arm pure_a_avg)" \
  --ref "k-means (rule a counts)=$(arm kmeans_a)" --ref "MCCFR (no abstraction)=$(arm mccfr)" --kmeans "k-means (rule a counts)"
echo "done: $OUT/pp_small_dir.md"
