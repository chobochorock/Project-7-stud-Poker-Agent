# job.sh GAME NAME PROG : one run of run_walltime.sh (same settings), skipped when exploitability.csv exists;
# a partial run left by a container restart is moved aside (NAME.lost_<time>), never reused
set -euo pipefail
SP=/tmp/claude-0/-home-user-Project-7-stud-Poker-Agent/b14be346-5af4-5c12-a577-98f801293c67/scratchpad; A=/home/user/Project-7-stud-Poker-Agent/agents/adaptive_buckets/aswiss
GAME=$1 name=$2 prog=$3; OUT=$SP/wt/$GAME; D=$OUT/tree; R=$OUT/runs_s1; d=$R/$name
[ -f "$d/exploitability.csv" ] && { echo "$GAME $name already done"; exit 0; }
SECS=1800; SEED=1
case $GAME in kuhn) EPISODES=2000; SECS=300 ;; leduc) EPISODES=1300 ;; *) EPISODES=1000 ;; esac
TAB="--mode es --plain 1 --iters 1000000000000 --epoch0 1000 --z 2 --seed $SEED --seconds $SECS"
MMD="--algo mmd --seed $SEED --seconds $SECS --episodes-total 1000000000000 --episodes $EPISODES --epochs 30 --minibatch 128"
MMD+=" --lr 5e-5 --value-lr 5e-5 --schedule sqrt --alpha 0.05 --eta 0.05 --horizon 10000000 --clip 0.3 --gamma 0.99"
MMD+=" --hidden1 128 --hidden2 128 --reward-scale 1"
ESC="--algo escher --seed $SEED --seconds $SECS --iterations 1000000000 --value-hands 1000 --traversals 1000 --batch 2048"
ESC+=" --regret-steps 500 --value-steps 500 --avg-steps 10000 --hidden1 256 --hidden2 128 --capacity 100000 --lr 1e-3"
ESC+=" --value-lr 1e-3 --reward-scale 1 --reinit 1 --value-reinit 1 --value-explore 0.01 --fallback argmax"
case $name in
  aswiss) ARGS="--buckets $D/bucket_timestep.bin --split cum --label window $TAB" ;;
  mccfr) ARGS="--buckets $D/bucket_identity.bin --split none $TAB" ;;
  kmeans) mkdir -p $R/kmeans_map
          [ -f $R/kmeans_map/km_map.bin ] || python3 $A/kmeans.py $D $R/aswiss/final_bucket_map.bin $R/kmeans_map/km_map.bin $SEED > $R/kmeans_map/kmeans.log
          ARGS="--buckets $R/kmeans_map/km_map.bin --split none $TAB" ;;
  mmd) ARGS="$MMD" ;;
  escher) ARGS="$ESC" ;;
esac
[ -d "$d" ] && mv "$d" "$d.lost_$(date +%s)"
mkdir -p "$d"
"$OUT/$prog" --tree "$D" --out "$d" $ARGS 2> "$d/stderr.log"
python3 "$A/eval.py" "$D" "$d" > "$d/eval.log" 2>&1
echo "$GAME $name done $(date -u +%H:%M)"
