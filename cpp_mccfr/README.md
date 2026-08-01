# C++ MCCFR v3 experiment

The planned 2-player Deep CFR trunk and safe 7th-street AsymP resolver are
specified in [`../DEEP_CFR_7STUD_ASYMP.md`](../DEEP_CFR_7STUD_ASYMP.md).

This folder is isolated from the Python runtime and implements the current
two-player EV experiment:

- betting rules v3: no 4th betting, per-player raise caps `1/2/3`
- the existing heuristic before the selected MCCFR start street
- online MCCFR or MCCFR+ starting on 5th, 6th, or 7th street
- paired-seat evaluation against the heuristic
- compact bucket keys and bucket growth diagnostics
- frozen Hellinger power centroids
- `power-recall`: current power bucket plus exact betting history; prior-street
  power buckets are deliberately omitted
- `power-range`: current power bucket plus a bounded 64-bin Bayesian summary
  of opponent actions under a frozen teacher policy
- optional periodically refreshed regret-based pruning

Build and check:

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_mccfr.cpp -o cpp_mccfr\stud_mccfr.exe
.\cpp_mccfr\stud_mccfr.exe --self-test
```

## Deep CFR IPC probe

The first C++/PyTorch integration uses batched localhost TCP. Build the probe:

```powershell
g++ -O3 -std=c++17 cpp_mccfr\deep_cfr_ipc_probe.cpp `
  -o cpp_mccfr\deep_cfr_ipc_probe.exe -lws2_32
```

Start the PyTorch server in one terminal:

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -B deep_cfr_ipc_server.py --port 28731 --hidden 256 --layers 2 --threads 1
```

Then send batches of real 5th-street states from another terminal:

```powershell
.\cpp_mccfr\deep_cfr_ipc_probe.exe `
  --port 28731 --batches 100 --batch-size 64 `
  --ante 1000 --stack-ante 1000 --seed 73
```

The server also accepts `--model checkpoint.pt` for a TorchScript module with
shape `[batch, 1832] -> [batch, 8]`.

## 7th-street Deep CFR trainer

This is the first real Deep CFR path, not the IPC throughput probe. H4 and the
5th/6th streets use the existing heuristic; C++ external-sampling traversals
start from reached 7th-street states. PyTorch trains two advantage networks and
one average-policy network from reservoir samples.

Build the traversal generator and evaluator:

```powershell
g++ -O3 -std=c++17 cpp_mccfr\deep_cfr_traverse.cpp `
  -o cpp_mccfr\deep_cfr_traverse.exe -lws2_32
g++ -O3 -std=c++17 cpp_mccfr\deep_cfr_evaluate.cpp `
  -o cpp_mccfr\deep_cfr_evaluate.exe -lws2_32
```

Short pipeline check:

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -B train_deep_cfr_7th.py `
  --run-dir models\deep_cfr_7th_smoke `
  --iterations 2 --traversals 10 --memory-capacity 1000 `
  --hidden 64 --layers 2 --batch-size 64 `
  --advantage-steps 20 --policy-steps 40 --threads 1 --build
```

First substantive run:

```powershell
& $py -B train_deep_cfr_7th.py `
  --run-dir models\deep_cfr_7th_v1 `
  --iterations 20 --traversals 1000 --memory-capacity 50000 `
  --hidden 256 --layers 2 --batch-size 512 `
  --advantage-steps 500 --policy-steps 1000 `
  --threads 1 --seed 17001
```

Resume with the same architecture and reservoir capacity while increasing the
target iteration:

```powershell
& $py -B train_deep_cfr_7th.py `
  --run-dir models\deep_cfr_7th_v1 `
  --iterations 50 --traversals 1000 --memory-capacity 50000 `
  --hidden 256 --layers 2 --batch-size 512 `
  --advantage-steps 500 --policy-steps 1000 `
  --threads 1 --seed 17001 --resume
```

Evaluate the average policy against the heuristic with paired seats:

```powershell
& $py -B evaluate_deep_cfr_7th.py `
  --model models\deep_cfr_7th_v1\policy.pt `
  --hands 10000 --threads 1 --seed 27001 --build
```

The run directory contains `checkpoint.pt`, `reservoirs.npz`, both advantage
TorchScript models, and `policy.pt`. The bundled PyTorch runtime is CPU-only;
`--device cuda` requires a CUDA-enabled PyTorch installation using a compatible
Python interpreter.

## Conditional-participation LBR baselines

Evaluate deterministic baselines without loading an MCCFR model. `fold`
always folds and therefore loses exactly one ante. `made-call` checks when
free and calls only with a current made-hand category at or above the
threshold. `made-bet` additionally opens for the minimum legal bet.

Category values are `0=high card`, `1=pair`, `2=two pair`, `3=trips`, through
`8=straight flush`.

```powershell
.\cpp_mccfr\stud_mccfr_conditional.exe --opponent policy-lbr --lbr-target fold --start-street 5 --iterations 0 --hands 10000 --ante 1000 --belief-particles 64
.\cpp_mccfr\stud_mccfr_conditional.exe --opponent policy-lbr --lbr-target made-call --made-min-category 1 --start-street 5 --iterations 0 --hands 10000 --ante 1000 --belief-particles 64
.\cpp_mccfr\stud_mccfr_conditional.exe --opponent policy-lbr --lbr-target made-bet --made-min-category 1 --start-street 5 --iterations 0 --hands 10000 --ante 1000 --belief-particles 64
```

## Posterior-range experiment

`power-range` learns a tiny table `P(action | street, final hand category)`
from teacher self-play. At lookup time it replays only the opponent's public
actions, updates a nine-category posterior, and quantizes it to at most 64
range buckets. It does not store exact betting history.

Create the range model and a warm-started policy:

```powershell
.\cpp_mccfr\stud_mccfr_range.exe `
  --bucket power-range `
  --load-atlas cpp_mccfr\power128_selfplay100m_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --fit-range-hands 20000 `
  --save-range cpp_mccfr\action_range_teacher100m_v1.bin `
  --imitate-from cpp_mccfr\root_mccfr_ante1000_100m.bin `
  --imitate-atlas cpp_mccfr\power64_v1.bin `
  --imitation-roots 20000 --imitation-strength 1000 `
  --hands 20000 --iterations 0 --opponent heuristic `
  --save cpp_mccfr\posterior_range_imitation20k.bin
```

Fine-tune it with MCCFR:

```powershell
.\cpp_mccfr\stud_mccfr_range.exe `
  --bucket power-range `
  --load-atlas cpp_mccfr\power128_selfplay100m_v1.bin `
  --load-range cpp_mccfr\action_range_teacher100m_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --load cpp_mccfr\posterior_range_imitation20k.bin `
  --root-iterations 10000 --root-report-every 5000 `
  --hands 20000 --iterations 0 --opponent heuristic `
  --save cpp_mccfr\posterior_range_imitation20k_ft10k.bin
```

## Five-player stack-conditioned experiment

`stud5_mccfr.cpp` trains only the 7th-street policy with five-player
external-sampling MCCFR+. The 5th and 6th streets use the existing heuristic.
Each root independently samples all five stacks log-uniformly from 50 to 1000
ante. A public-card-aware Monte Carlo power vector and the relative five-stack
vector are each assigned to a frozen hard k-means cluster.

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud5_mccfr.cpp -o cpp_mccfr\stud5_mccfr.exe
.\cpp_mccfr\stud5_mccfr.exe --self-test

.\cpp_mccfr\stud5_mccfr.exe `
  --root-iterations 10000 `
  --report-every 1000 `
  --eval-deals 1000 `
  --fit-roots 2000 `
  --clusters 64 `
  --stack-clusters 16 `
  --mc-samples 16 `
  --stack-min 50 `
  --stack-max 1000 `
  --seed 41103 `
  --save cpp_mccfr\stud5_hard_stack_compact_10k.bin
```

Continue the same regret table from 10k to a cumulative 100k roots:

```powershell
.\cpp_mccfr\stud5_mccfr.exe `
  --load cpp_mccfr\stud5_hard_stack_compact_10k.bin `
  --root-iterations 90000 `
  --report-every 5000 `
  --eval-deals 1000 `
  --fit-roots 2000 `
  --clusters 64 `
  --stack-clusters 16 `
  --mc-samples 16 `
  --stack-min 50 `
  --stack-max 1000 `
  --seed 41105 `
  --save cpp_mccfr\stud5_hard_stack_compact_100k.bin
```

Evaluate a frozen model with a new seed. `--eval-deals 5000` means 5,000
duplicate deals and 25,000 hands because the target rotates through all five
seats.

```powershell
.\cpp_mccfr\stud5_mccfr.exe `
  --load cpp_mccfr\stud5_hard_stack_compact_100k.bin `
  --root-iterations 0 `
  --eval-deals 5000 `
  --fit-roots 1 `
  --mc-samples 16 `
  --seed 41999
```

This is a five-player heuristic-field benchmark, not exploitability or a
multiplayer Nash-convergence result.

Fit and freeze a 64-centroid power atlas:

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power `
  --fit-hands 5000 `
  --clusters 64 `
  --power-samples 128 `
  --save-atlas cpp_mccfr\power64_v1.bin `
  --start-street 7 `
  --algorithm mccfr-plus `
  --hands 200 `
  --iterations 0 `
  --seed 7
```

Train from 7th street:

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 7 `
  --algorithm mccfr-plus `
  --hands 10000 `
  --iterations 16 `
  --report-every 1000 `
  --seed 7 `
  --save cpp_mccfr\mccfr_plus_v3.bin
```

Expand the same experiment to 5th street:

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr-plus `
  --hands 1000 `
  --iterations 4 `
  --seed 17 `
  --save cpp_mccfr\mccfr_plus_5th.bin
```

Evaluate a frozen table with zero online traversals:

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 7 `
  --algorithm mccfr-plus `
  --load cpp_mccfr\mccfr_plus_v3.bin `
  --hands 10000 `
  --iterations 0 `
  --seed 1007
```

### Batched partial AsymP

Warm-start from an MCCFR table and average external-sampling gradients over
multiple chance roots. Each batch preserves the AsymP alternating order:
update the perturbed player first, then update its opponent against the new
policy.

```powershell
.\cpp_mccfr\stud_mccfr_partial_asymp.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm asymp `
  --ante 1000 `
  --init-from cpp_mccfr\root_mccfr_ante1000_100m.bin `
  --root-iterations 10000 `
  --asymp-batch-roots 64 `
  --asymp-step 0.0005 `
  --asymp-mu 0.01 `
  --save cpp_mccfr\partial_asymp_100m_10k_b64.bin `
  --hands 2 `
  --iterations 0 `
  --seed 31011
```

Evaluate its approximate exploitability lower bound with the same policy-LBR
used for MCCFR:

```powershell
.\cpp_mccfr\stud_mccfr_partial_asymp.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm asymp `
  --ante 1000 `
  --load cpp_mccfr\partial_asymp_100m_10k_b64.bin `
  --hands 5000 `
  --iterations 0 `
  --opponent policy-lbr `
  --belief-particles 64 `
  --seed 31002
```

Evaluate the existing table with local Gaussian responsibilities. When
`--soft-top-p` is set, it keeps the smallest set of clusters whose cumulative
responsibility reaches the requested mass; `--soft-top-k` is ignored.

```powershell
.\cpp_mccfr\stud_mccfr_adaptive.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --load cpp_mccfr\root_mccfr_ante1000_10m.bin `
  --soft-top-p 0.99 `
  --soft-temperature 1 `
  --soft-local-bandwidth `
  --hands 10000 `
  --iterations 0 `
  --opponent policy-lbr `
  --belief-particles 32 `
  --seed 9401
```

Grow append-only clusters from the 10M checkpoint. Each cycle first tunes only
the local bandwidth scales, then adds one centroid when the parent cluster's
mean surrogate regret exceeds the threshold. For every existing betting
context on that street, the new cluster node is initialized from the current
responsibility-weighted mixed policy. A final temperature pass runs after the
last addition.

```powershell
.\cpp_mccfr\stud_mccfr_adaptive.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --load cpp_mccfr\root_mccfr_ante1000_10m.bin `
  --soft-top-p 0.99 `
  --soft-temperature 1 `
  --soft-local-bandwidth `
  --temperature-calibration-roots 20000 `
  --temperature-min-samples 100 `
  --cluster-growth-steps 3 `
  --cluster-growth-threshold 0.25 `
  --save cpp_mccfr\root_mccfr_ante1000_10m_adaptive.bin `
  --save-atlas cpp_mccfr\power64_10m_adaptive.bin `
  --save-temperatures cpp_mccfr\temperature_10m_adaptive.txt `
  --hands 2 `
  --iterations 0 `
  --seed 9402
```

Always load those three adaptive outputs together. Existing cluster IDs are
preserved and new IDs are appended. Physical merge and deletion are not
implemented.

Train soft policies from an empty regret table. All soft modes distribute each
MCCFR update over the selected local experts by Gaussian responsibility.

- `fixed`: keep the atlas and temperatures fixed. This is the soft control.
- `mix`: grow a mixed-strategy expert.
- `simple`: grow a one-action expert.

In adaptive modes, every adaptation interval the largest cluster-average
one-step regret either adds one centroid or, when it is below the threshold,
tunes local temperature.

Use `--soft-adapt-round-robin` to give streets 5, 6, and 7 separate adaptation
turns. A decreasing growth threshold can delay splitting until temperature
tuning has had a chance to improve the existing mixture.

```text
--cluster-growth-threshold 200
--cluster-growth-threshold-decay 0.9
--cluster-growth-threshold-min 1
--soft-adapt-round-robin
```

At adaptation step `j`, the active threshold is
`max(minimum, initial * decay^j)`.

Temperature-free mass growth uses a fixed 10% split of the selected parent's
state mass. A `POWERAT1` atlas loads with uniform masses; newly saved atlases
use `POWERAT2`.

- `--soft-growth point`: append the single highest-regret state as centroid.
- `--soft-growth residual`: append the regret-weighted centroid of the top 64
  residual states in the selected parent.

Both modes keep local temperature scales fixed at one. The child starts with
the highest-regret action and subsequent policy changes come only from CFR.

```powershell
.\cpp_mccfr\stud_mccfr_mass_growth.exe `
  --bucket power `
  --load-atlas cpp_mccfr\soft_seed8.bin `
  --save-atlas cpp_mccfr\mass_point_10k_atlas.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --soft-top-p 0.99 `
  --soft-temperature 1 `
  --soft-local-bandwidth `
  --soft-growth point `
  --soft-adapt-every 10000 `
  --cluster-growth-threshold 1 `
  --root-iterations 10000 `
  --save cpp_mccfr\mass_point_10k.bin `
  --hands 2 `
  --iterations 0 `
  --seed 16101
```

- `mix`: initialize the new expert from all positive action regrets.
- `simple`: initialize it as a one-hot policy on the largest-regret action.

```powershell
.\cpp_mccfr\stud_mccfr_soft_growth.exe `
  --bucket power `
  --load-atlas cpp_mccfr\soft_seed8.bin `
  --save-atlas cpp_mccfr\soft_mix_100k_atlas.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --soft-top-p 0.99 `
  --soft-temperature 1 `
  --soft-local-bandwidth `
  --soft-growth mix `
  --soft-adapt-every 10000 `
  --cluster-growth-threshold 1 `
  --temperature-min-samples 100 `
  --root-iterations 100000 `
  --root-report-every 10000 `
  --save cpp_mccfr\soft_mix_100k.bin `
  --save-temperatures cpp_mccfr\soft_mix_100k_temp.txt `
  --hands 2 `
  --iterations 0 `
  --seed 12101
```

Use `--soft-growth simple` and separate output paths for the one-action
variant. The model, atlas, and temperature file are one checkpoint and must
always be loaded together. The reported adaptation regret is a sampled
one-step action-gap surrogate, not formal exploitability.

For equal-compute comparisons, use a node-visit budget instead of a root
count. The final traversal may overshoot the requested budget by one root.

```powershell
.\cpp_mccfr\stud_mccfr_soft_growth.exe `
  --bucket power `
  --load-atlas cpp_mccfr\soft_seed8.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --soft-growth fixed `
  --soft-top-p 0.99 `
  --soft-temperature 1 `
  --soft-local-bandwidth `
  --root-node-budget 10000000 `
  --save cpp_mccfr\soft_fixed_10m_nodes.bin `
  --hands 2 `
  --iterations 0 `
  --seed 12101
```

Optional cold pruning:

```text
--prune-after 128 --prune-threshold 200 --prune-refresh 64
```

Pruning is off by default because the first experiment made play faster but
reduced EV.

## Five-player EV transfer

The frozen heads-up table can be evaluated in the existing five-player
equal-stack EV environment:

```powershell
python -B evaluate_five_player_ev.py `
  --model cpp_mccfr\root_mccfr_current_snapshot.bin `
  --target-count 1 `
  --deals 1000 `
  --ante 1000
```

Each deal is replayed with the MCCFR target rotated through all five seats.
The confidence interval uses deal-level averages, so the five rotations are
not incorrectly treated as independent samples. `--target-count 2`, `3`, or
`4` changes the field composition.

This is transfer evaluation, not five-player CFR. At each decision the
heads-up table sees the strongest visible active opponent as its representative;
the actual five-player pot, call amount, stack, legal actions, and public action
history are retained.

Unequal cash stacks can be sampled log-uniformly:

```powershell
python -B evaluate_five_player_ev.py `
  --exe cpp_mccfr\stud_mccfr_multi_stack.exe `
  --model cpp_mccfr\root_mccfr_ante1000_10m.bin `
  --target-count 1 `
  --deals 2000 `
  --ante 1000 `
  --stack-min-ante 50 `
  --stack-max-ante 1000
```

The five player caps are fixed across the five seat rotations of each deal.
The C++ projection receives separate own and representative-opponent caps.
This remains a frozen heads-up-policy transfer test, not stack-conditioned
five-player training.

The same frozen checkpoint can be tested at shorter heads-up effective stacks:

```powershell
python -B evaluate_stack_sensitivity.py `
  --model cpp_mccfr\root_mccfr_current_snapshot.bin `
  --stacks 20 50 100 200 `
  --deals 5000 `
  --ante 1000
```

Each base deal is played twice with the MCCFR policy in both seats. Confidence
intervals are computed from the paired deal averages.

Train a dedicated 20-ante policy:

```powershell
.\cpp_mccfr\stud_mccfr_stack.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --stack-ante 20 `
  --root-iterations 10000 `
  --hands 2 `
  --iterations 0 `
  --seed 31001 `
  --save cpp_mccfr\root_mccfr_stack20_10k.bin
```

`--stack-ante` defaults to `1000`, preserving previous behavior.

The binary model is intentionally local and compiler-dependent. It is a fast
checkpoint for this experiment, not a public interchange format.

See `BUCKET_GROWTH.md` for the exact bucket key and why the table grows faster
than the traversal counter. See `IMPLEMENTATION_GUIDE.md` for the minimum
environment/simulation path to reimplement yourself.
