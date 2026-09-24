# C++ MCCFR v3 experiment

The 7-stud epoch-weighted LightGBM regret-ensemble comparison with hard-256
MCCFR, per-hand Local BR-gap estimates, and lossless disk-backed residuals is
kept separately in [`../lightgbm_regret_ensemble/`](../lightgbm_regret_ensemble/README.md).

> **정리·확장성·알려진 결함은 [`REFACTORING.md`](REFACTORING.md).** 출력물은
> `logs/`·`results/` 로 이동됨. 새 프로브/에이전트는 모놀리스 수정 없이
> `#define STUD_MCCFR_NO_MAIN` + `#include "stud_mccfr.cpp"` 로 추가 가능.
> ⚠️ **빌드 시 `-static-libstdc++ -static-libgcc` 를 붙일 것.** 이 환경의
> libstdc++ DLL 이 깨져 있어, 없이 빌드하면 모든 exe 가 첫 iostream 출력에서
> 즉시 segfault 한다(코드 문제 아님). 자세한 내용은 REFACTORING.md §4.

The planned 2-player Deep CFR trunk and safe 7th-street AsymP resolver are
specified in [`../DEEP_CFR_7STUD_ASYMP.md`](../DEEP_CFR_7STUD_ASYMP.md).

The reward-audited, regret-preserving local bucket split proposed for the
power atlas is specified in [`ONLINE_LOCAL_BUCKET_SPLIT.md`](ONLINE_LOCAL_BUCKET_SPLIT.md).

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
- `power-tree`: fixed base power centroids followed by a hard local binary
  route tree with optional regret-aware online split audits
- optional periodically refreshed regret-based pruning

Build and check:

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_mccfr.cpp -o cpp_mccfr\stud_mccfr.exe
.\cpp_mccfr\stud_mccfr.exe --self-test
```

## Local power route tree

`--bucket power-tree` loads an existing `POWERAT1/2/3` atlas and promotes every
base centroid to a depth-zero route leaf. Base centroids never move. A local
split keeps the parent policy ID on child 0 and allocates one new policy ID for
child 1, so states in other base regions cannot be reassigned by the split.

Tree atlases are written as field-serialized `POWERAT5`. `POWERAT4` remains
loadable and starts every leaf cooldown counter at zero. Tree MCCFR models use
`MCCFRV6` and record the atlas generation checksum plus street-level base and
leaf counts. Loading a model with a different route-tree generation fails
instead of silently reinterpreting `power_cluster` IDs.

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power-tree `
  --load-atlas cpp_mccfr\power8_v1.bin `
  --start-street 5 --algorithm mccfr `
  --ante 1000 `
  --root-node-budget 100000000 `
  --root-report-every 1000 `
  --split-audit random `
  --split-audit-probability 0.01 `
  --split-rollouts 24 `
  --split-gap-threshold 0.04 `
  --split-gap-threshold-min 0.005 `
  --split-gap-touch-reference 100000 `
  --split-confidence-z 1.645 `
  --split-min-strong 8 `
  --split-min-touches 100000 `
  --split-reward-gain 0.20 `
  --split-reservoir 32 `
  --split-max-depth 8 `
  --split-max-leaves 128 `
  --save cpp_mccfr\power_tree_100m_nodes.bin `
  --save-atlas cpp_mccfr\power_tree_100m_nodes.atlas `
  --hands 2 --iterations 0 --seed 51001
```

The audit samples a bounded reservoir of visited states, estimates every legal
first action with common-random-number rollouts under the current average
policy, centers each action value by the current-policy value to form an
instantaneous regret vector, and accepts a split only when its one-sided
confidence bound and regret-vector SSE gain pass the configured thresholds. Atlas mutation happens
after both player traversals at the root boundary, at most once per paired root.
`--split-min-touches` defers an audit until the route leaf has received that
many weighted CFR visits. On split, the parent's cooldown touches are divided
between the children by the same observed child mass used for regret and
strategy mass; the sum is conserved. The strong-evidence threshold at leaf
touch count `N` is
`max(threshold_min, threshold * sqrt(reference / max(reference, N)))`.
Leaf-count and depth caps are checked before rollout work is scheduled.

Split progress metrics:

| Field | Meaning |
|---|---|
| `split_audits` | Completed action-value audit samples |
| `split_audit_node_visits` | Search nodes spent only on split audits |
| `split_touch_deferred` | Visits held back by the post-split cooldown |
| `split_capacity_deferred` | Visits skipped because that street hit its leaf cap |
| `split_depth_deferred` | Visits skipped because the route hit max depth |
| `split_gap_threshold_mean` | Mean touch-decayed strong-evidence threshold actually used |
| `split_gap_threshold_min_seen` | Smallest touch-decayed threshold used so far |
| `split_gap_threshold_max_seen` | Largest touch-decayed threshold used so far |
| `split_accepted_by_street` | Accepted splits for 5th/6th/7th |
| `split_inherited_touches` | Total parent cooldown touches conserved into accepted children |
| `split_last_child_touches` | Child 0/1 touch allocation at the latest accepted split |
| `split_leaves_by_street` | Current route leaves for 5th/6th/7th |
| `split_reward_rejected` | Fully collected candidates rejected by centered-regret split quality (legacy field name) |

Power-bucket assignment memoization is bounded with
`--power-cache-max N` (default `1000000`). The cache maps exact card
observations to their atlas leaf and is only a speed optimization; it is not
part of the learned model. When the limit is reached it is cleared and reused.
Use `--power-cache-max 0` to disable it. Progress JSON reports
`power_cache_entries`, `power_cache_limit`, `power_cache_hits`,
`power_cache_misses`, and `power_cache_resets`; `entries` must never exceed
`limit`. A smaller value such as `250000` is appropriate for long runs when
RAM matters more than recomputing occasional power vectors.

If every useful route leaf is already at its configured cap and no further
splits are expected, omit `--split-audit random`. This freezes the existing
tree while CFR training continues and avoids spending nodes on audits that can
only be deferred or rejected.

An accepted split preserves the exact parent mass by applying its observed
`q0/q1` partition to `regrets`, `raw_regrets`, `strategy_sum`, regret-node
`touches`, and route-leaf cooldown touches.
Audit rollout visits are included in `training_node_visits`, so use
`--root-node-budget` for fair static-vs-adaptive comparisons. A tree model and
its atlas are one checkpoint pair; `--split-audit random` therefore requires
both `--save` and `--save-atlas`.

`power-tree` is hard-only: global centroid append, soft/top-p assignment,
temperature calibration, imitation, and worker-delta merge are rejected.
The first implementation audits the determinized hidden state already sampled
by MCCFR. It does not yet resample hidden-hand particles from the policy
posterior, and the two checkpoint files are not written as one atomic
transaction.

## Deep CFR IPC probe

The staged public-belief search design is documented in
[`REBEL_7STUD_PLAN.md`](REBEL_7STUD_PLAN.md).
The executable 7th-street particle-PBS implementation and its commands are in
[`REBEL_7STUD_IMPLEMENTATION.md`](REBEL_7STUD_IMPLEMENTATION.md).
The first recursive stage, `V7 -> 6th depth-limited PBS search`, is documented
in [`REBEL_RECURSIVE_7STUD.md`](REBEL_RECURSIVE_7STUD.md).

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

## 5th-to-7th-street Deep CFR trainer

H4 uses the existing heuristic; C++ external-sampling traversals cover 5th,
6th, and 7th street by default. `--start-street 6|7` remains available for
smaller staged experiments. PyTorch trains two advantage networks and one
average-policy network from global-uniform reservoir samples. The model uses
separate card and betting-history branches followed by residual layers and
LayerNorm.

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
  --run-dir models\deep_cfr_5th_smoke --start-street 5 `
  --iterations 2 --traversals 10 --memory-capacity 1000 `
  --hidden 64 --layers 2 --batch-size 64 `
  --advantage-steps 20 --policy-steps 40 --threads 1 --build
```

First substantive run:

```powershell
& $py -B train_deep_cfr_7th.py `
  --run-dir models\deep_cfr_5th_plus_v2 --start-street 5 `
  --iterations 20 --traversals 1000 --memory-capacity 50000 `
  --hidden 128 --layers 2 --batch-size 512 `
  --advantage-steps 500 --policy-steps 1000 `
  --threads 1 --seed 17001
```

Resume with the same architecture and reservoir capacity while increasing the
target iteration:

```powershell
& $py -B train_deep_cfr_7th.py `
  --run-dir models\deep_cfr_5th_plus_v2 --start-street 5 `
  --iterations 50 --traversals 1000 --memory-capacity 50000 `
  --hidden 128 --layers 2 --batch-size 512 `
  --advantage-steps 500 --policy-steps 1000 `
  --threads 1 --seed 17001 --resume
```

Evaluate the average policy against the heuristic with paired seats:

```powershell
& $py -B evaluate_deep_cfr_7th.py `
  --model models\deep_cfr_5th_plus_v2\policy.pt --start-street 5 `
  --hands 10000 --threads 1 --seed 27001 --build
```

The run directory contains `checkpoint.pt`, `reservoirs.npz`, both advantage
TorchScript models, `policy.pt`, iteration checkpoints `policy_iN.pt`, and
`history.jsonl`. The bundled PyTorch runtime is CPU-only; `--device cuda`
requires a CUDA-enabled PyTorch installation using a compatible Python
interpreter. Corrected global reservoirs are intentionally incompatible with
the earlier street-stratified checkpoints.

Draw the in-training trajectory BR-gap history. This does not run LBR or any
other match evaluation:

```powershell
& $py -B analyze_deep_cfr_history.py `
  --run-dir models\deep_cfr_5th_plus_v2
```

Add `--hands 2000` for post-training heuristic checkpoint screens. LBR is
disabled by default; `--lbr-every N` explicitly enables the much more expensive
post-training Policy-LBR evaluation every `N` saved checkpoints. Neither
evaluation participates in training.

Each player generator records `trajectory_br_gap_mean_ante`, its standard
error, and `trajectory_br_gap_by_street_mean_ante`. The gap is accumulated per
training root from action values already computed by the external-sampling
traversal, so it adds no traversal or network query.

Evaluate one network directly with Policy-LBR:

```powershell
& $py -B evaluate_deep_cfr_7th.py `
  --model models\deep_cfr_5th_plus_v2\policy.pt --start-street 5 `
  --opponent policy-lbr --hands 10000 --belief-particles 240
```

See `DEEP_CFR_7STUD_AUDIT_2026-08-23_KO.md` for the implementation audit and
the corrected H128 experiment.

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

When fitting from an existing policy, `--fit-policy-epsilon E` replaces the
policy action with a uniformly sampled legal action with probability `E`.
This broadens trajectory coverage without changing later MCCFR training.

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power --fit-hands 100000 --clusters 128 `
  --power-samples 128 --fit-sample-cap 50000 `
  --fit-policy-model cpp_mccfr\root_mccfr_ante1000_100m.bin `
  --fit-policy-atlas cpp_mccfr\power64_v1.bin `
  --fit-policy-epsilon 0.1 `
  --save-atlas cpp_mccfr\power128_selfplay100m_eps10_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --hands 2 --iterations 0 --seed 37201
```

To fit from several behavior distributions, give each source a per-street
quota. Mixed fitting records each `(hand, street, seat)` once, so long betting
lines do not receive extra weight. LBR alternates seats between hands.

```powershell
.\cpp_mccfr\stud_mccfr_mixed_atlas.exe `
  --bucket power --fit-hands 500000 --clusters 256 `
  --power-samples 128 --fit-sample-cap 100000 `
  --fit-policy-model cpp_mccfr\root_mccfr_ante1000_100m.bin `
  --fit-policy-atlas cpp_mccfr\power64_v1.bin `
  --fit-lbr-model cpp_mccfr\made_call_pair_r1000_k256_eps20_1m.bin `
  --fit-lbr-atlas cpp_mccfr\power256_selfplay100m_eps20_v1.bin `
  --fit-lbr-sample-cap 50000 --fit-lbr-particles 64 `
  --fit-heuristic-sample-cap 50000 --fit-heuristic-epsilon 0.2 `
  --save-atlas cpp_mccfr\power256_mixed_self_lbr_epsheur_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --hands 2 --iterations 0 --seed 37401
```

For a frozen diagonal-Gaussian hard-EM atlas, the following uses equal
per-street quotas from epsilon-heuristic self-play and heuristic-vs-LBR. CFR
still assigns each state to exactly one bucket.

```powershell
.\cpp_mccfr\stud_mccfr_hard_em.exe `
  --bucket power --fit-hands 1000000 --clusters 512 `
  --power-samples 128 --fit-sample-cap 500000 `
  --fit-policy-epsilon 0.2 `
  --fit-lbr-target heuristic --fit-lbr-sample-cap 500000 `
  --fit-lbr-particles 64 --fit-clustering hard-em `
  --save-atlas cpp_mccfr\power512_hardem_epsheur_lbr50_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --hands 2 --iterations 0 --seed 40201
```

Street-specific cluster counts can follow the observed decision reach instead
of assigning the same resolution to every street. The following fits 16/64/256
power clusters for 5th/6th/7th street. Use `power-memory16` during training so
the final CFR key is the power cluster crossed with the cumulative past-street
betting summary.

```powershell
.\cpp_mccfr\stud_mccfr_street_clusters.exe `
  --bucket power-memory16 --fit-hands 250000 `
  --clusters-by-street 16,64,256 `
  --power-samples 128 --fit-sample-cap 250000 `
  --fit-heuristic-sample-cap 250000 --fit-heuristic-pool `
  --save-atlas cpp_mccfr\power16_64_256_pool500k_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --hands 2 --iterations 0 --seed 45101
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

### Exact-information 7th-street resolver

The evaluation-only resolver keeps the loaded table as the 5th/6th blueprint.
At each reached 7th-street root it runs CFR+ with exact observed cards and the
complete betting sequence. `--seventh-hand-history` additionally preserves
card arrival order instead of canonicalizing the current card sets.

```powershell
.\cpp_mccfr\stud_mccfr_seventh_resolver.exe `
  --bucket power-memory16 `
  --load-atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --start-street 5 --algorithm mccfr --ante 1000 `
  --load cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --seventh-resolve-iterations 32 --seventh-resolve-prior 100 `
  --seventh-hand-history `
  --hands 10000 --iterations 0 --opponent policy-lbr `
  --belief-particles 64 --seed 46103
```

This is sampled subgame resolving, not safe resolving: opponent private cards
are uniformly determinized and no trunk counterfactual-value gadget is used.
The exact information keys prevent state aliasing inside the sampled game, but
they do not make its root range exact.

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

See `REGRET_UPPER_BOUND.md` for the T=1 original-information-set
counterfactual-regret upper-bound estimator and its Monte Carlo limitations.
