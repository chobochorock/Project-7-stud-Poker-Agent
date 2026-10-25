# EVPA-inspired mini-stud endgame pilot

2026-10-17. This is a controlled component experiment, **not a full reproduction**
of Li & Huang's ICLR 2025 EVPA and not a 52-card seven-stud result.

## Question and prior work

Can a small held-out EV predictor produce better fixed hand buckets than card
features, and does EV-based pruning help or damage the lifted policy?

- [EVPA paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c1b5863a6b0f925617b917bb2f55be0-Paper-Conference.pdf),
  sections 4.1-4.3: approximate-equilibrium EV regression, ensemble pruning and
  depth/pot-weighted multi-node EV abstraction.
- [WEVA](https://arxiv.org/html/2605.10900v1): related warm-up EV clustering. This
  runner does not use test-game warm-up features except in the explicitly named
  `ev_oracle` diagnostic. Network arms use a frozen network on unseen public roots.

## Location and dependencies

Since 2026-10-25 the canonical workspace is `Project-7-stud-Poker-Agent`.
The solver, mini-stud environment and exact tree-BR test dependency are all in
this repository; a neighboring Toy checkout is not required. Local relative
symlinks at the former Toy paths preserve existing commands and artifact links.
Other Toy experiments were not moved.

Runtime packages: NumPy, PyTorch, scikit-learn and Matplotlib on Python 3.10+.
For a fresh environment: `python -m pip install numpy torch scikit-learn matplotlib`.

## Run (PowerShell, 7-stud project root)

Use the existing Python 3.10 installation with NumPy, PyTorch, scikit-learn and
Matplotlib. No package installation is performed by the runner.

```powershell
cd D:\Experiment\Project-7-stud-Poker-Agent
python -m unittest agents.evpa.test_experiment environments.mini_stud.test_game -v
python -m agents.evpa.experiment --help
python -u -m agents.evpa.experiment --output agents/evpa/data/pilot_new
```

The output directory must be new. Defaults: 36 training public roots, 12 validation
roots, 12 test roots; split seed 17; 3,000 teacher iterations; 10 independent network
initializations (seeds 17..26), each 1,500 Adam steps; 64x64 ReLU hidden layers; Huber
loss delta 0.1; learning rate 0.001; batch size 512. Standardization uses training
inputs only. Checkpoints are selected by validation MSE every 100 steps. Test labels
are used only for error/audit reports and the oracle arm, never for the learned arms.

Online solve: K=4 per player (capped by distinct type/features), 1,000 full CFR
iterations; clustering seeds 11, 29, 47; k-means++ with 10 initializations. The
uniform-time average uses own reach, not joint/opponent reach. Both players update
simultaneously from the same frozen policy, with signed regrets and no clipping.
One iteration is not one sampled hand.

## Environment and data source

`endgame.py` reuses `environments/mini_stud/game.py` preset `small`: rank-only
Q/K/A x4, three initial cards, fixed H4 discard/reveal, a public 5th card and a
private 7th card; betting cap 1 on both streets. It enumerates initial chance,
forces CHECK through 5th, and collects final-street public roots. There are 71
public roots and 2,364 full private histories in this prefix distribution.

Each root fixes the ordered public observations and betting prefix. The exact
joint private-type distribution retains card blocking and H4 selection bias;
it is NOT approximated by independent player ranges. Original private histories
(initial hand, discarded card, observed order and final private card) remain in
the unabstracted catalogue. Root-level train/validation/test sets are disjoint.
The check-prefix/H4 data distribution is an explicit restriction, not equilibrium
play in the full multi-street game.

The matrix solver factors identical public action trees. The existing expanded-tree
BR evaluator checks its results in tests, and terminal utilities are checked against
the original environment. Perfect recall within a fixed endgame is preserved by a
fixed type partition and distinct tables for every public action node. No public
cards or betting nodes are merged.

## Arms

| Arm | Partition | Pruning |
|---|---|---|
| `exact` | Original private types | None |
| `card_kmeans` | Made-hand rank counts + own discard rank | None |
| `ev_oracle` | Full held-out teacher multi-node EV | None; diagnostic, not deployable |
| `ev_network` | Frozen ensemble mean multi-node EV | None |
| `ev_network_pruned` | Same mapping as `ev_network` | Ensemble-based, bucket-conservative |
| `ev_prune_only` | Original private types | Ensemble-based |

All online policies start with zero regrets/average mass. The teacher policy and
teacher regrets are not installed into the tested solver. `card_kmeans` is a new
matched small-game baseline, not the archived 7-stud power-atlas result.

The network input uses only own observations and public history. Labels are the
finite teacher average-policy EV at each public node, divided by that node's pot.
Values average over the opponent's history-conditioned distribution before any
max. Off-path zero-opponent-reach cases use the root conditional prior; this is
explicitly a belief convention, not a uniquely determined equilibrium continuation.
The cluster vector weights predicted EV by `1 / (pot * max(depth, 0.2))`.

Pruning uses the local EVPA Algorithm 2 test:

```text
keep action a if
max_model EV(child_a) + delta * root_pot
    > min(max_other_action min_model EV(child), min_model EV(parent))
```

`delta=0.01`. Values in this comparison are converted back to antes. At least one
action is kept. Within each static hand bucket the union of member keep masks is
used: an action is removed only if every member rejects it. Thus all members share
one action set without later forgetting earlier mask distinctions.

Differences from the paper: much smaller networks/data, finite exact endgame teacher,
no depth-limit network recursion, no pre-root pruning, no root information sets
deleted, and the conservative bucket-union rule. The kernel still backs up the full
matrix tree, including masked branches. **This tests policy quality, not a runtime
speedup from pruning.** Correlated network min/max is not a calibrated confidence
interval; the audit measures errors against the finite teacher, not true Nash labels.

## Metrics and outputs

The completed `pilot_20261017` and `smoke_20261017` datasets, checkpoints, policy
exports, plots and logs are included in Git. Other generated runs remain ignored.
Historical `run.json` absolute paths and the pilot's `sources/` snapshots record
the original execution and are intentionally not rewritten by the migration.
See [migration manifest](MIGRATION_20261025.json) for old/new locations and checks.

- `run.json`: parameters, source hashes, root split, offline timings, prediction
  errors, complete status and summary. Model initialization seeds are NOT ten
  independent end-to-end experiment seeds.
- The completed `data/pilot_20261017/sources/` archive preserves the three source
  files matching that run's hashes, before Black-only formatting of the new solver
  files. Final saved policies were reloaded and their metrics checked after formatting.
- `teacher.csv`: exact teacher endgame exploitability and root Local BR-gap.
- `labels.npz`: compact offline feature/label arrays with explicit split IDs.
- `ev_models.pt`: inference checkpoint, reload with `experiment.load_models`.
  It does not contain optimizer state and is not a resumable training checkpoint.
- `policy_root*.npz`: bucket map, signed regrets, average masses and masks. Inference
  exports, not full experiment resumes. Action-tree definitions are from this code.
- `metrics.csv`: original-action exact exploitability=NashConv/2, current/average,
  Local BR-gap, L/C/H, bucket rows, table/diagnostic bytes and online time.
- `online_gap.csv`: every iteration's **pre-update current-policy root-actor** gap,
  integrating all private histories in that held-out root, not a single sampled hand.
- `pruning_audit.csv`: fraction of original information sets losing every action
  within 0.01 root pots of the reference-best action; reference is approximate.
- `comparison.png`: matched k-means overlay, log learning plots, H and MA100 of
  per-iteration root gaps. Values zero on logarithmic panels use display floor 1e-10.

Local gap is `E_private[max_a E_hidden[Q] - sum_a sigma(a) E_hidden[Q]]` at the root
actor. It is not a full LBR match or exploitability. The original-action BR always
permits **all original actions**, including actions removed by the tested policy.
L/C/H are exact accumulated raw original-action regret diagnostics (including
pruned actions). H is cancellation from grouping, not an exploitability lower bound.

Learning charts use equal ONLINE iteration counts, not equal total compute. Teacher
enumeration, labels and network fitting are logged separately. `training_history_visits`
counts equivalent represented history nodes per simultaneous sweep, not CPU operations.
Dense private-type payoff matrices, network storage, masks, mappings and raw regret
diagnostics must be included in a full memory comparison; `policy_table_bytes` alone
is not total RAM. Summaries average test roots equally, then clustering seeds; this
is not on-policy full-game weighting or a confidence interval.

Create the results summary and MA100 gap bands without retraining:

```powershell
python -m agents.evpa.report agents/evpa/data/pilot_20261017
```

The completed pilot is in [RESULTS.md](data/pilot_20261017/RESULTS.md). Gap boxes
show min/max and central 90% over roots, clustering seeds and 100-iteration windows;
these describe variability, not uncertainty about the mean. Some roots have at
most K private types, so the requested abstraction does not compress those cases.

## Limitations and next gate

No claim of full-game Nash safety, seven-stud improvement, PAC-safe pruning or exact
reproduction follows from this pilot. A stronger run should vary split/network seeds,
prefix policies/ranges and K, and compare total-time budgets. Before 52-card use,
validate private-range construction and safe re-solving; do not silently substitute
single-hand determinization for a full public-belief endgame.
