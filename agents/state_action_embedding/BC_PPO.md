# Heuristic imitation followed by PPO

Completed pilot: [2026-09-19 results](BC_PPO_RESULTS_20260919.md),
[test comparison](data/bc_ppo_20260919/comparison.png), and
[learning curves](data/bc_ppo_20260919/learning.png).

## Question

Can a small event Transformer imitate the existing heuristic, how many chips does
it lose against that heuristic, and does subsequent PPO improve its actual play?
This experiment uses the same seven-poker v3 engine as action2vec, not standard
casino Stud. The opponent is fixed. This is not self-play or an equilibrium solver.

For historical-policy self-play initialized from these checkpoints, see
[PPO_SELF_PLAY.md](PPO_SELF_PLAY.md). It has a separate CLI and SQLite run directory;
the original commands and prior results above remain fixed-opponent experiments.

## Policy and observations

- Reuse the 64D, two-layer, four-head causal event Transformer from action2vec.
- Input: complete player-visible event prefixes; chance stays in the input.
- No opponent private card values, terminal payouts, hand-strength oracle features,
  teacher strength scores or future actions enter the model.
- Policy outputs: 8 existing betting actions plus 12 ordered discard/reveal pairs
  `(discard_index, reveal_index)` for the original four-card order.
- Legal-action masks allow either the legal betting subset or all 12 initial
  discard/reveal choices. The acting player is always relative subject 0.
- Both kinds of decision are learned and executed by the model. It never delegates
  discard/reveal to the heuristic. It never falls back from an invalid action.
- The teacher uses its normal partial-observation API; it is the unchanged
  `agents/baselines/heuristic_agent.py`.

## Predeclared pilot

| Setting | Default |
|---|---|
| Rules | 2-player cash, 1000 initial chips, ante 1, engine betting rules v3 |
| Demonstrations | 12000 heuristic-vs-heuristic hands; both actor views |
| BC split | 80/10/10 by hand; all decisions and both views kept together |
| BC | 4000 updates, batch 64, Adam 0.0005; validation NLL every 500 |
| PPO | 128 batches of 128 fresh hands = 16384 hands per seed |
| PPO updates | 4 epochs/batch, minibatch 64, Adam 0.0001, eps 0.00001 |
| PPO clipping | policy probability ratio clipped to [0.8, 1.2] |
| GAE | gamma 1, lambda 0.95; complete learner trajectories, terminal bootstrap 0 |
| Reward | only final net chips / 100; no heuristic-shaped rewards |
| Loss | clipped policy loss + 0.5 * (0.5 * value MSE) - 0.01 * entropy |
| Stabilization | rollout-level advantage normalization, grad norm 0.5; stop epochs if minibatch approximate KL > 0.03 |
| Validation | 256 deal pairs = 512 hands every 16 PPO batches, including BC at update 0 |
| Test | 4000 new deal pairs = 8000 hands per policy per seed |
| Training seeds | 11, 22, 33 |

BC selects the lowest validation action NLL. PPO starts from that BC checkpoint,
with a newly initialized optimizer. The critic starts with the zero output already
present during BC. No critic-only warmup and no extra BC/KL-to-teacher penalty.
The PPO critic and policy share the backbone. BC weights are not frozen during PPO.

Only learner decisions have policy probabilities and PPO losses. Opponent/chance
events remain in the context, but are not learner actions. Every rollout batch is
collected under one frozen current policy; old log probabilities and GAE targets
remain fixed during its updates. Prefixes are recomputed with current weights;
no detached or stale Transformer cache is carried across optimizer updates.
Discounting is per learner decision with gamma=1, not per chance token.

This is a compact PPO-Clip/GAE implementation on the existing engine, not SB3.
References: [PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438). Unit tests check the surrogate clipping,
terminal returns, unchanged-policy likelihood ratios and nonzero parameter updates.

## Evaluation

Each fixed shuffled deck is played twice with learner seats exchanged. BC and PPO
use the same evaluation deal seeds and independent-from-chance action RNG seeds.
Different policies can still reach different histories and consume different
numbers of draws. This is paired deals, not artificially identical trajectories.

**Primary decoding is stochastic categorical for BOTH BC and PPO.** A greedy BC
policy and a stochastic PPO policy are not mixed in the reported comparison.
The heuristic is deterministic. Report win/tie/loss, total net chips, chips/hand,
and uncertainty grouped by deal pair. Win rate is not the primary objective.

Controls: heuristic-vs-heuristic (exact zero paired net) and uniform random vs
heuristic. Main comparison: BC vs `ppo_best` and BC vs `ppo_last`. `ppo_best` is
chosen ONLY by validation mean chips and includes the BC-at-update-0 candidate.
`ppo_last` always uses the final predeclared training budget, showing any collapse
hidden by best-checkpoint selection. Test games never select a checkpoint.

BC demonstrations use seeds 310000000 + hand. PPO training uses
410000000 + training_seed * 1000000 + hand. Validation uses 510000000 + pair.
Test uses 610000000 + pair. CLI budgets prevent overlaps of these partitions.
Action RNGs are separate from Python's deck RNG. Results store source SHA256,
budgets, training curves, selected checkpoints, raw returns and resource usage.

The report uses paired-deal bootstrap intervals conditional on the trained
models; training-seed dispersion is reported separately. A small pilot against
one fixed heuristic cannot establish general strength, exploitability, or Nash
convergence. PPO from scratch is not included in this requested BC-before/after
comparison, so the benefit of BC over scratch PPO is not identified here.

## Run

Run from `D:/Experiment/Project-7-stud-Poker-Agent`:

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/bc_ppo_20260919'
& $py -m unittest agents.state_action_embedding.test_bc_ppo -v
& $py -m agents.state_action_embedding.bc_ppo collect --out-dir $run
foreach ($seed in 11,22,33) {
    & $py -m agents.state_action_embedding.bc_ppo train --out-dir $run --seed $seed
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
}
foreach ($seed in 11,22,33) {
    & $py -m agents.state_action_embedding.bc_ppo evaluate --out-dir $run --seed $seed
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
}
python -m agents.state_action_embedding.bc_ppo_report --out-dir $run
```

Use a fresh run directory to repeat. Dataset, seed directories, and completed
evaluations are protected against accidental overwrite. `progress.json` is a
live log, not a resume checkpoint. Three saved policies per seed: `bc.pt`,
`ppo_best.pt`, `ppo_last.pt`. These contain model weights/schema/selection metadata,
not optimizer/RNG state, and are inference checkpoints only.

Load with `model, metadata = load_policy(path)` from `bc_ppo.py`; `DecisionAgent`
adapts the model to the instrumented existing `EventGame`. Collection, training and
evaluation modes do not alter the engine or previous action2vec checkpoints.
Plotting uses the ordinary Python installation with NumPy/Matplotlib, no Torch.

## Scaling comparison

Completed: [2026-09-19 scaling results](BC_PPO_SCALING_20260919.md) and
[comparison graph](data/bc_ppo_scaling_20260919/comparison.png).

`bc_ppo_scaling.py` compares three conditions with seeds 11, 22, 33:

| Condition | Model | BC updates | Fresh PPO hands |
|---|---|---:|---:|
| A: longer BC | 2 layers, 64D | 20000 | 16384 |
| B: longer PPO | 2 layers, 64D | 20000 | 65536 |
| C: larger model | 4 layers, 128D | 20000 | 65536 |

A and B share one training path: at update 128, the best-so-far and current policy
are copied to immutable budget snapshots before continuing to update 512. The
large model also keeps a 128-update snapshot, providing the fourth cell of the
size-by-PPO-budget comparison without duplicate training. Models start afresh;
old checkpoints are not presented as optimizer/RNG-resumable training state.

All runs reuse the exact pilot demonstration file/splits. BC validation remains
every 500 updates; a fixed 2048-row training probe adds a train/validation diagnostic
without changing the minibatch RNG. PPO settings and validation remain unchanged.
Depth and width change together, so this identifies their combined effect only.
`--layers` defaults to 2 and old depth-less BC/PPO checkpoints still load.

The final test uses new deck seeds `620000000 + pair`, 4000 seat-exchanged pairs
per policy/seed. The old pilot BC and PPO checkpoints are re-evaluated on these
same decks rather than comparing numbers from different tests. Validation-selected
and fixed-budget final policies are both reported. New runs must use a new folder.

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/bc_ppo_scaling_20260919'
& $py -m agents.state_action_embedding.bc_ppo_scaling run --out-dir $run --workers 3 --threads 4
python -m agents.state_action_embedding.bc_ppo_scaling report --out-dir $run
```

Each worker has its own log and seed folder; `progress.json` reports BC or PPO
progress. `plan.json` records the predeclared budgets and dataset hash. The runner
waits for all six training/evaluation workers and stops its own remaining workers
on failure. Windows processes use hidden windows. `--smoke` runs a separate tiny
two-model end-to-end check and must never be mixed with main results.

CPU runs are concurrent: their measured wall times include contention and are NOT
isolated throughput benchmarks. Bootstrap intervals condition on the trained
models, and seed dispersion is separate. More validation opportunities at the
longer PPO budget can affect checkpoint selection; fixed-budget results help expose
that effect. All claims remain specific to the fixed heuristic opponent.
