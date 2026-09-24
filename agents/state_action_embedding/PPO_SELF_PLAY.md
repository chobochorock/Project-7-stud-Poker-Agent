# Event Transformer PPO self-play

Question: does historical-policy self-play improve the existing event policy?
This is the original heads-up seven-poker v3 cash environment, not standard casino
Stud. Each hand starts with 1000 chips per player. Rules, observation visibility,
8 betting choices and 12 discard/reveal choices are unchanged.

## Run from PowerShell

```powershell
Set-Location 'D:/Experiment/Project-7-stud-Poker-Agent'
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/ppo_self_play_seed11'
$init = 'agents/state_action_embedding/data/bc_ppo_scaling_20260919/small/seed11/bc.pt'
& $py -m agents.state_action_embedding.ppo_self_play train --out-dir $run --init $init --seed 11 --updates 512 --rollout-hands 128 --eval-every 32 --eval-pairs 256
```

This is **65536 new training hands**, not 512 hands or optimizer steps. The default
monitoring budget is another 26112 hands: 17 checks (including update 0), three
opponents, 256 deal pairs, two seats. Evaluation hands never enter PPO updates.
The command starts from the longer-BC small policy, not a heuristic-trained PPO
checkpoint. Its dimensions are read from the file. `--init` can also load another
compatible BC/PPO checkpoint. Omit it to train a random model with `--dim 64
--layers 2`; this is a different experiment and must use a different directory.

Plot completed monitoring points during or after training:

```powershell
python -m agents.state_action_embedding.ppo_self_play plot --out-dir $run
python -m agents.state_action_embedding.ppo_self_play plot --out-dir $run --x-axis train_seconds
```

Use the ordinary `python` installation for plotting: the bundled `$py` runtime
used for Torch training does not have Matplotlib in this workspace.
`plot` requires NumPy/Matplotlib, not Torch. Available x axes are `train_hands`,
`learner_decisions`, `train_seconds` (rollout + PPO only), and `elapsed_seconds`
(also includes monitoring and persistence). It writes `learning_curve_<axis>.png`.

After the predeclared training budget, run the held-out test:

```powershell
& $py -m agents.state_action_embedding.ppo_self_play test --out-dir $run --test-pairs 4000
```

This adds **48000 evaluation hands**: initial and last policies, three opponents,
4000 paired deals, two seats. Do not use these test results to choose the checkpoint
or training duration and then report the same test as independent evidence.
If this evaluation is interrupted, repeat the same command: an already committed
initial benchmark is reused only with identical checkpoint hashes and pair budget.
A completed test is never overwritten.
For a code-only smoke test, use a NEW output directory, `--updates 2
--rollout-hands 4 --eval-every 1 --eval-pairs 2 --snapshot-every 1 --threads 1`.
Small smoke-test scores say nothing about policy strength.

## What self-play means here

- One learner is trained against one frozen historical policy per rollout batch.
  Both use the same player-visible event API and stochastic action decoding.
- The opponent is sampled uniformly from a bounded pool: initial policy plus the
  most recent snapshots. Defaults: `--pool-size 8`, `--snapshot-every 16` updates.
- The learner alternates physical seats; `--rollout-hands` must be even.
- Only the learner's actions have GAE/PPO loss. Opponent and chance events stay in
  context, but are not actions whose log probabilities the learner optimizes.
- The pool's tensors are independent frozen copies. Neither player changes during
  rollout collection. PPO uses the learner's recorded old log probabilities.
- This is historical-policy self-play, not simultaneous independent learners and
  not a guarantee of Nash convergence. No heuristic opponent is used for training
  (though BC initialization itself was learned from the heuristic).

PPO reuses `bc_ppo.py` unchanged loss/GAE semantics: terminal net chips / 100,
gamma 1, lambda .95, Adam lr .0001, clip .2, four epochs, batch 64, entropy .01,
shared actor/critic backbone, gradient norm .5, approximate-KL early stopping .03.
The final policy is chosen by training budget, not the best monitoring score.

## Which metrics matter

1. **Chips/hand against frozen opponents:** heuristic, uniform-random, and initial
   policy. Opponents, shuffled decks and physical-seat action RNG seeds stay fixed
   across monitoring checkpoints. This bank measures specific opponents, not all
   possible strategies; improvement can still be non-transitive or exploitative.
2. **Paired change from update 0:** compare the same deals/opponents/seats before
   and after learning. Store individual returns so another CI/bootstrap can be
   computed later, instead of keeping only a mean.
3. **Diagnostics:** entropy, approximate KL, clipping fraction, policy/value loss,
   training return, and opponent generation. Loss/entropy do not measure strength.

Playing a policy against itself is zero-sum: seat-balanced mean reward cannot tell
whether both sides improved. Training return also changes with the opponent pool.
Neither rising self-play reward nor falling PPO loss establishes improvement.
Use the fixed bank curve versus training hands/time, plus the independent final
test. Repeat with seeds 11/22/33 in separate directories for training variance.

Bands are pointwise 95% normal intervals over **deal-pair means**, conditional on
the trained policies. They are not simultaneous confidence bands, training-seed
uncertainty, or exact exploitability. Poker returns can have heavy tails; use more
pairs and the stored raw returns for robust follow-up analysis. Win rate is saved
but chip profit is the objective. There is no Elo or BR oracle hidden in this CLI.

## Stored files and SQL

- `metrics.sqlite`: configuration/source/checkpoint hashes, per-update diagnostics,
  monitoring/test metrics and every seat-paired evaluation return.
- `initial.pt`, `last.pt`: schema-compatible model weights. Last is atomically
  replaced after each completed update. Neither contains optimizer/RNG state.
- `checkpoints/update_NNNNNN.pt`: immutable evaluation/pool snapshots, usable for
  later cross-play via `bc_ppo.evaluate_policy(..., opponent=other_model)`.
- `learning_curve_<axis>.png`: generated on demand, never required for training.

The SQLite tables are `metadata`, `updates`, `evaluations`, `pair_returns`. No
expanded trajectory/observation JSON is accumulated. Metadata values are small
JSON objects; numeric metrics and pair returns are typed SQL columns. For example:

```sql
SELECT update_index, train_hands, opponent, chips_per_hand, ci_low, ci_high,
       delta_chips, delta_ci_low, delta_ci_high
FROM evaluations WHERE split = 'monitor'
ORDER BY update_index, opponent;
```

Training deal seeds: `1000000000000 + seed * 1000000000 + hand_index`.
Monitoring: `2000000000000 + pair_index`; held-out test: `3000000000000 + pair_index`.
These do not overlap the original BC/PPO datasets. Action RNGs are independent of
the deck RNG; evaluation action RNGs are attached to physical seats, so identical
policies produce exactly zero total return across each seat-swapped pair.

Existing output directories are refused. Ctrl+C keeps committed SQLite rows and
the last completed weight checkpoint; `--init last.pt` in a new directory is a
warm start with a fresh optimizer/pool/counters, **not exact resume**. A forced
process termination can leave status `running`; inspect the run before treating
it as a finished test candidate. Monitoring can be plotted while training via WAL.

## Checks

```powershell
& $py -m unittest agents.state_action_embedding.test_bc_ppo agents.state_action_embedding.test_ppo_self_play -v
```

Checks cover legal/private observations (existing suite), frozen opponent weights,
on-policy likelihoods, pool eviction, seat-pair cancellation, actual PPO updates,
checkpoint reload, SQLite integrity/paired deltas and overwrite protection.
