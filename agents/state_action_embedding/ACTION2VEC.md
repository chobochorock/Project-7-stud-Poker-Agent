# Chance-aware action2vec and reward forecasting

Completed pilot: [2026-09-19 results](ACTION2VEC_RESULTS_20260919.md),
[comparison](data/action2vec_20260919/comparison.png), and
[raw metrics](data/action2vec_20260919/summary.json).

## Question and scope

Keep observed chance events in the input, but train the main sequence predictor
only at **opponent betting action** targets. Do these representations predict
terminal chip rewards from a still-unresolved history?

The first implementation uses the **after-observing-chance** interpretation.
For `raise -> deal X -> opponent call`, the call prediction sees X if X is visible
to the observer. It does NOT claim to predict the response before that draw.

Existing seven-poker v3 engine, two-player cash, 1000 starting chips and ante 1.
Rules are unchanged. This is not standard casino Stud or a trained poker policy.

## Events and privacy

Each local token has six fields: kind, relative subject, card, action, street,
actual chips paid. Small learned embeddings and a numeric projection combine them.
It does not encode an entire state vector at every event or use a state-ID table.

- START, ANTE, private/public DEAL, DISCARD, REVEAL, STREET, TURN, BET.
- Chance ownership is the recipient; kind identifies a chance event.
- Opponent private deals and discards retain their public existence, but card=0.
- All values shown to the observing player are retained, including own discards.
- TURN records the publicly known acting seat before its action; it contains no
  selected action, card, amount, pot, reward or hidden strategy ID.
- Terminal payoff and showdown disclosures are labels only, never model inputs.
- Full prefixes fit within 128 events; longer histories fail instead of truncating.
- Engine hooks record events without replacing dealing/betting rules. Observation
  cards and chip stacks are reconstructed from every sampled prefix and checked
  against the engine during collection.

## Controlled comparison

Three methods have the same two-layer, four-head, 64D causal Transformer, heads,
initialization, data and supervised reward-training budget:

| Method | Sequence pretraining |
|---|---|
| `none` | No pretraining; reward model starts from random initialization |
| `players` | Opponent betting-action CE only; chance remains in input |
| `all` | Opponent-action CE plus visible dealt-card CE |

`all` is a player+chance target baseline, not a literal full-vocabulary LM.
Automatic boundaries, discard/reveal choices, own betting targets, and unobserved
private cards are not prediction targets. Known legal betting actions mask logits
in BOTH training/evaluation; legality comes from the exact engine and is available
at that decision. Chance targets are only observable dealt cards, not oracle cards.
The all objective averages over both action and chance target counts; players
averages over action targets only. Thus equal update counts are not equal action
loss weighting, nor equal target counts. This is the intended budget-allocation
comparison, not proof that chance prediction itself is always harmful.

There is no direct chance CE in `players`. Later action losses can still update
card/chance embeddings through attention. Tests verify this distinction.

Pretraining checkpoint selection uses validation opponent-action NLL for both
methods. `none` is saved without pretraining. No test-driven model selection.

## Data and reward benchmark

Equal numbers of uniform-random vs random, existing heuristic vs heuristic, and
mixed hands (heuristic seat alternated). This explicitly contains heuristic
behavior as DATA, not as new reward labels, rules, hand-strength features or a
claim of heuristic-free learning. Policy type is not an input. Results are broken
down by cohort. This tests held-out deals of known policies, not unseen opponents.

Stratified 80/10/10 hand split, with both player views always together. Select one
betting decision per hand, independently of its eventual reward, then forecast
both players' final net chips using only the prefix ending at TURN, BEFORE that
decision's action. No folded/resolved terminal prefix can reveal the answer.
Both views are correlated; they are not independent test samples.

1. Freeze the pretrained/random Transformer. Fit a ridge reward probe, choosing
   regularization by validation MSE. This tests accessible information in z_t.
2. Train the full Transformer and identical small value head on terminal net-chip
   MSE. Compare to the same model trained only for reward (`none`).
3. Compare to training-mean return and a raw-current-state ridge baseline. The raw
   baseline alone receives current cards/stacks/legal-mask features; the event
   models do not. It has a different input dimension, so it is a reference, not
   a parameter-matched architecture. Its action slot is a constant dummy CHECK,
   never the actual unknown target action.

Reward targets are standardized by TRAIN statistics. Value checkpoint selection
uses validation RMSE (including step 0, the constant predictor). Test metrics:
RMSE/MAE in chips, R2 against the test mean, mean prediction error, per-cohort
results, and paired hand-bootstrap differences in the report.

Terminal returns are noisy outcomes under these collection policies. They are
not optimal Q, CFR regrets, causal action effects or exploitability. Strong reward
prediction does not by itself make a solver. Pretraining methods spend extra
compute compared to none; time/memory/budgets are separately recorded.

## Run and load

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/action2vec_20260919'
& $py -m unittest agents.state_action_embedding.test_action2vec -v
& $py -m agents.state_action_embedding.action2vec collect --out-dir $run --hands 9000 --seed 20260919
foreach ($seed in 11,22,33) {
    foreach ($method in 'none','players','all') {
        & $py -m agents.state_action_embedding.action2vec train --out-dir $run --method $method --seed $seed
        if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    }
}
python -m agents.state_action_embedding.action2vec_report --out-dir $run
```

Use new directories; existing data and runs are not overwritten. `events.npz`
contains player-visible event tokens, legal masks, lengths, reward anchors, raw
reference states, rewards, hand IDs, split IDs and cohort IDs. See dataset.json.

Each run stores pretrained.pt and value.pt, metrics.json and held-out predictions.
Load a trusted checkpoint with `model, metadata = load_model(path)` in action2vec.py.
For value.pt, chip prediction is `model.value(tokens, anchors) * metadata['scale']
+ metadata['mean']`. These are inference checkpoints, not optimizer/RNG resume
checkpoints. Every saved value model is reloaded and its predictions verified.
Both tokens and anchor indices must use torch.long. For batched prefixes, PAD
all events after each row's anchor, as the provided predict_value helper does.

Memory uses the existing 10 ms process-RSS sampler and CUDA allocator peaks when
CUDA is available. Phase measurements include validation inside their loops.
GPU measurements are null on a CPU-only PyTorch installation, not zero VRAM.
