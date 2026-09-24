# State-action embedding: forward Skip-gram vs CPC

The subsequent chance-event / player-only prediction experiment is documented in
[ACTION2VEC.md](ACTION2VEC.md); its checkpoints and results are separate.

The subsequent actor-view heuristic imitation and on-policy improvement experiment
is documented in [BC_PPO.md](BC_PPO.md), including actual game evaluation.

This is a representation-learning benchmark, not a CFR/RL solver. It uses the
existing Python seven-poker v3 environment (not standard casino seven-card Stud).
The game engine and existing replay/checkpoint files are left unchanged.

## Architecture and comparison

- Local encoder: current observation + the just-selected public action -> 64D.
  It never receives `betting_history`. Same local event gives the same embedding.
- Input: four 52-card multi-hot groups (own hidden/public/discard, opponent public),
  street, relative actor, selected action, acting player's legal mask, 14 scalars.
  Opponent private cards and terminal rewards are never encoder inputs.
- Forward Skip-gram: local embedding predicts events 1 or 2 betting actions ahead.
  This is an **inductive, directional Skip-gram analogue**, not the original
  word2vec lookup table or a bidirectional-window reproduction.
- CPC: the same local encoder followed by a two-layer, four-head causal Transformer;
  its prefix representation predicts the SAME future events.
- Both use the same sampled anchor/target pairs, negative batches, initial encoder
  weights, optimizer steps and embedding width for each seed. CPC has extra model
  parameters and processes more tokens, so this is not equal FLOPs or wall time.
- A batch has one future street/horizon, to reduce trivial street discrimination.
  Other views/events from the same hand are excluded as negatives. Candidate
  card identity can still be a shortcut; this is a documented pilot limitation.
- Contrastive loss uses normalized dot products / temperature 0.1 and in-batch
  softmax classification. This is InfoNCE-like, not the exact negative-sampling
  binary loss used by original word2vec. Target encoders also receive gradients.

All prefixes are bounded by `--context` (default 16). CPC need not read unbounded
history, nor does CPC inherently require a Transformer. This implementation uses
one to test the proposed architecture. No pretrained language model is downloaded.

## Dataset and honest benchmarks

Fresh heads-up cash hands: 1,000 starting chips, ante 1, uniform legal actions,
random discard/reveal. Only betting events are tokens; setup/discard decisions
are not modeled. Both fixed player perspectives are recorded at every betting
action. Own discarded-card knowledge is retained as part of current observation.
Both views of a hand always belong to the same 80/10/10 train/validation/test split.
`trajectories.npz` has `x`, `lengths`, `actions`, `streets`, `returns`, `split`, `hand`.
The feature vectors are padded float32; IDs and lengths are integer arrays.

Primary comparison: **freeze each local encoder**, fit the same linear probe, and
evaluate held-out hands. One reproducibly sampled non-final anchor per player/hand
avoids overweighting long hands and trivially settled outcomes.

1. Final net-chip return RMSE in chips, lower is better. Targets are observed
   returns under the uniform behavior policy, NOT optimal Q or regret.
2. Next public betting action negative log-likelihood, lower is better; accuracy
   is secondary. No illegal-action mask is applied to the future prediction head.
3. Raw current features, untrained random encoder, and constant/frequency baselines.

Probe: standardized train features; linear 64D -> return + 8 action logits, Adam,
fixed budget. The two output tasks select their own checkpoints by validation loss.
Raw inputs have more dimensions and are an information-preservation baseline, not
a parameter-matched encoder. Context mean pooling is provided for both encoders;
CPC contextual output is a separate, higher-information evaluation, not a fair
current-only embedding comparison. Seed SD is not a test-data confidence interval;
the two player views are correlated. All seeds use the same dataset split.

Training retrieval accuracy is diagnostic only. Embedding targets change across
models and can exploit easy card identity, so it is NOT the winner criterion.
Uniform opponents contain no private-card-dependent betting signal. These metrics
do not establish useful bluff/range inference, poker strength or exploitability.
The next stage is an identical RL budget with held-out nonrandom opponents and
paired deals, reporting net return with hand-clustered uncertainty.

## Memory

Each `memory` call must be a fresh process. Reports include:

- FP32 parameter bytes, encoder-only bytes, actual initialized Adam state bytes.
- 10 ms sampled process working-set baseline, peak and increase. This includes
  Python/PyTorch overhead and can miss short peaks; it is not exact tensor peak RAM.
- CUDA allocated/reserved allocator peaks if a CUDA PyTorch build is available;
  null means **not measured**, not zero GPU usage. Driver overhead is excluded.
- Synthetic dense B x L inputs for context lengths 8/16/32/64, ten horizon-1 training steps.
  These isolate length scaling from real hand lengths. Actual-training RAM is
  separately recorded after the dataset has been loaded into host RAM.
- Inference phases `encoder` and `context` use no gradients, with the full model
  resident. Dropping CPC's Transformer after pretraining leaves the SAME local
  encoder parameter count and compute as Skip-gram. No KV cache is implemented.

For a standard full-attention implementation, attention scores may scale as L^2;
actual PyTorch kernels and memory reuse affect measured peaks. Model and Adam
storage scale with parameter count. Do not infer GPU memory from CPU RSS.

## Run

Use a new output directory; existing datasets and run outputs are not overwritten.
The shared dependency is a CPython 3.12 PyTorch CPU build on this machine.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/pilot_20260919'
& $py -m unittest agents.state_action_embedding.test_experiment -v
& $py -m agents.state_action_embedding.experiment collect --out-dir $run --hands 6000 --seed 20260919
foreach ($seed in 11,22,33) {
    foreach ($method in 'skipgram','cpc') {
        & $py -m agents.state_action_embedding.experiment train --out-dir $run --method $method --seed $seed --steps 800
        if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    }
}
foreach ($length in 8,16,32,64) {
    foreach ($method in 'skipgram','cpc') {
        & $py -m agents.state_action_embedding.experiment memory --out-dir $run --method $method --context $length
        if ($LASTEXITCODE -ne 0) { throw 'Memory measurement failed' }
    }
}
foreach ($phase in 'encoder','context') {
    foreach ($method in 'skipgram','cpc') {
        & $py -m agents.state_action_embedding.experiment memory --out-dir $run --method $method --phase $phase --batch 1
    }
}
# Default Python 3.10 has Matplotlib; reporting does not import PyTorch.
python -m agents.state_action_embedding.report --out-dir $run
& $py -m agents.state_action_embedding.experiment audit --out-dir $run
```

`checkpoint.pt` contains architecture and trained weights. Load with
`load_model(path)` from `experiment.py`; `model.encoder(x)` returns local embeddings.
It is an inference checkpoint, not a resumable optimizer/RNG checkpoint. Only load
trusted checkpoint files. CPC history inference uses `model.summarize(x, lengths)`.
Action selection would require a query token or evaluating candidate actions;
these completed-action embeddings are not a deployable policy by themselves.

The optional `audit` reloads every saved checkpoint and checks held-out retrieval
with the full prefix versus only the last event. This is a shortcut diagnostic,
not another primary benchmark. For CPC, shortening also changes positional input.

References: [CPC](https://arxiv.org/abs/1807.03748),
[word2vec](https://arxiv.org/abs/1310.4546).

## Separate State / Action Distillation

The independent 10k-hand, low-fold corpus and train-only skip-gram teacher ->
two-hidden-layer 32D MLP experiment is documented in
[SEPARATE_SKIPGRAM.md](SEPARATE_SKIPGRAM.md). It does not change the experiments
above or train a combined state-action encoder.

## Atomic Actions And Exact-Endpoint Paths

[ACTION_PATH_EMBEDDING.md](ACTION_PATH_EMBEDDING.md) compares atomic event SGNS
with variable-length paths sharing exactly the same current observations at both
ends. It preserves private-target masking and distinguishes observation equality
from perfect-recall information-set equality. No game rules or earlier runs change.

## Additive Observation Targets

[ADDITIVE_OBSERVATION.md](ADDITIVE_OBSERVATION.md) applies frozen atomic-action SGNS
vectors to cumulative observation targets, fits a current-observation MLP, and
checks positional controls, visible-information probes and same-endpoint consistency.
