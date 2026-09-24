# Autoencoder Abstraction

AE + k-means and single-code VQ-VAE, with raw-feature and random-encoder
k-means controls. This family contains model code; all datasets, checkpoints
and plots live in `data/`. No previous model or experiment is overwritten.

Completed pilot: [2026-09-23 results](RESULTS_20260923.md).
Frozen-bucket solver commands: [100k AE/VQ MCCFR and evaluation](CFR.md).
Completed solver pilot: [100k training and LBR results](CFR_RESULTS_20260923.md).
User-completed 1M runs: [verification, paired 100k comparison and plots](data/cfr_1m_review_20260923/README.md).
Default baseline overlay: [matched k-means evaluation, log-log curves and ratios](data/cfr_kmeans_overlay_20260924/README.md).
User-completed VQ 10M: [verification, matched long-run k-means controls and LBR](data/vq_10m_review_20260924/README.md).
Expanded LBR evaluation: [five fresh seeds, 20,000 hands per policy and matched k-means](data/vq_10m_lbr_multiseed_20260924/README.md).

## Scope

The primary `--scope power` experiment uses the SAME 18-dimensional power
features, completion sample limit 128 and 5th/6th/7th-street domain as the
existing C++ hard-256 atlas. There are 256 CARD codes **per street**, not
256 complete information-set buckets. Training pools both acting seats.
Final solver keys must preserve actor, street, exact legal-action mask and
the solver's retained betting context; `bucket_keys` enforces this contract.

The game is this project's heads-up C++ 7-stud v3, not casino Stud or
Stud-Leduc: ante 1000, stack 1000 antes, fixed heuristic H4 discard/reveal,
no H4 betting, existing 1/2/3 raise caps. The collector reuses the main solver;
the CFR bridge adds atlas lookup hooks without changing the learning rule.
Collecting data is played trajectories, NOT MCCFR traversals.

Behavior policy at each legal decision:

| Street | Fold | Each other legal action |
|---|---:|---|
| 5th | 0.02625 | `(1 - 0.02625) / number_of_nonfold_actions` |
| 6th | 0.06125 | `(1 - 0.06125) / number_of_nonfold_actions` |
| 7th | 0.0875 | `(1 - 0.0875) / number_of_nonfold_actions` |

If fold is unavailable, use uniform legal actions. A sole legal action has
probability one. These probabilities are not per-hand street fold rates.
Aggregate frequencies of nonfold actions need not match because legal sets vary.
H4 is still heuristic; the trajectory source is not entirely heuristic-free.

The historical atlas was fitted using epsilon-0.2 self-play. The representation
pilot's four methods instead fit the same freshly collected low-fold data.
Downstream CFR plots also retain the historical atlas/table as a separately
labelled reference, now with matched evaluation. This does not eliminate its
different training data or actor-shared keys; it is not an encoder-only ablation.

## Models And Budget

- AE: `18 -> hidden(64) -> latent(8) -> hidden(64) -> 18`, ReLU hidden layers.
- VQ-VAE: same encoder/decoder, nearest-neighbor codebook of 256 vectors.
- Training: CPU, Adam lr 0.001, batch 256, 2000 updates including 200 continuous
  reconstruction warmup updates. VQ codebook is initialized by k-means on
  train-only warmup embeddings, then uses reconstruction + codebook +
  `0.25 * commitment` loss and a straight-through gradient.
- Both networks start with identical encoder/decoder weights per seed and use
  identical training minibatch sequences. Validation reconstruction chooses
  the checkpoint, never test rewards. No regret, reward or strategy labels
  enter encoder or cluster training.
- AE latent k-means and raw/random controls use sklearn KMeans, n_init=3,
  max_iter=100, same seed and at most 20,000 train rows per street. The VQ
  initializer is an extra fitting cost, included in its measured run time.
- The raw baseline operates on original power coordinates without scaling;
  AE reconstruction also uses these bounded coordinates. This is a fresh
  sklearn fit, not a bitwise reproduction of the older C++ atlas initializer.
- Hyperparameters `--hidden`, `--latent`, `--codes`, `--steps`, `--warmup`,
  `--batch`, `--fit-cap`, `--seeds`, and `--threads` are configurable.

## Full Observation Option And Recall

The compact dataset also retains the existing Deep-CFR observer input:
ordered own hidden/public cards, own discard, ordered opponent public cards,
all betting-history tokens (street/relative actor/action), seat/street,
public chip scalars and legal mask. Opponent hidden/discard and future deck
are excluded. All inputs precede the current action.

`--scope infoset` expands these fields into the existing 1832D representation.
It is a separate input-scope experiment, not the historical 18D k-means scope.
The history capacity is 24 tokens and overflow fails rather than truncating.
This path passed a small smoke run; the completed 3-seed main pilot is `power`.
The current full-input AE uses plain MSE, so sparse one-hot reconstruction
can favor padding/frequent values. It needs a separately specified mixed
categorical/numeric objective before treating it as a strong benchmark.

Full observable input does NOT make the compressed code perfect recall.
Different observed histories can still share a code. Retaining original
history only in the dataset also does not repair the deployed policy.
Perfect recall requires an appropriate history-preserving strategy key or a
proved abstraction condition; no such guarantee is claimed here.

## Run

Run from the 7-stud project root using the system Python 3.10 environment
with its already-installed torch/sklearn/matplotlib. No new dependencies.

```powershell
$run = 'agents/autoencoder_abstraction/data/NEW_RUN'
python -m unittest agents.autoencoder_abstraction.test_experiment -v
python -m agents.autoencoder_abstraction.experiment collect --out-dir $run --hands 10000 --data-seed 20260923
python -m agents.autoencoder_abstraction.experiment train --out-dir $run --scope power --codes 256 --steps 2000 --warmup 200 --seeds 11 22 33
python -m agents.autoencoder_abstraction.experiment report --out-dir $run --scope power
# Separate scope; run directories include scope so old results are not overwritten.
python -m agents.autoencoder_abstraction.experiment train --out-dir $run --scope infoset --seeds 11
```

Collection builds `bin/collect.exe` and runs its self-test. The verified
compact NPZ replaces only this run's temporary binary; no raw row copies
or old files are deleted. Split is by whole hand, 80/10/10, before any model
fitting. Both seats and all streets of a hand stay in the same partition.

## Load And Assign

```python
import numpy as np
import torch
from agents.autoencoder_abstraction.experiment import assign, bucket_keys, inputs

checkpoint = torch.load('PATH/checkpoint.pt', weights_only=True, map_location='cpu')
with np.load('RUN/dataset.npz', allow_pickle=False) as data:
    rows = data['rows']
rows = rows[rows['street'] == checkpoint['street']]
codes = assign(checkpoint, inputs(rows, checkpoint['scope']))
# Supply the same public betting-context keys that your solver already retains:
# keys = bucket_keys(rows, codes, retained_context=solver_context_keys)
```

`assign` is a batch diagnostic API, not a fast C++ per-node solver adapter.
Each saved file contains architecture/weights and centroids or codebook.
It is an inference checkpoint, not optimizer/RNG state for exact resumption.
Do not load untrusted checkpoints. Dataset `action` and `outcome` are excluded
from `inputs`, and no true hidden state is supplied to the models.

## Metrics And Limits

- `power_bucket_mse`: held-out original power-vector error against each
  code's TRAIN mean power vector. All methods use the same diagnostic decoder.
  This objective naturally favors Euclidean k-means; it is not a strategic metric.
- Used codes, largest-code mass and effective codes `exp(entropy)` diagnose
  dead codes and imbalance. Empty train codes use the train-wide mean in the probe.
- Behavior-return RMSE: fit a TRAIN mean outcome per actor/legal-mask/code;
  evaluate the first decision per hand/actor/street, with group-mean fallback
  for unseen keys. This is noisy observed-return prediction under this behavior
  policy, NOT counterfactual value, Q*, regret, Local BR-gap or exploitability.
- Times include fitting, assignment, diagnostics and checkpoint round-trip.
  Network memory is sampled process RSS growth, not isolated tensor memory or
  a fair fresh-process per-method peak; GPU training was not used.
- Seed SD is not a confidence interval; seeds share one collected dataset.
- No CFR training, LBR league or exploitability measurement is part of this
  representation pilot. The separate C++ solver experiment is documented in
  [CFR.md](CFR.md); compression quality alone cannot choose a poker winner.
