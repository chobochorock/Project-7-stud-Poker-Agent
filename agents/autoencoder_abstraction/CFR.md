# Frozen AE/VQ Buckets With MCCFR

Run from the project root using the system Python with torch, sklearn, numpy
and matplotlib. The runner builds C++ when necessary, validates bucket parity,
then trains and evaluates. Existing output directories are rejected.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
python -m agents.autoencoder_abstraction.run_cfr --method ae_kmeans --hands 100000 --out-dir agents/autoencoder_abstraction/data/ae_cfr_100k
python -m agents.autoencoder_abstraction.run_cfr --method vqvae --hands 100000 --out-dir agents/autoencoder_abstraction/data/vq_cfr_100k
```

These are independent runs, not an ensemble. Existing k-means files are not
modified. Choose a new `--out-dir` for every run. Options: `--help`.

## VQ 10M Run

User-run extension planned on 2026-09-24. This starts MCCFR from zero with the
same frozen representation and seeds, NOT from the saved 1M policy. There is
no exact-resume option. 10M hands = 20M traversals; the VQ encoder stays fixed.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
python -m agents.autoencoder_abstraction.run_cfr --method vqvae --hands 10000000 --representation-seed 11 --seed 7 --eval-every 10000 --eval-roots 128 --eval-particles 32 --eval-seed 307 --lbr-pairs 500 --lbr-particles 64 --out-dir agents/autoencoder_abstraction/data/vq_cfr_10m
```

Current/average root Local BR-gap is recorded every 10k in
`solver/local_gap.csv`; final LBR plays 500 seat-swapped pairs per policy.
`run.log` and `solver/progress.csv` retain progress and memory measurements.
Only the final `solver/policy_10000000.bin` is saved, so interrupting training
does not leave an intermediate resumable model. Completed reports include
the existing hard-256 curve, whose cached matched measurements stop at 1M.
The current descriptive slope summary still covers [100k, 1M], not [1M, 10M].

Correction after the user's reminder: longer-trained k-means models DO exist.
`../cpp_mccfr/data/made_call_r1000_k512_epsheur20_memory16_10m.bin` and
`../cpp_mccfr/data/made_call_r1000_k512_epsheur20_memory16_30m.bin` have completed
10M/30M root-training logs and use `power512_epsheur20_memory16_v1.bin`.
They are signed MCCFR, 512-card-cluster, actor-shared power-memory16 models,
not checkpoints continuing the seed-7 hard-256 curve. They can serve as separate
historical-model controls after matching the evaluator. Their existing LBR
reports use different settings, so neither those scores nor their names should
be spliced into the hard-256 curve. No new evaluation was run during this
inventory correction; automatic reports still use the cached hard-256 curve.

### Completed 10M Review

The user-run 10M experiment is now complete. The
[review](data/vq_10m_review_20260924/README.md) validates its first-1M prefix,
adds matched frozen K512 10M/30M and archived K64 "100M" evaluations, and
compares early versus late log-log slopes. These references are isolated
points, not extensions of the K256 curve. K64's exact warm-start budget
remains unverified; "100M" is its archived label.

```powershell
# Evaluation only; validates existing runs and rejects existing output paths.
python -m agents.autoencoder_abstraction.analyze_10m --out-dir agents/autoencoder_abstraction/data/vq_10m_review_new
```

The review fits [1M, 10M] slopes while retaining [100k, 1M] comparisons.
Do not confuse those late-window values with the original training report's
default early-window slopes. LBR's heavy-tailed uncertainty is documented.

## Multi-Seed Frozen LBR

```powershell
python -m agents.autoencoder_abstraction.reevaluate_lbr --out-dir agents/autoencoder_abstraction/data/vq_10m_lbr_new
```

This evaluates VQ 10M and the archived K512 10M/30M and K64 "100M" models,
without training. Defaults: new evaluation seeds 2026092401..2026092405,
2000 seat-swapped deal pairs per seed and policy, 64 LBR particles. Thus each
current/average policy plays 20,000 hands. `--pairs` and `--seeds` override
the fixed evaluation budget. Existing outputs are rejected.

Before new evaluation, the loader must exactly reproduce VQ's old seed-307
500-pair LBR and root-gap results. That old sample is excluded from all new
aggregate statistics. All models use identical new deal streams. Both policies
are retained, all outcomes included, and there is no outcome-based early stopping.
`run.json` records the plan before evaluation; raw CSVs, `summary.json`, and
`lbr_comparison.png` retain results. The output also contains 128 new root gaps
per seed/model, but this experiment's primary metric is current VQ LBR.

The report includes normal intervals and a 2000-replicate bootstrap resampling
deal pairs within each seed; model differences use matched pairs. These are
evaluation seeds, NOT independently trained models. More deals reduce sampling
uncertainty but do not fix weak-attack bias or prove low exploitability.
One ante/hand is the user's reference, not a universal research threshold.
Completed default run: [new-seed results and paired comparisons](data/vq_10m_lbr_multiseed_20260924/README.md).

## Default k-means Overlay

`metrics.png` now includes the retained legacy hard-256 k-means baseline for
both current and average policies. Top: **log-log Local BR-gap**. Middle:
**model gap / k-means gap**, where 1 is the baseline and below 1 is lower gap.
Bottom: measured LBR endpoints with deal-paired 95% intervals; its y-axis stays
linear because LBR profits/intervals can be zero or negative. No smoothing,
interpolation, or extrapolation is used. Hand 0 and nonpositive gaps remain in
CSV/JSON but are excluded from logarithmic axes. Nonpositive CI lower bounds
are not drawn on log axes.

The reference is `data/kmeans_reference_matched_20260924`. It reevaluates the
original seed-7 10k..1M checkpoints without retraining, using the **same**
128 roots / 32 particles / seed 307 as AE/VQ and the same 500-pair / 64-particle
LBR streams at 100k and 1M. Legacy data generation and actor-shared strategy
keys still differ: this is a historical-model comparison, not a controlled
ablation of encoders. Raw evaluation CSVs and input hashes are retained.

```powershell
# Only needed to regenerate the reference; existing outputs are rejected.
python -m agents.autoencoder_abstraction.comparison --out-dir agents/autoencoder_abstraction/data/kmeans_reference_new
# Validate four completed AE/VQ runs and plot all three families together.
python -m agents.autoencoder_abstraction.analyze_1m --out-dir agents/autoencoder_abstraction/data/comparison_new
```

The second command uses the default retained reference. To select a regenerated
one, pass `--baseline-dir`. Future training reports use the default reference
automatically. Missing or mismatched evaluator settings produce a visible
warning; ratios are withheld when gap evaluation settings differ. Reference
curves stop at their actual measured budget, even for runs exceeding 1M.

`metrics.json` / `comparison.json` records descriptive OLS slopes
`d log(gap) / d log(hands)` over measured checkpoints in [100k, 1M], endpoint
percentage reductions, and `2**slope` (fitted gap factor per doubling). These
are not convergence exponents or forecasts: shared roots correlate evaluations,
Monte Carlo noise remains, and all training runs use only one seed. The ratio
curve shows sampled mean crossings, not statistically established superiority.

## Matched Conditions

- Environment: existing C++ heads-up 7-stud v3, fixed H4 discard/reveal,
  betting from fifth street, raise caps 1/2/3, ante 1000, stack 1000 antes.
- Source: `data/power_lowfold_10k_20260923`, representation seed 11 selected
  before downstream evaluation. `--representation-seed 22` or `33` selects
  the other already-trained representations. No new representation fitting.
- Each street has a frozen 18 -> 64 ReLU -> 8 encoder and 256 centers (AE)
  or codebook entries (VQ). No decoder, splitting, merging, or 10k replacement.
- A strategy key retains seat, street, card code, exact legal mask, and the
  solver's cumulative betting summary. 256 card codes != 256 strategy buckets.
- Signed external-sampling MCCFR, NOT CFR+, full-tree CFR, or CFR-D. Default
  solver seed 7, zero regrets, uniform legal initial policy. A hand is one
  fresh root traversed for player 0 then player 1: **100k hands = 200k traversals**.
  Traverser actions are enumerated; non-traverser actions are sampled.
- Current policy uses regret matching on cumulative signed regret. Average
  policy uses the existing external-sampling non-traverser `strategy_sum`
  updates, without linear weighting; zero average mass falls back to current
  regret matching. Both are saved in the same table file.
- Low-fold probabilities (2.625% / 6.125% / 8.75%) apply ONLY to representation
  data collection, not MCCFR self-play or evaluation.
- AE/VQ share root/evaluation seeds; action trajectories can differ. This
  single representation/solver seed experiment is a pilot, not a robust ranking.

## Evaluation

At hand 0, each 10k, and the final hand, `--eval-roots 128 --eval-particles 32`
estimates root-only Local BR-gap on fixed independent fifth-street roots.
For each legal action, rollout values are averaged over H4-conditioned hidden
worlds BEFORE taking the max; subtract the target policy value. Continuations
use the frozen current/average policy. The unit is ante. Finite-particle max
bias remains, and root-mean normal 95% CIs do not remove it. This is not a
later-street gap measurement, exact BR, or exploitability.

Final `--lbr-pairs 500 --lbr-particles 64` uses the existing posterior-aware
PolicyLBR: same deal with LBR in both seats, **1000 played hands per target
policy**. Positive LBR ante/hand means the exploiter wins. Its continuation
model is approximate, not exact exploitability; CIs are not formal bounds.
Use paired-deal means, not individual seat-hands, as the CI sampling unit.
Evaluation seed 307; LBR uses a separate deterministic stream. No evaluation
trains the target. Tables are reloaded before LBR after checking policy equality.

`--eval-every` changes logging frequency, not abstraction. Gaps are per-checkpoint
means, not running cumulative means. `train_seconds` excludes evaluations;
progress elapsed includes gaps but excludes final LBR. `run.json` wall time
includes validation/training/LBR/plots but excludes compilation. Training peak
RSS is a process high-water mark, not just serialized model size.

## Outputs And Loading

New artifacts stay in `agents/autoencoder_abstraction/data/<run>/`:

| File | Meaning |
|---|---|
| `neural_atlas.bin` | Frozen encoder and centers for all three streets |
| `solver/policy_100000.bin` | Signed regret and strategy-sum tables |
| `solver/progress.csv` | Hands, traversals, nodes, buckets, time, training peak RSS |
| `solver/local_gap.csv` | Per-root gaps at each checkpoint |
| `solver/lbr_pairs.csv`, `solver/lbr_queries.csv` | Seat payoffs and missing-key counts |
| `metrics.png`, `summary.json` | Curves, LBR means and approximate 95% CIs |
| `run.json`, `run.log`, `validation.txt` | Arguments, hashes, checks and progress |

Only final tables are saved; no intermediate checkpoints or temporary parity
rows are retained. Keep atlas, policy and run.json together. These are inference
checkpoints, NOT exact resume: the legacy table serializer does not save RNG
or counters. Never pair a policy with a different encoder/seed/raw-kmeans atlas.
Verify run.json hashes before external reuse: the legacy table file does not
embed encoder identity. Loading inside the runner's C++ translation unit:

```cpp
neural7::NeuralAtlas atlas("RUN/neural_atlas.bin");
MCCFR solver(false, 7, 5, &atlas);
solver.use_cumulative_street_summary(true);
solver.load("RUN/solver/policy_100000.bin");
auto average = solver.policy(observed_state, observed_state.actor);
auto current = solver.instantaneous_policy(observed_state, observed_state.actor);
```

`solver.choose` may train online; frozen evaluation instead samples `policy`.
An unseen complete strategy key falls back to uniform legal actions. A card
always gets a code; this does not imply every betting/key combination is trained.

## Limits And Checks

18D card codes and betting summaries remain imperfect-recall abstractions.
No original-game convergence guarantee is claimed. Historical hard-256 tables
used actor-shared keys and may differ in data/initialization/budget, so are not
a controlled comparison here. AE/VQ use identical seat-separated key rules.

Each run checks all source observations for Python/C++ assignment agreement,
rejects truncated/nonfinite/trailing weights, checks privacy, seat partition,
legal policies and final save/load equality. C++ uses float32 dense layers and
double Euclidean distances; parity is empirical, not a bitwise guarantee at
every possible code boundary. Only hard assignments are supported.

```powershell
python -m agents.autoencoder_abstraction.run_cfr --method vqvae --hands 10 --eval-every 10 --eval-roots 2 --eval-particles 4 --lbr-pairs 2 --lbr-particles 4 --out-dir agents/autoencoder_abstraction/data/cfr_smoke_new
agents/autoencoder_abstraction/bin/neural_mccfr.exe --engine-test
python -m unittest agents.autoencoder_abstraction.test_experiment -v
```
