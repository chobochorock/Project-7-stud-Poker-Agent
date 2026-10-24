# Frozen Reconstruction Buckets + MCCFR

The experiment reuses `reconstruction_20261002/transformer/seed11/best.pt` and
its saved raw/latent 512-centroid atlases. It does not use the newer initial-four-card
model, refit the encoder, or overwrite previous clustering results.

## Question and Prior Work

Does a reconstruction-trained full-history representation produce a better policy
than raw current-observation k-means with the same clustering budget and solver?
This is a controlled comparison of established components, not a novel solver.

- [Lanctot et al. 2009](https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html): outcome-sampling MCCFR, reused through `raw_cfr_ppo.outcome_targets`.
- [Ganzfried and Sandholm 2014](https://ojs.aaai.org/index.php/AAAI/article/view/8816): poker clustering abstraction, but based on future strength distributions/EMD rather than reconstruction latent coordinates.
- [Yang et al. 2017](https://proceedings.mlr.press/v70/yang17b.html): joint deep representation/clustering; related, but our encoder and centroids stay frozen.

No exact match was established by this limited literature check; that is not evidence of novelty.

## Protocol

- Original custom seven-poker v3, heads-up cash, 1000 chips, ante 1. BET decisions on streets 5/6/7 only.
- Uniform fixed discard/reveal for both players and all opponents. Only betting is learned.
- Signed outcome-sampling MCCFR, alternating traverser, epsilon=0.6 for the traverser; opponent sampled from its current policy. Initial regrets and strategy sums zero.
- One hand means one sampled trajectory, not two traversals. Policy fixed within each 128-hand batch; workers sample independently and their updates are added in hand order. Comparisons use the same batching and seeds.
- Table key is `(absolute seat, own betting-decision depth, centroid ID)`. Centroids already partition street and exact legal-action mask. Depth prevents revisiting one abstract information set on the same trajectory; it does NOT guarantee perfect recall. 512 is the number of centroids, not the number of table entries.
- Average increments use own policy reach divided by sampled action-prefix reach. The omitted chance/H4 factor is constant within each street in this fixed uniform-dealing/uniform-H4 setup and cancels when normalizing each bucket's average strategy. This is not a general unequal-chance estimator. Exact-enumeration Kuhn tests check action-reach correction separately from regret estimation.
- Unknown legal partitions use explicit legal-uniform fallback and are counted. They never map into another legal-action partition.
- Training payoff is net chips/100; reported evaluation payoff is net chips/hand. Source/model/atlas hashes are checked for resume. Per-hand seeds and batch boundaries make resume reproducible.
- Original encoder/clustering data used low-fold trajectories (2.625/6.125/8.75%), unlike the evolving CFR sampling distribution. Raw sees the current observation; latent sees observable event history, so information input is not identical.

## State Prediction

This evaluates reconstruction of the current observation, not next-state forecasting.
Cards are ranked independently in four groups: own hidden, own public, own discard,
opponent public. `k` is computed from observable event kinds/recipients, not target
card identities. Report exact-set and recall@k, plus cross-group collisions without
repairing the predictions. Use original logits, not calibrated-probability claims.
Legal-mask exact match, street/actor accuracy, chip MAE and count MAE remain separate.
Static scores average prefixes within each hand first. Policy-panel scores first
average that player's decisions within each hand, then average hands; skipped/no-decision
hands do not enter reconstruction metrics. Both current and average policies are evaluated.

## Commands

Run at the project root with the existing Python 3.12 runtime (Torch dependencies are reused).

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/reconstruction_cfr_20261002'
& $py -m unittest agents.state_action_embedding.test_reconstruction_cfr agents.state_action_embedding.test_raw_cfr_ppo -v
& $py -m agents.state_action_embedding.reconstruction_cfr audit --out-dir $run
& $py -m agents.state_action_embedding.reconstruction_cfr train --out-dir $run --method raw --hands 10000 --workers 4
& $py -m agents.state_action_embedding.reconstruction_cfr train --out-dir $run --method latent --hands 10000 --workers 4
# Continue the same tables, retaining 10k/100k/1M checkpoint evaluations:
& $py -m agents.state_action_embedding.reconstruction_cfr train --out-dir $run --method raw --hands 1000000 --workers 4 --resume
& $py -m agents.state_action_embedding.reconstruction_cfr train --out-dir $run --method latent --hands 1000000 --workers 4 --resume
python -m agents.state_action_embedding.reconstruction_cfr_report --run $run
```

`last.npz` is resumable, including regret/average tables, visits and exact hand index.
`checkpoint_*.npz` retains the requested milestones; `evaluation_*.json` contains paired
payoffs, reconstruction diagnostics and fallback counts. No large replay is retained.

## Evaluation Limits

200 held-out deal pairs per fixed opponent (uniform, heuristic), each in both seats.
The same evaluation panel is reused across checkpoints, raw/latent and current/average.
Normal-approximation intervals are descriptive and fragile with heavy-tailed poker
returns; these small panels and one training seed cannot establish strategic superiority.
Do not choose hyperparameters on this repeatedly viewed panel and call it an untouched test.

**Local BR-gap, LBR profit and exact exploitability are not measured by this adapter.**
2026-10-03 supplement: Local BR-gap now has a separate post-hoc evaluator;
see [refinement criteria and gap commands](REFINEMENT_AND_BR_GAP.md).
The original 2026-10-02 payoff figures are not relabeled as BR-gap results.
The old C++ evaluator is not compatible with this event encoder and uniform-H4 protocol.
Graphs overlay raw k-means at matched budgets and show signed payoffs on a linear y-axis,
not fabricated log-log BR-gap curves. Missing metrics/baselines must remain explicit.
The 10k/100k/1M commands are execution budgets, not claims that those runs have completed.

## Completed Run

`data/reconstruction_cfr_20261002` completed both methods at 10k/100k/1M on 2026-10-02.
See [results and paired intervals](data/reconstruction_cfr_20261002/RESULTS.md),
[policy comparison](data/reconstruction_cfr_20261002/comparison.png) and
[top-k reconstruction](data/reconstruction_cfr_20261002/state_topk.png).
At 1M, average-policy heuristic profit was raw -17.235 vs latent -17.870 chips/hand;
paired latent-minus-raw interval [-8.749, 7.479] does not establish latent superiority.

### Post-hoc Local BR-gap (2026-10-03)

Saved 10k/100k/1M policies were evaluated without retraining on 128 common fifth-street
first-decision information sets and 64 conditional worlds per root. See the
[gap report](data/reconstruction_gap_20261003/RESULTS.md) and
[log-scale graph](data/reconstruction_gap_20261003/local_br_gap.png).
The [same-root 512-particle audit](data/reconstruction_gap_particle_audit_20261003/RESULTS.md)
shows substantial sampling sensitivity. Consult both before interpreting absolute gaps.
These are original-information-set diagnostics, not LBR profit or exact exploitability;
refinement training has not run. Definitions and commands are in
[REFINEMENT_AND_BR_GAP.md](REFINEMENT_AND_BR_GAP.md).
