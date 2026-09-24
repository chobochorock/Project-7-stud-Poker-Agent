# Frozen Seven-Stud League

Run from `D:/Experiment/Project-7-stud-Poker-Agent`:

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -u agents/evaluation/league/run_league.py --build --hands 10000 --seed 1307
# Redraw/validate a completed run without replaying games:
& $py agents/evaluation/league/run_league.py --analyze SAVED_RUN_DIRECTORY
# Small end-to-end test; output directory must not exist:
& $py agents/evaluation/league/run_league.py --hands 20 --out-dir NEW_SMOKE_DIRECTORY
```

`--hands` is the TOTAL per matchup, not per seat or per player. The default
17-player roster has 136 matchups: 1,360,000 played hands in total and
160,000 hands per participant. Each matchup uses the same 5,000 independently shuffled
decks twice, with players swapping seats. The entire league reuses those
5,000 deals; action RNGs are reproducible and specific to entrant and seat.

## Participants

- Epoch LightGBM ensemble: all 100 models from the completed 1M run, current.
- Hard-256 1M: current and average, from that same run.
- Power-64 archived 100M MCCFR: current and average.
- Power-512 memory16 30M and 10M: current and average at each checkpoint.
- Power-512 memory81 10M: current and average.
- Power-512 without betting memory 10M: current and average.
- Local route-tree 100M **nodes**: current and average; not 100M hands.
- Existing baseline heuristic and uniform legal-action policy.

Exact checkpoint paths, atlas pairings and hashes are in `manifest.json`.
The roster is defined in `make_roster` before any match result is observed.
The 10M/30M/100M names identify archived checkpoints; this is not an equal
training-budget comparison. Neither table size nor a filename is treated as
a reliable count of training node visits. Current/average views share the
same loaded table and do not train separately during evaluation.

As requested, online resolvers and neural/Python adapters are outside this
first league. Smoke checkpoints, duplicate snapshots and soft-bucket models
requiring additional nonserialized runtime settings are also excluded.
Memory16 10M is intentionally retained alongside 30M as a progress reference.

## Rules And Safety

`evaluate_league.cpp` calls the existing `play_hand_match`; it does not
reimplement betting, dealing, H4 selection or payouts. Same two-player v3
game: fixed heuristic H4, 5th-7th betting, ante 1000 chips, stacks 1000 antes.
Each hand starts with fresh stacks and a shuffled deck. No online search,
regret updates or adaptation occur; policies are frozen.

Hard-model headers determine the betting-memory flags. The normal model
loader checks format, abstraction and starting street; tree models also
check atlas generation and shape. Unsupported abstractions fail instead of
being silently interpreted. Unseen table/bucket queries use the original
policy fallback and are counted per matchup.

Each run checks legal normalized distributions, hidden-card invariance,
sampling reproducibility, zero-sum viewpoint reversal, and the always-fold
-1 ante sanity check in both seats. Table counts must remain unchanged.
The 20-hand smoke covers all 136 matchups but is NOT a performance result.

## Statistics

Positive payoff is profit earned by the ROW player, in ante/hand; unlike
LBR reports, higher is better for that player. The matrix is antisymmetric.
The primary head-to-head estimate is the mean of 5,000 seat-pair averages.
Normal-approximation 95% intervals use independent deals, not independent
seats. Hand wins/ties/losses are supplementary, not the ranking criterion.

Two equal-opponent-weight scores are shown: all rivals, and learned rivals
only (exclude heuristic/uniform as opponents). All participants are still
shown in both rankings. Aggregate score intervals first average rivals
WITHIN each deal block, preserving the correlation from shared deals.
The resulting rankings depend on this roster, which includes both current
and average views. They are not Nash rankings or exploitability estimates.

Matrix asterisks mark pointwise 95% intervals excluding zero. They are not
adjusted for 136 comparisons, and intervals omit training-seed uncertainty.
Colors use a signed-log scale with a linear region +/-0.1 ante; printed
cell values and statistics are NOT logarithm-transformed.

## Files

- `manifest.json`, `roster.tsv`: fixed roster, scope, input and source hashes.
- `sources/`: evaluation code and binary snapshots; models are not duplicated.
- `progress.log`: named matchup/hand/profit/timing fields and peak C++ memory.
- `matches/paired_payoffs.csv`: raw terminal payoffs for both seats.
- `matches/matches.csv`: engine means, timings and policy lookup diagnostics.
- `summary.json`: complete match statistics and both rankings.
- `payoff_matrix.csv`, `payoff_matrix.png`, `ranking.png`: tables and charts.

The analyzer rejects incomplete/duplicate matches, mismatched deal seeds,
invalid payoffs and disagreement with engine summaries. Existing output
directories are never overwritten by the runner. Re-analysis only updates
derived statistics and graphs. A failed run's partial results are preserved
but never ranked as a complete league.

## Completed Run

[2026-09-20: 17 policies, 10,000 hands per matchup](data/league_1m_30m_100m_20260920/README.md)
contains the interpretation, full payoff matrix, ranking and raw data links.
