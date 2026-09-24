# 7-stud Epoch Regret Ensemble

## Implementation Correction: 2026-09-20

**The archived learner/exporter used joint leaf-ID signatures, not the
user's intended cumulative regret at each leaf of each tree.** The archived
1M run (all 100 models), earlier 100k run and both 10k pilots use `EPOCH7V1`
joint-signature tables. Their measurements describe that implemented variant,
not success or failure of the intended leaf-wise method. Raw results remain
unchanged. [Evidence and corrected design](LEAF_REGRET_DESIGN.md).

The new default is **per-tree leaf storage** (`EPOCH7L1`); `--aggregation joint`
explicitly retains the legacy method. Both model formats can be loaded.
The refined boundary changed from .9 to .95 for future fits only.
Completed: [refined 100k / 10k epochs](data/leaf_refined_100k_seed7_20260920/README.md).
Held-out current-policy gap was lower than hard-256, but average-policy gap
was higher; one seat's starting group remained a single leaf. This is not a
whole-game strength result. Soft-CE and K-means follow-ups have not run.
The subsequent [10k-hand-per-target LBR evaluation](data/leaf_refined_100k_seed7_20260920/lbr_seed307_p240/README.md)
found higher losses against this attacker for both ensemble current and
average policies than for their hard-256 counterparts. Root-gap superiority
did not establish full-hand defensive strength. No further training was run.

**Raw-data cleanup, 2026-09-20:** at the user's request, `rows_*.bin` and
empty residual spools were removed from the corrected leaf 100k run
(38.91 GiB) and archived `seven_stud_1m_seed7_20260919_204332` run
(127.61 GiB). Total deleted file bytes: 178,799,306,580 (166.52 GiB).
Models, all baseline checkpoints, evaluation/training metrics, graphs and
source snapshots remain. Both frozen loaders passed self-tests afterwards.
Offline raw-row refitting from these two runs now requires regenerated
data. Other runs were not cleaned; the runner's future retention default
has not changed.

**Further cleanup, 2026-09-22:** the user requested safe cleanup of other
unused intermediates. Removed 51 raw row files and nine empty spools from
15 older/smoke/pilot runs, totaling 37,220,219,316 bytes (34.66 GiB).
[File-by-file manifest and preservation checks](data/storage_cleanup_20260922.json).
All 14 runs with ensemble exports passed the frozen LBR self-test after
deletion; one abandoned label smoke had no ensemble to test. Models,
baseline checkpoints, plots, metrics and sources remain unchanged.
These deleted raw rows cannot be refitted without regenerating data.
This is disk cleanup, not a change to future retention or fitting RAM.

## Folder Guide

Canonical family: `agents/lightgbm_regret_ensemble/`.

```text
lightgbm_regret_ensemble/
  README.md                    scope, commands and file guide
  stud_epoch_ensemble.cpp       C++ learner and model inference
  run_epoch_ensemble.py         training and LightGBM fitting
  evaluate_epoch_ensemble.cpp  held-out root-gap evaluation
  evaluate_epoch_lbr.cpp        full-hand Policy-LBR matches
  analyze_epoch_*.py            reports and graphs
  test_epoch_ensemble.py        representation tests
  bin/                         executables
  data/                        runs, models, datasets and evaluation outputs
```

The archived joint-signature 100k run is `data/seven_stud_100k_seed7_spooled/`.
Its ten model exports are `epoch_001/model.bin` through `epoch_010/model.bin`.
LBR results are in its `lbr_seed307_p240/` subfolder. Older incomplete runs
and smoke tests remain separately named and are not combined with it.
The common game/MCCFR engine and frozen atlas stay in `../cpp_mccfr/`;
they are referenced, not duplicated. Old paths are compatibility links.

Archived joint-signature reports: [100k training](data/seven_stud_100k_seed7_spooled/README.md),
[Policy-LBR evaluation](data/seven_stud_100k_seed7_spooled/lbr_seed307_p240/README.md).
The latter did not establish superiority over hard-256's current policy;
the conditional automatic 1M run was not started. The user subsequently
completed a manual 1M run. Its [training and LBR report](data/seven_stud_1m_seed7_20260919_204332/README.md)
includes linear/log gap plots and the completed 5,000-pair LBR evaluation.

The 2026-09-19 layout move preserved 1,421 files and their byte total;
13 main model/CSV hashes were checked. A rebuilt evaluator reproduced the
pre-move smoke payoff CSV exactly. A separate 20-hand, two-epoch training /
save / reload smoke also passed at the new path. It is not a 1M run and is
stored separately as `data/layout_pipeline_smoke_20260919/`.

## Current And Average: Which Training Run?

The reported hard-256 **current** and **average** strategies both come from
the same newly trained **100k-hand** baseline, `hard256_100000.bin`. They are
not the older 30M or 100M policies, and average-strategy training was not a
separate run:

- Current: normalize the positive parts of the final cumulative regrets
  (`current_strategy` / `instantaneous_policy` in the shared MCCFR code).
- Average: normalize `strategy_sum`, accumulated at sampled non-traverser
  nodes during those same external-sampling traversals. With no strategy
  mass for an entry, the implementation falls back to its current strategy.
- The proposed ensemble result is its final combined current policy, not
  a separately trained average-policy model.

New `EPOCH7L1` models also preserve per-leaf strategy sums. The held-out
evaluator reports `ensemble_average` when all loaded models support it:
average tree masses within each epoch, sum with epoch weights 1..t, then
normalize over legal actions. Zero mass uses uniform play. This is a
compressed approximation of the accumulated policy, not an exact full-game
CFR average. Old `EPOCH7V1` models cannot reconstruct this statistic.

The `100m` in the **atlas** filename describes the source of the reused
card clustering. Its centroids/feature preprocessing are reused, but no
30M/100M regret table or policy is loaded. This is therefore cold-start
policy learning with a previously fitted representation, not a wholly
data-free initialization.

The cumulative mean shown in the gap graph is an average of **measured gap
values**. It is not the gap of an average policy and is not `strategy_sum`.

## Manual 1M Run

The command now uses the leaf default. Add `--aggregation joint` only to
explicitly reproduce the archived representation; no new 1M run is started
automatically by this documentation change.

The following command trains **both** the proposed method and hard-256
baseline for 1,000,000 hands from scratch, with the same 10,000-hand epochs,
8-particle per-hand evaluation and epoch weights 1 through 100. It does
**not** resume the existing 100k checkpoint for another 900k hands: exact
resumption is not implemented. It uses a fresh timestamped directory and
does not overwrite the 100k models. The user starts this long run manually.

```powershell
$py = 'C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$family = 'D:\Experiment\Project-7-stud-Poker-Agent\agents\lightgbm_regret_ensemble'
$run = Join-Path $family ('data\seven_stud_1m_seed7_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))

& $py -u "$family\run_epoch_ensemble.py" --build `
  --hands 1000000 --epoch-hands 10000 --particles 8 --seed 7 `
  --out-dir $run

# After completion, or to redraw later with $run set to the saved directory:
& $py "$family\analyze_epoch_ensemble.py" $run
```

No game/algorithm changes are needed to accept 1M. Full 1M memory/time
behavior has not been measured. Do not close the training terminal:
completed epoch exports remain usable for inference, but are not exact
resume checkpoints. Increasing the number of frozen models makes later
policy queries more expensive; 1M need not take just ten times the 100k
wall time. Rebuild the final held-out/LBR evaluators before evaluating 1M:
they now require explicit `HANDS EPOCH_HANDS` arguments and load that exact
baseline checkpoint plus all its epoch models. Older binaries target 100k.

### Recorded Metrics

- `per_hand.csv`: one row **per training hand** for both policies' estimated
  first-5th-infoset Local BR-gaps, node counts, table sizes, cumulative
  training/evaluation time, and combined C++ peak working set. The stream
  flushes every 100 hands and at epoch boundaries; a crash can lose the
  unflushed tail. It is not one record per 10k epoch.
- `boundary_audit.csv`: fixed 128 roots x 128 particles at hand zero and
  before/after every new abstraction. These are additional evaluations.
- `local_br_gap.png`: automatically refreshed after every completed epoch.
  `analyze_epoch_ensemble.py` additionally creates `comparison.png` and
  `comparison_log.png` (logarithmic gap y axes) and `analysis.json`, including
  the fixed-audit curve. Counts remain linear; nonpositive values are masked
  on log axes without modifying the underlying measurements.
- The CSV is enough for later graphing; `rows_*.bin` are the much larger
  abstraction-training datasets, not needed merely to draw the curves.

`--particles 8` controls the **per-hand metric estimator**, not the number
of MCCFR training traversals. The metric remains a noisy root-only
one-step gap, not LBR match profit or full-game exploitability.

### Console Progress

Every 100 training hands the worker prints named fields:

```text
PROGRESS hand=277600 ensemble_gap_ante=0.255921 hard256_gap_ante=0.748312 epoch_infosets=3590799 cpp_train_eval_seconds=4925.75
```

- `hand`: completed training hands (one traversal per player for each method).
- `ensemble_gap_ante`, `hard256_gap_ante`: current policies' estimated local
  gaps at this hand's shared evaluation root, in antes. These are single-root
  samples, not rolling/cumulative averages or whole-game exploitability.
- `epoch_infosets`: exact temporary infoset entries for the current epoch,
  backed by the disk spool. Not final bucket count or RAM bytes; resets when
  the epoch table is released.
- `cpp_train_eval_seconds`: accumulated C++ training plus per-hand evaluation
  seconds for both methods. Excludes Python fitting, boundary audits, model
  export/load and other overhead, so it is not total wall time or an ETA.

`AUDITED hand=... ensemble_mean_gap_ante=... hard256_mean_gap_ante=...`
instead reports the fixed 128-root audit means. `EPOCH hand=...` marks the
completed batch export, before Python fits the next abstraction. CSV columns
are unchanged; every hand's metric is still recorded there.

Labels take effect on the next build/run. An already-running executable
continues its original format; do not interrupt a long run just to change
the labels, since exact resumption is not implemented.

### Memory And Storage

Observed during the archived joint-signature 100k run (not per-agent isolated
figures). These are NOT the corrected leaf-wise run's measurements; see
[Raw Rows Are Not Trees](#raw-rows-are-not-trees) and its linked run report.

| Resource | Observed value |
|---|---:|
| C++ process: both learners plus evaluation, peak working set | about 0.36 GiB |
| Python process: tree fitting, peak working set | about 10.92 GiB |
| Proposed model exports, ten epochs | 203.66 MiB |
| All raw `rows_*.bin` datasets | 23.84 GiB |

The process peaks are not a simultaneous combined peak. The C++ CSV does
not include Python fitting RAM. Raw infosets use a disk spool to avoid
keeping the whole temporary table in RAM, but LightGBM grouping still
allocates arrays for an epoch's data.

At 1M there are 100 frozen models rather than ten. Holding the epoch length
at 10k means fitting RAM need not grow tenfold, but future visit counts,
group sizes and retained models can increase it. **There is no measured
1M peak or enforced global RAM cap.** In particular, the C++ free-memory
guard is not a guard on Python fitting. The 10.92 GiB observation is a
planning baseline, not a promise that 11 GiB is sufficient.

A simple tenfold storage extrapolation gives roughly **2 GiB of proposed
model exports and 240 GiB of raw datasets**, before baseline checkpoint
versions and scratch space. These are estimates, not upper bounds; budget
additional free disk and monitor usage. All raw datasets are retained by
default. No prior datasets are automatically deleted. Files in this
family's `data/` remain separate from source code.

## Game Scope

This is the **existing two-player 7-stud v3 C++ game**, not Stud-Leduc.
H4 discard/reveal is the existing fixed heuristic. There is no H4 betting;
5th, 6th and 7th betting use the existing legal actions and raise caps 1/2/3.
Ante is 1000 chips, initial stack is 1000 antes. Both policies start without
learned regrets. One training hand means a fresh deal and one external-sampling
traversal for each player, not a single played trajectory.

The baseline is signed MCCFR with the existing frozen hard-256 power atlas
and cumulative betting-aggression summary. There are **256 card clusters per
street**, not 256 total regret entries: betting context and legal masks also
distinguish entries. The atlas was fitted previously on epsilon-0.2 self-play
data; its provenance is `data/power256_selfplay100m_eps20_v1.json`. That offline
atlas fitting cost is not included in this run. No learned baseline regret
checkpoint is loaded.

## Learning

For epoch k, the proposed policy is RM(sum_{j<k} j R_j(f_j(I)) + k D_k(I)).
Negative regrets remain signed. D_k uses lossless observable keys (own cards,
own discard, ordered public cards, betting history and public chip state).
It never uses opponent hidden cards/discard or the simulation deck. At the
boundary the full epoch data are first exported. After the pre-compression
audit, the temporary C++ table is released before Python fitting; its signed
regret totals and routing probes remain for validating the returned model.
All exported rows participate in bucket aggregation. A 40-million-infoset or
less-than-2-GiB-free-RAM guard aborts rather than silently dropping information.

The final version stores temporary exact infosets in an append-only disk spool,
sharded by the actor's initial 5th-street observation. Only the current hand's
two shards are mutable in RAM. A repeated initial observation reloads its
previous accumulated regrets and strategy sums; evaluation queries also load
stored shards when necessary. Latest segments, not obsolete versions, are
exported at the boundary. This is lossless storage, not an additional bucket
abstraction or a reset of D_k between hands. A repeated-root regression test
compares its updates and node visits with the original all-in-RAM version.
The temporary spool is truncated after export and the pre-boundary audit.

Labels come from the epoch's external-sampling strategy accumulator. As in
the existing MCCFR implementation, it accumulates at sampled non-traverser
nodes. Nodes without this mass use their final combined current policy.
Labels: illegal, [0,.1), [.1,.5), [.5,.9], (.9,1]. Group by player, street,
and exact legal-action mask before fitting.

The historical default fits one categorical LightGBM classifier per legal action, 3 boosting rounds,
4 leaves per tree, using at most 20k uniformly sampled rows per group. This
factorizes the label vector instead of forming an unbounded joint class set.
All rows, including those outside the fitting sample, are routed afterwards.
By default, sum **only this epoch's new signed regrets** into each tree's
individual leaves. Query one leaf per tree and average these vectors within
the epoch, then apply the epoch weight. Leaf payloads are sums over rows, not
row averages. This prevents tree count from multiplying an epoch's influence.
No joint-signature lookup is used; only entirely unseen groups can be missing.
`--aggregation joint` instead uses the archived joint signature table.

In legacy joint mode, Python groups the leaf IDs in lossless two-bit packed form (at most four
leaves per tree), then exports the original leaf IDs. This changes neither
the partition nor bucket regrets.

Features are the existing 18-dimensional power vector plus 16 public betting
features. C++ runs traversal, updates, exported-tree inference and rollouts;
Python only fits LightGBM at boundaries and plots the results. Models are kept
in C++ for inference; this is not an all-infoset cached Stud-Leduc experiment.

## Metric

Every training hand is followed by an independent evaluation deal shared by
both methods. Evaluate **the first actor's 5th-street information set**:

1. Sample hidden hands from the card prior conditioned on the viewer's known
   cards, public cards and the fixed H4 discard/reveal rule. This is a root
   with no betting history, so no action-history likelihood is needed.
2. For every legal first action, roll out both players' current policies to
   the terminal using the same particle and random-number seed across actions.
3. Average action values over particles **before** taking max_a Q(a)-pi.Q.

The output is an approximate local one-step deviation gain in **ante units**.
The existing terminal function already normalizes by ante; do not divide twice.
With 8 particles it is noisy and the max operator has selection bias. It is
not exact Local BR-gap, full-game exploitability, a Nash guarantee, or a
measurement over all streets. The cumulative mean is a mean of measured
gaps, not the gap of a time-average policy. Both methods are evaluated using
their current policy, not current versus average.

At hand 0 and before/after every abstraction boundary, an independent fixed
set of 128 evaluation roots uses 128 particles per root. The same conditional
particles and rollout seeds are reused across checkpoints and methods. These
audit roots are not used for training or selecting parameters. This reduces
sampling noise but does not eliminate finite-sample max bias. Results are in
`boundary_audit.csv`; the audit cost is additional to per-hand evaluation time.

`evaluate_epoch_ensemble.cpp` reloads the requested baseline checkpoint and
all `HANDS / EPOCH_HANDS` models for a separate final test. Its command is
`exe RUN_DIR ATLAS SEED ROOTS PARTICLES HANDS EPOCH_HANDS`; missing models or
incorrect weights fail rather than silently evaluating a shorter prefix.
The original 100k independent sample used seed 107, 256 new roots and
512 particles/root. It includes the baseline's average
policy and a uniform policy as additional references, and reports how many
epoch models contain a bucket for each root. This does not turn the root-only
metric into whole-game exploitability. Do not compare these numbers directly
with older trajectory-relaxed sum-of-gaps logs.

## Usage

From `D:/Experiment/Project-7-stud-Poker-Agent`, with the Python runtime that
provides NumPy/matplotlib. The runner reuses the installed LightGBM dependency
under the sibling Toy project's `.lightgbm_experiment` when present.

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --build --hands 100000 `
  --epoch-hands 10000 --particles 8 --seed 7 --aggregation leaf `
  --label-mode refined --tree-budget 24 `
  --out-dir agents/lightgbm_regret_ensemble/data/new_run

& $py agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --plot RUN_DIRECTORY
& $py agents/lightgbm_regret_ensemble/analyze_epoch_ensemble.py RUN_DIRECTORY
& $py -m unittest discover -s agents/lightgbm_regret_ensemble -p test_epoch_ensemble.py -v
agents/lightgbm_regret_ensemble/bin/stud_epoch_ensemble.exe --engine-self-test
agents/lightgbm_regret_ensemble/bin/stud_epoch_ensemble.exe --spool-test `
  agents/cpp_mccfr/data/power256_selfplay100m_eps20_v1.bin `
  agents/lightgbm_regret_ensemble/data/spool_test.bin
```

Use a **new** output directory. `--hands 200 --epoch-hands 100` is a pipeline
smoke test, not the main experimental schedule. Build uses static MinGW C++
runtime libraries because the existing dynamic runtime is incompatible.

## Policy-LBR Evaluation Of The 100k Models

`evaluate_epoch_lbr.cpp` uses the existing `PolicyLBR` and
`play_hand_policy_lbr` without changing the game or opponent search. It plays
5th through 7th, conditions particles on known cards and the fixed H4 rule,
and weights them by the frozen target's observed action probabilities.
Opponent hidden cards, opponent discard and the future deck are not inputs
to its decisions. Unlike the root gap, evaluation records **actual terminal
profit earned by the LBR**, in ante/hand. Lower is better for the target.

This is the existing local fold/call-equity search: future betting is
approximated by check/call-down, and raises use the target's immediate fold
probability. It replans at each real decision. This is neither full BR nor
exact exploitability. A weak finite-particle attack can miss weaknesses.
Particle posterior uses the existing 1e-12 action-probability floor; H4
rejection proposals are capped at 64 times the requested particle count.
Effective particle counts and policy-lookup misses are logged.

The original [Lisy and Bowling LBR paper](https://arxiv.org/abs/1612.07547)
also evaluates actual winnings under a one-action lookahead/check-down
approximation. This code adapts that idea to stud with sampled hidden hands
and the project's H4 rule; it does not reproduce the paper's exact hold'em
range enumeration. Even if LBR earns less against target A than target B,
that alone does not order their true exploitabilities.

Each independent deal is played with the LBR in each seat against each
target: ensemble current, hard-256 current, hard-256 average, uniform.
If every epoch uses `EPOCH7L1`, ensemble average is appended as a fifth
target; legacy checkpoints retain the original four. The analyzer accepts
both formats. `evaluation.json` also records the combined evaluator's peak
working set, not a separate RAM measurement for each target.
Policies are frozen. All targets share deal/seat random seeds, but different
actions can consume different random draws. Standard errors use the mean
of the two seat outcomes per deal; paired differences compare the same
deal across targets. CIs do not measure variation across training seeds.

```powershell
g++ -O3 -std=c++17 -DNOMINMAX -static-libstdc++ -static-libgcc `
  agents/lightgbm_regret_ensemble/evaluate_epoch_lbr.cpp `
  -o agents/lightgbm_regret_ensemble/bin/evaluate_epoch_lbr.exe -lpsapi
$run = 'agents/lightgbm_regret_ensemble/data/seven_stud_100k_seed7_spooled'
$atlas = 'agents/cpp_mccfr/data/power256_selfplay100m_eps20_v1.bin'
agents/lightgbm_regret_ensemble/bin/evaluate_epoch_lbr.exe --self-test $run $atlas 100000 10000
agents/lightgbm_regret_ensemble/bin/evaluate_epoch_lbr.exe $run $atlas "$run/lbr_seed307_p240" 5000 240 307 100000 10000
& $py agents/lightgbm_regret_ensemble/analyze_epoch_lbr.py "$run/lbr_seed307_p240"
```

The evaluator refuses an existing output directory. `5000` pairs means
10,000 hands **per target**, not 5,000 training traversals. Run the analyzer
after completion. The small seed-207/64-particle run is a smoke test only.
The shared ensemble `found` flag reports a matched joint signature in legacy
models, or an existing group with fully stored leaves in leaf models. Rebuild
evaluators before loading the new format. Archived model bytes and legacy
action probabilities are unchanged. The self-test covers private-card
invariance, valid distributions, deterministic replay, missing-bucket flags,
and the +1 ante/hand always-fold target sanity check.

### LBR Evaluation Of The 1M Models

Rebuild using the command above, then run from the project root:

```powershell
$run = 'agents/lightgbm_regret_ensemble/data/seven_stud_1m_seed7_20260919_204332'
$atlas = 'agents/cpp_mccfr/data/power256_selfplay100m_eps20_v1.bin'
agents/lightgbm_regret_ensemble/bin/evaluate_epoch_lbr.exe --self-test $run $atlas 1000000 10000
agents/lightgbm_regret_ensemble/bin/evaluate_epoch_lbr.exe $run $atlas "$run/lbr_seed307_p240" 5000 240 307 1000000 10000
& $py agents/lightgbm_regret_ensemble/analyze_epoch_lbr.py "$run/lbr_seed307_p240"
```

The final two arguments are mandatory, including for the self-test. Output
prints `CHECKPOINT training_hands=1000000 epoch_models=100` before playing;
`evaluation.json` records those counts. The original four targets, particle
count and seat-pairing are unchanged. This is a frozen-policy evaluation,
not more training. `lbr_comparison.png` stays linear;
`lbr_comparison_symlog.png` adds a signed-log view, linear within +/-1 ante,
so negative profits and confidence bounds are retained. The Local BR-gap
log plot uses ordinary logarithmic y axes.

## Saved Training Outputs

- `config.json`: seed, scope, source and atlas SHA256 hashes.
- `per_hand.csv`: both gaps, each method's training node counts, table sizes,
  combined C++ process peak working set, cumulative training/evaluation times.
- `residual_spool.bin`: scratch storage of exact current-epoch residuals;
  emptied after a successful boundary export/audit. It is not a saved policy.
- `rows_N.bin`: little-endian ROWS7V02 header, uint64 row count, fixed 340-byte
  records: uint32 group, float32[34] features, float64[8] new regret,
  float64[8] normalized teacher policy, uint64 visits, float64[8] strategy sum.
  Legacy ROWS7V01 omits the last field and is 276 bytes; the legacy offline
  `compare_labels.py` reads only those archived rows.
- `epoch_NNN/model.bin`: EPOCH7L1, epoch weight, grouped numerical trees and
  each tree's leaf regret vectors, strategy sums and row support counts.
  Legacy EPOCH7V1 stores only joint-signature regrets. LightGBM text models
  are retained in both modes, but do not contain the regret payloads.
- `hard256_N.bin`: baseline's existing model format.
- `local_br_gap.png`, `metrics_summary.json`, progress and error logs.

These are **model/data exports, not exact-resume checkpoints**: the per-epoch
lossless keys and RNG state are not serialized. To repeat training use the
recorded seed and command. Source self-tests check private-information
invariance, card uniqueness, zero-sum payout and signed regret; each fitted
model checks exported routing and conservation before the next epoch starts.

### Raw Rows Are Not Trees

For the corrected `data/leaf_refined_100k_seed7_20260920/` run:

| File | Role | Size |
|---|---|---:|
| `rows_10000.bin` | Exact-infoset statistics from hands 1..10,000 | 3.10 GiB |
| `epoch_001/model.bin` | Trees and aggregated leaf payloads from those rows | 1.01 MiB |
| `rows_100000.bin` | Exact-infoset statistics from hands 90,001..100,000 | 1.97 GiB |
| `epoch_010/model.bin` | Trees and aggregated leaf payloads from those rows | 0.70 MiB |
| All ten `model.bin` files | Complete frozen ensemble | 7.49 MiB |

`rows_100000.bin` has 6,217,051 uncompressed 340-byte records plus a 16-byte
header: exactly 2,113,797,356 bytes. A row is an interval accumulator for an
exact information set, not a hand, trajectory, or LightGBM tree node.
External sampling branches over the traverser's actions, so 10k hands can
produce millions of rows. Many rows are summarized in a single tree leaf.
The final ensemble needs **all ten** model exports and the shared power
atlas for features, not just `epoch_010/model.bin`. Frozen evaluation never
opens `rows_*.bin`; their purpose is auditing/refitting, not inference.
The two runs named in the cleanup notice no longer retain these raw files.

The observed 17.47 GiB Python fitting peak is not tree size. `fit_epoch`
memory-maps the raw file, then makes group-sized feature/teacher/regret/
strategy copies and routes **all** group rows through the trees. Predictions
and aggregation need additional temporary arrays. The 20k fitting cap limits
LightGBM training rows, not these allocations. The largest interval contains
35,648,078 rows (`rows_30000.bin`, 11.29 GiB), much more than the final interval.
Disk size, mapped resident pages, temporary arrays and model size are distinct.

A future memory reduction can keep the same fit sample and stream full-row
routing/leaf aggregation in chunks. This is not implemented here. Deleting
saved raw files frees disk space, not the working memory of a future fit.
The exported models are lossy abstractions, so their
small size alone does not establish equivalent policy quality.

Same hand budget does not imply the same node budget: policy-dependent
external sampling changes traversal size. The CSV records both. C++ peak RAM
is combined for both methods and excludes Python LightGBM RAM; it does not
establish a per-method memory advantage. The exact temporary table is large,
and representation-memory reduction is an empirical question, not a claim.

## Label Experiments (2026-09-20)

`--label-mode` selects `coarse` (historical default), `refined`, or
`cross_entropy`. Existing checkpoint files are unchanged. Refined edges are
exactly `[0, .05, .1, .2, .3, .4, .5, .95, 1]`: left-closed/right-open bins,
except `[.95, 1]`. The previous interval is `[.5, .95)`; .9 is no longer
a refined boundary. Archived results used .9 and are not relabeled.
Illegal actions remain excluded by the exact legal mask;
an illegal action is never confused with a legal action of probability zero.

Cross-entropy uses a separate soft Bernoulli target for each legal action:
`p = teacher_probability`, `q = sigmoid(tree_ensemble_score)`, and loss
`-p*log(q) - (1-p)*log(1-q)`. No sampling into a 0/1 label or probability
quantization occurs. For an unweighted row the gradient is `q-p` and Hessian
is `q*(1-q)`. LightGBM selects FEATURE thresholds that reduce its regularized
loss approximation. The target is continuous but the resulting tree leaves
and their leaf buckets are still finite and hard-routed.

The per-action predictions need not sum to one. They are NOT used as the
deployed policy: only the fitted tree routes are used. As before, new signed
regrets are summed into each individual tree leaf by default, epochs are weighted,
and regret matching determines actions. This is not regret regression,
soft bucket membership, a new CFR variant, or a convergence guarantee.

`--tree-budget 24` caps trees per action, not boosting rounds. For fully
represented labels this gives coarse: 6 rounds x 4 classes, refined:
3 rounds x 8 classes, soft cross-entropy: 24 rounds x 1 output. Missing
classes and early stopping of unsplittable trees can lower actual counts.
All trees still have at most four leaves. Without `--tree-budget`, all modes
use three rounds for backwards compatibility; that is NOT a matched-size
comparison. Config and epoch summaries record the chosen mode and budget.

Run from the project root, using a NEW output directory for each command:

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$source = 'agents/lightgbm_regret_ensemble/data/seven_stud_1m_seed7_20260919_204332'
# Cheap offline check of the three splitters, no more RL training:
& $py agents/lightgbm_regret_ensemble/compare_labels.py --source-run $source --tree-budget 24
# Fresh training, not resumption of the archived 1M ensemble:
& $py -u agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --build --label-mode refined --tree-budget 24 --hands 100000 --out-dir agents/lightgbm_regret_ensemble/data/NEW_REFINED_RUN
& $py -u agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --label-mode cross_entropy --tree-budget 24 --hands 100000 --out-dir agents/lightgbm_regret_ensemble/data/NEW_SOFT_RUN
# Runnable checks, including held-out label/regret isolation:
& $py -m unittest discover -s agents/lightgbm_regret_ensemble -p test_epoch_ensemble.py
```

`compare_labels.py` samples 200k rows from each of the saved 10k, 500k and
1M boundaries by default. Within each player/street/legal-mask group, all
identical 34-feature vectors are assigned to the same fold (80/20 by unique
feature vectors). Each mode gets exactly the same rows and at most 20k fit
rows per group. ONLY training-fold rows enter the regret tables. Booster
text files and compatible EPOCH7V1 exports stay under the new run's
`hand_<N>/<mode>/` directory; these individual epoch refits do NOT form a
freshly trained 1M ensemble. Train/test/fit source indices are saved.

Offline metrics:

- `prediction_tv`: half the L1 distance to the held-out teacher after
  normalizing the eight per-action probability predictions. Categorical
  predictions use TRAINING class-mean probabilities, not test means.
- `partition_tv`: held-out teacher versus the training teacher mean within
  its route. Unseen routes use that group's training teacher mean. This is
  a diagnostic decoder, NOT the deployed regret policy.
- `matched_partition_tv`: the same error on matched routes only. Different
  modes can match different subsets; compare with coverage, not alone.
- `unseen_route_fraction`: held-out joint signatures absent from training.
  In deployed inference an unmatched epoch contributes zero regret, not a
  predicted probability or training-mean teacher policy.
- `delta_regret_rm_tv`: RM of the stored bucket's epoch increment versus RM
  of the held-out individual increment, restricted to nonzero increments.
  This omits older epochs, is noisy, and is NOT CFR regret or exploitability.
- `model_bytes`: sampled epoch inference export size, NOT process RAM or
  full-training checkpoint size. Actual tree and bucket counts are saved.

There are no source hand IDs in ROWS7V01. Descendants of one hand may cross
folds even after identical-feature grouping. These are representation
diagnostics on one archived training run, not independent-hand confidence
intervals, LBR, or evidence of stronger game play. Data are predominantly
later-street infosets; aggregate metrics weight sampled rows, not streets
equally. A fresh online training comparison remains a separate experiment.

Official references: [LightGBM objectives and tree counts](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective),
[soft-label objective source](https://github.com/microsoft/LightGBM/blob/master/src/objective/xentropy_objective.hpp).

Completed offline results: [2026-09-20 label ablation](data/label_ablation_20260920/README.md).

Fresh self-play pilot (separate from the offline teacher-data refit):
[10k hands each, 1k-hand epochs, refined vs soft cross-entropy](data/online_labels_10k_20260920/README.md).
This uses independent cold-start policies, identical initial data and an
identical Hard-256 control. Subsequent training data follow each method's
own changing policy. Run `analyze_label_runs.py RUN_PAIR_DIRECTORY` after
both final independent audits to validate the pair and redraw its graphs.
