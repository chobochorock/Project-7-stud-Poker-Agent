# MCCFR bucket growth

## What a bucket is

The current MCCFR bucket is not a learned cluster. It is a dictionary entry
created from a hand-written information-set abstraction:

```text
bucket =
  hand category
  x primary-rank band
  x own public-card features
  x opponent public-card features
  x pot-odds band
  x remaining-stack/pot band
  x own raises this street
  x exact relative betting history
  x exact legal-action mask
```

The Python implementation serializes this tuple to JSON. The C++ experiment
packs the same fields into `InfoKey`; this changes lookup cost, not the
abstraction.

## When it grows

Every MCCFR traversal performs these steps:

1. Sample the opponent's hidden cards.
2. At an opponent node, sample one action from regret matching.
3. At a traverser node, evaluate every legal action.
4. For every visited information set, look up its key.
5. If the key is absent, create a zero-regret node immediately.

Therefore one traversal visits and may create many buckets:

```text
new buckets per traversal
  = newly encountered information sets along all traverser branches
```

It is not limited to the real decision root. There is no merge, eviction, or
minimum-visit rule. A freshly sampled hidden hand or a new betting history can
create another permanent entry.

## Why history is the main multiplier

With `h` legal histories and `c` card/ratio combinations, the reachable table
is roughly:

```text
|I_abstract| <= h * c
```

The v3 per-player raise caps make `h` finite, but they do not make it small.
`CHECK -> raise -> CALL` and `raise -> re-raise -> CALL` are deliberately
different information sets. The legal mask also changes after a check or when
a player's street raise cap is exhausted.

This is correct for perfect recall, but expensive. Removing history blindly
would merge states whose legal actions and strategic meaning differ.

## Reading the diagnostics

- `misses`: new buckets created during this run.
- `hits`: an existing bucket was reused.
- `single_touch_ratio`: table fraction seen only once.
- `new_buckets_per_traversal`: direct growth rate.
- `buckets_by_history_length`: where the branching multiplier appears.
- `buckets_by_hand_category`: whether rare made hands fragment the table.

A useful table should eventually show rising hit rate and falling new-bucket
rate. If the median remains one touch, most regret updates cannot accumulate
before the state distribution moves elsewhere.

## What C++ does and does not fix

C++ removes Python `deepcopy`, JSON serialization, and dictionary-object
overhead. It makes the same experiment much faster.

It does not improve coverage. If one-touch buckets remain dominant, the next
change should be a measured abstraction change, such as merging only card
features while retaining public betting history and legal actions. Speed and
state abstraction are separate problems.

## Frozen power buckets

The `power` mode replaces exact card/history products with:

```text
street
x frozen power centroid
x pot-odds and stack/pot bands
x own/opponent raise counts
x checked flag
x last public action class
x coarse betting-intent class
x legal-action mask
```

The power vector contains the final hand-category distribution, transformed
with `sqrt(p)` so dot products correspond to Bhattacharyya similarity. It also
contains expected primary tie-break rank and opponent public-card pressure.

Centroids are fitted before MCCFR and then frozen. Moving them during training
would change the meaning of existing regret entries.

In the first identical-seed 2,000-hand seventh-street comparison:

```text
legacy: 268,027 buckets, 61.3% single-touch, about -4.12 ante/hand
power:    2,604 buckets,  4.9% single-touch, about -0.50 ante/hand
```

This is evidence for continuing the experiment, not an equilibrium claim:
both agents learn online against one fixed heuristic. The power confidence
interval still included zero in this run.
