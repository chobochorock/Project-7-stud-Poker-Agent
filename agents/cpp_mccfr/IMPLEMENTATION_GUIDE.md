# Minimal implementation guide

## What to write yourself

For understanding, write a tiny Kuhn or Leduc implementation from scratch,
then review this 7-stud implementation. Writing the small version teaches the
algorithm; reviewing the larger version teaches engineering constraints.
Rewriting all 7-stud rules first mostly teaches debugging.

## Environment boundary

Only four operations are required:

```cpp
uint8_t valid_mask(const State&, int actor);
ActionResult apply_action(State&, int actor, Action);
double terminal_net_search(const State&, int player);
InfoKey make_information_key(const State&, int viewer);
```

`State` owns cards, public betting history, pot, contributions, current actor,
street, and terminal flags. `valid_mask` is the only authority for legal
actions. `apply_action` changes chips and history but does not choose actions.

The information key may contain only what `viewer` can observe:

```text
own hidden and public cards
opponent public cards
public betting history
pot/contributions/legal actions
```

Never use the opponent's sampled hidden cards in the viewer's key.

## One real hand

```text
shuffle deck
→ ante
→ H4 discard/reveal
→ deal 4th public
→ deal 5th public and bet
→ deal 6th public and bet
→ deal 7th hidden and bet
→ showdown
→ net chips
```

The outer match loop calls either the heuristic or MCCFR agent whenever the
environment requests an action.

## One MCCFR simulation

At a real decision root:

```text
copy the public state
→ keep the root player's private cards
→ sample opponent hidden/discarded cards
→ sample the future deck
→ traverse until fold or showdown
```

This is root determinization. It does not reveal the sampled opponent cards to
the root player's information key.

Traversal:

```text
terminal
→ return net chips / ante

opponent node
→ sample one regret-matching action

traverser node
→ evaluate every active legal action
→ compute node value
→ add action_value - node_value to regret
```

At a betting-round boundary, the simulator deals the next street from its
sampled deck and continues. The returned terminal value is propagated through
normal function returns; no separate tree object is required.

## Power atlas

Before CFR training:

```text
collect heuristic information states
→ enumerate/sample the viewer's remaining cards
→ estimate final hand-category probabilities p
→ transform to sqrt(p)
→ fit k-means separately for 5th/6th/7th
→ freeze and save centroids
```

During MCCFR, nearest-centroid assignment is cached by visible card masks.
Regret is indexed by the frozen centroid ID and coarse public betting state.

## Trash-line pruning

Permanent deletion is only safe for duplicate actions that produce the same
next state. Other bad actions keep their parent statistics:

```text
raw cumulative regret sufficiently negative
and current strategy probability zero
→ skip descendant generation temporarily
→ periodically evaluate all actions again
```

This experiment leaves pruning disabled by default. Its first setting roughly
doubled speed but worsened heuristic-match EV.
