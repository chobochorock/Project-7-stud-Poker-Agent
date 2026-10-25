"""Exact best response and exploitability on a cached two-player zero-sum game tree (NumPy).

Usage (from code, any Python with NumPy):
    levels = tree_levels(tree)
    value = exploitability(tree, levels, policy, action_counts)
Input: `tree` = (players, information, degree, children, chance, utility) in the format of
sampled_ira_native.cache_tree / mini_stud_ira.dense_tree (preorder, children padded with -1,
utility = player 0 return at terminals); `policy[information_set, position]` with rows that
sum to one over the legal positions; `action_counts[information_set]`.
Output: NashConv/2 in the utility unit (antes for the mini-stud family), the same quantity as
OpenSpiel's exploitability for two-player zero-sum games.

The best response takes, at every information set of the responder, the action with the
largest counterfactual value summed over its histories (hidden histories averaged BEFORE the
max). It needs every history of an information set to lie at the same tree depth, which
check_same_depth() verifies; perfect recall of the game is assumed. Checked against OpenSpiel
on Kuhn and Stud-Leduc in test_mini_stud_ira.py.
"""

from __future__ import annotations

import numpy as np

TERMINAL_NODE = -2
CHANCE_NODE = -1


def tree_levels(tree) -> list[np.ndarray]:
    """Node ids grouped by depth (root first)."""
    children = tree[3]
    levels = []
    frontier = np.array([0], dtype=np.int64)
    while frontier.size:
        levels.append(frontier)
        following = children[frontier]
        frontier = following[following >= 0].astype(np.int64)
    return levels


def check_same_depth(tree, levels, information_sets: int) -> None:
    information = tree[1]
    depth = np.empty(len(information), dtype=np.int64)
    for level, nodes in enumerate(levels):
        depth[nodes] = level
    decisions = np.flatnonzero(information >= 0)
    low = np.full(information_sets, np.iinfo(np.int64).max)
    high = np.full(information_sets, -1)
    np.minimum.at(low, information[decisions], depth[decisions])
    np.maximum.at(high, information[decisions], depth[decisions])
    if np.any(low != high):
        raise ValueError(
            "some information set spans several tree depths; single-pass BR invalid"
        )


def best_response_value(
    tree, levels, policy: np.ndarray, action_counts: np.ndarray, responder: int
) -> float:
    """Value for `responder` of a best response to the other player's `policy`."""
    players, information, _, children, chance, utility = tree
    width = children.shape[1]
    valid = children >= 0
    reach = np.zeros(
        len(players)
    )  # chance x opponent reach (counterfactual for the responder)
    reach[0] = 1.0
    for nodes in levels:
        active = nodes[players[nodes] != TERMINAL_NODE]
        if not active.size:
            continue
        kinds = players[active]
        weight = np.zeros((active.size, width))
        is_chance = kinds == CHANCE_NODE
        weight[is_chance] = chance[active[is_chance]]
        is_opponent = kinds == 1 - responder
        weight[is_opponent] = policy[information[active[is_opponent]]]
        weight[kinds == responder] = 1.0
        mask = valid[active]
        reach[children[active][mask]] = (reach[active][:, None] * weight)[mask]

    values = utility * (1.0 if responder == 0 else -1.0)
    totals = np.zeros((len(action_counts), width))
    positions = np.arange(width)
    for nodes in reversed(levels):
        active = nodes[players[nodes] != TERMINAL_NODE]
        if not active.size:
            continue
        kinds = players[active]
        mask = valid[active]
        child_values = np.where(mask, values[np.where(mask, children[active], 0)], 0.0)
        result = np.empty(active.size)
        is_chance = kinds == CHANCE_NODE
        result[is_chance] = (chance[active[is_chance]] * child_values[is_chance]).sum(
            axis=1
        )
        is_opponent = kinds == 1 - responder
        result[is_opponent] = (
            policy[information[active[is_opponent]]] * child_values[is_opponent]
        ).sum(axis=1)
        is_own = kinds == responder
        if is_own.any():
            own_nodes = active[is_own]
            sets = information[own_nodes]
            np.add.at(totals, sets, child_values[is_own] * reach[own_nodes][:, None])
            legal = positions[None, :] < action_counts[sets][:, None]
            best = np.argmax(np.where(legal, totals[sets], -np.inf), axis=1)
            result[is_own] = child_values[is_own][np.arange(own_nodes.size), best]
        values[active] = result
    return float(values[0])


def exploitability(
    tree,
    levels,
    policy: np.ndarray,
    action_counts: np.ndarray,
    utility_sum: float = 0.0,
) -> float:
    """NashConv / 2 of the profile `policy` (both players' rows in one array)."""
    total = sum(
        best_response_value(tree, levels, policy, action_counts, responder)
        for responder in (0, 1)
    )
    return (total - utility_sum) / 2.0
