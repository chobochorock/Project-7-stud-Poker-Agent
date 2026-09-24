from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
TOY_ROOT = (
    next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "PROJECT_LAYOUT.json").is_file()
    )
).parent / "Toy-Card-Game-Agent"
for dependency in (TOY_ROOT / ".open_spiel", TOY_ROOT / ".deep_cfr_deps"):
    if dependency.exists():
        sys.path.insert(0, str(dependency))

import numpy as np
import pyspiel
import torch
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, expected_game_score, exploitability

from stud_leduc_game import (
    Action,
    DECK,
    MAX_RAISES,
    information_key,
    load_game,
    rank,
)


VALUE_SCALE = 13.0
FEATURE_SIZE = 29


@dataclasses.dataclass(frozen=True)
class PublicKey:
    public_ranks: tuple[int, int]
    history: tuple[int, ...]
    contributions: tuple[int, int]


@dataclasses.dataclass
class PublicBelief:
    key: PublicKey
    mass: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )

    @property
    def reach(self) -> float:
        return float(self.mass.sum())

    @property
    def posterior(self) -> np.ndarray:
        total = self.reach
        if total <= 0.0:
            raise RuntimeError("empty public belief")
        return self.mass / total

    def features(self) -> np.ndarray:
        values = list(self.posterior.reshape(-1))
        for public_rank in self.key.public_ranks:
            values.extend(float(public_rank == candidate) for candidate in range(3))
        values.extend(value / VALUE_SCALE for value in self.key.contributions)
        for slot in range(4):
            action = self.key.history[slot] if slot < len(self.key.history) else -1
            values.extend(
                (
                    float(action == -1),
                    float(action == Action.CHECK_CALL),
                    float(action == Action.BET_RAISE),
                )
            )
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (FEATURE_SIZE,):
            raise RuntimeError(f"unexpected feature shape: {result.shape}")
        return result


@dataclasses.dataclass(frozen=True)
class BettingState:
    actor: int = 0
    round_bet: tuple[int, int] = (0, 0)
    raises: int = 0
    actions: tuple[int, ...] = ()
    contributions: tuple[int, int] = (1, 1)


@dataclasses.dataclass(frozen=True)
class Transition:
    state: BettingState | None
    folded: int | None = None
    round_complete: bool = False


def legal_actions(state: BettingState) -> tuple[int, ...]:
    call = max(state.round_bet) - state.round_bet[state.actor]
    actions = (
        [int(Action.FOLD), int(Action.CHECK_CALL)] if call else [int(Action.CHECK_CALL)]
    )
    if state.raises < MAX_RAISES:
        actions.append(int(Action.BET_RAISE))
    return tuple(actions)


def transition(state: BettingState, action: int, bet_size: int) -> Transition:
    player = state.actor
    call = max(state.round_bet) - state.round_bet[player]
    previous = state.actions[-1] if state.actions else None
    actions = state.actions + (int(action),)
    if action == Action.FOLD:
        return Transition(None, folded=player)

    round_bet = list(state.round_bet)
    contributions = list(state.contributions)
    if action == Action.BET_RAISE:
        paid = call + bet_size
        round_bet[player] += paid
        contributions[player] += paid
        return Transition(
            BettingState(
                actor=1 - player,
                round_bet=tuple(round_bet),
                raises=state.raises + 1,
                actions=actions,
                contributions=tuple(contributions),
            )
        )
    if call:
        round_bet[player] += call
        contributions[player] += call
        return Transition(
            BettingState(
                actor=player,
                round_bet=tuple(round_bet),
                raises=state.raises,
                actions=actions,
                contributions=tuple(contributions),
            ),
            round_complete=True,
        )
    if previous == Action.CHECK_CALL:
        return Transition(
            BettingState(
                actor=player,
                round_bet=tuple(round_bet),
                raises=state.raises,
                actions=actions,
                contributions=tuple(contributions),
            ),
            round_complete=True,
        )
    return Transition(
        BettingState(
            actor=1 - player,
            round_bet=tuple(round_bet),
            raises=state.raises,
            actions=actions,
            contributions=tuple(contributions),
        )
    )


def folded_payoff(contributions: tuple[int, int], folded: int, player: int) -> float:
    pot = sum(contributions)
    award = pot if player == 1 - folded else 0.0
    return award - contributions[player]


def showdown_payoff(
    private_ranks: tuple[int, int],
    public_ranks: tuple[int, int],
    contributions: tuple[int, int],
    player: int,
) -> float:
    values = [
        (
            int(private_ranks[index] == public_ranks[index]),
            max(private_ranks[index], public_ranks[index]),
            min(private_ranks[index], public_ranks[index]),
        )
        for index in range(2)
    ]
    winners = [index for index, value in enumerate(values) if value == max(values)]
    award = sum(contributions) / len(winners) if player in winners else 0.0
    return award - contributions[player]


class RegretTable:
    def __init__(self):
        self.regrets: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}
        self.legal: dict[str, tuple[int, ...]] = {}
        self.owner: dict[str, int] = {}
        self.iterations = 0

    def strategy(
        self, key: str, legal: tuple[int, ...], exploration: float = 0.0
    ) -> np.ndarray:
        self.legal.setdefault(key, legal)
        regret = self.regrets.setdefault(key, np.zeros(3, dtype=np.float64))
        positive = np.maximum(regret, 0.0)
        result = np.zeros(3, dtype=np.float64)
        total = positive[list(legal)].sum()
        if total > 0.0:
            result[list(legal)] = positive[list(legal)] / total
        else:
            result[list(legal)] = 1.0 / len(legal)
        if exploration > 0.0:
            result[list(legal)] = (1.0 - exploration) * result[
                list(legal)
            ] + exploration / len(legal)
        return result

    def average_strategy(self, key: str, legal: tuple[int, ...]) -> np.ndarray:
        total = self.strategy_sum.get(key)
        if total is None or total[list(legal)].sum() <= 0.0:
            return self.strategy(key, legal)
        result = np.zeros(3, dtype=np.float64)
        result[list(legal)] = total[list(legal)] / total[list(legal)].sum()
        return result

    def policy_map(self) -> dict[str, dict[int, float]]:
        result = {}
        for key, legal in self.legal.items():
            strategy = self.average_strategy(key, legal)
            result[key] = {action: float(strategy[action]) for action in legal}
        return result

    def diagnostics(self) -> dict[str, float]:
        maxima = [
            float(np.maximum(value, 0.0).max()) for value in self.regrets.values()
        ]
        positive = [
            float(x) for value in self.regrets.values() for x in np.maximum(value, 0.0)
        ]
        return {
            "infosets": len(self.regrets),
            "positive_regret_mean": float(np.mean(positive)) if positive else 0.0,
            "regret_decomposition_proxy": sum(maxima) / max(1, self.iterations),
        }


class FinalRoundSolver(RegretTable):
    def __init__(self, belief: PublicBelief):
        super().__init__()
        self.belief = belief

    def _key(
        self, player: int, private_ranks: tuple[int, int], state: BettingState
    ) -> str:
        history = tuple((0, action) for action in self.belief.key.history) + tuple(
            (1, action) for action in state.actions
        )
        return information_key(
            player,
            1,
            private_ranks[player],
            self.belief.key.public_ranks,
            history,
        )

    def _walk(
        self,
        private_ranks: tuple[int, int],
        state: BettingState,
        traverser: int,
        chance_reach: float,
        reaches: tuple[float, float],
        delta: dict[str, np.ndarray],
    ) -> float:
        player = state.actor
        legal = legal_actions(state)
        key = self._key(player, private_ranks, state)
        self.owner[key] = player
        strategy = self.strategy(key, legal)
        utilities = np.zeros(3, dtype=np.float64)
        for action in legal:
            child = transition(state, action, bet_size=4)
            child_reaches = list(reaches)
            child_reaches[player] *= strategy[action]
            if child.folded is not None:
                utilities[action] = folded_payoff(
                    child.state.contributions if child.state else state.contributions,
                    child.folded,
                    traverser,
                )
            elif child.round_complete:
                utilities[action] = showdown_payoff(
                    private_ranks,
                    self.belief.key.public_ranks,
                    child.state.contributions,
                    traverser,
                )
            else:
                utilities[action] = self._walk(
                    private_ranks,
                    child.state,
                    traverser,
                    chance_reach,
                    tuple(child_reaches),
                    delta,
                )
        node_value = float(np.dot(strategy, utilities))
        if player == traverser:
            counterfactual_reach = chance_reach * reaches[1 - player]
            update = delta.setdefault(key, np.zeros(3, dtype=np.float64))
            for action in legal:
                update[action] += counterfactual_reach * (
                    utilities[action] - node_value
                )
        return node_value

    def _accumulate_average(self, weight: float) -> None:
        seen: set[str] = set()

        def visit(
            private_ranks: tuple[int, int],
            state: BettingState,
            reaches: tuple[float, float],
        ):
            player = state.actor
            legal = legal_actions(state)
            key = self._key(player, private_ranks, state)
            strategy = self.strategy(key, legal)
            if key not in seen:
                seen.add(key)
                total = self.strategy_sum.setdefault(key, np.zeros(3, dtype=np.float64))
                total += weight * reaches[player] * strategy
            for action in legal:
                child = transition(state, action, bet_size=4)
                if child.folded is not None or child.round_complete:
                    continue
                child_reaches = list(reaches)
                child_reaches[player] *= strategy[action]
                visit(private_ranks, child.state, tuple(child_reaches))

        for deal in np.argwhere(self.belief.posterior > 0.0):
            visit(
                tuple(int(x) for x in deal),
                BettingState(contributions=self.belief.key.contributions),
                (1.0, 1.0),
            )

    def solve(self, iterations: int) -> None:
        posterior = self.belief.posterior
        for _ in range(iterations):
            for traverser in range(2):
                delta: dict[str, np.ndarray] = {}
                for deal in np.argwhere(posterior > 0.0):
                    private_ranks = tuple(int(x) for x in deal)
                    self._walk(
                        private_ranks,
                        BettingState(contributions=self.belief.key.contributions),
                        traverser,
                        float(posterior[private_ranks]),
                        (1.0, 1.0),
                        delta,
                    )
                for key, update in delta.items():
                    self.regrets[key] = np.maximum(0.0, self.regrets[key] + update)
            self.iterations += 1
            self._accumulate_average(float(self.iterations))

    def _expected(
        self,
        private_ranks: tuple[int, int],
        state: BettingState,
        player: int,
    ) -> float:
        actor = state.actor
        legal = legal_actions(state)
        strategy = self.average_strategy(self._key(actor, private_ranks, state), legal)
        value = 0.0
        for action in legal:
            child = transition(state, action, bet_size=4)
            if child.folded is not None:
                child_value = folded_payoff(
                    child.state.contributions if child.state else state.contributions,
                    child.folded,
                    player,
                )
            elif child.round_complete:
                child_value = showdown_payoff(
                    private_ranks,
                    self.belief.key.public_ranks,
                    child.state.contributions,
                    player,
                )
            else:
                child_value = self._expected(private_ranks, child.state, player)
            value += strategy[action] * child_value
        return value

    def value_target(self) -> tuple[np.ndarray, np.ndarray]:
        posterior = self.belief.posterior
        values = np.zeros((2, 3), dtype=np.float32)
        masks = np.zeros((2, 3), dtype=np.float32)
        for player in range(2):
            for private_rank in range(3):
                conditional = (
                    posterior[private_rank, :]
                    if player == 0
                    else posterior[:, private_rank]
                )
                mass = float(conditional.sum())
                if mass <= 0.0:
                    continue
                total = 0.0
                for opponent_rank, probability in enumerate(conditional):
                    if probability <= 0.0:
                        continue
                    deal = (
                        (private_rank, opponent_rank)
                        if player == 0
                        else (opponent_rank, private_rank)
                    )
                    total += probability * self._expected(
                        deal,
                        BettingState(contributions=self.belief.key.contributions),
                        player,
                    )
                values[player, private_rank] = total / mass
                masks[player, private_rank] = 1.0
        return values, masks


class ValueNetwork(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_SIZE, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 6),
        )

    def forward(self, inputs):
        return self.model(inputs)


class Replay:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, belief: PublicBelief, values: np.ndarray, masks: np.ndarray) -> None:
        item = (
            belief.features(),
            values.reshape(-1) / VALUE_SCALE,
            masks.reshape(-1),
            belief.reach,
        )
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = item

    def batch(self, size: int) -> tuple[torch.Tensor, ...]:
        chosen = self.rng.choices(self.items, k=min(size, len(self.items)))
        columns = list(zip(*chosen))
        return tuple(
            torch.as_tensor(np.stack(column), dtype=torch.float32)
            for column in columns[:3]
        )


def train_step(network, optimizer, replay: Replay, batch_size: int) -> float:
    if not replay.items:
        return 0.0
    features, targets, masks = replay.batch(batch_size)
    predictions = network(features)
    squared = (predictions - targets) ** 2 * masks
    loss = squared.sum() / masks.sum().clamp_min(1.0)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach()) * VALUE_SCALE * VALUE_SCALE


def predict_values(network, beliefs: list[PublicBelief]) -> dict[PublicKey, np.ndarray]:
    if not beliefs:
        return {}
    features = torch.as_tensor(
        np.stack([belief.features() for belief in beliefs]), dtype=torch.float32
    )
    with torch.no_grad():
        outputs = network(features).cpu().numpy().reshape(-1, 2, 3) * VALUE_SCALE
    return {belief.key: output for belief, output in zip(beliefs, outputs)}


class RootSolver(RegretTable):
    def __init__(self, exploration: float):
        super().__init__()
        self.exploration = exploration

    @staticmethod
    def _key(player: int, private_cards: tuple[int, int], state: BettingState) -> str:
        return information_key(
            player,
            0,
            rank(private_cards[player]),
            None,
            tuple((0, action) for action in state.actions),
        )

    def collect_beliefs(self, average: bool) -> list[PublicBelief]:
        beliefs: dict[PublicKey, PublicBelief] = {}

        def visit(private_cards, state, reach_probability):
            player = state.actor
            legal = legal_actions(state)
            key = self._key(player, private_cards, state)
            strategy = (
                self.average_strategy(key, legal)
                if average
                else self.strategy(key, legal, self.exploration)
            )
            for action in legal:
                probability = strategy[action]
                if probability <= 0.0:
                    continue
                child = transition(state, action, bet_size=2)
                if child.folded is not None:
                    continue
                if child.round_complete:
                    remaining = [card for card in DECK if card not in private_cards]
                    public_probability = 1.0 / (len(remaining) * (len(remaining) - 1))
                    for first in remaining:
                        for second in remaining:
                            if second == first:
                                continue
                            public_ranks = (rank(first), rank(second))
                            public_key = PublicKey(
                                public_ranks,
                                child.state.actions,
                                child.state.contributions,
                            )
                            belief = beliefs.setdefault(
                                public_key, PublicBelief(public_key)
                            )
                            private_ranks = (
                                rank(private_cards[0]),
                                rank(private_cards[1]),
                            )
                            belief.mass[private_ranks] += (
                                reach_probability * probability * public_probability
                            )
                    continue
                visit(private_cards, child.state, reach_probability * probability)

        private_probability = 1.0 / (len(DECK) * (len(DECK) - 1))
        for first in DECK:
            for second in DECK:
                if second != first:
                    visit((first, second), BettingState(), private_probability)
        return sorted(beliefs.values(), key=lambda belief: repr(belief.key))

    def _leaf_value(
        self,
        private_cards: tuple[int, int],
        state: BettingState,
        traverser: int,
        leaf_values: dict[PublicKey, np.ndarray],
    ) -> float:
        remaining = [card for card in DECK if card not in private_cards]
        probability = 1.0 / (len(remaining) * (len(remaining) - 1))
        value = 0.0
        for first in remaining:
            for second in remaining:
                if first == second:
                    continue
                key = PublicKey(
                    (rank(first), rank(second)),
                    state.actions,
                    state.contributions,
                )
                estimate = leaf_values.get(key)
                if estimate is not None:
                    value += (
                        probability
                        * estimate[traverser, rank(private_cards[traverser])]
                    )
        return value

    def _walk(
        self,
        private_cards: tuple[int, int],
        state: BettingState,
        traverser: int,
        chance_reach: float,
        reaches: tuple[float, float],
        leaf_values: dict[PublicKey, np.ndarray],
        delta: dict[str, np.ndarray],
    ) -> float:
        player = state.actor
        legal = legal_actions(state)
        key = self._key(player, private_cards, state)
        self.owner[key] = player
        strategy = self.strategy(key, legal, self.exploration)
        utilities = np.zeros(3, dtype=np.float64)
        for action in legal:
            child = transition(state, action, bet_size=2)
            child_reaches = list(reaches)
            child_reaches[player] *= strategy[action]
            if child.folded is not None:
                utilities[action] = folded_payoff(
                    child.state.contributions if child.state else state.contributions,
                    child.folded,
                    traverser,
                )
            elif child.round_complete:
                utilities[action] = self._leaf_value(
                    private_cards, child.state, traverser, leaf_values
                )
            else:
                utilities[action] = self._walk(
                    private_cards,
                    child.state,
                    traverser,
                    chance_reach,
                    tuple(child_reaches),
                    leaf_values,
                    delta,
                )
        node_value = float(np.dot(strategy, utilities))
        if player == traverser:
            update = delta.setdefault(key, np.zeros(3, dtype=np.float64))
            counterfactual_reach = chance_reach * reaches[1 - player]
            for action in legal:
                update[action] += counterfactual_reach * (
                    utilities[action] - node_value
                )
        return node_value

    def _accumulate_average(self, weight: float) -> None:
        seen: set[str] = set()

        def visit(private_cards, state, reaches):
            player = state.actor
            legal = legal_actions(state)
            key = self._key(player, private_cards, state)
            strategy = self.strategy(key, legal, self.exploration)
            if key not in seen:
                seen.add(key)
                total = self.strategy_sum.setdefault(key, np.zeros(3, dtype=np.float64))
                total += weight * reaches[player] * strategy
            for action in legal:
                child = transition(state, action, bet_size=2)
                if child.folded is not None or child.round_complete:
                    continue
                child_reaches = list(reaches)
                child_reaches[player] *= strategy[action]
                visit(private_cards, child.state, tuple(child_reaches))

        for first in DECK:
            for second in DECK:
                if first != second:
                    visit((first, second), BettingState(), (1.0, 1.0))

    def search(self, iterations: int, target_network) -> None:
        private_probability = 1.0 / (len(DECK) * (len(DECK) - 1))
        for _ in range(iterations):
            for traverser in range(2):
                beliefs = self.collect_beliefs(average=False)
                leaf_values = predict_values(target_network, beliefs)
                delta: dict[str, np.ndarray] = {}
                for first in DECK:
                    for second in DECK:
                        if first == second:
                            continue
                        self._walk(
                            (first, second),
                            BettingState(),
                            traverser,
                            private_probability,
                            (1.0, 1.0),
                            leaf_values,
                            delta,
                        )
                for key, update in delta.items():
                    self.regrets[key] = np.maximum(0.0, self.regrets[key] + update)
            self.iterations += 1
            self._accumulate_average(float(self.iterations))


class TablePolicy(policy_lib.Policy):
    def __init__(self, game, table):
        super().__init__(game, [0, 1])
        self.table = table

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else player_id
        key = state.information_state_string(player)
        probabilities = self.table.get(key)
        if probabilities is not None:
            return probabilities
        legal = state.legal_actions(player)
        return {action: 1.0 / len(legal) for action in legal}


def resolve_beliefs(
    beliefs: list[PublicBelief],
    iterations: int,
) -> tuple[
    dict[str, dict[int, float]],
    list[tuple[PublicBelief, np.ndarray, np.ndarray]],
    float,
]:
    policies: dict[str, dict[int, float]] = {}
    targets = []
    regret_proxy = 0.0
    for belief in beliefs:
        solver = FinalRoundSolver(belief)
        solver.solve(iterations)
        policies.update(solver.policy_map())
        values, masks = solver.value_target()
        targets.append((belief, values, masks))
        regret_proxy += (
            solver.diagnostics()["regret_decomposition_proxy"] * belief.reach
        )
    return policies, targets, regret_proxy


def target_mae(network, targets) -> tuple[float, float]:
    if not targets:
        return 0.0, 0.0
    predictions = predict_values(network, [target[0] for target in targets])
    absolute_sum = 0.0
    mask_sum = 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for belief, values, masks in targets:
        error = np.abs(predictions[belief.key] - values) * masks
        count = float(masks.sum())
        absolute_sum += float(error.sum())
        mask_sum += count
        weighted_sum += float(error.sum()) * belief.reach / max(1.0, count)
        weight_total += belief.reach
    return absolute_sum / max(1.0, mask_sum), weighted_sum / max(1e-12, weight_total)


def evaluate_policy(game, table) -> tuple[float, float]:
    policy = TablePolicy(game, table)
    value = expected_game_score.policy_value(game.new_initial_state(), [policy, policy])
    return float(exploitability.exploitability(game, policy)), float(value[0])


def run(args) -> list[dict[str, float]]:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    game = load_game()
    network = ValueNetwork(args.hidden)
    target_network = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    replay = Replay(args.replay_capacity, args.seed)
    root = RootSolver(args.search_exploration)
    baseline = cfr.CFRPlusSolver(game)
    metrics: list[dict[str, float]] = []
    started = time.perf_counter()

    if args.load_checkpoint:
        saved = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        network.load_state_dict(saved["network"])
        target_network.load_state_dict(saved["target_network"])
        optimizer.load_state_dict(saved["optimizer"])
        root.regrets = saved["root_regrets"]
        root.strategy_sum = saved["root_strategy_sum"]
        root.legal = saved["root_legal"]
        root.owner = saved["root_owner"]
        root.iterations = saved["root_iterations"]
        if args.eval_only:
            beliefs = root.collect_beliefs(average=True)
            local_policy, targets, local_regret = resolve_beliefs(
                beliefs, args.leaf_iterations
            )
            table = root.policy_map()
            table.update(local_policy)
            exact_exploitability, policy_value = evaluate_policy(game, table)
            mae, weighted_mae = target_mae(network, targets)
            report = {
                "evaluation": "checkpoint-leaf-budget-ablation",
                "root_cfr_iterations": root.iterations,
                "leaf_cfr_iterations": args.leaf_iterations,
                "public_belief_states": len(beliefs),
                "exploitability": exact_exploitability,
                "policy_value_p0": policy_value,
                "value_mae": mae,
                "reach_weighted_value_mae": weighted_mae,
                "leaf_reach_weighted_regret_proxy": local_regret,
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(json.dumps(report))
            return [report]

    # Uniform-root exact solves give the first target network a meaningful scale.
    if not args.load_checkpoint:
        warm_beliefs = root.collect_beliefs(average=False)
        _, warm_targets, _ = resolve_beliefs(warm_beliefs, args.leaf_iterations)
        for belief, values, masks in warm_targets:
            replay.add(belief, values, masks)
            train_step(network, optimizer, replay, args.batch_size)
        for _ in range(args.warmup_steps):
            train_step(network, optimizer, replay, args.batch_size)
        target_network.load_state_dict(network.state_dict())

    for outer in range(1, args.outer_iterations + 1):
        root.search(args.root_iterations, target_network)
        beliefs = root.collect_beliefs(average=True)
        local_policy, targets, local_regret = resolve_beliefs(
            beliefs, args.leaf_iterations
        )
        pre_mae, pre_weighted_mae = target_mae(network, targets)
        losses = []
        for belief, values, masks in targets:
            replay.add(belief, values, masks)
            losses.append(train_step(network, optimizer, replay, args.batch_size))
        for _ in range(args.train_steps):
            losses.append(train_step(network, optimizer, replay, args.batch_size))
        post_mae, post_weighted_mae = target_mae(network, targets)
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_network.parameters(), network.parameters()
            ):
                target_parameter.mul_(1.0 - args.target_tau).add_(
                    parameter, alpha=args.target_tau
                )

        table = root.policy_map()
        table.update(local_policy)
        exact_exploitability, policy_value = evaluate_policy(game, table)
        for _ in range(args.baseline_iterations):
            baseline.evaluate_and_update_policy()
        baseline_exploitability = float(
            exploitability.exploitability(game, baseline.average_policy())
        )
        report = {
            "outer_iteration": outer,
            "root_cfr_iterations": root.iterations,
            "leaf_cfr_iterations": args.leaf_iterations,
            "public_belief_states": len(beliefs),
            "replay_size": len(replay.items),
            "replay_seen": replay.seen,
            "exploitability": exact_exploitability,
            "policy_value_p0": policy_value,
            "baseline_cfrplus_iterations": outer * args.baseline_iterations,
            "baseline_exploitability": baseline_exploitability,
            "value_mae_before_update": pre_mae,
            "value_mae_after_update": post_mae,
            "reach_weighted_value_mae_before": pre_weighted_mae,
            "reach_weighted_value_mae_after": post_weighted_mae,
            "value_training_mse": float(np.mean(losses)) if losses else 0.0,
            "root_regret_proxy": root.diagnostics()["regret_decomposition_proxy"],
            "leaf_reach_weighted_regret_proxy": local_regret,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics.append(report)
        print(json.dumps(report), flush=True)
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            with args.metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report) + "\n")
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "network": network.state_dict(),
                    "target_network": target_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "root_regrets": root.regrets,
                    "root_strategy_sum": root.strategy_sum,
                    "root_legal": root.legal,
                    "root_owner": root.owner,
                    "root_iterations": root.iterations,
                    "metrics": metrics,
                    "config": vars(args),
                },
                args.checkpoint,
            )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Isolated online PBS-search/value-learning experiment"
    )
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--root-iterations", type=int, default=10)
    parser.add_argument("--leaf-iterations", type=int, default=50)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--target-tau", type=float, default=0.2)
    parser.add_argument("--search-exploration", type=float, default=0.02)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        args.outer_iterations = 2
        args.root_iterations = 2
        args.leaf_iterations = 3
        args.baseline_iterations = 2
        args.hidden = 16
        args.batch_size = 8
        args.warmup_steps = 3
        args.train_steps = 2
        args.metrics = None
        args.checkpoint = None
        args.load_checkpoint = None
        args.eval_only = False
    positive = (
        args.outer_iterations,
        args.root_iterations,
        args.leaf_iterations,
        args.baseline_iterations,
        args.hidden,
        args.batch_size,
        args.replay_capacity,
        args.threads,
    )
    if min(positive) <= 0 or args.warmup_steps < 0 or args.train_steps < 0:
        parser.error("counts and sizes must be positive")
    if not 0.0 < args.target_tau <= 1.0:
        parser.error("--target-tau must be in (0, 1]")
    if not 0.0 <= args.search_exploration < 1.0:
        parser.error("--search-exploration must be in [0, 1)")
    if args.eval_only and not args.load_checkpoint:
        parser.error("--eval-only requires --load-checkpoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    metrics = run(args)
    if args.self_test:
        assert len(metrics) == 2
        assert all(math.isfinite(row["exploitability"]) for row in metrics)
        assert all(row["public_belief_states"] > 0 for row in metrics)
        print('{"self_test":"ok"}')


if __name__ == "__main__":
    main()
