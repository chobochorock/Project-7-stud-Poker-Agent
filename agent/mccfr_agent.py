from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agent.heuristic_agent import HeuristicPokerAgent
from agent.uct_agent import _EVSimulation
from poker_env import (
    BETTING_RULES_VERSION,
    Card,
    get_best_hand,
    get_public_betting_priority,
)


MCCFR_START_STREETS = ("6th", "7th_hidden")


@dataclass
class _RegretNode:
    regrets: dict[str, float] = field(default_factory=dict)
    strategy_sum: dict[str, float] = field(default_factory=dict)

    def strategy(self, actions: Sequence[str]) -> dict[str, float]:
        for action in actions:
            self.regrets.setdefault(action, 0.0)
            self.strategy_sum.setdefault(action, 0.0)
        positive = {action: max(0.0, self.regrets[action]) for action in actions}
        total = sum(positive.values())
        if total == 0.0:
            return {action: 1.0 / len(actions) for action in actions}
        return {action: positive[action] / total for action in actions}

    def average_strategy(self, actions: Sequence[str]) -> dict[str, float]:
        self.strategy(actions)
        total = sum(self.strategy_sum[action] for action in actions)
        if total == 0.0:
            return self.strategy(actions)
        return {action: self.strategy_sum[action] / total for action in actions}


class MCCFRPokerAgent(HeuristicPokerAgent):
    """Heuristic early streets with bucketed MCCFR on seventh street."""

    def __init__(
        self,
        name: str,
        iterations: int = 16,
        seed: int | None = None,
        raise_cap: int = 2,
        start_street: str = "7th_hidden",
        freeze_seventh: bool = False,
        decision_strategy: str = "average",
    ):
        super().__init__(name)
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if raise_cap < 0:
            raise ValueError("raise_cap must be non-negative")
        if start_street not in MCCFR_START_STREETS:
            raise ValueError(f"Unsupported MCCFR start street: {start_street}")
        if freeze_seventh and start_street != "6th":
            raise ValueError("Frozen seventh-street continuation requires start_street=6th")
        if decision_strategy not in {"average", "current"}:
            raise ValueError("decision_strategy must be average or current")
        self.iterations = iterations
        self.raise_cap = raise_cap
        self.start_street = start_street
        self.freeze_seventh = freeze_seventh
        self.decision_strategy = decision_strategy
        self.rng = random.Random(seed)
        self.nodes: dict[str, _RegretNode] = {}
        self.decisions = 0
        self.heuristic_decisions = 0
        self.traversals = 0
        self.last_strategy: dict[str, float] = {}
        self.last_bucket = ""

    def choose_action(
        self, state: dict[str, Any], valid_actions: Sequence[str]
    ) -> str | None:
        actions = tuple(valid_actions)
        if not actions:
            return None
        street = str(state.get("street"))
        if (
            state.get("game_mode") != "ev"
            or state.get("seat_count") != 2
        ):
            self.heuristic_decisions += 1
            return super().choose_action(state, actions)
        if self.freeze_seventh and street == "7th_hidden":
            return self._choose_from_table(state, actions)
        if not self._mccfr_active(street):
            self.heuristic_decisions += 1
            return super().choose_action(state, actions)

        root_seat = int(state.get("seat_index", 0))
        for iteration in range(self.iterations):
            simulation = _EVSimulation(state, self.rng, raise_cap=self.raise_cap)
            traverser = root_seat if iteration % 2 == 0 else 1 - root_seat
            self._traverse(simulation, traverser, [1.0, 1.0])
            self.traversals += 1

        return self._choose_from_table(state, actions)

    def _choose_from_table(
        self, state: dict[str, Any], actions: Sequence[str]
    ) -> str:
        filtered_actions = tuple(
            action
            for action in actions
            if self.raise_count_allows(state, action)
        )
        key = self._bucket_key(state)
        node = self.nodes.setdefault(key, _RegretNode())
        use_current = self.decision_strategy == "current" and not (
            self.freeze_seventh and state.get("street") == "7th_hidden"
        )
        self.last_strategy = (
            node.strategy(filtered_actions)
            if use_current
            else node.average_strategy(filtered_actions)
        )
        self.last_bucket = key
        self.decisions += 1
        return self._sample(self.last_strategy)

    def raise_count_allows(self, state: dict[str, Any], action: str) -> bool:
        return not (
            action in {"DDADANG", "QUARTER", "HALF"}
            and int(state.get("raise_count", 0)) >= self.raise_cap
        )

    def _traverse(
        self,
        simulation: _EVSimulation,
        traverser: int,
        reach: list[float],
    ) -> float:
        if simulation.terminal:
            return simulation.terminal_net(traverser) / max(1.0, simulation.ante)

        if self.freeze_seventh and simulation.street == "7th_hidden":
            actor = simulation.actor
            actions = tuple(simulation.valid_actions())
            node = self.nodes.get(self._bucket_key(simulation.observation(actor)))
            strategy = (
                node.average_strategy(actions)
                if node is not None
                else {action: 1.0 / len(actions) for action in actions}
            )
            simulation.apply(self._sample(strategy))
            return self._traverse(simulation, traverser, reach)

        actor = simulation.actor
        actions = tuple(simulation.valid_actions())
        key = self._bucket_key(simulation.observation(actor))
        node = self.nodes.setdefault(key, _RegretNode())
        strategy = node.strategy(actions)

        if actor != traverser:
            action = self._sample(strategy)
            next_reach = reach.copy()
            next_reach[actor] *= strategy[action]
            simulation.apply(action)
            return self._traverse(simulation, traverser, next_reach)

        for action in actions:
            node.strategy_sum[action] += reach[actor] * strategy[action]
        action_values: dict[str, float] = {}
        for action in actions:
            child = copy.deepcopy(simulation)
            next_reach = reach.copy()
            next_reach[actor] *= strategy[action]
            child.apply(action)
            action_values[action] = self._traverse(child, traverser, next_reach)

        node_value = sum(strategy[action] * action_values[action] for action in actions)
        for action in actions:
            node.regrets[action] += action_values[action] - node_value
        return node_value

    def _sample(self, probabilities: dict[str, float]) -> str:
        threshold = self.rng.random()
        cumulative = 0.0
        last = next(iter(probabilities))
        for action, probability in probabilities.items():
            last = action
            cumulative += probability
            if threshold <= cumulative:
                return action
        return last

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        return super().choose_discard_and_reveal(hidden_cards)

    def _bucket_payload(self, state: dict[str, Any]) -> tuple[Any, ...]:
        own_hidden = self._cards(state.get("my_hidden_cards", []))
        own_public = self._cards(state.get("my_public_cards", []))
        opponent_public = self._cards(
            state.get("opponents", [{}])[0].get("public_cards", [])
        )
        score = get_best_hand(own_hidden + own_public)
        primary_rank = score[1] if len(score) > 1 else 0
        call = float(state.get("call_amount", 0) or 0)
        pot = max(1.0, float(state.get("pot", 0) or 0))
        chips = max(0.0, float(state.get("my_chips", 0) or 0))
        history = tuple(
            (event.get("actor"), event.get("action"))
            for event in state.get("betting_history", [])
            if event.get("street") == state.get("street")
        )
        payload = (
            score[0],
            min(3, max(0, (primary_rank - 2) // 4)),
            self._public_features(own_public),
            self._public_features(opponent_public),
            self._ratio_bucket(call / max(1.0, pot + call), (0.1, 0.2, 0.33, 0.5)),
            self._ratio_bucket(chips / pot, (0.5, 1.0, 2.0)),
            min(self.raise_cap, int(state.get("raise_count", 0))),
            history,
            tuple(state.get("valid_actions", ())),
        )
        if self.start_street == "6th":
            payload = (str(state.get("street")), *payload)
        return payload

    def _bucket_key(self, state: dict[str, Any]) -> str:
        return json.dumps(self._bucket_payload(state), separators=(",", ":"))

    def _mccfr_active(self, street: str) -> bool:
        if self.freeze_seventh:
            return street == "6th"
        return MCCFR_START_STREETS.index(street) >= MCCFR_START_STREETS.index(
            self.start_street
        ) if street in MCCFR_START_STREETS else False

    def _public_features(self, cards: Sequence[Card]) -> tuple[int, int, int]:
        if not cards:
            return -1, 0, 0
        category = get_public_betting_priority(cards)[0]
        max_suit = max(Counter(card.suit for card in cards).values())
        values = {card.value for card in cards}
        if 14 in values:
            values.add(1)
        connected = max(
            len(values.intersection(range(start, start + 5)))
            for start in range(1, 11)
        )
        return category, max_suit, connected

    @staticmethod
    def _ratio_bucket(value: float, thresholds: Sequence[float]) -> int:
        return sum(value > threshold for threshold in thresholds)

    @staticmethod
    def _cards(values: Sequence[Any]) -> list[Card]:
        cards = []
        for value in values:
            if isinstance(value, Card):
                cards.append(value)
            else:
                label = str(value)
                cards.append(Card(label[0], label[1:]))
        return cards

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def reset_average_strategy(self) -> None:
        for node in self.nodes.values():
            node.strategy_sum.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "iterations_per_decision": self.iterations,
            "search_raise_cap": self.raise_cap,
            "start_street": self.start_street,
            "freeze_seventh": self.freeze_seventh,
            "decision_strategy": self.decision_strategy,
            "decisions": self.decisions,
            "heuristic_decisions": self.heuristic_decisions,
            "traversals": self.traversals,
            "buckets": len(self.nodes),
            "last_bucket": self.last_bucket,
            "last_strategy": self.last_strategy,
        }

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "version": 1,
            "metadata": metadata or {},
            "config": {
                "iterations": self.iterations,
                "raise_cap": self.raise_cap,
                "start_street": self.start_street,
                "freeze_seventh": self.freeze_seventh,
                "average_strategy_update": "traverser-reach",
                "betting_rules_version": BETTING_RULES_VERSION,
            },
            "nodes": {
                key: {
                    "regrets": node.regrets,
                    "strategy_sum": node.strategy_sum,
                }
                for key, node in self.nodes.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        temporary.replace(path)

    def load(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported MCCFR table version")
        saved_rules = int(
            payload.get("config", {}).get("betting_rules_version", 1)
        )
        if saved_rules != BETTING_RULES_VERSION:
            raise ValueError(
                f"Table betting rules are v{saved_rules}, expected v{BETTING_RULES_VERSION}"
            )
        saved_raise_cap = int(payload.get("config", {}).get("raise_cap", self.raise_cap))
        if saved_raise_cap != self.raise_cap:
            raise ValueError(
                f"Table raise cap is {saved_raise_cap}, requested {self.raise_cap}"
            )
        saved_start_street = str(
            payload.get("config", {}).get("start_street", "7th_hidden")
        )
        if saved_start_street != self.start_street:
            raise ValueError(
                f"Table starts at {saved_start_street}, requested {self.start_street}"
            )
        saved_freeze_seventh = bool(
            payload.get("config", {}).get("freeze_seventh", False)
        )
        if saved_freeze_seventh != self.freeze_seventh:
            raise ValueError(
                "Table frozen-seventh setting does not match the requested agent"
            )
        self.nodes = {
            key: _RegretNode(
                regrets={action: float(value) for action, value in values["regrets"].items()},
                strategy_sum={
                    action: float(value)
                    for action, value in values["strategy_sum"].items()
                },
            )
            for key, values in payload["nodes"].items()
        }
        return dict(payload.get("metadata", {}))

    def initialize_from_seventh_street(self, path: Path) -> None:
        if self.start_street != "6th":
            raise ValueError("Seventh-street initialization requires a 6th+ agent")
        source = MCCFRPokerAgent(
            self.name,
            iterations=self.iterations,
            raise_cap=self.raise_cap,
            start_street="7th_hidden",
        )
        source.load(path)
        self.nodes = {
            json.dumps(
                ["7th_hidden", *json.loads(key)], separators=(",", ":")
            ): _RegretNode(
                regrets=dict(node.regrets),
                strategy_sum=dict(node.strategy_sum),
            )
            for key, node in source.nodes.items()
        }


def mccfr_bucket_vector(payload: Sequence[Any]) -> tuple[float, ...]:
    offset = int(bool(payload) and isinstance(payload[0], str))
    own_public = payload[offset + 2]
    opponent_public = payload[offset + 3]
    return (
        float(payload[offset]),
        float(payload[offset + 1]),
        *(float(value) for value in own_public),
        *(float(value) for value in opponent_public),
        float(payload[offset + 4]),
        float(payload[offset + 5]),
        float(payload[offset + 6]),
    )


def mccfr_bucket_signature(payload: Sequence[Any]) -> str:
    return json.dumps([payload[-2], payload[-1]], separators=(",", ":"))


class MCCFRKMeansAgent(MCCFRPokerAgent):
    """Nearest-centroid policy over a compressed seventh-street table."""

    def __init__(
        self,
        name: str,
        iterations: int = 0,
        seed: int | None = None,
        raise_cap: int = 2,
    ):
        super().__init__(
            name, iterations=iterations, seed=seed, raise_cap=raise_cap
        )
        self.feature_mean = np.zeros(0, dtype=np.float32)
        self.feature_scale = np.ones(0, dtype=np.float32)
        self.cluster_groups: dict[str, tuple[np.ndarray, list[_RegretNode]]] = {}
        self.cluster_hits = 0
        self.cluster_fallbacks = 0

    def load_clustered(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported MCCFR k-means table version")
        if int(payload.get("betting_rules_version", 1)) != BETTING_RULES_VERSION:
            raise ValueError("MCCFR k-means table uses incompatible betting rules")
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float32)
        self.cluster_groups = {}
        self.nodes = {}
        for group in payload["groups"]:
            signature = json.dumps(group["signature"], separators=(",", ":"))
            nodes = [
                _RegretNode(
                    regrets={
                        action: float(value)
                        for action, value in node["regrets"].items()
                    },
                    strategy_sum={
                        action: float(value)
                        for action, value in node["strategy_sum"].items()
                    },
                )
                for node in group["nodes"]
            ]
            self.cluster_groups[signature] = (
                np.asarray(group["centers"], dtype=np.float32),
                nodes,
            )
            for cluster, node in enumerate(nodes):
                self.nodes[f"{signature}#{cluster}"] = node
        for key, node in payload.get("fallback_nodes", {}).items():
            self.nodes[key] = _RegretNode(
                regrets={
                    action: float(value)
                    for action, value in node["regrets"].items()
                },
                strategy_sum={
                    action: float(value)
                    for action, value in node["strategy_sum"].items()
                },
            )
        return dict(payload.get("metadata", {}))

    def load(self, path: Path) -> dict[str, Any]:
        return self.load_clustered(path)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        clustered_keys = {
            f"{signature}#{cluster}"
            for signature, (_, nodes) in self.cluster_groups.items()
            for cluster in range(len(nodes))
        }
        payload = {
            "version": 1,
            "betting_rules_version": BETTING_RULES_VERSION,
            "metadata": metadata or {},
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_scale": self.feature_scale.astype(float).tolist(),
            "groups": [
                {
                    "signature": json.loads(signature),
                    "centers": centers.astype(float).tolist(),
                    "nodes": [
                        {
                            "regrets": node.regrets,
                            "strategy_sum": node.strategy_sum,
                        }
                        for node in nodes
                    ],
                }
                for signature, (centers, nodes) in self.cluster_groups.items()
            ],
            "fallback_nodes": {
                key: {
                    "regrets": node.regrets,
                    "strategy_sum": node.strategy_sum,
                }
                for key, node in self.nodes.items()
                if key not in clustered_keys
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        temporary.replace(path)

    def reset_regrets_and_average_strategy(self) -> None:
        for node in self.nodes.values():
            node.regrets.clear()
            node.strategy_sum.clear()

    def _bucket_key(self, state: dict[str, Any]) -> str:
        payload = self._bucket_payload(state)
        signature = mccfr_bucket_signature(payload)
        group = self.cluster_groups.get(signature)
        if group is None:
            raw = json.dumps(payload, separators=(",", ":"))
            return f"raw:{raw}"
        centers, _ = group
        vector = np.asarray(mccfr_bucket_vector(payload), dtype=np.float32)
        vector = (vector - self.feature_mean) / self.feature_scale
        cluster = int(np.argmin(np.sum((centers - vector) ** 2, axis=1)))
        return f"{signature}#{cluster}"

    def _choose_from_table(
        self, state: dict[str, Any], actions: Sequence[str]
    ) -> str:
        payload = self._bucket_payload(state)
        group = self.cluster_groups.get(mccfr_bucket_signature(payload))
        if group is None:
            self.cluster_fallbacks += 1
            self.heuristic_decisions += 1
            return HeuristicPokerAgent.choose_action(self, state, actions)
        self.cluster_hits += 1
        return super()._choose_from_table(state, actions)

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        result.update(
            {
                "cluster_groups": len(self.cluster_groups),
                "cluster_hits": self.cluster_hits,
                "cluster_fallbacks": self.cluster_fallbacks,
                "clusters": sum(len(group[1]) for group in self.cluster_groups.values()),
            }
        )
        return result

    def learn_from_database(self, database=None) -> dict[str, Any]:
        return self.diagnostics()
