import random
from abc import ABC, abstractmethod
from typing import Any, Sequence


class BasePokerAgent(ABC):
    """Common interface for every player controller."""

    ACTIONS = (
        "CHECK", "BBING", "DDADANG", "QUARTER", "HALF", "FULL", "CALL", "FOLD"
    )

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        """Choose one betting action from valid_actions."""

    @abstractmethod
    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        """Choose one hidden-card index to discard and another to reveal."""

    def observe_reward(self, reward: int, final_state: dict[str, Any] | None = None) -> None:
        """Receive a hand-level chip reward. Non-learning agents ignore it."""

    @abstractmethod
    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        """Train or summarize a model from stored trajectories."""


class PokerAgent(BasePokerAgent):
    """Reference random agent.

    This class is intentionally simple: it documents the environment-facing
    methods that stronger agents should implement.
    """

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None

        chosen_action = random.choice(list(valid_actions))
        print(f"[{self.name}] random action: {chosen_action}")
        return chosen_action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        if len(hidden_cards) < 2:
            raise ValueError("At least two hidden cards are needed to discard and reveal.")

        discard_idx = random.randrange(len(hidden_cards))
        reveal_candidates = [i for i in range(len(hidden_cards)) if i != discard_idx]
        reveal_idx = random.choice(reveal_candidates)
        return discard_idx, reveal_idx

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "trained": False,
            "reason": "Random agents do not train from trajectory databases.",
        }
