from __future__ import annotations

from typing import Any, Sequence

from agent.base import BasePokerAgent


class HumanAgent(BasePokerAgent):
    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        print(f"\n[{self.name}]")
        print(f"chips={state['my_chips']} pot={state['pot']} call={state['call_amount']}")
        print(f"hidden={state['my_hidden_cards']} public={state['my_public_cards']}")
        print(f"valid actions={list(valid_actions)}")

        while True:
            action = input("action: ").strip().upper()
            if action in valid_actions:
                return action
            print("Please choose one of the valid actions.")

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        print(f"\n[{self.name}] hidden cards:")
        for index, card in enumerate(hidden_cards):
            print(f"  {index}: {card}")

        while True:
            try:
                discard_idx = int(input("discard index: "))
                reveal_idx = int(input("reveal index: "))
            except ValueError:
                print("Please enter numeric indices.")
                continue
            if discard_idx != reveal_idx and 0 <= discard_idx < len(hidden_cards) and 0 <= reveal_idx < len(hidden_cards):
                return discard_idx, reveal_idx
            print("Discard and reveal must be different valid indices.")

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"agent": type(self).__name__, "trained": False, "reason": "Human agents do not train."}


class WebHumanAgent(BasePokerAgent):
    """Marker used while browser actions are handled by the web controller."""

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        raise RuntimeError("Web human actions are submitted through the browser.")

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        raise RuntimeError("Web human discard/reveal choices are submitted through the browser.")

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"agent": type(self).__name__, "trained": False, "reason": "Human web agents do not train."}
