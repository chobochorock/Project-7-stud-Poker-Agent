from __future__ import annotations

import os
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from agent.base import BasePokerAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "cpp_mccfr" / "stud_mccfr.exe"
DEFAULT_ATLAS = ROOT / "cpp_mccfr" / "power64_v1.bin"
DEFAULT_MODEL = ROOT / "cpp_mccfr" / "root_mccfr_current_snapshot.bin"
if not DEFAULT_MODEL.is_file():
    DEFAULT_MODEL = ROOT / "cpp_mccfr" / "root_mccfr_ante1000_10m.bin"
STREETS = {"5th", "6th", "7th_hidden"}


class CppMCCFRAgent(BasePokerAgent):
    """Persistent bridge to the C++ table-policy process."""

    def __init__(
        self,
        name: str,
        *,
        exe: str | os.PathLike[str] | None = None,
        atlas: str | os.PathLike[str] | None = None,
        model: str | os.PathLike[str] | None = None,
        seed: int | None = None,
    ):
        super().__init__(name)
        exe_path = Path(exe or os.environ.get("CPP_MCCFR_EXE", DEFAULT_EXE))
        atlas_path = Path(atlas or os.environ.get("CPP_MCCFR_ATLAS", DEFAULT_ATLAS))
        model_path = Path(model or os.environ.get("CPP_MCCFR_MODEL", DEFAULT_MODEL))
        for path in (exe_path, atlas_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        if seed is None:
            try:
                seed = 7 + int(name.rsplit("_", 1)[-1])
            except ValueError:
                seed = 7
        self.model_path = model_path
        self.action_counts: Counter[str] = Counter()
        self.representative_counts: Counter[str] = Counter()
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                str(exe_path),
                "--agent-stdio",
                "--bucket", "power",
                "--load-atlas", str(atlas_path),
                "--start-street", "5",
                "--algorithm", "mccfr",
                "--load", str(model_path),
                "--seed", str(seed),
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=creation_flags,
        )
        ready = self._process.stdout.readline().strip()
        if ready != "READY":
            error = self._process.stderr.read().strip()
            self.close()
            raise RuntimeError(error or f"C++ agent failed to start: {ready}")

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None
        if state.get("game_mode") != "ev" or int(state.get("seat_count", 0)) < 2:
            raise ValueError("C++ MCCFR requires EV mode with at least two players.")
        if state["street"] not in STREETS:
            raise ValueError(f"C++ MCCFR cannot act on {state['street']}.")

        opponent = self._representative_opponent(state["opponents"])
        self.representative_counts[opponent["seat"]] += 1
        own_stack = int(state["effective_stack"])
        opponent_stack = int(opponent.get("effective_stack") or own_stack)
        effective_stack = (
            str(own_stack)
            if own_stack == opponent_stack
            else f"{own_stack},{opponent_stack}"
        )
        history = ";".join(
            f"{event['street']}:{0 if event['actor'] == 'self' else 1}:{event['action']}"
            for event in state["betting_history"]
        ) or "-"
        mask = sum(1 << self.ACTIONS.index(action) for action in valid_actions)
        fields = [
            "ACT",
            state["street"],
            str(state["ante"]),
            str(state["pot"]),
            str(state["current_highest_bet"]),
            effective_stack,
            str(state["my_invested"]),
            str(state["my_round_bet"]),
            str(opponent["invested"]),
            str(opponent["round_bet"]),
            self._cards(state["my_hidden_cards"]),
            self._cards(state["my_public_cards"]),
            state["my_discarded_card"] or "-",
            self._cards(opponent["public_cards"]),
            history,
            str(mask),
        ]
        response = self._ask("\t".join(fields)).split("\t")
        if len(response) != 2 or response[0] != "ACTION" or response[1] not in valid_actions:
            raise RuntimeError(f"Invalid C++ action response: {response}")
        self.action_counts[response[1]] += 1
        return response[1]

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        response = self._ask(f"DISCARD\t{self._cards(hidden_cards)}").split("\t")
        if len(response) != 3 or response[0] != "DISCARD":
            raise RuntimeError(f"Invalid C++ discard response: {response}")
        return int(response[1]), int(response[2])

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"agent": type(self).__name__, "trained": False, "reason": "Frozen C++ table policy."}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "actions": dict(self.action_counts),
            "representative_opponents": dict(self.representative_counts),
        }

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.wait(timeout=1)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.kill()

    def _ask(self, command: str) -> str:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(self._process.stderr.read().strip() or "C++ agent stopped.")
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
            response = self._process.stdout.readline().strip()
        if response.startswith("ERROR\t"):
            raise RuntimeError(response.removeprefix("ERROR\t"))
        return response

    @staticmethod
    def _cards(cards: Sequence[Any]) -> str:
        return ",".join(map(str, cards)) or "-"

    @staticmethod
    def _representative_opponent(opponents: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from poker_env import Card, get_public_betting_priority

        if not opponents:
            raise ValueError("At least one opponent is required.")
        active = [
            opponent for opponent in opponents
            if not opponent.get("is_folded") and not opponent.get("is_eliminated")
        ]
        candidates = active or list(opponents)

        def priority(opponent: dict[str, Any]) -> tuple[Any, ...]:
            cards = [
                Card(str(card)[0], str(card)[1:])
                for card in opponent.get("public_cards", [])
            ]
            return (
                get_public_betting_priority(cards),
                int(opponent.get("round_bet", 0)),
                int(opponent.get("invested", 0)),
                -int(opponent.get("seat_index", 0)),
            )

        return max(candidates, key=priority)

    def __del__(self) -> None:
        self.close()
