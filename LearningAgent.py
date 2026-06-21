import gzip
import json
import os
import random
import time
import uuid
from typing import Any, Sequence

from agent import PokerAgent


DB_VERSION = 2


def _fresh_database() -> dict[str, Any]:
    return {
        "version": DB_VERSION,
        "metadata": {
            "format": "shared anonymous 7-stud trajectory database",
            "compression": "Use a .json.gz filename to store the same readable JSON through gzip.",
            "identity_policy": "No player names, human labels, or opponent agent types are stored.",
        },
        "q_values": {},
        "episodes": [],
    }


class SharedTrajectoryDatabase:
    """Readable JSON store with optional gzip compression."""

    def __init__(self, filename: str):
        self.filename = filename

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.filename):
            return _fresh_database()

        try:
            with self._open("rt") as db_file:
                raw = json.load(db_file)
        except (OSError, json.JSONDecodeError) as exc:
            database = _fresh_database()
            database["metadata"]["load_error"] = str(exc)
            return database

        if isinstance(raw, dict) and raw.get("version") == DB_VERSION:
            raw.setdefault("metadata", {})
            raw.setdefault("q_values", {})
            raw.setdefault("episodes", [])
            return raw

        database = _fresh_database()
        database["metadata"]["legacy_entries_ignored"] = len(raw) if isinstance(raw, dict) else 0
        database["metadata"]["legacy_reason"] = (
            "Legacy state keys included player names, so they are not reused under the anonymity policy."
        )
        return database

    def save(self, database: dict[str, Any]) -> None:
        with self._open("wt") as db_file:
            json.dump(database, db_file, ensure_ascii=False, indent=2, sort_keys=True)

    def _open(self, mode: str):
        if self.filename.endswith(".gz"):
            return gzip.open(self.filename, mode, encoding="utf-8")
        return open(self.filename, mode, encoding="utf-8")


class LearningAgent(PokerAgent):
    """Tabular Monte Carlo agent backed by a shared anonymous trajectory DB."""

    db_filename = "LearningAgent_Shared_db.json"
    _shared_databases: dict[str, dict[str, Any]] = {}
    _stores: dict[str, SharedTrajectoryDatabase] = {}

    def __init__(
        self,
        name: str,
        db_filename: str | None = None,
        exploration_rate: float = 0.30,
    ):
        super().__init__(name)
        self.db_filename = db_filename or type(self).db_filename
        self.exploration_rate = exploration_rate
        self.trajectory: list[dict[str, Any]] = []

        if self.db_filename not in type(self)._stores:
            type(self)._stores[self.db_filename] = SharedTrajectoryDatabase(self.db_filename)
        if self.db_filename not in type(self)._shared_databases:
            type(self)._shared_databases[self.db_filename] = type(self)._stores[self.db_filename].load()

        self.memory = type(self)._shared_databases[self.db_filename]
        self.store = type(self)._stores[self.db_filename]

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None

        state = self._sanitize_state(state)
        state_key = self._state_to_key(state)
        q_entry = self._ensure_q_entry(state_key, valid_actions)

        if random.random() < self.exploration_rate:
            chosen_action = random.choice(list(valid_actions))
            policy = "explore"
        else:
            valid_values = {action: q_entry[action]["value"] for action in valid_actions}
            best_value = max(valid_values.values())
            best_actions = [action for action, value in valid_values.items() if value == best_value]
            chosen_action = random.choice(best_actions)
            policy = "exploit"

        self.trajectory.append(
            {
                "step": len(self.trajectory),
                "state_key": state_key,
                "state": state,
                "valid_actions": list(valid_actions),
                "action": chosen_action,
                "policy": policy,
            }
        )

        print(f"[{self.name}] {policy} action: {chosen_action}")
        return chosen_action

    def observe_reward(self, reward: int, final_state: dict[str, Any] | None = None) -> None:
        if not self.trajectory:
            return

        sanitized_final_state = self._sanitize_state(final_state or {})
        for step in self.trajectory:
            action = step["action"]
            q_entry = self._ensure_q_entry(step["state_key"], step["valid_actions"])
            action_value = q_entry[action]
            action_value["visits"] += 1
            action_value["value"] += (reward - action_value["value"]) / action_value["visits"]

        self.memory["episodes"].append(
            {
                "episode_id": uuid.uuid4().hex,
                "created_at": int(time.time()),
                "agent_class": type(self).__name__,
                "reward": reward,
                "final_state": sanitized_final_state,
                "trajectory": self.trajectory,
            }
        )
        self.store.save(self.memory)
        self.trajectory = []

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        database = database or self.memory
        q_values = database.get("q_values", {})
        episodes = database.get("episodes", [])
        return {
            "agent": type(self).__name__,
            "trained": True,
            "method": "tabular Monte Carlo value update",
            "states": len(q_values),
            "episodes": len(episodes),
            "note": "Replace this method with function approximation when the table becomes too sparse.",
        }

    def _ensure_q_entry(self, state_key: str, valid_actions: Sequence[str]) -> dict[str, Any]:
        q_values = self.memory.setdefault("q_values", {})
        q_entry = q_values.setdefault(state_key, {})
        for action in valid_actions:
            q_entry.setdefault(action, {"value": 0.0, "visits": 0})
        return q_entry

    def _state_to_key(self, state: dict[str, Any]) -> str:
        return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _sanitize_state(self, value: Any) -> Any:
        identity_keys = {"name", "player_name", "agent_name", "human_name", "opponent_name"}
        if isinstance(value, dict):
            return {
                key: self._sanitize_state(child)
                for key, child in value.items()
                if key not in identity_keys
            }
        if isinstance(value, list):
            return [self._sanitize_state(child) for child in value]
        return value
