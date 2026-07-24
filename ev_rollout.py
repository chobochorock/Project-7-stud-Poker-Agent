from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import random
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from agent import PokerAgent
from poker_env import DEFAULT_EV_STACK_ANTE, EV_RAISE_CAP, PokerGame


SCHEMA_VERSION = 2
SUITS = {"s": 0, "h": 1, "d": 2, "c": 3}
RANK_LABELS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANKS = {rank: index for index, rank in enumerate(RANK_LABELS)}
SUIT_PERMUTATIONS = tuple(itertools.permutations(range(4)))
STREETS = {"4th": 0, "5th": 1, "6th": 2, "7th_hidden": 3}
ACTIONS = {
    action: index
    for index, action in enumerate(
        ("CHECK", "BBING", "DDADANG", "QUARTER", "HALF", "CALL", "FOLD")
    )
}
STREET_NAMES = {value: key for key, value in STREETS.items()}
ACTION_NAMES = {value: key for key, value in ACTIONS.items()}


class _NullWriter:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


class UniformRolloutAgent(PokerAgent):
    def __init__(self, name: str, seed: int):
        super().__init__(name)
        self.rng = random.Random(seed)
        self.trajectory: list[tuple[str, str, str, int]] = []

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None
        action = self.rng.choice(list(valid_actions))
        self.trajectory.append((canonical_state(state), action, state["street"], state["raise_count"]))
        return action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        discard = self.rng.randrange(len(hidden_cards))
        reveal = self.rng.choice([index for index in range(len(hidden_cards)) if index != discard])
        return discard, reveal


class RolloutTable:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE q_values (
                state_json TEXT NOT NULL,
                action TEXT NOT NULL,
                visits INTEGER NOT NULL,
                return_sum REAL NOT NULL,
                PRIMARY KEY (state_json, action)
            ) WITHOUT ROWID;
            CREATE TABLE coverage (
                street TEXT NOT NULL,
                raise_depth INTEGER NOT NULL,
                action TEXT NOT NULL,
                visits INTEGER NOT NULL,
                PRIMARY KEY (street, raise_depth, action)
            ) WITHOUT ROWID;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )

    def flush(
        self,
        q_updates: dict[tuple[str, str], list[float]],
        coverage_updates: Counter[tuple[str, int, str]],
    ) -> None:
        if not q_updates and not coverage_updates:
            return
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO q_values (state_json, action, visits, return_sum)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (state_json, action) DO UPDATE SET
                    visits = visits + excluded.visits,
                    return_sum = return_sum + excluded.return_sum
                """,
                ((state, action, int(values[0]), values[1]) for (state, action), values in q_updates.items()),
            )
            self.connection.executemany(
                """
                INSERT INTO coverage (street, raise_depth, action, visits)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (street, raise_depth, action) DO UPDATE SET
                    visits = visits + excluded.visits
                """,
                ((street, depth, action, visits) for (street, depth, action), visits in coverage_updates.items()),
            )
        q_updates.clear()
        coverage_updates.clear()

    def finish(self, metadata: dict[str, Any]) -> tuple[int, int]:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ((key, str(value)) for key, value in metadata.items()),
            )
        rows = int(self.connection.execute("SELECT COUNT(*) FROM q_values").fetchone()[0])
        samples = int(self.connection.execute("SELECT COALESCE(SUM(visits), 0) FROM q_values").fetchone()[0])
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()
        return rows, samples

    def disk_bytes(self) -> int:
        paths = (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm"))
        return sum(path.stat().st_size for path in paths if path.exists())


def canonical_state(state: dict[str, Any]) -> str:
    opponent = state["opponents"][0]
    card_groups = _canonical_card_groups(
        state["my_hidden_cards"],
        state["my_public_cards"],
        state.get("my_discarded_card"),
        opponent["public_cards"],
    )
    history = [
        [STREETS[event["street"]], 0 if event["actor"] == "self" else 1, ACTIONS[event["action"]]]
        for event in state["betting_history"]
    ]
    payload = [
        SCHEMA_VERSION,
        STREETS[state["street"]],
        card_groups,
        state["ante"],
        state["pot"],
        state["my_invested"],
        opponent["invested"],
        state["my_round_bet"],
        opponent["round_bet"],
        state["call_amount"],
        state["raise_count"],
        history,
    ]
    return json.dumps(payload, separators=(",", ":"))


def decode_state(encoded: str | Sequence[Any]) -> dict[str, Any]:
    payload = json.loads(encoded) if isinstance(encoded, str) else list(encoded)
    if len(payload) == 11:
        schema_version = 0
        fields = payload
    elif len(payload) == 12 and payload[0] == SCHEMA_VERSION:
        schema_version = payload[0]
        fields = payload[1:]
    else:
        raise ValueError("Unsupported EV state schema.")

    street, cards, ante, pot, my_invested, opponent_invested, my_round_bet, opponent_round_bet, call_amount, raise_count, history = fields
    return {
        "schema_version": schema_version,
        "street": STREET_NAMES[street],
        "cards": {
            "my_hidden": [_decode_card(code) for code in cards[0]],
            "my_public": [_decode_card(code) for code in cards[1]],
            "my_discarded": _decode_card(cards[2]) if cards[2] >= 0 else None,
            "opponent_public": [_decode_card(code) for code in cards[3]],
        },
        "ante": ante,
        "pot": pot,
        "my_invested": my_invested,
        "opponent_invested": opponent_invested,
        "my_round_bet": my_round_bet,
        "opponent_round_bet": opponent_round_bet,
        "call_amount": call_amount,
        "raise_count": raise_count,
        "betting_history": [
            {
                "street": STREET_NAMES[event[0]],
                "actor": "self" if event[1] == 0 else "opponent",
                "action": ACTION_NAMES[event[2]],
            }
            for event in history
        ],
    }


def _canonical_card_groups(
    hidden: Sequence[str],
    public: Sequence[str],
    discarded: str | None,
    opponent_public: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]:
    best = None
    for permutation in SUIT_PERMUTATIONS:
        candidate = (
            tuple(_card_code(card, permutation) for card in hidden),
            tuple(_card_code(card, permutation) for card in public),
            _card_code(discarded, permutation) if discarded else -1,
            tuple(_card_code(card, permutation) for card in opponent_public),
        )
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def _card_code(label: str, suit_permutation: Sequence[int]) -> int:
    rank = label[1:]
    rank = "T" if rank == "10" else rank
    return RANKS[rank] * 4 + suit_permutation[SUITS[label[0]]]


def _decode_card(code: int) -> str:
    if not 0 <= code < 52:
        raise ValueError(f"Invalid card code: {code}")
    rank = RANK_LABELS[code // 4]
    return f"u{code % 4}{rank}"


def inspect_table(path: Path, limit: int = 5) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        rows = connection.execute(
            "SELECT state_json, action, visits, return_sum FROM q_values LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    return {
        "metadata": metadata,
        "rows": [
            {
                "state": decode_state(state_json),
                "action": action,
                "visits": visits,
                "average_return": return_sum / visits,
            }
            for state_json, action, visits, return_sum in rows
        ],
    }


def run_rollouts(
    output: Path,
    *,
    seconds: float | None = None,
    hands: int | None = None,
    ante: int = 1000,
    seed: int = 2026,
    flush_hands: int = 250,
    progress_seconds: float = 10.0,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    if seconds is None and hands is None:
        raise ValueError("seconds or hands is required.")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive.")
    if hands is not None and hands <= 0:
        raise ValueError("hands must be positive.")
    if flush_hands <= 0:
        raise ValueError("flush_hands must be positive.")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    agents = {
        "Player_1": UniformRolloutAgent("Player_1", seed * 2),
        "Player_2": UniformRolloutAgent("Player_2", seed * 2 + 1),
    }
    game = PokerGame(["Player_1", "Player_2"], log_file=None, ante=ante, game_mode="ev")
    table = RolloutTable(output)
    q_updates: dict[tuple[str, str], list[float]] = {}
    coverage_updates: Counter[tuple[str, int, str]] = Counter()
    hand_count = 0
    sample_count = 0
    interrupted = False
    stopped_by = "unknown"
    start = time.perf_counter()
    next_progress = start + progress_seconds

    try:
        while True:
            now = time.perf_counter()
            if seconds is not None and now - start >= seconds:
                stopped_by = "seconds"
                break
            if hands is not None and hand_count >= hands:
                stopped_by = "hands"
                break

            with contextlib.redirect_stdout(_NullWriter()):
                game.play_hand(agents)
            if sum(player.chips for player in game.players) != 0:
                raise AssertionError("EV rewards must remain zero-sum.")

            for player in game.players:
                agent = agents[player.name]
                reward = player.chips
                for state_json, action, street, raise_depth in agent.trajectory:
                    values = q_updates.setdefault((state_json, action), [0.0, 0.0])
                    values[0] += 1
                    values[1] += reward
                    coverage_updates[(street, raise_depth, action)] += 1
                    sample_count += 1
                agent.trajectory.clear()

            hand_count += 1
            if hand_count % flush_hands == 0:
                table.flush(q_updates, coverage_updates)
                if max_bytes is not None and table.disk_bytes() >= max_bytes:
                    stopped_by = "size_limit"
                    break
            if progress_seconds > 0 and now >= next_progress:
                elapsed = now - start
                print(f"{hand_count:,} hands, {sample_count:,} samples, {hand_count / elapsed:,.1f} hands/s")
                next_progress = now + progress_seconds
    except KeyboardInterrupt:
        interrupted = True
        stopped_by = "interrupt"
    finally:
        table.flush(q_updates, coverage_updates)

    elapsed = time.perf_counter() - start
    rows, stored_samples = table.finish(
        {
            "mode": "ev",
            "schema_version": SCHEMA_VERSION,
            "ante": ante,
            "raise_cap": EV_RAISE_CAP,
            "effective_stack_ante": DEFAULT_EV_STACK_ANTE,
            "seed": seed,
            "hands": hand_count,
            "samples": sample_count,
            "elapsed_seconds": elapsed,
            "interrupted": interrupted,
            "stopped_by": stopped_by,
            "size_limit_bytes": max_bytes or 0,
        }
    )
    size_bytes = output.stat().st_size
    return {
        "output": str(output.resolve()),
        "hands": hand_count,
        "samples": stored_samples,
        "unique_state_actions": rows,
        "elapsed_seconds": elapsed,
        "hands_per_second": hand_count / elapsed if elapsed else 0.0,
        "samples_per_hand": stored_samples / hand_count if hand_count else 0.0,
        "table_bytes": size_bytes,
        "bytes_per_unique_state_action": size_bytes / rows if rows else 0.0,
        "interrupted": interrupted,
        "stopped_by": stopped_by,
        "size_limit_bytes": max_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run uniform stackless heads-up EV rollouts.")
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--seconds", type=float, help="Wall-clock duration")
    limit.add_argument("--hands", type=int, help="Exact number of hands")
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--flush-hands", type=int, default=250)
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--max-gib", type=float, help="Stop when SQLite plus WAL reaches this size")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", type=Path, help="Decode and print an existing rollout table")
    parser.add_argument("--limit", type=int, default=5, help="Rows to print with --inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect is not None:
        print(json.dumps(inspect_table(args.inspect, args.limit), ensure_ascii=False, indent=2))
        return
    output = args.output or Path("replays") / f"ev_rollout_{datetime.now():%Y%m%d_%H%M%S}.sqlite3"
    seconds = args.seconds if args.seconds is not None else None if args.hands is not None else 60.0
    result = run_rollouts(
        output,
        seconds=seconds,
        hands=args.hands,
        ante=args.ante,
        seed=args.seed,
        flush_hands=args.flush_hands,
        progress_seconds=args.progress_seconds,
        max_bytes=int(args.max_gib * 1024**3) if args.max_gib is not None else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
