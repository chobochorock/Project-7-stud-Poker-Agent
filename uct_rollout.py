from __future__ import annotations

import argparse
import contextlib
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Sequence

from agent.uct_agent import UCTPokerAgent, UCTSearchRecord
from ev_rollout import ACTIONS
from poker_env import DEFAULT_EV_STACK_ANTE, PokerGame


UCT_SCHEMA_VERSION = 2


class _NullWriter:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


class UCTNodeTable:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        action_columns = ",\n".join(
            f"{action.lower()}_visits INTEGER NOT NULL, "
            f"{action.lower()}_return_sum REAL NOT NULL, "
            f"{action.lower()}_return_sq_sum REAL NOT NULL"
            for action in ACTIONS
        )
        self.connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS uct_nodes (
                state_json TEXT NOT NULL,
                seat_index INTEGER NOT NULL,
                search_version TEXT NOT NULL,
                opponent_policy TEXT NOT NULL,
                simulation_budget INTEGER NOT NULL,
                searches INTEGER NOT NULL,
                simulations INTEGER NOT NULL,
                legal_mask INTEGER NOT NULL,
                last_action TEXT NOT NULL,
                {action_columns},
                PRIMARY KEY (
                    state_json,
                    seat_index,
                    search_version,
                    opponent_policy,
                    simulation_budget
                )
            ) WITHOUT ROWID;
            """
        )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(uct_nodes)")
        }
        if "ddadang_visits" not in columns:
            self.connection.close()
            raise ValueError("UCT database uses obsolete betting rules; use a new shard")
        self._insert_sql = self._build_insert_sql()

    def flush(self, records: list[UCTSearchRecord]) -> None:
        if not records:
            return
        with self.connection:
            self.connection.executemany(self._insert_sql, (self._row(record) for record in records))
        records.clear()

    def finish(self, metadata: dict[str, Any]) -> tuple[int, int]:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ((f"uct_{key}", str(value)) for key, value in metadata.items()),
            )
        nodes, simulations = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(simulations), 0) FROM uct_nodes"
        ).fetchone()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()
        return int(nodes), int(simulations)

    def disk_bytes(self) -> int:
        paths = (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm"))
        return sum(path.stat().st_size for path in paths if path.exists())

    @staticmethod
    def _build_insert_sql() -> str:
        action_names = [action.lower() for action in ACTIONS]
        action_columns = [
            column
            for action in action_names
            for column in (
                f"{action}_visits",
                f"{action}_return_sum",
                f"{action}_return_sq_sum",
            )
        ]
        columns = [
            "state_json",
            "seat_index",
            "search_version",
            "opponent_policy",
            "simulation_budget",
            "searches",
            "simulations",
            "legal_mask",
            "last_action",
            *action_columns,
        ]
        updates = [
            "searches = searches + excluded.searches",
            "simulations = simulations + excluded.simulations",
            "legal_mask = legal_mask | excluded.legal_mask",
            "last_action = excluded.last_action",
            *(f"{column} = {column} + excluded.{column}" for column in action_columns),
        ]
        placeholders = ", ".join("?" for _ in columns)
        return f"""
            INSERT INTO uct_nodes ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (
                state_json,
                seat_index,
                search_version,
                opponent_policy,
                simulation_budget
            ) DO UPDATE SET {', '.join(updates)}
        """

    @staticmethod
    def _row(record: UCTSearchRecord) -> tuple[Any, ...]:
        action_values = []
        for action in ACTIONS:
            action_values.extend(
                (
                    record.action_visits[action],
                    record.return_sums[action],
                    record.return_squared_sums[action],
                )
            )
        return (
            record.state_json,
            record.seat_index,
            record.search_version,
            record.opponent_policy,
            record.simulation_budget,
            1,
            sum(record.action_visits.values()),
            record.legal_mask,
            record.chosen_action,
            *action_values,
        )


def run_uct_rollouts(
    output: Path,
    hands: int | None = 1000,
    seconds: float | None = None,
    simulations: int = 256,
    exploration: float = 2**0.5,
    ante: int = 1000,
    effective_stack_ante: int = DEFAULT_EV_STACK_ANTE,
    seed: int = 7,
    flush_hands: int = 10,
    progress_seconds: float = 10.0,
    max_bytes: int | None = None,
    opponent_policy: str = "random",
    record_tree_nodes: bool = False,
    record_min_visits: int = 1,
    min_simulations: int | None = None,
    simulation_batch: int = 32,
    epsilon_ante: float | None = None,
) -> dict[str, Any]:
    if hands is not None and hands <= 0:
        raise ValueError("hands must be positive or None.")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive or None.")
    if hands is None and seconds is None:
        raise ValueError("At least one of hands or seconds is required.")
    if flush_hands <= 0:
        raise ValueError("flush_hands must be positive.")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive or None.")

    random.seed(seed)
    game = PokerGame(
        ["Player_1", "Player_2"],
        log_file=None,
        ante=ante,
        game_mode="ev",
        ev_stack_ante=effective_stack_ante,
    )
    agents = {
        "Player_1": UCTPokerAgent(
            "Player_1",
            simulations=simulations,
            exploration=exploration,
            seed=seed * 2 + 1,
            opponent_policy=opponent_policy,
            record_tree_nodes=record_tree_nodes,
            record_min_visits=record_min_visits,
            min_simulations=min_simulations,
            simulation_batch=simulation_batch,
            epsilon_ante=epsilon_ante,
        ),
        "Player_2": UCTPokerAgent(
            "Player_2",
            simulations=simulations,
            exploration=exploration,
            seed=seed * 2 + 2,
            opponent_policy=opponent_policy,
            record_tree_nodes=record_tree_nodes,
            record_min_visits=record_min_visits,
            min_simulations=min_simulations,
            simulation_batch=simulation_batch,
            epsilon_ante=epsilon_ante,
        ),
    }
    table = UCTNodeTable(output)
    pending_records: list[UCTSearchRecord] = []
    started = time.perf_counter()
    last_progress = started
    completed_hands = 0
    collected_searches = 0
    collected_records = 0
    interrupted = False
    stopped_by = "hands"

    try:
        while hands is None or completed_hands < hands:
            elapsed = time.perf_counter() - started
            if seconds is not None and elapsed >= seconds:
                stopped_by = "seconds"
                break

            searches_before = sum(agent.searches for agent in agents.values())
            with contextlib.redirect_stdout(_NullWriter()):
                game.play_hand(agents)
            completed_hands += 1
            collected_searches += sum(agent.searches for agent in agents.values()) - searches_before

            for agent in agents.values():
                records = agent.drain_search_records()
                collected_records += len(records)
                pending_records.extend(records)

            if completed_hands % flush_hands == 0:
                table.flush(pending_records)
                if max_bytes is not None and table.disk_bytes() >= max_bytes:
                    stopped_by = "size_limit"
                    break

            now = time.perf_counter()
            if progress_seconds > 0 and now - last_progress >= progress_seconds:
                elapsed = now - started
                print(
                    f"{completed_hands:,} hands, {collected_searches:,} roots, "
                    f"{collected_records:,} records, "
                    f"{completed_hands / max(elapsed, 1e-9):.2f} hands/s"
                )
                last_progress = now
    except KeyboardInterrupt:
        interrupted = True
        stopped_by = "interrupted"
    finally:
        table.flush(pending_records)

    elapsed = time.perf_counter() - started
    actual_simulations = sum(agent.simulations_run for agent in agents.values())
    converged_searches = sum(agent.converged_searches for agent in agents.values())
    final_ci_radius_sum = sum(agent.final_ci_radius_sum for agent in agents.values())
    nodes, stored_simulations = table.finish(
        {
            "schema_version": UCT_SCHEMA_VERSION,
            "search_version": next(iter(agents.values())).search_version,
            "simulation_budget": simulations,
            "min_simulations": min_simulations,
            "simulation_batch": simulation_batch,
            "epsilon_ante": epsilon_ante,
            "actual_simulations": actual_simulations,
            "converged_searches": converged_searches,
            "exploration": exploration,
            "opponent_policy": opponent_policy,
            "record_tree_nodes": record_tree_nodes,
            "record_min_visits": record_min_visits,
            "ante": ante,
            "effective_stack_ante": effective_stack_ante,
            "last_run_hands": completed_hands,
            "last_run_searches": collected_searches,
            "last_run_seconds": elapsed,
            "last_run_seed": seed,
        }
    )
    result = {
        "output": str(output.resolve()),
        "effective_stack_ante": effective_stack_ante,
        "hands": completed_hands,
        "search_roots": collected_searches,
        "stored_records": collected_records,
        "unique_uct_nodes": nodes,
        "stored_simulations": stored_simulations,
        "elapsed_seconds": elapsed,
        "hands_per_second": completed_hands / max(elapsed, 1e-9),
        "roots_per_hand": collected_searches / max(completed_hands, 1),
        "records_per_hand": collected_records / max(completed_hands, 1),
        "seconds_per_search_root": elapsed / max(collected_searches, 1),
        "actual_root_simulations": actual_simulations,
        "average_simulations_per_root": actual_simulations / max(collected_searches, 1),
        "simulations_per_hand": actual_simulations / max(completed_hands, 1),
        "configured_max_simulations_per_hand": (
            collected_searches * simulations / max(completed_hands, 1)
        ),
        "converged_searches": converged_searches,
        "convergence_rate": (
            converged_searches / max(collected_searches, 1)
            if epsilon_ante is not None
            else None
        ),
        "average_final_ci95_radius_ante": (
            final_ci_radius_sum / max(collected_searches, 1)
            if epsilon_ante is not None
            else None
        ),
        "table_bytes": table.disk_bytes(),
        "size_limit_bytes": max_bytes,
        "interrupted": interrupted,
        "stopped_by": stopped_by,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect heads-up EV UCT search nodes.")
    parser.add_argument("--output", type=Path, default=Path("replays/uct_rollout.sqlite3"))
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--exploration", type=float, default=2**0.5)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument(
        "--effective-stack-ante", type=int, default=DEFAULT_EV_STACK_ANTE
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flush-hands", type=int, default=10)
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--max-gib", type=float)
    parser.add_argument("--opponent-policy", choices=("random", "uct"), default="random")
    parser.add_argument("--record-tree-nodes", action="store_true")
    parser.add_argument("--record-min-visits", type=int, default=1)
    parser.add_argument("--min-simulations", type=int)
    parser.add_argument("--simulation-batch", type=int, default=32)
    parser.add_argument("--epsilon-ante", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    hands = None if args.hands == 0 else args.hands
    result = run_uct_rollouts(
        output=args.output,
        hands=hands,
        seconds=args.seconds,
        simulations=args.simulations,
        exploration=args.exploration,
        ante=args.ante,
        effective_stack_ante=args.effective_stack_ante,
        seed=args.seed,
        flush_hands=args.flush_hands,
        progress_seconds=args.progress_seconds,
        max_bytes=int(args.max_gib * 1024**3) if args.max_gib is not None else None,
        opponent_policy=args.opponent_policy,
        record_tree_nodes=args.record_tree_nodes,
        record_min_visits=args.record_min_visits,
        min_simulations=args.min_simulations,
        simulation_batch=args.simulation_batch,
        epsilon_ante=args.epsilon_ante,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
