from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

from agent.mccfr_agent import (
    MCCFRKMeansAgent,
    MCCFRPokerAgent,
    MCCFR_START_STREETS,
)
from poker_env import PokerGame


def train(
    output: Path,
    *,
    hands: int,
    iterations: int,
    raise_cap: int,
    ante: int,
    checkpoint_hands: int,
    progress_seconds: float,
    seed: int,
    resume: bool,
    reset_average_strategy: bool,
    start_street: str,
    initialize_from: Path | None,
    freeze_seventh: bool,
    cluster_template: Path | None,
) -> dict[str, object]:
    if hands <= 0 or iterations <= 0 or checkpoint_hands <= 0 or ante <= 0:
        raise ValueError("hands, iterations, checkpoint_hands, and ante must be positive")
    if output.exists() and not resume:
        raise FileExistsError(f"{output} already exists; pass --resume to continue")
    if reset_average_strategy and not (resume and output.exists()):
        raise ValueError("--reset-average-strategy requires an existing --resume table")
    if resume and initialize_from is not None:
        raise ValueError("--initialize-from cannot be combined with --resume")
    if initialize_from is not None and start_street != "6th":
        raise ValueError("--initialize-from requires --start-street 6th")
    if freeze_seventh and start_street != "6th":
        raise ValueError("--freeze-seventh requires --start-street 6th")
    if freeze_seventh and not resume and initialize_from is None:
        raise ValueError("--freeze-seventh requires --initialize-from for a new table")
    if cluster_template is not None and (
        start_street != "7th_hidden" or initialize_from is not None or freeze_seventh
    ):
        raise ValueError("--cluster-template currently supports only standalone 7th training")

    if cluster_template is None:
        agent: MCCFRPokerAgent = MCCFRPokerAgent(
            "MCCFR",
            iterations=iterations,
            raise_cap=raise_cap,
            seed=seed,
            start_street=start_street,
            freeze_seventh=freeze_seventh,
        )
    else:
        agent = MCCFRKMeansAgent(
            "MCCFR", iterations=iterations, raise_cap=raise_cap, seed=seed
        )
    completed = 0
    initialized_from = str(initialize_from.resolve()) if initialize_from else None
    if resume and output.exists():
        metadata = agent.load(output)
        completed = int(metadata.get("completed_hands", 0))
        initialized_from = metadata.get("initialized_from")
        if reset_average_strategy:
            agent.reset_average_strategy()
        agent.set_seed(seed + completed)
    elif cluster_template is not None:
        agent.load(cluster_template)
        agent.reset_regrets_and_average_strategy()
    elif initialize_from is not None:
        agent.initialize_from_seventh_street(initialize_from)

    started = time.perf_counter()
    last_progress = started
    target = completed + hands
    with open(os.devnull, "w", encoding="utf-8") as sink:
        for hand_index in range(completed, target):
            random.seed(seed + hand_index)
            game = PokerGame(
                ["Player_1", "Player_2"],
                log_file=None,
                ante=ante,
                game_mode="ev",
            )
            with contextlib.redirect_stdout(sink):
                game.play_hand({"Player_1": agent, "Player_2": agent})

            current = hand_index + 1
            now = time.perf_counter()
            if current % checkpoint_hands == 0 or current == target:
                agent.save(
                    output,
                    {
                        "completed_hands": current,
                        "seed": seed,
                        "ante": ante,
                        "average_strategy_update": "traverser-reach",
                        "initialized_from": initialized_from,
                    },
                )
            if progress_seconds > 0 and now - last_progress >= progress_seconds:
                elapsed = now - started
                print(
                    f"{current:,}/{target:,} hands, "
                    f"{agent.traversals:,} traversals, {len(agent.nodes):,} buckets, "
                    f"{(current - completed) / max(elapsed, 1e-9):.2f} hands/s"
                )
                last_progress = now

    elapsed = time.perf_counter() - started
    return {
        "output": str(output.resolve()),
        "trained_hands": hands,
        "completed_hands": target,
        "elapsed_seconds": elapsed,
        "hands_per_second": hands / max(elapsed, 1e-9),
        **agent.diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the bucketed seventh-street MCCFR table."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--raise-cap", type=int, default=2)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--checkpoint-hands", type=int, default=1000)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--start-street", choices=MCCFR_START_STREETS, default="7th_hidden"
    )
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--freeze-seventh", action="store_true")
    parser.add_argument("--cluster-template", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-average-strategy", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.output,
                hands=args.hands,
                iterations=args.iterations,
                raise_cap=args.raise_cap,
                ante=args.ante,
                checkpoint_hands=args.checkpoint_hands,
                progress_seconds=args.progress_seconds,
                seed=args.seed,
                resume=args.resume,
                reset_average_strategy=args.reset_average_strategy,
                start_street=args.start_street,
                initialize_from=args.initialize_from,
                freeze_seventh=args.freeze_seventh,
                cluster_template=args.cluster_template,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
