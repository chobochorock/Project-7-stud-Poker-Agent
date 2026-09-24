"""Train the existing event Transformer with historical-policy PPO self-play.

Usage (project root, the existing Python 3.12/PyTorch runtime):
    python -m agents.state_action_embedding.ppo_self_play train --out-dir RUN --init BC.pt
    python -m agents.state_action_embedding.ppo_self_play test --out-dir RUN --test-pairs 4000
    python -m agents.state_action_embedding.ppo_self_play plot --out-dir RUN

Input: optional BC/PPO inference checkpoint; otherwise a fresh random policy.
Output: metrics.sqlite, initial/last policies, immutable numbered checkpoints, PNG.
Metric: seat-paired chips/hand vs frozen opponents, NOT exploitability or NashConv.
Checkpoints are weight-only warm starts, not optimizer/RNG-resumable state.
See PPO_SELF_PLAY.md for the opponent pool, seed partitions and statistical limits.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np


MONITOR_DEAL_BASE = 2_000_000_000_000
TEST_DEAL_BASE = 3_000_000_000_000
MONITOR_ACTION_BASE = 5_000_000_000_000
TEST_ACTION_BASE = 6_000_000_000_000
BENCHMARKS = ("heuristic", "random", "initial")
SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE updates (
    update_index INTEGER PRIMARY KEY, train_hands INTEGER, learner_decisions INTEGER,
    train_seconds REAL, elapsed_seconds REAL, opponent_update INTEGER, pool_size INTEGER,
    rollout_chips REAL, policy_loss REAL, value_loss REAL, entropy REAL,
    approx_kl REAL, clip_fraction REAL, optimizer_steps INTEGER, kl_stopped INTEGER
);
CREATE TABLE evaluations (
    split TEXT, update_index INTEGER, opponent TEXT,
    train_hands INTEGER, learner_decisions INTEGER, train_seconds REAL, elapsed_seconds REAL,
    pairs INTEGER, chips_per_hand REAL, ci_low REAL, ci_high REAL,
    win_rate REAL, tie_rate REAL, delta_chips REAL, delta_ci_low REAL, delta_ci_high REAL,
    PRIMARY KEY (split, update_index, opponent)
);
CREATE TABLE pair_returns (
    split TEXT, update_index INTEGER, opponent TEXT, pair_index INTEGER,
    chips_seat0 INTEGER, chips_seat1 INTEGER,
    PRIMARY KEY (split, update_index, opponent, pair_index)
);
"""


def set_metadata(db: sqlite3.Connection, key: str, value: object) -> None:
    with db:
        db.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            (key, json.dumps(value, allow_nan=False)),
        )


def frozen_copy(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


def add_snapshot(pool: dict, update: int, model, capacity: int) -> None:
    """Keep the initial policy and the most recent capacity-1 snapshots."""
    pool[update] = frozen_copy(model)
    while len(pool) > capacity:
        del pool[min(key for key in pool if key != 0)]


def save_checkpoint(path: Path, model, update: int, hands: int, decisions: int) -> None:
    from .bc_ppo import save_policy

    temporary = path.with_suffix(".tmp")
    save_policy(
        temporary,
        model,
        stage="ppo_self_play",
        selected_update=update,
        train_hands=hands,
        learner_decisions=decisions,
    )
    temporary.replace(path)


def stored_returns(db: sqlite3.Connection, split: str, update: int, opponent: str):
    return np.asarray(
        db.execute(
            "SELECT chips_seat0, chips_seat1 FROM pair_returns "
            "WHERE split=? AND update_index=? AND opponent=? ORDER BY pair_index",
            (split, update, opponent),
        ).fetchall(),
        dtype=np.int32,
    )


def benchmark(
    db: sqlite3.Connection,
    model,
    initial,
    split: str,
    update: int,
    pairs: int,
    counters: tuple[int, int, float],
    started: float,
) -> None:
    from .bc_ppo import evaluate_policy, profit_metrics, torch

    torch.set_num_threads(1)
    deal_base = MONITOR_DEAL_BASE if split == "monitor" else TEST_DEAL_BASE
    action_base = MONITOR_ACTION_BASE if split == "monitor" else TEST_ACTION_BASE
    opponents = (None, "random", initial)
    results = {
        name: evaluate_policy(model, pairs, deal_base, action_base, opponent=opponent)
        for name, opponent in zip(BENCHMARKS, opponents)
    }
    scores = {}
    # A checkpoint's whole benchmark bank commits together; interruption leaves no partial point.
    with db:
        for name, values in results.items():
            baseline = values if update == 0 else stored_returns(db, split, 0, name)
            if baseline.shape != values.shape:
                raise ValueError(
                    "Baseline and current evaluation must use identical pairs"
                )
            metric = profit_metrics(values)
            delta = profit_metrics(values - baseline)
            db.execute(
                "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    split,
                    update,
                    name,
                    *counters,
                    time.perf_counter() - started,
                    pairs,
                    metric["chips_per_hand"],
                    *metric["pair_normal_95pct"],
                    metric["win_rate"],
                    metric["tie_rate"],
                    delta["chips_per_hand"],
                    *delta["pair_normal_95pct"],
                ),
            )
            db.executemany(
                "INSERT INTO pair_returns VALUES (?,?,?,?,?,?)",
                [
                    (split, update, name, pair, int(a), int(b))
                    for pair, (a, b) in enumerate(values)
                ],
            )
            scores[name] = round(metric["chips_per_hand"], 5)
    print(f"{split} update={update} chips/hand={scores}", flush=True)


def train(args: argparse.Namespace) -> None:
    from . import bc_ppo as ppo

    if args.out_dir.exists():
        raise FileExistsError(
            "Use a new --out-dir; existing runs are never overwritten"
        )
    ppo.torch.set_num_threads(args.threads)
    ppo.torch.manual_seed(args.seed)
    device = "cuda" if ppo.torch.cuda.is_available() else "cpu"
    model = (
        ppo.load_policy(args.init, device)[0]
        if args.init
        else ppo.ActorCritic(args.dim, args.layers).to(device)
    )
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "checkpoints").mkdir()
    initial = frozen_copy(model)
    pool = {0: initial}
    pool_rng = np.random.default_rng(args.seed + 200)
    update_rng = np.random.default_rng(args.seed + 100)
    optimizer = ppo.torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    train_base = 1_000_000_000_000 + args.seed * 1_000_000_000
    hashes = ppo.source_hashes()
    for path in (Path(__file__), Path(__file__).with_name("experiment.py")):
        hashes[str(path.relative_to(ppo.ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dim": model.dim,
        "layers": len(model.layers),
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "init_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest()
        if args.init
        else None,
        "source_sha256": hashes,
        "rules_version": ppo.BETTING_RULES_VERSION,
        "train_deal_base": train_base,
        "monitor_deal_base": MONITOR_DEAL_BASE,
        "test_deal_base": TEST_DEAL_BASE,
        "monitor_action_base": MONITOR_ACTION_BASE,
        "test_action_base": TEST_ACTION_BASE,
        "opponent_sampling": "uniform over initial + recent frozen snapshots, once per rollout batch",
        "ppo_lr": 1e-4,
        "ppo_clip": 0.2,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "reward_scale": ppo.REWARD_SCALE,
        "entropy_coefficient": 0.01,
        "value_loss_coefficient_on_mse": 0.25,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "decoding": "stochastic categorical, independent physical-seat action RNGs",
        "checkpoint_selection": "fixed budget last; no best-of-monitor selection",
        "checkpoint_format": "weights only; --init is NOT exact resume",
    }
    started = time.perf_counter()
    hands = decisions = 0
    train_seconds = 0.0
    db = sqlite3.connect(args.out_dir / "metrics.sqlite")
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        set_metadata(db, "config", config)
        set_metadata(db, "status", "running")
        save_checkpoint(args.out_dir / "initial.pt", model, 0, 0, 0)
        save_checkpoint(args.out_dir / "last.pt", model, 0, 0, 0)
        benchmark(
            db, model, initial, "monitor", 0, args.eval_pairs, (0, 0, 0.0), started
        )
        for update in range(1, args.updates + 1):
            tick = time.perf_counter()
            opponent_update = int(pool_rng.choice(list(pool)))
            ppo.torch.set_num_threads(1)
            data, profits = ppo.rollout(
                model,
                args.rollout_hands,
                train_base + hands,
                4_000_000_000_000 + args.seed * 1_000_000_000 + update,
                opponent=pool[opponent_update],
            )
            ppo.torch.set_num_threads(args.threads)
            metric = ppo.update_ppo(model, optimizer, data, update_rng, args, device)
            train_seconds += time.perf_counter() - tick
            hands += args.rollout_hands
            decisions += len(data["action"])
            save_checkpoint(args.out_dir / "last.pt", model, update, hands, decisions)
            with db:
                db.execute(
                    "INSERT INTO updates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        update,
                        hands,
                        decisions,
                        train_seconds,
                        time.perf_counter() - started,
                        opponent_update,
                        len(pool),
                        float(profits.mean()),
                        metric["policy_loss"],
                        metric["value_loss"],
                        metric["entropy"],
                        metric["approx_kl"],
                        metric["clip_fraction"],
                        metric["optimizer_steps"],
                        int(metric["kl_stopped"]),
                    ),
                )
            snapshot_due = update % args.snapshot_every == 0
            eval_due = update % args.eval_every == 0 or update == args.updates
            if snapshot_due or eval_due:
                save_checkpoint(
                    args.out_dir / "checkpoints" / f"update_{update:06d}.pt",
                    model,
                    update,
                    hands,
                    decisions,
                )
            if snapshot_due:
                add_snapshot(pool, update, model, args.pool_size)
            if eval_due:
                benchmark(
                    db,
                    model,
                    initial,
                    "monitor",
                    update,
                    args.eval_pairs,
                    (hands, decisions, train_seconds),
                    started,
                )
            elif update % 8 == 0:
                print(
                    f"update={update} hands={hands} opponent_update={opponent_update}",
                    flush=True,
                )
        set_metadata(db, "status", "completed")
    except KeyboardInterrupt:
        set_metadata(db, "status", "interrupted")
        raise
    except Exception as error:
        set_metadata(db, "status", {"failed": str(error)})
        raise
    finally:
        db.close()
    print(
        f"Saved {args.out_dir / 'metrics.sqlite'}; use the plot and test commands.",
        flush=True,
    )


def test(args: argparse.Namespace) -> None:
    """Evaluate initial and fixed-budget last policies on a new, held-out deal set."""
    from .bc_ppo import load_policy

    database = args.out_dir / "metrics.sqlite"
    if not database.is_file():
        raise FileNotFoundError(database)
    db = sqlite3.connect(database)
    try:
        status = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()[0]
        )
        if status == "running":
            raise ValueError("Wait for training to finish before running held-out test")
        if db.execute(
            "SELECT 1 FROM evaluations WHERE split='test' AND update_index>0"
        ).fetchone():
            raise FileExistsError(
                "Held-out test already exists; refusing repeated test selection"
            )
        initial, _ = load_policy(args.out_dir / "initial.pt")
        model, saved = load_policy(args.out_dir / "last.pt")
        update = saved["selected_update"]
        if update == 0:
            raise ValueError("No completed PPO update to test")
        counters = db.execute(
            "SELECT train_hands, learner_decisions, train_seconds FROM updates WHERE update_index=?",
            (update,),
        ).fetchone()
        if counters is None:
            raise ValueError("Last policy has no committed training record")
        test_config = {
            "pairs": args.test_pairs,
            "update": update,
            "initial_sha256": hashlib.sha256(
                (args.out_dir / "initial.pt").read_bytes()
            ).hexdigest(),
            "last_sha256": hashlib.sha256(
                (args.out_dir / "last.pt").read_bytes()
            ).hexdigest(),
        }
        previous = db.execute("SELECT value FROM metadata WHERE key='test'").fetchone()
        if previous and json.loads(previous[0]) != test_config:
            raise ValueError(
                "An interrupted test must retain its original policies and pair budget"
            )
        set_metadata(db, "test", test_config)
        started = time.perf_counter()
        if not db.execute("SELECT 1 FROM evaluations WHERE split='test'").fetchone():
            benchmark(
                db, initial, initial, "test", 0, args.test_pairs, (0, 0, 0.0), started
            )
        benchmark(
            db, model, initial, "test", update, args.test_pairs, counters, started
        )
    finally:
        db.close()


def plot(args: argparse.Namespace) -> None:
    """Read committed SQLite rows, including from an ongoing run; requires no Torch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    database = (args.out_dir / "metrics.sqlite").resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        points = db.execute(
            "SELECT * FROM evaluations WHERE split='monitor' ORDER BY update_index"
        ).fetchall()
        updates = db.execute("SELECT * FROM updates ORDER BY update_index").fetchall()
    if not points:
        raise ValueError("No complete monitoring evaluation yet")
    labels = {
        "train_hands": "Training hands (evaluation excluded)",
        "learner_decisions": "Learner decisions collected",
        "train_seconds": "Training seconds (evaluation excluded)",
        "elapsed_seconds": "Wall seconds (including evaluation)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    colors = ("#167d9a", "#a33c57", "#419348")
    for name, color in zip(BENCHMARKS, colors):
        rows = [row for row in points if row["opponent"] == name]
        x = [row[args.x_axis] for row in rows]
        for axis, mean, low, high in (
            (axes[0], "chips_per_hand", "ci_low", "ci_high"),
            (axes[1], "delta_chips", "delta_ci_low", "delta_ci_high"),
        ):
            axis.plot(x, [r[mean] for r in rows], marker=".", color=color, label=name)
            axis.fill_between(
                x,
                [r[low] for r in rows],
                [r[high] for r in rows],
                color=color,
                alpha=0.13,
            )
    axes[0].set(title="Frozen opponents: chips / hand", ylabel="Net chips / hand")
    axes[1].set(
        title="Paired improvement over initial policy", ylabel="Change in chips / hand"
    )
    for axis in axes[:2]:
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.legend()
    axes[2].plot(
        [r[args.x_axis] for r in updates],
        [r["entropy"] for r in updates],
        color="#7357a2",
    )
    axes[2].set(title="Policy entropy (diagnostic, not strength)", ylabel="Nats")
    for axis in axes:
        axis.set_xlabel(labels[args.x_axis])
        axis.grid(alpha=0.2)
    fig.suptitle(
        "PPO self-play monitoring | pointwise paired-deal 95% normal intervals, not exploitability",
        fontsize=12,
    )
    output = args.out_dir / f"learning_curve_{args.x_axis}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser(
        "train", help="fresh run; optional weight-only initialization"
    )
    trainer.add_argument("--out-dir", type=Path, required=True)
    trainer.add_argument(
        "--init", type=Path, help="BC/PPO weights; dimensions come from this file"
    )
    trainer.add_argument("--seed", type=int, default=11)
    trainer.add_argument(
        "--updates", type=int, default=512, help="fresh rollout batches"
    )
    trainer.add_argument(
        "--rollout-hands", type=int, default=128, help="complete 1v1 hands per batch"
    )
    trainer.add_argument("--ppo-epochs", type=int, default=4)
    trainer.add_argument(
        "--batch",
        type=int,
        default=64,
        help="learner decisions per optimizer minibatch",
    )
    trainer.add_argument(
        "--snapshot-every",
        type=int,
        default=16,
        help="updates between frozen opponents",
    )
    trainer.add_argument(
        "--pool-size", type=int, default=8, help="initial plus recent opponents"
    )
    trainer.add_argument(
        "--eval-every", type=int, default=32, help="updates between monitoring checks"
    )
    trainer.add_argument(
        "--eval-pairs",
        type=int,
        default=256,
        help="seat-swapped deal pairs per opponent",
    )
    trainer.add_argument("--threads", type=int, default=4)
    trainer.add_argument(
        "--dim", type=int, default=64, help="random-start width; ignored with --init"
    )
    trainer.add_argument(
        "--layers", type=int, default=2, help="random-start depth; ignored with --init"
    )
    tester = commands.add_parser(
        "test", help="one held-out initial/last benchmark, not model selection"
    )
    tester.add_argument("--out-dir", type=Path, required=True)
    tester.add_argument(
        "--test-pairs",
        type=int,
        default=4000,
        help="paired deals per opponent and policy",
    )
    plotter = commands.add_parser(
        "plot", help="plot SQLite monitoring data, also while training"
    )
    plotter.add_argument("--out-dir", type=Path, required=True)
    plotter.add_argument(
        "--x-axis",
        choices=(
            "train_hands",
            "learner_decisions",
            "train_seconds",
            "elapsed_seconds",
        ),
        default="train_hands",
    )
    args = parser.parse_args(argv)
    if args.command == "train":
        positive = (
            args.updates,
            args.rollout_hands,
            args.ppo_epochs,
            args.batch,
            args.snapshot_every,
            args.eval_every,
            args.threads,
            args.dim,
            args.layers,
        )
        if (
            min(positive) < 1
            or args.pool_size < 2
            or args.eval_pairs < 2
            or args.dim % 4
        ):
            parser.error(
                "Positive budgets, pool>=2, pairs>=2, and dim divisible by 4 required"
            )
        if (
            not 0 <= args.seed <= 999
            or args.updates * args.rollout_hands >= 1_000_000_000
        ):
            parser.error("Seed must be in [0,999], with fewer than 1B training hands")
        if args.rollout_hands % 2:
            parser.error("Use an even --rollout-hands for balanced learner seats")
    elif args.command == "test" and args.test_pairs < 2:
        parser.error("At least two test pairs required")
    {"train": train, "test": test, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
