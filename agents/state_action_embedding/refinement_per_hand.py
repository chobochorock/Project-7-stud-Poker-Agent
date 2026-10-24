"""Replay the unchanged batched trainer; evaluate one held-out root per hand.

Usage (project root, existing Torch Python):
    python -m agents.state_action_embedding.refinement_per_hand --out-dir RUN
    python -m agents.state_action_embedding.refinement_per_hand --out-dir RUN --resume
Plot separately using the existing Python with matplotlib (no Torch required):
    python -m agents.state_action_embedding.refinement_per_hand --out-dir RUN --plot-only
Default: first 1,000 hands, raw/fixed policies, current AND average, 32 particles.
Output: per_hand_gap.jsonl, per_hand_gap.png, per_hand_summary.json plus checkpoints.
The policy is the pre-update policy actually used by each training hand. Batch128
is unchanged; every hand gets a distinct held-out fifth-street information set.
The plot uses a trailing 100-hand mean, NOT a same-policy fixed-panel estimate.
This is measurement instrumentation, not CFR+IRA or an exploitability evaluator.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

FAMILY = Path(__file__).resolve().parent


def policy_budget(hand, batch=128, interval=10000):
    if hand < 1 or batch < 1 or interval < 1:
        raise ValueError("Positive hand, batch and interval required")
    boundary = ((hand - 1) // interval) * interval
    return boundary + ((hand - 1 - boundary) // batch) * batch


def moving_mean(values, window=100):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < window or window < 1:
        raise ValueError("A full window is required")
    sums = np.r_[0.0, np.cumsum(values)]
    return (sums[window:] - sums[:-window]) / window


def evaluate_one(task):
    from . import refinement_cfr as r

    root_task, budget = task
    row = r.evaluate_root(root_task)
    row["policy_hands"] = budget
    return row


def run(args):
    from . import refinement_cfr as r

    r.c.torch.set_num_threads(1)
    source = json.loads((args.source_run / "config.json").read_text(encoding="utf-8"))
    if (
        args.hands > source["hands"]
        or source["batch"] != 128
        or any(
            h % source["interval"] and h != source["hands"]
            for h in source["milestones"]
        )
    ):
        raise ValueError("Stay inside the source run's budget and batch128 protocol")
    budgets = sorted(
        {policy_budget(h, 128, source["interval"]) for h in range(1, args.hands + 1)}
    )
    stop = min(
        source["hands"],
        ((args.hands - 1) // source["interval"] + 1) * source["interval"],
    )
    conditions = dict(source)
    conditions.update(
        reference=Path(source["reference"]),
        baseline_run=Path(source["baseline_run"]),
        out_dir=args.out_dir,
        workers=args.workers,
        resume=args.resume,
        hands=stop,
        milestones=sorted((set(budgets) | {stop}) - {0}),
    )
    r.train(SimpleNamespace(**conditions))
    for arm in r.ARMS:
        original = args.source_run / arm / f"checkpoint_{stop:07d}.npz"
        if not original.exists():
            raise FileNotFoundError(
                f"Need a source checkpoint at replay endpoint: {original}"
            )
        before, after = r.c.load_tables(original), r.c.load_tables(
            args.out_dir / arm / original.name
        )
        for key in ("regret", "average", "visits", "rules"):
            if not np.array_equal(before[key], after[key]):
                raise AssertionError(f"Replay changed training: {arm}/{key}")
        zero = args.out_dir / arm / "initial_policy.npz"
        if not zero.exists():
            # Only the model fields read by the existing evaluator; no replay copies.
            np.savez_compressed(
                zero,
                regret=np.zeros_like(after["regret"]),
                average=np.zeros_like(after["average"]),
                rules="{}",
            )

    tasks, snapshots = [], {}
    for arm in ("raw", "fixed"):
        for budget in budgets:
            path = args.out_dir / arm / f"after_review_{budget:07d}.npz"
            if not path.exists():
                path = (
                    args.out_dir
                    / arm
                    / (
                        f"checkpoint_{budget:07d}.npz"
                        if budget
                        else "initial_policy.npz"
                    )
                )
            snapshots[str(path.resolve())] = r.c.r.sha(path)
        for hand in range(1, args.hands + 1):
            budget = policy_budget(hand, 128, source["interval"])
            path = args.out_dir / arm / f"after_review_{budget:07d}.npz"
            if not path.exists():
                path = (
                    args.out_dir
                    / arm
                    / (
                        f"checkpoint_{budget:07d}.npz"
                        if budget
                        else "initial_policy.npz"
                    )
                )
            for policy in ("current", "average"):
                tasks.append(
                    (
                        (
                            str(path.resolve()),
                            arm,
                            hand,
                            "per_hand",
                            policy,
                            hand - 1,
                            args.particles,
                            args.eval_seed,
                        ),
                        budget,
                    )
                )
    meta = dict(
        hands=args.hands,
        particles=args.particles,
        seed=args.eval_seed,
        window=100,
        batch=128,
        interval=source["interval"],
        arms=["raw", "fixed"],
        policies=["current", "average"],
        snapshots=snapshots,
        measurement="one distinct held-out original root per hand; policy used BEFORE batched update",
        source_run=str(args.source_run.resolve()),
        replay_verified_at_hands=stop,
        training_config_sha256=r.c.r.sha(args.out_dir / "config.json"),
        script_sha256=r.c.r.sha(Path(__file__)),
    )
    output, metadata = (
        args.out_dir / "per_hand_gap.jsonl",
        args.out_dir / "per_hand_config.json",
    )
    if metadata.exists():
        if not args.resume or json.loads(metadata.read_text(encoding="utf-8")) != meta:
            raise ValueError("Evaluation exists or conditions differ")
    else:
        if output.exists():
            raise FileExistsError(output)
        r.c.r.write_json(metadata, meta)
    rows = (
        [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        if output.exists()
        else []
    )
    done = {(v["arm"], v["hands"], v["policy"]) for v in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate evaluation rows")
    pending = [t for t in tasks if (t[0][1], t[0][2], t[0][4]) not in done]
    started, logged = time.perf_counter(), 0.0
    with mp.get_context("spawn").Pool(
        args.workers, r.init_worker, (source["reference"],)
    ) as pool:
        for count, row in enumerate(pool.imap(evaluate_one, pending), 1):
            r.write_row(output, row)
            now = time.perf_counter()
            if now - logged >= 30 or count == len(pending):
                print(
                    f"PER_HAND completed={len(done)+count} total={len(tasks)} arm={row['arm']} hands={row['hands']} policy={row['policy']} elapsed_seconds={now-started:.1f}",
                    flush=True,
                )
                logged = now
    r.write_row(
        args.out_dir / "evaluation_sessions.jsonl",
        dict(rows=len(pending), seconds=time.perf_counter() - started),
    )
    print(
        f"PER_HAND_DATA_COMPLETE output={output}; render with --plot-only", flush=True
    )


def plot(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((directory / "per_hand_config.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(v)
        for v in (directory / "per_hand_gap.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    panels, identities = {}, {}
    for row in rows:
        key = row["arm"], row["policy"]
        panel = panels.setdefault(key, {})
        hand = row["hands"]
        if hand in panel or row["particles"] != meta["particles"] or row["missing"]:
            raise ValueError("Duplicate row, inconsistent particles or missing bucket")
        signature = row["root_seed"], row["observation_sha256"], tuple(row["legal"])
        if identities.setdefault(hand, signature) != signature:
            raise ValueError("Unmatched root between comparison arms/policies")
        if row["policy_hands"] != policy_budget(hand, meta["batch"], meta["interval"]):
            raise ValueError("Wrong policy timing")
        panel[hand] = row
    if set(panels) != {(a, p) for a in meta["arms"] for p in meta["policies"]}:
        raise ValueError("Missing comparison arm")
    n, window = meta["hands"], meta["window"]
    x, mean_x = np.arange(1, n + 1), np.arange(window, n + 1)
    means, summary = {}, {}
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, policy in enumerate(meta["policies"]):
        for arm, color, label in (
            ("raw", "#227c70", "Raw k-means"),
            ("fixed", "#4765b3", "Latent fixed"),
        ):
            panel = panels[(arm, policy)]
            if set(panel) != set(range(1, n + 1)):
                raise ValueError("Incomplete per-hand trace")
            values = np.array([panel[h]["local_br_gap"] for h in x])
            smoothed = moving_mean(values, window)
            means[(arm, policy)] = smoothed
            positive = values > 0
            axes[0, col].scatter(
                x[positive], values[positive], s=4, alpha=0.10, color=color
            )
            positive_mean = smoothed > 0
            axes[0, col].plot(
                mean_x[positive_mean],
                smoothed[positive_mean],
                color=color,
                label=f"{label}: MA100",
            )
            summary[f"{arm}_{policy}"] = dict(
                final_ma100=float(smoothed[-1]),
                raw_zero_count=int((values == 0).sum()),
                mean_gap=float(values.mean()),
                mean_crossfit=float(
                    np.mean([v["crossfit_deviation"] for v in panel.values()])
                ),
            )
        denominator = means[("raw", policy)]
        valid = denominator > 0
        axes[1, col].plot(
            mean_x[valid],
            means[("fixed", policy)][valid] / denominator[valid],
            color="#4765b3",
            label="Latent MA100 / raw MA100",
        )
        axes[1, col].axhline(1, color="#555555", linewidth=0.8)
        axes[0, col].set_title(
            f"{policy.title()}: one evaluated information set per hand"
        )
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylabel("Local BR-gap (ante, log)")
        axes[1, col].set_title(f"{policy.title()}: matched moving-mean ratio")
    for panel in axes.flat:
        panel.set_xscale("log")
        panel.set_xlabel("Training hand index (policy frozen within each batch128)")
        panel.grid(alpha=0.2)
        panel.legend(fontsize=8)
    figure.suptitle(f"First {n:,} hands only: raw samples + trailing 100-hand mean")
    figure.text(
        0.5,
        0.025,
        f"{meta['particles']} particles/root; zeros retained in MA but omitted from log scatter. No full BR/LBR/exploitability.\nMA100 mixes 100 roots and potentially different policies; it does not remove finite-sample max bias.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.96))
    figure.savefig(directory / "per_hand_gap.png", dpi=150)
    plt.close(figure)
    (directory / "per_hand_summary.json").write_text(
        json.dumps(
            dict(
                metrics=summary,
                rows=len(rows),
                rollouts=sum(v["rollouts"] for v in rows),
                policy_queries=sum(v["policy_queries"] for v in rows),
                missing=sum(v["missing"] for v in rows),
                protocol=meta["measurement"],
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        f"PER_HAND_COMPLETE rows={len(rows)} plot={directory / 'per_hand_gap.png'}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-run", type=Path, default=FAMILY / "data/refinement_dense_20261003"
    )
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=307)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if not args.out_dir.is_relative_to((FAMILY / "data").resolve()):
        parser.error("Output must be inside the model family's data/")
    if (
        args.hands < 100
        or args.workers < 1
        or args.eval_seed < 1
        or args.particles < 4
        or args.particles % 2
    ):
        parser.error(
            "Need at least 100 hands, positive workers/seed and even particles >=4"
        )
    plot(args.out_dir) if args.plot_only else run(args)


if __name__ == "__main__":
    main()
