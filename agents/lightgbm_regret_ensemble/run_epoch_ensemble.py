"""Run the C++ 7-stud epoch ensemble against cold-start hard-256 MCCFR.

Usage (project root):
    python agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --hands 100000 --build
    python agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --hands 200 --epoch-hands 100 --build
    python agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --plot RUN_DIRECTORY
    python agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --label-mode refined --tree-budget 24
    python agents/lightgbm_regret_ensemble/run_epoch_ensemble.py --label-mode cross_entropy --tree-budget 24

Input: existing frozen power256 atlas; epoch teacher-policy targets, not oracle labels.
Output: data/<run> with every-hand metrics, epoch models and plots.
Metric: estimated first-5th-infoset one-step gap; not full-game exploitability.
The .bin epoch models are inference exports, not exact-resume checkpoints.
Default storage: per-tree leaf regrets and strategy sums (EPOCH7L1).
--aggregation joint retains legacy joint-leaf exports for explicit controls.
See README.md for scope, costs and finite-sample evaluation bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "PROJECT_LAYOUT.json").exists())
CORE = ROOT / "agents/cpp_mccfr"
VENDOR = ROOT.parent / "Toy-Card-Game-Agent" / ".lightgbm_experiment"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "data/mpl-cache"))
import lightgbm as lgb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DTYPE = np.dtype(
    [
        ("group", "<u4"),
        ("features", "<f4", 34),
        ("regrets", "<f8", 8),
        ("teacher", "<f8", 8),
        ("visits", "<u8"),
    ]
)
LEAF_DTYPE = np.dtype(DTYPE.descr + [("strategy_sum", "<f8", 8)])
LABEL_MODES = ("coarse", "refined", "cross_entropy")
REFINED_EDGES = (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.95, 1)


def tree_nodes(tree):
    """Export numerical <= splits with LightGBM's exact leaf IDs."""
    nodes = []

    def visit(node):
        index = len(nodes)
        nodes.append(None)
        if "leaf_index" in node or "split_index" not in node:
            nodes[index] = (-1, -1, -1, node.get("leaf_index", 0), 0.0)
        else:
            if node["decision_type"] != "<=":
                raise ValueError("Only numerical LightGBM splits are supported")
            left = visit(node["left_child"])
            right = visit(node["right_child"])
            nodes[index] = (node["split_feature"], left, right, -1, node["threshold"])
        return index

    visit(tree)
    return nodes


def labels_for(probabilities, mode="coarse"):
    """Legal labels start at 1; 0 is reserved for illegal actions.

    Refined bins are left-closed, right-open, except the final [.95, 1].
    Coarse preserves the original inclusive .9 boundary for reproducibility.
    """
    probabilities = np.asarray(probabilities)
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if mode == "refined":
        return (
            np.searchsorted(REFINED_EDGES[1:-1], probabilities, side="right").astype(
                np.int32
            )
            + 1
        )
    if mode != "coarse":
        raise ValueError("cross_entropy uses probabilities directly, not class labels")
    labels = np.ones(probabilities.shape, dtype=np.int32)
    labels[probabilities >= 0.1] = 2
    labels[probabilities >= 0.5] = 3
    labels[probabilities > 0.9] = 4
    return labels


def fit_action(features, probabilities, seed, mode="coarse", tree_budget=None):
    """Shared splitter fit for online epochs and offline label comparisons.

    Return the booster and training class-mean probabilities (diagnostics only).
    tree_budget caps trees per action, not boosting rounds; multiclass uses
    complete rounds, so it may leave fewer than num_class trees unused.
    None preserves the historical three boosting rounds.
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"unknown label mode: {mode}")
    p = np.asarray(probabilities, dtype=float)
    labels_for(p)  # Validate soft targets as well as categorical targets.
    params = dict(
        num_leaves=4,
        learning_rate=0.3,
        min_data_in_leaf=10,
        min_data_in_bin=1,
        verbosity=-1,
        num_threads=1,
        deterministic=True,
        force_col_wise=True,
        seed=seed,
    )
    if mode == "cross_entropy":
        target, means, classes_per_round = p, np.array([p.mean()]), 1
        params["objective"] = "cross_entropy"
        constant = np.ptp(p) == 0
    else:
        classes, target = np.unique(labels_for(p, mode), return_inverse=True)
        means = np.array([p[target == i].mean() for i in range(len(classes))])
        classes_per_round = len(classes)
        params.update(objective="multiclass", num_class=classes_per_round)
        constant = classes_per_round < 2
    if constant:
        return None, means
    rounds = 3 if tree_budget is None else tree_budget // classes_per_round
    if rounds < 1:
        raise ValueError("tree budget must allow at least one complete boosting round")
    model = lgb.train(
        params, lgb.Dataset(features, label=target), num_boost_round=rounds
    )
    return model, means


def write_group(stream, group_id, trees, signatures, regrets):
    """Write one group in the existing C++ EPOCH7V1 inference format."""
    stream.write(struct.pack("<II", int(group_id), len(trees)))
    for tree in trees:
        stream.write(struct.pack("<I", len(tree)))
        for node in tree:
            stream.write(struct.pack("<iiiid", *node))
    stream.write(struct.pack("<I", len(signatures)))
    for signature, regret in zip(signatures, regrets):
        stream.write(np.asarray(signature, dtype="<i4").tobytes())
        stream.write(np.asarray(regret, dtype="<f8").tobytes())


def unique_leaf_signatures(signatures):
    """Losslessly pack 4-leaf tree IDs before grouping millions of rows."""
    rows, trees = signatures.shape
    if not trees:
        return signatures[:1].astype("<i4"), np.zeros(rows, dtype=int)
    if np.any(signatures > 3) or np.any(signatures < 0):
        raise ValueError("2-bit route packing requires at most four leaves")
    packed = np.zeros((rows, (trees + 3) // 4), dtype=np.uint8)
    for t in range(trees):
        packed[:, t // 4] |= signatures[:, t].astype(np.uint8) << (2 * (t % 4))
    codes, routes = np.unique(packed, axis=0, return_inverse=True)
    unique = np.empty((len(codes), trees), dtype="<i4")
    for t in range(trees):
        unique[:, t] = (codes[:, t // 4] >> (2 * (t % 4))) & 3
    return unique, routes


def leaf_statistics(leaf_ids, regrets, strategy_sum, leaf_count):
    """Sum this epoch's increments in one tree; never aggregate joint routes."""
    leaf_ids = np.asarray(leaf_ids, dtype=np.intp)
    if leaf_ids.ndim != 1 or not len(leaf_ids):
        raise ValueError("nonempty leaf assignments required")
    if np.any(leaf_ids < 0) or np.any(leaf_ids >= leaf_count):
        raise ValueError("invalid leaf assignment")
    if regrets.shape != (len(leaf_ids), 8) or strategy_sum.shape != regrets.shape:
        raise ValueError("expected one eight-action vector per row")
    if not np.isfinite(regrets).all() or not np.isfinite(strategy_sum).all():
        raise ValueError("nonfinite leaf payload")
    if np.any(strategy_sum < 0):
        raise ValueError("negative strategy mass")
    counts = np.bincount(leaf_ids, minlength=leaf_count).astype("<u8")
    if np.any(counts == 0):
        raise ValueError("fitted leaf missing from aggregation rows")
    tables = []
    for values in (regrets, strategy_sum):
        table = np.column_stack(
            [
                np.bincount(leaf_ids, weights=values[:, a], minlength=leaf_count)
                for a in range(8)
            ]
        )
        if not np.allclose(table.sum(axis=0), values.sum(axis=0)):
            raise ValueError("leaf aggregation lost mass")
        tables.append(table)
    return tables[0], tables[1], counts


def write_leaf_group(stream, group_id, trees, statistics):
    """EPOCH7L1: nodes, then each tree's leaf regret/strategy/count payloads."""
    if not trees or len(trees) != len(statistics):
        raise ValueError("each leaf model requires at least one tree")
    stream.write(struct.pack("<II", int(group_id), len(trees)))
    for tree in trees:
        stream.write(struct.pack("<I", len(tree)))
        for node in tree:
            stream.write(struct.pack("<iiiid", *node))
    for regret, strategy, counts in statistics:
        stream.write(struct.pack("<I", len(counts)))
        for r, s, count in zip(regret, strategy, counts):
            stream.write(np.asarray(r, dtype="<f8").tobytes())
            stream.write(np.asarray(s, dtype="<f8").tobytes())
            stream.write(struct.pack("<Q", int(count)))


def fit_epoch(
    rows_path,
    epoch,
    seed,
    fit_cap=20000,
    label_mode="coarse",
    tree_budget=None,
    aggregation="leaf",
):
    started = perf_counter()
    with rows_path.open("rb") as stream:
        magic = stream.read(8)
        if magic not in (b"ROWS7V01", b"ROWS7V02"):
            raise ValueError("Unknown row format")
        count = struct.unpack("<Q", stream.read(8))[0]
    if aggregation not in ("leaf", "joint"):
        raise ValueError("unknown aggregation")
    if aggregation == "leaf" and magic != b"ROWS7V02":
        raise ValueError(
            "leaf export requires preserved strategy sums, not legacy teacher-only rows"
        )
    dtype = LEAF_DTYPE if magic == b"ROWS7V02" else DTYPE
    if rows_path.stat().st_size != 16 + count * dtype.itemsize:
        raise ValueError("Truncated rows")
    rows = np.memmap(rows_path, dtype=dtype, mode="r", offset=16, shape=(count,))
    out_dir = rows_path.parent / f"epoch_{epoch:03d}"
    out_dir.mkdir(exist_ok=False)
    model_path = out_dir / "model.bin"
    group_ids = np.unique(rows["group"])
    rng = np.random.default_rng(seed + epoch * 1009)
    total_buckets = 0
    total_trees = 0
    total_fit_rows = 0
    with model_path.open("wb") as stream:
        model_magic = b"EPOCH7L1" if aggregation == "leaf" else b"EPOCH7V1"
        stream.write(model_magic + struct.pack("<II", epoch, len(group_ids)))
        for group_id in group_ids:
            indices = np.flatnonzero(rows["group"] == group_id)
            features = np.ascontiguousarray(rows["features"][indices])
            teacher = rows["teacher"][indices]
            mask = int(group_id) % 256
            legal = np.array([bool(mask & (1 << a)) for a in range(8)])
            assert np.allclose(teacher.sum(axis=1), 1)
            assert np.all(teacher[:, ~legal] == 0)
            training = np.arange(len(indices))
            if len(training) > fit_cap:
                training = np.sort(rng.choice(training, fit_cap, replace=False))
            total_fit_rows += len(training)
            all_trees = []
            leaf_columns = []
            statistics = []
            regrets = rows["regrets"][indices]
            strategy_sum = (
                rows["strategy_sum"][indices] if magic == b"ROWS7V02" else None
            )
            if strategy_sum is not None:
                assert np.all(strategy_sum[:, ~legal] == 0)
            for action in np.flatnonzero(legal):
                model, _ = fit_action(
                    features[training],
                    teacher[training, action],
                    seed + epoch * 1009 + int(group_id) * 8 + int(action),
                    label_mode,
                    tree_budget,
                )
                if model is None:
                    continue
                model.save_model(str(out_dir / f"group_{group_id}_action_{action}.txt"))
                trees = [
                    tree_nodes(t["tree_structure"])
                    for t in model.dump_model()["tree_info"]
                ]
                leaves = np.asarray(
                    model.predict(features, pred_leaf=True, num_threads=1),
                    dtype=np.uint8,
                )
                leaves = leaves.reshape(len(indices), -1)
                assert len(trees) == leaves.shape[1]
                # Check exported threshold routing independently from LightGBM.
                for t, tree in enumerate(trees):
                    for row in np.linspace(
                        0, len(features) - 1, min(10, len(features)), dtype=int
                    ):
                        n = 0
                        while tree[n][3] < 0:
                            f, left, right, _, threshold = tree[n]
                            n = left if features[row, f] <= threshold else right
                        assert tree[n][3] == leaves[row, t]
                all_trees.extend(trees)
                if aggregation == "leaf":
                    for t, tree in enumerate(trees):
                        leaf_count = max(node[3] for node in tree) + 1
                        statistics.append(
                            leaf_statistics(
                                leaves[:, t], regrets, strategy_sum, leaf_count
                            )
                        )
                else:
                    leaf_columns.append(leaves)
            if aggregation == "leaf":
                if not all_trees:
                    all_trees = [[(-1, -1, -1, 0, 0.0)]]
                    statistics = [
                        leaf_statistics(
                            np.zeros(len(indices), dtype=int), regrets, strategy_sum, 1
                        )
                    ]
                write_leaf_group(stream, group_id, all_trees, statistics)
                total_buckets += sum(len(stat[2]) for stat in statistics)
            else:
                signatures = (
                    np.column_stack(leaf_columns)
                    if leaf_columns
                    else np.zeros((len(indices), 0), dtype="<i4")
                )
                unique, routes = unique_leaf_signatures(signatures)
                table = np.zeros((len(unique), 8), dtype="<f8")
                np.add.at(table, routes, regrets)
                assert np.allclose(table.sum(axis=0), regrets.sum(axis=0))
                assert np.all(table[:, ~legal] == 0)
                write_group(stream, group_id, all_trees, unique, table)
                total_buckets += len(unique)
            total_trees += len(all_trees)
    summary = {
        "epoch": epoch,
        "rows": int(count),
        "fit_rows": total_fit_rows,
        "groups": len(group_ids),
        "buckets": total_buckets,
        "trees": total_trees,
        "fit_seconds": perf_counter() - started,
        "model_bytes": model_path.stat().st_size,
        "label_mode": label_mode,
        "aggregation": aggregation,
        "model_format": model_magic.decode("ascii"),
        "tree_budget_per_action": tree_budget,
        "refined_edges": REFINED_EDGES if label_mode == "refined" else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return model_path, summary


def plot_run(directory):
    rows = np.genfromtxt(directory / "per_hand.csv", delimiter=",", names=True)
    rows = np.atleast_1d(rows)
    if not len(rows):
        return
    x = rows["hand"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    stats = {}
    block = max(1, min(1000, len(rows) // 20))
    for prefix, name, color in [
        ("ensemble", "Epoch LightGBM", "#bc3347"),
        ("hard256", "Hard-256 MCCFR", "#168776"),
    ]:
        y = rows[f"{prefix}_root_local_gap_ante"]
        n = len(y) // block
        means = y[: n * block].reshape(n, block).mean(axis=1)
        xs = x[: n * block].reshape(n, block).mean(axis=1)
        axes[0].plot(xs, means, label=name, color=color)
        axes[1].plot(
            x, np.cumsum(y) / np.arange(1, len(y) + 1), label=name, color=color
        )
        tail = y[-min(10000, len(y)) :]
        stats[prefix] = {
            "all_hand_gap_mean": float(y.mean()),
            "last_10k_gap_mean": float(tail.mean()),
        }
    axes[0].set_title(
        f"7-stud: first 5th-street infoset Local BR-gap ({block}-hand bins)"
    )
    axes[1].set_title(
        "Cumulative mean of measured gaps (not gap of the average policy)"
    )
    for ax in axes:
        ax.set_xlabel("Training hands (two external-sampling traversals per hand)")
        ax.set_ylabel("Estimated one-step gain / ante")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(directory / "local_br_gap.png", dpi=160)
    plt.close(fig)
    stats.update(
        {
            "hands": int(x[-1]),
            "cpp_peak_rss_bytes": int(rows["peak_cpp_rss_bytes"].max()),
            "train_seconds": float(rows["train_seconds"][-1]),
            "eval_seconds": float(rows["eval_seconds"][-1]),
            "ensemble_training_nodes": int(rows["ensemble_training_nodes"][-1]),
            "hard256_training_nodes": int(rows["hard256_training_nodes"][-1]),
        }
    )
    (directory / "metrics_summary.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument("--epoch-hands", type=int, default=10000)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="coarse")
    parser.add_argument("--aggregation", choices=("leaf", "joint"), default="leaf")
    parser.add_argument(
        "--tree-budget",
        type=int,
        help="maximum trees per legal action; default: historical 3 boosting rounds",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--atlas", type=Path, default=CORE / "data/power256_selfplay100m_eps20_v1.bin"
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot:
        plot_run(args.plot.resolve())
        return
    if (
        min(args.hands, args.epoch_hands, args.particles) < 1
        or args.hands % args.epoch_hands
    ):
        parser.error("positive counts and complete epochs required")
    if args.tree_budget is not None and args.tree_budget < 8:
        parser.error("--tree-budget must be >=8 to support all refined classes")
    exe = HERE / "bin/stud_epoch_ensemble.exe"
    if args.build:
        exe.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                shutil.which("g++") or "g++",
                "-O3",
                "-std=c++17",
                "-DNOMINMAX",
                "-static-libstdc++",
                "-static-libgcc",
                str(HERE / "stud_epoch_ensemble.cpp"),
                "-o",
                str(exe),
                "-lpsapi",
            ],
            check=True,
        )
    subprocess.run([str(exe), "--self-test", str(args.atlas)], check=True)
    directory = (
        args.out_dir or HERE / "data" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["binary_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    config["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            HERE / "stud_epoch_ensemble.cpp",
            CORE / "stud_mccfr.cpp",
            CORE / "local_split_helpers.hpp",
            Path(__file__),
            ROOT / "environments/seven_stud/stud_rules.hpp",
            args.atlas,
        ]
    }
    config["scope"] = "2-player 7-stud v3; H4 fixed heuristic; 5th-7th betting"
    config["regret_combination"] = "epoch k weight=k; current delta weight=t+1; " + (
        "within-epoch tree mean"
        if args.aggregation == "leaf"
        else "joint signature lookup"
    )
    config["average_combination"] = (
        "epoch k weight=k on strategy sums; normalize after combining"
        if args.aggregation == "leaf"
        else "not stored in legacy model"
    )
    config["refined_edges"] = REFINED_EDGES if args.label_mode == "refined" else None
    config[
        "metric"
    ] = "current-policy first-5th-infoset particle-rollout gap; not exploitability"
    (directory / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (directory / "sources").mkdir()
    for source in [
        HERE / "stud_epoch_ensemble.cpp",
        CORE / "stud_mccfr.cpp",
        CORE / "local_split_helpers.hpp",
        Path(__file__),
        ROOT / "environments/seven_stud/stud_rules.hpp",
    ]:
        shutil.copy2(source, directory / "sources" / source.name)
    with (directory / "stderr.log").open("w", encoding="utf-8") as errors, (
        directory / "progress.log"
    ).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(exe),
                str(directory),
                str(args.atlas),
                str(args.seed),
                str(args.particles),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            bufsize=1,
        )

        def wait_for(prefix):
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                log.write(line)
                log.flush()
                if line.startswith(prefix):
                    return
            raise RuntimeError(
                f"C++ exited early ({process.wait()}), see {directory / 'stderr.log'}"
            )

        try:
            wait_for("READY")
            process.stdin.write("AUDIT\n")
            process.stdin.flush()
            wait_for("AUDITED")
            for epoch in range(1, args.hands // args.epoch_hands + 1):
                process.stdin.write(f"TRAIN {args.epoch_hands}\n")
                process.stdin.flush()
                wait_for("EPOCH")
                process.stdin.write("AUDIT\n")
                process.stdin.flush()
                wait_for("AUDITED")
                process.stdin.write("RELEASE\n")
                process.stdin.flush()
                wait_for("RELEASED")
                model, summary = fit_epoch(
                    directory / f"rows_{epoch * args.epoch_hands}.bin",
                    epoch,
                    args.seed,
                    label_mode=args.label_mode,
                    tree_budget=args.tree_budget,
                    aggregation=args.aggregation,
                )
                print(json.dumps(summary), flush=True)
                process.stdin.write(f"LOAD {model}\n")
                process.stdin.flush()
                wait_for("LOADED")
                process.stdin.write("AUDIT\n")
                process.stdin.flush()
                wait_for("AUDITED")
                plot_run(directory)
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            if process.wait() != 0:
                raise RuntimeError("C++ worker failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
    print(f"RESULTS {directory}", flush=True)


if __name__ == "__main__":
    main()
