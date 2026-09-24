"""Offline splitter comparison on SAVED 7-stud epoch rows, not new RL training.

Usage (project root):
    python agents/lightgbm_regret_ensemble/compare_labels.py --source-run RUN_DIR
    python agents/lightgbm_regret_ensemble/compare_labels.py --source-run RUN_DIR --hands 1000000 --sample-rows 20000

Compares coarse, refined [0,.05,.1,.2,.3,.4,.5,.9,1], and soft cross-entropy.
Same sampled rows, feature-grouped holdout and fit sample for every mode.
No original hand IDs exist in ROWS7V01: this is NOT an independent-hand test,
exploitability estimate, or three independently trained 1M policies.
Outputs stay in this model family's data/ folder. Existing runs are untouched.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import struct
from time import perf_counter

from run_epoch_ensemble import (
    DTYPE,
    HERE,
    LABEL_MODES,
    REFINED_EDGES,
    fit_action,
    lgb,
    np,
    plt,
    tree_nodes,
    unique_leaf_signatures,
    write_group,
)


def feature_split(features, rng):
    """Keep every identical feature vector in one fold (not a hand split)."""
    _, inverse = np.unique(features, axis=0, return_inverse=True)
    groups = rng.permutation(int(inverse.max()) + 1)
    if len(groups) < 2:
        raise ValueError("need two different feature vectors for holdout")
    heldout = groups[: max(1, len(groups) // 5)]
    test = np.flatnonzero(np.isin(inverse, heldout))
    train = np.flatnonzero(~np.isin(inverse, heldout))
    assert not np.intersect1d(inverse[train], inverse[test]).size
    return train, test


def regret_policy(regrets, legal):
    positive = np.maximum(regrets, 0) * legal
    mass = positive.sum(axis=1, keepdims=True)
    result = np.broadcast_to(legal / legal.sum(), positive.shape).copy()
    np.divide(positive, mass, out=result, where=mass > 0)
    return result


def compare_group(
    rows, train, test, fitting, group_id, seed, mode, budget, folder, stream
):
    features = np.ascontiguousarray(rows["features"])
    teacher = rows["teacher"]
    legal = ((int(group_id) % 256) & (1 << np.arange(8))) != 0
    if not (
        np.all(np.isfinite(features))
        and np.all(np.isfinite(rows["regrets"]))
        and np.allclose(teacher.sum(axis=1), 1)
        and np.all(teacher[:, ~legal] == 0)
        and np.all(rows["regrets"][:, ~legal] == 0)
    ):
        raise ValueError("invalid features, teacher or regret mask")
    predictions = np.zeros((len(test), 8))
    trees, leaves = [], []
    for action in np.flatnonzero(legal):
        model, means = fit_action(
            features[fitting],
            teacher[fitting, action],
            seed + int(group_id) * 8 + int(action),
            mode,
            budget,
        )
        if model is None:
            predictions[:, action] = means[0]
            continue
        model.save_model(str(folder / f"group_{group_id}_action_{action}.txt"))
        output = model.predict(features[test], num_threads=1)
        predictions[:, action] = output if mode == "cross_entropy" else output @ means
        exported = [
            tree_nodes(t["tree_structure"]) for t in model.dump_model()["tree_info"]
        ]
        routes = np.asarray(
            model.predict(features, pred_leaf=True, num_threads=1), dtype=np.uint8
        )
        routes = routes.reshape(len(rows), -1)
        assert len(exported) == routes.shape[1]
        for t, tree in enumerate(exported):
            for index in test[:8]:
                node = 0
                while tree[node][3] < 0:
                    f, left, right, _, threshold = tree[node]
                    node = left if features[index, f] <= threshold else right
                assert tree[node][3] == routes[index, t]
        trees.extend(exported)
        leaves.append(routes)
    signatures = (
        np.column_stack(leaves) if leaves else np.zeros((len(rows), 0), dtype=np.uint8)
    )
    unique, route = unique_leaf_signatures(signatures)
    counts = np.bincount(route[train], minlength=len(unique))
    table = np.zeros((len(unique), 8))
    teacher_sum = np.zeros_like(table)
    np.add.at(table, route[train], rows["regrets"][train])
    np.add.at(teacher_sum, route[train], teacher[train])
    assert np.allclose(table.sum(axis=0), rows["regrets"][train].sum(axis=0))
    seen = counts > 0
    # Export only train routes; test labels and regrets never enter the model.
    write_group(stream, group_id, trees, unique[seen], table[seen])
    matched = seen[route[test]]
    bucket_teacher = np.broadcast_to(
        teacher[train].mean(axis=0), (len(unique), 8)
    ).copy()
    np.divide(
        teacher_sum, counts[:, None], out=bucket_teacher, where=counts[:, None] > 0
    )
    partition_tv = 0.5 * np.abs(bucket_teacher[route[test]] - teacher[test]).sum(axis=1)
    raw_prediction = predictions.copy()
    mass = predictions.sum(axis=1, keepdims=True)
    np.divide(predictions, mass, out=predictions, where=mass > 0)
    predictions[mass[:, 0] == 0] = legal / legal.sum()
    prediction_tv = 0.5 * np.abs(predictions - teacher[test]).sum(axis=1)
    bucket_policy = regret_policy(table[route[test]], legal)
    reference = regret_policy(rows["regrets"][test], legal)
    nonzero = np.any(rows["regrets"][test] != 0, axis=1)
    regret_tv = 0.5 * np.abs(bucket_policy - reference).sum(axis=1)
    squared_error = (raw_prediction[:, legal] - teacher[test][:, legal]) ** 2
    return dict(
        group=int(group_id),
        train_rows=len(train),
        test_rows=len(test),
        fit_rows=len(fitting),
        trees=len(trees),
        buckets=int(seen.sum()),
        prediction_tv_sum=float(prediction_tv.sum()),
        partition_tv_sum=float(partition_tv.sum()),
        matched_partition_tv_sum=float(partition_tv[matched].sum()),
        matched_rows=int(matched.sum()),
        unseen_rows=int((~matched).sum()),
        action_mse_sum=float(squared_error.sum()),
        action_count=int(squared_error.size),
        delta_regret_rm_tv_sum=float(regret_tv[nonzero].sum()),
        nonzero_regret_test_rows=int(nonzero.sum()),
    )


def summarize(groups):
    def total(key):
        return sum(g[key] for g in groups)

    n = total("test_rows")
    return dict(
        test_rows=n,
        train_rows=total("train_rows"),
        fit_rows=total("fit_rows"),
        trees=total("trees"),
        buckets=total("buckets"),
        prediction_tv=total("prediction_tv_sum") / n,
        partition_tv=total("partition_tv_sum") / n,
        matched_partition_tv=total("matched_partition_tv_sum")
        / max(1, total("matched_rows")),
        unseen_route_fraction=total("unseen_rows") / n,
        action_mse=total("action_mse_sum") / total("action_count"),
        delta_regret_rm_tv=total("delta_regret_rm_tv_sum")
        / max(1, total("nonzero_regret_test_rows")),
        nonzero_regret_test_rows=total("nonzero_regret_test_rows"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument(
        "--hands", nargs="+", type=int, default=[10000, 500000, 1000000]
    )
    parser.add_argument("--sample-rows", type=int, default=200000)
    parser.add_argument("--fit-cap", type=int, default=20000)
    parser.add_argument("--tree-budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=921)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if (
        args.sample_rows < 1000
        or args.fit_cap < 20
        or args.tree_budget < 8
        or args.seed < 0
    ):
        parser.error("need sample-rows>=1000, fit-cap>=20, tree-budget>=8, seed>=0")
    if len(set(args.hands)) != len(args.hands) or min(args.hands) < 1:
        parser.error("hand checkpoints must be positive and unique")
    source = args.source_run.resolve()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    epoch_hands = int(source_config["epoch_hands"])
    if any(h % epoch_hands for h in args.hands):
        parser.error("checkpoints must be complete source epochs")
    directory = (
        args.out_dir
        or HERE / "data" / datetime.now().strftime("label_ablation_%Y%m%d_%H%M%S")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "sources").mkdir()
    for path in [
        Path(__file__),
        HERE / "run_epoch_ensemble.py",
        HERE / "test_epoch_ensemble.py",
    ]:
        shutil.copy2(path, directory / "sources" / path.name)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        scope="offline sampled epoch compression; feature-grouped holdout, NOT independent hands or RL training",
        modes=LABEL_MODES,
        refined_edges=REFINED_EDGES,
        lightgbm_version=lgb.__version__,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), HERE / "run_epoch_ensemble.py"]
        },
    )
    (directory / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for hand in args.hands:
        started = perf_counter()
        path = source / f"rows_{hand}.bin"
        before = path.stat()
        with path.open("rb") as stream:
            if stream.read(8) != b"ROWS7V01":
                raise ValueError("unknown source rows format")
            count = struct.unpack("<Q", stream.read(8))[0]
        if before.st_size != 16 + count * DTYPE.itemsize:
            raise ValueError("truncated source rows")
        data = np.memmap(path, dtype=DTYPE, mode="r", offset=16, shape=(count,))
        rng = np.random.default_rng(args.seed + hand)
        indices = np.sort(
            rng.choice(count, min(count, args.sample_rows), replace=False)
        )
        sample = data[indices]
        del data
        case = directory / f"hand_{hand}"
        case.mkdir()
        group_ids = np.unique(sample["group"])
        diagnostics = {mode: [] for mode in LABEL_MODES}
        durations = {mode: 0.0 for mode in LABEL_MODES}
        train_global, test_global, fit_global = [], [], []
        with ExitStack() as stack:
            streams = {}
            for mode in LABEL_MODES:
                (case / mode).mkdir()
                streams[mode] = stack.enter_context(
                    (case / mode / "model.bin").open("wb")
                )
                streams[mode].write(
                    b"EPOCH7V1"
                    + struct.pack("<II", hand // epoch_hands, len(group_ids))
                )
            for group_id in group_ids:
                positions = np.flatnonzero(sample["group"] == group_id)
                rows = sample[positions]
                train, test = feature_split(rows["features"], rng)
                fitting = np.sort(
                    rng.choice(train, min(len(train), args.fit_cap), replace=False)
                )
                train_global.extend(indices[positions[train]])
                test_global.extend(indices[positions[test]])
                fit_global.extend(indices[positions[fitting]])
                for mode in LABEL_MODES:
                    tick = perf_counter()
                    result = compare_group(
                        rows,
                        train,
                        test,
                        fitting,
                        group_id,
                        args.seed + hand,
                        mode,
                        args.tree_budget,
                        case / mode,
                        streams[mode],
                    )
                    diagnostics[mode].append(result)
                    durations[mode] += perf_counter() - tick
                print(
                    f"GROUP hand={hand} group={group_id} train={len(train)} test={len(test)} elapsed_seconds={perf_counter()-started:.1f}",
                    flush=True,
                )
        assert not np.intersect1d(train_global, test_global).size
        np.savez_compressed(
            case / "split_indices.npz",
            train=np.array(train_global),
            test=np.array(test_global),
            fit=np.array(fit_global),
        )
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError("source changed during evaluation")
        provenance = dict(
            path=str(path),
            rows=count,
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            sampled_rows=len(sample),
            sampled_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
        )
        (case / "input.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        for mode in LABEL_MODES:
            result = dict(
                hand=hand,
                mode=mode,
                **summarize(diagnostics[mode]),
                model_bytes=(case / mode / "model.bin").stat().st_size,
                fit_route_export_seconds=durations[mode],
            )
            (case / mode / "diagnostics.json").write_text(
                json.dumps(dict(summary=result, groups=diagnostics[mode]), indent=2)
                + "\n",
                encoding="utf-8",
            )
            results.append(result)
            print("RESULT " + json.dumps(result), flush=True)
        (directory / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (key, title, scale) in zip(
        axes.flat,
        [
            ("prediction_tv", "Held-out probability prediction TV (lower better)", 1),
            ("partition_tv", "Held-out bucket mean-policy TV (lower better)", 1),
            ("unseen_route_fraction", "Unseen held-out routes (%)", 100),
            ("model_bytes", "Sampled bucket export size (MiB)", 1 / 2**20),
        ],
    ):
        for index, (mode, color) in enumerate(
            zip(LABEL_MODES, ["#626c78", "#167e87", "#b63c58"])
        ):
            values = [
                next(r for r in results if r["hand"] == h and r["mode"] == mode)[key]
                * scale
                for h in args.hands
            ]
            ax.bar(
                np.arange(len(args.hands)) + (index - 1) * 0.25,
                values,
                width=0.24,
                label=mode,
                color=color,
            )
        ax.set_xticks(range(len(args.hands)), [f"{h:,}" for h in args.hands])
        ax.set(title=title, xlabel="Source training checkpoint (hands)")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Frozen 7-stud data refits; feature-grouped holdout, not new RL training"
    )
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)
    print(f"RESULTS {directory}", flush=True)


if __name__ == "__main__":
    main()
