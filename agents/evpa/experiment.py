"""EVPA-inspired held-out mini-stud endgame pilot (not the full ICLR reproduction).

Usage (7-stud project root; Python with numpy, torch, sklearn, matplotlib):
    python -m agents.evpa.experiment --output agents/evpa/data/<NEW run>
    python -m unittest agents.evpa.test_experiment -v
Input: exact mini-stud check-prefix roots; disjoint public-root train/validation/test.
Output: model-only EV checkpoint, labels, policies, metrics, per-iteration root gaps,
comparison.png and run.json. Outputs are not resumable solver checkpoints.
Metrics: original-action endgame exploitability, exact root Local BR-gap, L/C/H.
Limit: small MLPs, finite CFR teacher, conservative bucket pruning, fixed root prior.
The traversal still visits masked branches, so this pilot does NOT test speedups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from .endgame import CFR, Endgame, collect_roots


def model(input_size, hidden):
    return nn.Sequential(
        nn.Linear(input_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def train_ensemble(x, y, groups, train, valid, output, *, members, steps, seed, hidden):
    train_mask, valid_mask = np.isin(groups, train), np.isin(groups, valid)
    mean, scale = x[train_mask].mean(axis=0), x[train_mask].std(axis=0)
    scale = np.maximum(scale, 1e-5)
    features = torch.from_numpy(((x - mean) / scale).astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32)[:, None])
    train_x, train_y = features[train_mask], targets[train_mask]
    valid_x, valid_y = features[valid_mask], targets[valid_mask]
    networks, logs = [], []
    for member in range(members):
        torch.manual_seed(seed + member)
        network = model(x.shape[1], hidden)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        best, best_state = float("inf"), None
        for step in range(1, steps + 1):
            selection = torch.randint(len(train_x), (min(512, len(train_x)),))
            loss = nn.functional.huber_loss(
                network(train_x[selection]), train_y[selection], delta=0.1
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == steps:
                with torch.no_grad():
                    score = float(((network(valid_x) - valid_y) ** 2).mean())
                if score < best:
                    best = score
                    best_state = {
                        key: value.detach().clone()
                        for key, value in network.state_dict().items()
                    }
        network.load_state_dict(best_state)
        network.eval()
        networks.append(network)
        logs.append({"member": member, "seed": seed + member, "validation_mse": best})
        print(
            f"EV_MODEL member={member+1}/{members} validation_rmse={best**0.5:.6f}",
            flush=True,
        )
    torch.save(
        {
            "state_dicts": [n.state_dict() for n in networks],
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "input_size": x.shape[1],
            "hidden": hidden,
        },
        output / "ev_models.pt",
    )
    return networks, mean, scale, logs


def load_models(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    networks = []
    for state in saved["state_dicts"]:
        network = model(saved["input_size"], saved["hidden"])
        network.load_state_dict(state)
        network.eval()
        networks.append(network)
    return networks, saved["mean"].numpy(), saved["scale"].numpy()


def predict(networks, mean, scale, x):
    tensor = torch.from_numpy(((x - mean) / scale).astype(np.float32))
    with torch.no_grad():
        return np.stack([network(tensor).numpy().ravel() for network in networks])


def cluster(features, k, seed):
    k = min(k, len(features), len(np.unique(features, axis=0)))
    labels = KMeans(
        n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=seed
    ).fit_predict(features)
    return np.unique(labels, return_inverse=True)[1]


def ev_features(game, values):
    # Paper Algorithm 3's importance: EV / (pot * max(depth, 0.2)).
    nodes = game.decision
    weights = np.asarray(
        [
            1
            / (
                sum(game.nodes[i].state.invested)
                / game.nodes[i].state.config.ante
                * max(game.nodes[i].depth, 0.2)
            )
            for i in nodes
        ]
    )
    return [v[:, nodes] * weights[None, :] for v in values]


def retained_actions(game, predictions, delta):
    """EVPA Algorithm 2 local keep test; ensemble spread is NOT a confidence interval.

    We do not prune roots or remove information sets. A downstream static bucket
    uses the union of its members' retained actions, preserving shared legal sets.
    """
    result = {}
    for i in game.decision:
        node = game.nodes[i]
        p = node.state.actor
        low, high = predictions[p].min(axis=0), predictions[p].max(axis=0)
        keep = np.zeros((game.sizes[p], len(node.actions)), bool)
        for action, child in enumerate(node.children):
            siblings = [c for c in node.children if c != child]
            threshold = np.minimum(low[:, siblings].max(axis=1), low[:, i])
            keep[:, action] = high[:, child] + delta * game.root_pot > threshold
        # Defensive nonempty action set under finite / correlated predictions.
        empty = ~keep.any(axis=1)
        scores = predictions[p].mean(axis=0)[:, node.children]
        keep[np.flatnonzero(empty), scores[empty].argmax(axis=1)] = True
        result[i] = keep
    return result


def pruning_audit(game, policy, reference):
    """Count excluded reference-best original infoset actions, not a certified error rate."""
    values, reaches = game.values(reference), game.reaches(reference)
    wrong, total = 0, 0
    for i in game.decision:
        p = game.nodes[i].state.actor
        q = np.stack(
            [
                game.conditional(values[c], reaches, p, i)
                for c in game.nodes[i].children
            ],
            axis=1,
        )
        support = policy.masks[i][policy.mappings[p]]
        lost = q.max(axis=1) - np.where(support, q, -np.inf).max(axis=1)
        wrong += int((lost > 0.01 * game.root_pot).sum())
        total += len(lost)
    return wrong / total


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, gaps, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "exact": "#333333",
        "card_kmeans": "#1768bf",
        "ev_oracle": "#a77500",
        "ev_network": "#cf303b",
        "ev_network_pruned": "#16816c",
        "ev_prune_only": "#875ca8",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for arm, color in colors.items():
        selected = [r for r in rows if r["arm"] == arm]
        points = sorted({r["iteration"] for r in selected})
        for ax, field in zip(
            axes.ravel()[:3], ["average_exploitability", "average_root_local_gap", "H"]
        ):
            values = [
                np.mean([r[field] for r in selected if r["iteration"] == t])
                for t in points
            ]
            ax.plot(points, np.maximum(values, 1e-10), label=arm, color=color)
            ax.set(
                xscale="log", yscale="log", xlabel="Full CFR iterations", ylabel=field
            )
            ax.grid(alpha=0.2)
        series = np.mean([g for a, g in gaps if a == arm], axis=0)
        window = min(100, len(series))
        smooth = np.convolve(series, np.ones(window) / window, mode="valid")
        axes[1, 1].plot(
            np.arange(window, len(series) + 1), smooth, color=color, label=arm
        )
    axes[1, 1].set(
        xlabel="Full CFR iterations", ylabel="Current root Local BR-gap (MA100)"
    )
    axes[1, 1].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "EVPA-inspired mini-stud endgames | held-out public roots\n"
        "Equal online iterations; offline teacher/training excluded from this axis; oracle is diagnostic"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-iterations", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cluster-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--test-roots", type=int, default=12)
    parser.add_argument("--train-roots", type=int, default=36)
    parser.add_argument("--validation-roots", type=int, default=12)
    parser.add_argument("--delta", type=float, default=0.01)
    args = parser.parse_args()
    for name in (
        "teacher_iterations",
        "iterations",
        "steps",
        "members",
        "hidden",
        "k",
        "test_roots",
        "train_roots",
        "validation_roots",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if (
        args.delta < 0
        or not args.cluster_seeds
        or min(args.cluster_seeds) < 0
        or args.seed < 0
    ):
        parser.error("delta and seeds must be nonnegative")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = vars(args).copy()
    manifest["output"] = str(args.output.resolve())
    manifest.update(
        status="running",
        scope="mini-stud small final street; forced CHECK prefix; full chance catalogue",
        algorithm="simultaneous signed full CFR, uniform-time own-reach average",
        deviations="2x64 MLP by default; no root pruning; bucket-union masks; full masked traversal",
        metrics="original-action exact endgame NashConv/2 and root-actor conditional Local BR-gap in antes",
        memory="raw per-type regrets and teacher labels are off-policy diagnostic/offline storage, counted separately",
    )
    manifest["sources"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).with_name("endgame.py"),
            Path("environments/mini_stud/game.py"),
        ]
    }
    run_path = args.output / "run.json"
    run_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    start = time.perf_counter()
    groups = collect_roots()
    order = np.random.default_rng(args.seed).permutation(len(groups))
    n_train, n_valid = args.train_roots, args.validation_roots
    if n_train + n_valid + args.test_roots > len(groups):
        raise ValueError("requested more public roots than available")
    train = order[:n_train].tolist()
    valid = order[n_train : n_train + n_valid].tolist()
    test = order[n_train + n_valid : n_train + n_valid + args.test_roots].tolist()
    chosen = train + valid + test
    games, references, labels, x_rows, y_rows, group_rows, teacher_rows = (
        {},
        {},
        {},
        [],
        [],
        [],
        [],
    )
    for count, index in enumerate(chosen):
        game = Endgame(groups[index])
        games[index] = game
        teacher = CFR(game)
        teacher.run(args.teacher_iterations)
        reference = teacher.policy(average=True)
        references[index] = reference
        ev = game.evs(reference)
        labels[index] = ev
        pot = np.asarray(
            [sum(n.state.invested) / n.state.config.ante for n in game.nodes]
        )
        for p in (0, 1):
            x = game.network_inputs(p)
            x_rows.append(x)
            y_rows.append((ev[p] / pot[None, :]).ravel())
            group_rows.extend([index] * len(x))
        teacher_rows.append(
            {
                "root": index,
                "split": "train"
                if index in train
                else "valid"
                if index in valid
                else "test",
                "types0": game.sizes[0],
                "types1": game.sizes[1],
                "public_nodes": len(game.nodes),
                "joint_support": int((game.joint > 0).sum()),
                **game.metrics(reference),
            }
        )
        print(
            f"TEACHER root={count+1}/{len(chosen)} types={game.sizes} exploitability={teacher_rows[-1]['exploitability']:.6f}",
            flush=True,
        )
    manifest["teacher_seconds"] = time.perf_counter() - start
    manifest["split"] = {"train": train, "valid": valid, "test": test}
    write_csv(args.output / "teacher.csv", teacher_rows)
    x, y, ids = np.concatenate(x_rows), np.concatenate(y_rows), np.asarray(group_rows)
    np.savez_compressed(
        args.output / "labels.npz",
        x=x,
        y=y,
        root=ids,
        train=np.asarray(train),
        valid=np.asarray(valid),
        test=np.asarray(test),
    )
    started = time.perf_counter()
    networks, mean, scale, train_log = train_ensemble(
        x,
        y,
        ids,
        train,
        valid,
        args.output,
        members=args.members,
        steps=args.steps,
        seed=args.seed,
        hidden=args.hidden,
    )
    loaded, saved_mean, saved_scale = load_models(args.output / "ev_models.pt")
    np.testing.assert_allclose(
        predict(networks, mean, scale, x[:10]),
        predict(loaded, saved_mean, saved_scale, x[:10]),
    )
    manifest["network_training_seconds"] = time.perf_counter() - started
    manifest["network_training"] = train_log
    prediction = predict(networks, mean, scale, x)
    manifest["ev_errors"] = {
        name: {
            "rmse_pot_units": float(
                np.sqrt(np.mean((prediction.mean(axis=0)[mask] - y[mask]) ** 2))
            ),
            "mae_pot_units": float(
                np.mean(abs(prediction.mean(axis=0)[mask] - y[mask]))
            ),
            "ensemble_range_coverage": float(
                np.mean(
                    (y[mask] >= prediction.min(axis=0)[mask])
                    & (y[mask] <= prediction.max(axis=0)[mask])
                )
            ),
        }
        for name, mask in [
            ("train", np.isin(ids, train)),
            ("validation", np.isin(ids, valid)),
            ("test", np.isin(ids, test)),
        ]
    }
    run_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, gap_series, audits = [], [], []
    reports = {
        1,
        10,
        25,
        50,
        100,
        200,
        500,
        args.iterations,
        *range(1000, args.iterations + 1, 1000),
    }
    with (args.output / "online_gap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        gap_writer = csv.writer(stream)
        gap_writer.writerow(
            ["root", "arm", "seed", "iteration", "current_preupdate_root_local_gap"]
        )
        for root in test:
            game = games[root]
            pot = np.asarray(
                [sum(n.state.invested) / n.state.config.ante for n in game.nodes]
            )
            predicted = [
                predict(networks, mean, scale, game.network_inputs(p)).reshape(
                    args.members, game.sizes[p], -1
                )
                * pot[None, None, :]
                for p in (0, 1)
            ]
            ev_means = [v.mean(axis=0) for v in predicted]
            masks = retained_actions(game, predicted, args.delta)
            for seed in args.cluster_seeds:
                maps = {
                    "card_kmeans": [
                        cluster(game.card_features(p), args.k, seed) for p in (0, 1)
                    ],
                    "ev_oracle": [
                        cluster(f, args.k, seed)
                        for f in ev_features(game, labels[root])
                    ],
                    "ev_network": [
                        cluster(f, args.k, seed) for f in ev_features(game, ev_means)
                    ],
                }
                cases = [(name, mapping, None) for name, mapping in maps.items()]
                cases.append(("ev_network_pruned", maps["ev_network"], masks))
                if seed == args.cluster_seeds[0]:
                    cases += [("exact", None, None), ("ev_prune_only", None, masks)]
                for arm, mapping, retained in cases:
                    solver = CFR(game, mapping, retained)
                    lost = pruning_audit(game, solver, references[root])
                    audits.append(
                        {
                            "root": root,
                            "arm": arm,
                            "seed": seed,
                            "reference_best_excluded_fraction": lost,
                        }
                    )
                    seconds = 0.0
                    gaps = []
                    for iteration in range(1, args.iterations + 1):
                        tick = time.perf_counter()
                        gap = solver.step()
                        seconds += time.perf_counter() - tick
                        gaps.append(gap)
                        gap_writer.writerow([root, arm, seed, iteration, gap])
                        if iteration in reports:
                            current = game.metrics(solver.policy())
                            average = game.metrics(solver.policy(True))
                            rows.append(
                                {
                                    "root": root,
                                    "arm": arm,
                                    "seed": seed,
                                    "iteration": iteration,
                                    "training_seconds": seconds,
                                    "training_history_visits": iteration
                                    * len(game.nodes)
                                    * int((game.joint > 0).sum()),
                                    **{f"current_{k}": v for k, v in current.items()},
                                    **{f"average_{k}": v for k, v in average.items()},
                                    **solver.diagnostics(),
                                }
                            )
                    gap_series.append((arm, np.asarray(gaps)))
                    exported = {}
                    for i in game.decision:
                        exported[f"regret_{i}"] = solver.regrets[i]
                        exported[f"strategy_sum_{i}"] = solver.sums[i]
                        exported[f"mask_{i}"] = solver.masks[i]
                    exported.update(
                        mapping0=solver.mappings[0], mapping1=solver.mappings[1]
                    )
                    np.savez_compressed(
                        args.output / f"policy_root{root}_{arm}_seed{seed}.npz",
                        **exported,
                    )
                    print(
                        f"SOLVE root={root} arm={arm} seed={seed} average_exploitability={rows[-1]['average_exploitability']:.6f} seconds={seconds:.3f}",
                        flush=True,
                    )
                write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "pruning_audit.csv", audits)
    plot(rows, gap_series, args.output / "comparison.png")
    final = [r for r in rows if r["iteration"] == args.iterations]
    fields = [
        "average_exploitability",
        "current_exploitability",
        "average_root_local_gap",
        "bucket_rows",
        "retained_fraction",
        "policy_table_bytes",
        "diagnostic_bytes",
        "training_seconds",
    ]
    manifest["summary"] = {
        arm: {
            field: float(np.mean([r[field] for r in final if r["arm"] == arm]))
            for field in fields
        }
        for arm in sorted({r["arm"] for r in final})
    }
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.perf_counter() - start
    manifest[
        "root_mean_definition"
    ] = "unweighted mean over held-out public roots and clustering seeds; not a confidence interval"
    run_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
