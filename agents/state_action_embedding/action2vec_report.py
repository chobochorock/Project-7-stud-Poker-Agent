"""Summarize action2vec runs; requires NumPy/Matplotlib, not PyTorch.

Usage: python -m agents.state_action_embedding.action2vec_report --out-dir RUN
Writes summary.json, comparison.png and calibration.png. Plot error bars show
training-seed SD. Paired bootstrap resamples HANDS, retaining both player views;
its CI is conditional on this dataset and these trained models, not seed variance.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("none", "players", "all")
COHORTS = ("overall", "random", "heuristic", "mixed")
COLORS = ("#777777", "#19806e", "#ae4566")


def stats(values):
    return {"mean": float(np.mean(values)), "seed_sd": float(np.std(values))}


def paired_rmse(left, right, truth, hands):
    """Average squared errors across seeds, NOT predictions (no ensemble benefit)."""
    unique, inverse = np.unique(hands, return_inverse=True)
    counts = np.bincount(inverse)
    errors = []
    for prediction in (left, right):
        mse = np.square(np.asarray(prediction) - truth).mean(axis=0)
        errors.append(np.bincount(inverse, weights=mse) / counts)
    rng = np.random.default_rng(20260919)
    differences = []
    for _ in range(2000):
        sample = rng.integers(len(unique), size=len(unique))
        differences.append(
            np.sqrt(errors[0][sample].mean()) - np.sqrt(errors[1][sample].mean())
        )
    return {
        "rmse_difference_chips": float(
            np.sqrt(errors[0].mean()) - np.sqrt(errors[1].mean())
        ),
        "hand_bootstrap_95pct": np.quantile(differences, [0.025, 0.975]).tolist(),
        "hands": len(unique),
    }


def audit_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    pairs = data["hand"].reshape(-1, 2)
    np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])
    splits = data["split"].reshape(-1, 2)
    np.testing.assert_array_equal(splits[:, 0], splits[:, 1])
    np.testing.assert_array_equal(data["reward"].reshape(-1, 2).sum(1), 0)
    events = data["tokens"]
    private = np.isin(events[..., 0], [3, 5]) & (events[..., 1] == 1)
    assert (events[..., 2][private] == 0).all(), "Opponent private card leak"
    anchors = data["anchor"].astype(int)
    assert (events[np.arange(len(events)), anchors, 0] == 8).all()
    opponent = (events[..., 0] == 9) & (events[..., 1] == 1)
    action = events[..., 3] - 1
    train_targets = action[opponent & (data["split"] == 0)[:, None]]
    frequencies = np.bincount(train_targets, minlength=8) + 1
    baseline = {}
    for cohort, name in enumerate(COHORTS):
        rows = data["split"] == 2
        if cohort:
            rows &= data["cohort"] == cohort - 1
        mask = opponent & rows[:, None]
        probability = data["legal"][mask] * frequencies
        probability = probability / probability.sum(-1, keepdims=True)
        target = action[mask]
        baseline[name] = {
            "nll": float(-np.log(probability[np.arange(len(target)), target]).mean()),
            "accuracy": float((probability.argmax(-1) == target).mean()),
        }
    audit = {}
    for split, name in enumerate(("train", "validation", "test")):
        reward = data["reward"][data["split"] == split].astype(float)
        square = np.sort(reward**2)
        audit[name] = {
            "hands": len(reward) // 2,
            "absolute_reward_quantiles_50_90_95_99_100": np.quantile(
                np.abs(reward), [0.5, 0.9, 0.95, 0.99, 1]
            ).tolist(),
            "top_1pct_share_zero_prediction_squared_error": float(
                square[-max(1, len(square) // 100) :].sum() / square.sum()
            ),
            "constant_rmse_chips": float(np.sqrt(square.mean())),
        }
    return audit, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    groups, predictions = {}, {}
    for method in METHODS:
        paths = sorted(args.out_dir.glob(f"{method}_seed*/metrics.json"))
        if not paths:
            parser.error(f"No completed runs for {method}")
        groups[method] = [json.loads(p.read_text()) for p in paths]
        predictions[method] = []
        for path in paths:
            with np.load(path.parent / "predictions.npz", allow_pickle=False) as data:
                predictions[method].append(dict(data))
    reference = predictions["none"][0]
    seeds = [v["seed"] for v in groups["none"]]
    for method in METHODS:
        assert [
            v["seed"] for v in groups[method]
        ] == seeds, "Need matched training seeds"
        for data in predictions[method]:
            for key in ("truth", "hand", "cohort", "anchor", "raw", "constant"):
                np.testing.assert_array_equal(data[key], reference[key])
    summary = {"seeds": seeds, "reward": {}, "action": {}, "resources": {}}
    summary["dataset_audit"], summary["action_frequency_baseline"] = audit_dataset(
        args.out_dir / "events.npz"
    )
    for method, runs in groups.items():
        summary["reward"][method] = {
            feature: {
                cohort: {
                    metric: stats(
                        [r["reward_test"][feature][cohort][metric] for r in runs]
                    )
                    for metric in ("rmse_chips", "mae_chips", "r2")
                }
                for cohort in COHORTS
            }
            for feature in ("value", "frozen", "raw", "constant")
        }
        summary["action"][method] = {
            cohort: {
                metric: stats([r["action_test"][cohort][metric] for r in runs])
                for metric in ("nll", "accuracy", "uniform_legal_nll")
            }
            for cohort in COHORTS
        }
        summary["resources"][method] = {
            phase: {
                metric: stats([r[f"{phase}_memory"][metric] for r in runs])
                for metric in ("rss_sampled_peak_mib", "seconds")
            }
            for phase in ("pretrain", "value")
        }
    left = [d["value"] for d in predictions["players"]]
    summary["paired_comparisons"] = {}
    for label, right in (
        ("players_minus_none", [d["value"] for d in predictions["none"]]),
        ("players_minus_all", [d["value"] for d in predictions["all"]]),
        ("players_minus_constant", [reference["constant"]]),
        ("players_minus_raw", [reference["raw"]]),
    ):
        summary["paired_comparisons"][label] = paired_rmse(
            left, right, reference["truth"], reference["hand"]
        )
    check = paired_rmse(
        [reference["truth"]],
        [reference["truth"]],
        reference["truth"],
        reference["hand"],
    )
    assert check["hand_bootstrap_95pct"] == [0, 0]
    summary[
        "bootstrap_note"
    ] = "Seed-mean squared errors; paired hand resampling; conditional on trained models, not a CI over training randomness. Negative difference favors players."
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    for offset, feature, hatch in ((-0.18, "frozen", "//"), (0.18, "value", "")):
        metrics = [
            summary["reward"][m][feature]["overall"]["rmse_chips"] for m in METHODS
        ]
        ax.bar(
            np.arange(3) + offset,
            [v["mean"] for v in metrics],
            width=0.35,
            yerr=[v["seed_sd"] for v in metrics],
            capsize=3,
            color=COLORS,
            hatch=hatch,
            label=feature,
        )
    for feature, style in (("constant", "--"), ("raw", ":")):
        value = summary["reward"]["none"][feature]["overall"]["rmse_chips"]["mean"]
        ax.axhline(
            value, color="black", linestyle=style, label=f"{feature}: {value:.2f}"
        )
    ax.set_xticks(range(3), METHODS)
    ax.set_title("Pre-action terminal return RMSE (chips; lower better)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        metric = [summary["action"][method][c]["nll"] for c in COHORTS[1:]]
        ax.bar(
            np.arange(3) + (i - 1) * 0.24,
            [v["mean"] for v in metric],
            width=0.24,
            yerr=[v["seed_sd"] for v in metric],
            capsize=3,
            color=color,
            label=method,
        )
    ax.set_xticks(range(3), COHORTS[1:])
    ax.plot(
        range(3),
        [summary["action_frequency_baseline"][c]["nll"] for c in COHORTS[1:]],
        "kx",
        label="train-frequency baseline",
    )
    ax.set_title("Opponent-action NLL after pretraining (lower better)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        metric = [summary["reward"][method]["value"][c]["r2"] for c in COHORTS[1:]]
        ax.bar(
            np.arange(3) + (i - 1) * 0.24,
            [v["mean"] for v in metric],
            width=0.24,
            yerr=[v["seed_sd"] for v in metric],
            capsize=3,
            color=color,
            label=method,
        )
    ax.set_xticks(range(3), COHORTS[1:])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Fine-tuned reward R2 (zero = mean predictor)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for method, color in zip(METHODS, COLORS):
        curves = [
            [p for p in r["curve"] if p["stage"] == "value"] for r in groups[method]
        ]
        values = np.array([[p["rmse_chips"] for p in c] for c in curves])
        steps = [p["step"] for p in curves[0]]
        ax.plot(steps, values.mean(0), color=color, label=method)
        ax.fill_between(
            steps,
            values.mean(0) - values.std(0),
            values.mean(0) + values.std(0),
            color=color,
            alpha=0.15,
        )
    ax.set_title("Validation reward RMSE (model selection only)")
    ax.set_xlabel("Reward fine-tuning updates")
    ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Chance-aware action2vec | held-out 7-poker v3 hands\n{len(seeds)} matched seeds; error bars = seed SD, not confidence intervals",
        fontsize=13,
    )
    fig.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for method, color, ax in zip(METHODS, COLORS, axes):
        data = predictions[method][0]
        order = np.argsort(data["value"])
        bins = np.array_split(order, 1 if data["value"].std() < 1e-6 else 10)
        x = [data["value"][b].mean() for b in bins]
        y = [data["truth"][b].mean() for b in bins]
        ax.scatter(data["value"], data["truth"], s=4, alpha=0.08, color=color)
        ax.plot(
            x,
            y,
            "o-",
            color="black",
            markersize=4,
            label=f"{len(bins)} equal-count bin(s)",
        )
        bounds = [min(min(x), min(y), -1), max(max(x), max(y), 1)]
        ax.plot(bounds, bounds, "--", color="gray")
        ax.set_title(method)
        ax.set_xlabel("Predicted final net chips")
        ax.set_ylabel("Actual final net chips")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Reward forecasts before an unresolved action | seed {seeds[0]} (preselected)\nDots = individual noisy returns; black line = binned observed means; no ensemble"
    )
    fig.savefig(args.out_dir / "calibration.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary["paired_comparisons"], indent=2))


if __name__ == "__main__":
    main()
