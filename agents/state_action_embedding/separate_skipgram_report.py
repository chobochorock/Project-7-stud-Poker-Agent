"""Plot separate SGNS/distillation results with the existing Matplotlib Python.

Usage: python -m agents.state_action_embedding.separate_skipgram_report --out-dir RUN
Input: state_seed*/metrics.json, action_seed*/metrics.json, dataset.json.
Output: comparison.png and summary.json. No PyTorch import or model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads((args.out_dir / "dataset.json").read_text(encoding="utf-8"))
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.out_dir.glob("*_seed*/metrics.json"))
    ]
    if not runs:
        parser.error("No completed experiments")
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
    colors = {"state": "#087e8b", "action": "#c84b31"}
    summary = {"dataset": dataset, "runs": []}
    for run in runs:
        stream = run["stream"]
        label = f"{stream}, seed {run['seed']}"
        color = colors[stream]
        teacher = run["teacher_curve"]
        curve = run["student_curve"]
        steps = [row["step"] for row in curve]
        axes[0, 0].plot(
            [row["step"] for row in teacher],
            [row["loss"] for row in teacher],
            label=label,
            color=color,
        )
        axes[0, 1].plot(
            steps, [row["standardized_mse"] for row in curve], label=label, color=color
        )
        column = int(stream == "action")
        for partition, style in (("train", "--"), ("val", "-")):
            axes[1, column].plot(
                steps,
                [row[partition]["retrieval_top1"] for row in curve],
                style,
                color=color,
                label=partition,
            )
            axes[2, column].plot(
                steps,
                [row[partition]["retrieval_nll"] for row in curve],
                style,
                color=color,
                label=partition,
            )
        axes[1, column].axhline(
            1 / (run["negatives"] + 1),
            color="gray",
            linestyle=":",
            label="constant / random rank",
        )
        axes[1, column].set_ylim(0, 1)
        baseline = run["mean_baseline"]["retrieval_nll"]
        axes[2, column].axhline(
            baseline,
            color="gray",
            linestyle=":",
            label="constant / uniform probability",
        )
        largest_nll = max(
            row[partition]["retrieval_nll"]
            for row in curve
            for partition in ("train", "val")
        )
        axes[2, column].set_ylim(0, max(2.05, largest_nll * 1.05))
        axes[2, column].set_title(f"{stream}: context NLL (lower is better)")
        summary["runs"].append(
            {
                key: run[key]
                for key in (
                    "stream",
                    "seed",
                    "train_vocabulary",
                    "val_known_token_fraction",
                    "teacher",
                    "final",
                    "mean_baseline",
                    "seconds",
                )
            }
        )
    titles = [
        "Teacher sampled SGNS NLL / term",
        "Student standardized regression MSE",
        "State: context top-1 (6 candidates by default)",
        "Action: next betting-event top-1",
    ]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
    for axis in axes.flat:
        axis.set_xlabel("Optimizer steps within this stage")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Train-only skip-gram teachers -> local MLPs | {dataset['hands']:,} hands | showdown {dataset['showdown_rate']:.1%}\nValidation context prediction is not poker strength; unseen states have no teacher MSE",
        fontsize=13,
    )
    figure.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(figure)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.out_dir / "comparison.png")


if __name__ == "__main__":
    main()
