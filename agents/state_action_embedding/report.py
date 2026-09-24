"""Plot completed runs without PyTorch; Python with NumPy/Matplotlib is enough.

Usage: python -m agents.state_action_embedding.report --out-dir RUN
Writes comparison.png, memory.png, and summary.json inside RUN.
Error bars are training-seed standard deviations, NOT confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        json.loads(path.read_text())
        for path in sorted(args.out_dir.glob("*_seed*/metrics.json"))
    ]
    if not runs:
        parser.error("No completed metrics.json files")
    groups = {}
    for run in runs:
        for feature, metrics in run["probes"].items():
            if (
                feature in ("raw_current", "random_encoder", "constant")
                and run["method"] != "skipgram"
            ):
                continue
            label = (
                feature
                if feature in ("raw_current", "random_encoder", "constant")
                else f"{run['method']}/{feature}"
            )
            groups.setdefault(label, []).append(metrics)
    summary = {}
    for label, values in groups.items():
        summary[label] = {
            metric: {
                "mean": float(np.mean([v[metric] for v in values])),
                "seed_std": float(np.std([v[metric] for v in values])),
            }
            for metric in (
                "return_rmse_chips",
                "next_action_nll",
                "next_action_accuracy",
            )
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    labels = [
        name
        for name in (
            "constant",
            "raw_current",
            "random_encoder",
            "skipgram/encoder",
            "cpc/encoder",
            "skipgram/mean_history",
            "cpc/mean_history",
            "cpc/context",
        )
        if name in summary
    ]
    colors = [
        "#7a7a7a",
        "#aaaaaa",
        "#777799",
        "#207f9d",
        "#b3415a",
        "#4fa4b1",
        "#d17b90",
        "#397b56",
    ][: len(labels)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("return_rmse_chips", "next_action_nll", "next_action_accuracy"),
        (
            "Final return RMSE (chips, lower better)",
            "Next-action NLL (lower better)",
            "Next-action accuracy (higher better)",
        ),
    ):
        ax.barh(
            labels,
            [summary[n][metric]["mean"] for n in labels],
            xerr=[summary[n][metric]["seed_std"] for n in labels],
            color=colors,
            capsize=3,
        )
        ax.invert_yaxis()
        maximum = max(summary[n][metric]["mean"] for n in labels)
        for index, name in enumerate(labels):
            value = summary[name][metric]["mean"]
            text = f"{value:.2f}" if metric == "return_rmse_chips" else f"{value:.3f}"
            ax.text(value + maximum * 0.025, index, text, va="center", fontsize=9)
        ax.set_xlim(0, maximum * 1.2)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "7-poker v3 | frozen linear probes | held-out hands\n"
        "Encoder rows: current-only; mean_history/context: prefix\n"
        "Mean and training-seed SD (3 seeds); not confidence intervals",
        fontsize=12,
    )
    fig.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(fig)
    memory = [
        json.loads(path.read_text()) for path in args.out_dir.glob("memory_*.json")
    ]
    if memory:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for method, color in (("skipgram", "#207f9d"), ("cpc", "#b3415a")):
            selected = sorted(
                (v for v in memory if v["method"] == method and v["phase"] == "train"),
                key=lambda v: v["context"],
            )
            for ax, key, title in zip(
                axes,
                ("rss_sampled_peak_mib", "rss_peak_delta_mib"),
                (
                    "Sampled peak process RAM (MiB)",
                    "Increase above import baseline (MiB)",
                ),
            ):
                ax.plot(
                    [v["context"] for v in selected],
                    [v["memory"][key] for v in selected],
                    "o-",
                    label=method,
                    color=color,
                )
                ax.set_xlabel("Context length (synthetic dense inputs)")
                ax.set_title(title, fontsize=11)
                ax.grid(alpha=0.2)
                ax.legend()
        fig.suptitle("Fresh-process FP32 training memory | B=64 | 10 ms RSS sampling")
        fig.savefig(args.out_dir / "memory.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
