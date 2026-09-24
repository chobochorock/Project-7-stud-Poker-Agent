"""Plot saved action/path experiments without PyTorch.

Usage: python -m agents.state_action_embedding.action_path_report --out-dir RUN
Read-only inputs: completed per-seed metrics. Outputs: comparison.png, summary.json.
Different panels have different targets; retrieval is not policy performance.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(
        p
        for p in args.out_dir.glob("seed*")
        if (p / "sgns_endpoint_metrics.json").exists()
    )
    if not runs:
        raise ValueError("No completed seeds")
    results = {
        p.name: {
            name: read(p / f"{name}.json")
            for name in (
                "sgns_metrics",
                "endpoint_metrics",
                "sgns_endpoint_metrics",
                "baselines",
            )
        }
        for p in runs
    }
    audit_path = args.out_dir / "frequency_audit.json"
    audit = read(audit_path) if audit_path.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = ["#167a9f", "#ad3758", "#b77b16"]
    for condition, label, color in (
        ("val", "Global negatives", colors[0]),
        ("val_matched", "Kind/actor/street matched", colors[1]),
    ):
        curves = [
            [
                row["metrics"][condition]["retrieval_top1"]
                for row in r["sgns_metrics"]["curve"]
            ]
            for r in results.values()
        ]
        steps = [
            row["step"] for row in next(iter(results.values()))["sgns_metrics"]["curve"]
        ]
        mean, std = np.mean(curves, 0), np.std(curves, 0, ddof=int(len(runs) > 1))
        axes[0].plot(steps, mean, label=label, color=color)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
    if audit is not None:
        axes[0].axhline(
            audit["val_matched"]["top1"],
            color="#333333",
            linestyle="--",
            label="Matched frequency baseline",
        )
    for name, label, color in (
        ("endpoint", "Endpoint GRU", colors[0]),
        ("sgns_endpoint", "SGNS + endpoint GRU", colors[1]),
    ):
        curves = [
            [row["metrics"]["val"]["top1"] for row in r[name + "_metrics"]["curve"]]
            for r in results.values()
        ]
        steps = [
            row["step"]
            for row in next(iter(results.values()))[name + "_metrics"]["curve"]
        ]
        mean, std = np.mean(curves, 0), np.std(curves, 0, ddof=int(len(runs) > 1))
        axes[1].plot(steps, mean, label=label, color=color)
        axes[1].fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
    mean_sgns = np.mean(
        [r["baselines"]["sgns_mean"]["val"]["top1"] for r in results.values()]
    )
    axes[1].axhline(
        mean_sgns, color=colors[2], linestyle="--", label="Frozen SGNS mean"
    )
    axes[1].axhline(1, color="#448545", linestyle=":", label="Public payment sum")
    summary = {"seeds": [int(p.name[4:]) for p in runs], "test": {}, "sgns": {}}
    summary["frequency_audit"] = audit
    for name in ("val", "val_matched"):
        values = [
            r["sgns_metrics"]["curve"][-1]["metrics"][name]["retrieval_top1"]
            for r in results.values()
        ]
        summary["sgns"][name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=int(len(runs) > 1))),
            "values": values,
        }
    for k, (name, label) in enumerate(
        (
            ("sgns_mean", "SGNS mean"),
            ("endpoint", "Endpoint"),
            ("sgns_endpoint", "SGNS+end"),
            ("payment", "Payment"),
        )
    ):
        summary["test"][name] = {}
        for j, panel in enumerate(("test", "test_unseen_template")):
            values = [
                r["baselines"][name][panel]["top1"]
                if name in ("sgns_mean", "payment")
                else r[name + "_metrics"]["final"][panel]["top1"]
                for r in results.values()
            ]
            avg, sd = float(np.mean(values)), float(
                np.std(values, ddof=int(len(runs) > 1))
            )
            summary["test"][name][panel] = {"mean": avg, "sd": sd, "values": values}
            axes[2].bar(
                k + (j - 0.5) * 0.35,
                avg,
                0.35,
                yerr=sd,
                color=colors[j],
                label=("New roots", "New roots + held-out query pattern")[j]
                if k == 0
                else None,
            )
    axes[2].set_xticks(range(4), ["SGNS mean", "Endpoint", "SGNS+end", "Payment"])
    for ax, title in zip(
        axes,
        (
            "A. Atomic action skip-gram",
            "B. Different-length path validation",
            "C. Final exact-endpoint retrieval",
        ),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.06)
        ax.axhline(1 / 6, color="#777777", linestyle=":", linewidth=1)
        ax.set_ylabel("Top-1 among 6 candidates")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_xlabel("SGNS optimizer updates")
    axes[1].set_xlabel("Path optimizer updates (pretraining excluded)")
    fig.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(fig)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
