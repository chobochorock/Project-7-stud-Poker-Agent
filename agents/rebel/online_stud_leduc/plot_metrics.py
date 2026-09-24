from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    iterations = [row["root_cfr_iterations"] for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(
        iterations,
        [row["exploitability"] for row in rows],
        marker="o",
        label="online PBS + V2",
    )
    axes[0, 0].plot(
        iterations,
        [row["baseline_exploitability"] for row in rows],
        marker="s",
        label="flat CFR+",
    )
    axes[0, 0].set_title("Exact exploitability")
    axes[0, 0].set_ylabel("exploitability (lower is better)")
    axes[0, 0].legend()

    axes[0, 1].plot(
        iterations,
        [row["value_mae_before_update"] for row in rows],
        label="before online update",
    )
    axes[0, 1].plot(
        iterations,
        [row["value_mae_after_update"] for row in rows],
        label="after online update",
    )
    axes[0, 1].set_title("Private-type CFV error")
    axes[0, 1].set_ylabel("MAE (chips)")
    axes[0, 1].legend()

    axes[1, 0].plot(iterations, [row["root_regret_proxy"] for row in rows])
    axes[1, 0].set_title("Root search regret proxy")
    axes[1, 0].set_ylabel("sum max positive regret / T")

    axes[1, 1].plot(
        iterations, [row["replay_size"] for row in rows], label="replay size"
    )
    axes[1, 1].plot(
        iterations,
        [row["public_belief_states"] for row in rows],
        label="PBS per outer step",
    )
    axes[1, 1].set_title("Online data coverage")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("root CFR iterations")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
