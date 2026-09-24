from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hands: list[int] = []
    heuristic: list[float] = []
    cluster: list[float] = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            hands.append(int(row["hand"]))
            heuristic.append(float(row["heuristic_gap_ante"]))
            cluster.append(float(row["cluster_gap_ante"]))
    return np.asarray(hands), np.asarray(heuristic), np.asarray(cluster)


def geometric_bins(
    hands: np.ndarray, values: np.ndarray, count: int = 80
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.geomspace(1, len(hands) + 1, count + 1).astype(int))
    centers: list[float] = []
    means: list[float] = []
    errors: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        window = values[left - 1 : right - 1]
        if len(window) < 16:
            continue
        centers.append(math.sqrt(int(left) * max(int(left), int(right) - 1)))
        means.append(float(np.mean(window)))
        errors.append(
            float(1.96 * np.std(window, ddof=1) / math.sqrt(len(window)))
            if len(window) > 1
            else 0.0
        )
    return np.asarray(centers), np.asarray(means), np.asarray(errors)


def cumulative_at_log_points(
    values: np.ndarray, count: int = 160
) -> tuple[np.ndarray, np.ndarray]:
    points = np.unique(np.geomspace(1, len(values), count).astype(int))
    cumulative = np.cumsum(values)
    return points, cumulative[points - 1] / points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hands, heuristic, cluster = load(args.input)
    if not len(hands):
        raise RuntimeError("empty BR-gap CSV")

    hx, hm, he = geometric_bins(hands, heuristic)
    cx, cm, ce = geometric_bins(hands, cluster)
    cumulative_x, cumulative_h = cumulative_at_log_points(heuristic)
    _, cumulative_c = cumulative_at_log_points(cluster)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    top, bottom = axes
    floor = 1e-6
    for x, mean, error, label, color in (
        (hx, hm, he, "Fixed heuristic", "#c44e52"),
        (cx, cm, ce, "k512 MCCFR cold start", "#4c72b0"),
    ):
        top.plot(x, np.maximum(mean, floor), color=color, label=label)
        lower = mean - error
        top.fill_between(
            x,
            np.maximum(lower, floor),
            np.maximum(mean + error, floor),
            where=lower > 0,
            color=color,
            alpha=0.18,
        )
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylabel("Log-binned sampled BR-gap proxy (ante)")
    top.set_title("7-Stud: trajectory-relaxed counterfactual deviation gap")
    top.grid(True, which="both", alpha=0.22)
    top.legend()

    bottom.plot(cumulative_x, cumulative_h, color="#c44e52", label="Fixed heuristic")
    bottom.plot(
        cumulative_x, cumulative_c, color="#4c72b0", label="k512 MCCFR cold start"
    )
    bottom.set_xscale("log")
    bottom.set_xlabel("Training hands")
    bottom.set_ylabel("Cumulative mean gap (ante)")
    bottom.grid(True, which="both", alpha=0.22)
    bottom.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
