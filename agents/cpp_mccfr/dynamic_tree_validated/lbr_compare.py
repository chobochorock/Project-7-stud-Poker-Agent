"""Compare two frozen DT LBR evaluations with identical k-means controls.

Usage (project root): python -m agents.cpp_mccfr.dynamic_tree_validated.lbr_compare
    --dynamic LBR_OUTPUT --frozen LBR_OUTPUT --output NEW_REPORT_DIRECTORY
Writes summary.json and comparison.png under this family's data directory.
Metric: approximate LBR attacker ante/hand, not exact exploitability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run import DATA
from ..abstraction_lbr.run import NAMES, estimate


def read_samples(directory: Path) -> tuple[dict, dict, dict]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary["input_unchanged"]:
        raise ValueError("Evaluation input changed")
    samples = {name: [] for name in NAMES}
    streams = {}
    for seed in summary["seeds"]:
        with (directory / str(seed) / "paired_payoffs.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != summary["pairs_per_seed"]:
            raise ValueError("Wrong pair count")
        streams[seed] = [(row["pair"], row["deal_seed"]) for row in rows]
        for name in NAMES:
            samples[name].append([
                [float(row[name + "_seat0"]), float(row[name + "_seat1"])]
                for row in rows
            ])
    samples = {name: np.asarray(values) for name, values in samples.items()}
    for name, values in samples.items():
        if not np.isfinite(values).all() or (np.abs(values) > 1000).any():
            raise ValueError("Invalid payoff")
        np.testing.assert_allclose(values.mean(), summary["policies"][name]["mean"])
    return summary, samples, streams


def compare(dynamic: Path, frozen: Path, output: Path) -> dict:
    left, a, stream_a = read_samples(dynamic)
    right, b, stream_b = read_samples(frozen)
    for field in ("training_hands", "seeds", "pairs_per_seed", "particles", "exe_sha256"):
        if left[field] != right[field]:
            raise ValueError("Mismatched evaluation: " + field)
    if stream_a != stream_b:
        raise ValueError("Mismatched deal/seat pairs")
    for name in ("baseline_current", "baseline_average", "uniform"):
        np.testing.assert_array_equal(a[name], b[name])
    policies = {}
    differences = {}
    for policy in ("current", "average"):
        values = {
            "dynamic": a["model_" + policy].mean(axis=2),
            "frozen": b["model_" + policy].mean(axis=2),
            "kmeans": a["baseline_" + policy].mean(axis=2),
        }
        policies[policy] = {name: estimate(x) for name, x in values.items()}
        differences[policy] = {
            first + "_minus_" + second: estimate(values[first] - values[second])
            for first, second in (("dynamic", "kmeans"), ("dynamic", "frozen"),
                                  ("frozen", "kmeans"))
        }
    result = {
        "sources": {"dynamic": str(dynamic), "frozen": str(frozen)},
        **{key: left[key] for key in ("training_hands", "seeds", "pairs_per_seed", "particles")},
        "metric": "LBR attacker ante/hand; lower is better for target; not exploitability",
        "matched_deals_and_duplicate_controls_identical": True,
        "policies": policies,
        "paired_differences": differences,
        "uniform": estimate(a["uniform"].mean(axis=2)),
        "limits": [
            "Two evaluation seeds, one training seed; approximate particle/check-down LBR.",
            "Both players use the original heuristic H4 selection, not learned 12-action H4.",
            "Split schedules differ from the start, not only after 1M.",
            "95% bootstrap resamples deal pairs within seed; rare all-ins widen uncertainty.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, output)
    return result


def plot(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharey="row")
    colors = ["#c62828", "#277b42", "#1565c0"]
    for column, policy in enumerate(("current", "average")):
        groups = [
            (result["policies"][policy], ["dynamic", "frozen", "kmeans"],
             ["Dynamic DT", "Split-frozen DT", "k-means"], "LBR attacker profit (ante/hand)"),
            (result["paired_differences"][policy],
             ["dynamic_minus_kmeans", "dynamic_minus_frozen", "frozen_minus_kmeans"],
             ["Dynamic - KM", "Dynamic - frozen", "Frozen - KM"], "Paired profit difference (ante/hand)"),
        ]
        for row, (stats, names, labels, ylabel) in enumerate(groups):
            ax = axes[row, column]
            for index, (name, color) in enumerate(zip(names, colors)):
                value = stats[name]
                lo, hi = value["bootstrap_ci95"]
                ax.vlines(index, lo, hi, color=color, linewidth=2)
                ax.scatter(index, value["mean"], color=color, s=45, zorder=3)
                ax.scatter(index + np.linspace(-.08, .08, len(value["seed_means"])),
                           value["seed_means"], color=color, marker="x", alpha=.6)
                ax.annotate(f'{value["mean"]:.3f}', (index, value["mean"]),
                            xytext=(6, 7), textcoords="offset points", fontsize=9)
            ax.axhline(0, color="#444444", linewidth=.8)
            ax.set_xticks(range(len(names)), labels)
            ax.set_xlim(-.5, len(names) - .5)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.2)
            if row == 0:
                ax.set_title(policy.title() + " policy | lower is better for target")
    hands = len(result["seeds"]) * result["pairs_per_seed"] * 2
    fig.suptitle(f"7-stud 10M: {hands:,} hands/policy, {result['particles']} LBR particles\n"
                 "95% within-seed paired-deal bootstrap; x: seed means; not exact exploitability")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(DATA.resolve()):
        parser.error("Use a NEW output under dynamic_tree_validated data")
    result = compare(args.dynamic.resolve(), args.frozen.resolve(), output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
