"""Summarize a completed paired PolicyLBR evaluation of frozen models.

Usage: python agents/lightgbm_regret_ensemble/analyze_epoch_lbr.py LBR_DIRECTORY
Writes summary.json, lbr_comparison.png and lbr_comparison_symlog.png.
The signed-log view remains linear within +/-1 ante, retaining zero/negative
profits and confidence bounds. Positive payoff means the LBR
exploiter wins. Intervals use independent deal pairs, not individual seats.
They do not cover training-seed variability or quantify exact exploitability.
New leaf checkpoints include ensemble_average; archived four-policy runs work unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import mean, stdev

from run_epoch_ensemble import np, plt

NAMES = [
    "ensemble_current",
    "hard256_current",
    "hard256_average",
    "uniform",
    "ensemble_average",
]
LABELS = [
    "Epoch LightGBM\ncurrent",
    "Hard-256\ncurrent",
    "Hard-256\naverage",
    "Uniform",
    "Epoch LightGBM\naverage",
]
COLORS = ["#bf354b", "#168776", "#3979b6", "#777777", "#c68636"]


def estimate(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("At least two finite pairs required")
    average = mean(values)
    se = stdev(values) / math.sqrt(len(values))
    return {
        "mean": average,
        "se": se,
        "ci95": [average - 1.96 * se, average + 1.96 * se],
    }


def analyze(directory):
    meta = json.loads((directory / "evaluation.json").read_text())
    names = [name for name in NAMES if name in meta["policies"]]
    if names[:4] != NAMES[:4] or set(names) != set(meta["policies"]):
        raise ValueError("Unexpected evaluation policies")
    labels = [LABELS[NAMES.index(name)] for name in names]
    colors = [COLORS[NAMES.index(name)] for name in names]
    with (directory / "paired_payoffs.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != meta["pairs"] or [int(r["pair"]) for r in rows] != list(
        range(len(rows))
    ):
        raise ValueError("Incomplete or nonconsecutive paired evaluation")
    if len({r["deal_seed"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation deals")
    samples = {}
    summary = {**meta, "complete": True, "paired_differences": {}}
    for name in names:
        seats = [[float(r[f"{name}_seat{s}"]) for r in rows] for s in (0, 1)]
        if any(abs(v) > 1000 or not math.isfinite(v) for seat in seats for v in seat):
            raise ValueError("Invalid terminal payoff")
        values = [(a + b) / 2 for a, b in zip(*seats)]
        stats = estimate(values)
        if not math.isclose(
            stats["mean"], meta["policies"][name]["mean"], abs_tol=1e-8
        ):
            raise ValueError("CSV/engine payoff mismatch")
        stats["seat_means"] = [mean(seat) for seat in seats]
        stats["max_abs_pair"] = max(map(abs, values))
        stats["pair_quantiles"] = dict(
            zip(
                ["p0", "p1", "p50", "p99", "p100"],
                np.quantile(values, [0, 0.01, 0.5, 0.99, 1]).tolist(),
            )
        )
        summary["policies"][name].update(stats)
        samples[name] = values
    for name in names[1:]:
        delta = [a - b for a, b in zip(samples[names[0]], samples[name])]
        summary["paired_differences"][f"ensemble_minus_{name}"] = estimate(delta)
    if "ensemble_average" in samples:
        summary["paired_differences"][
            "ensemble_average_minus_hard256_average"
        ] = estimate(
            [
                a - b
                for a, b in zip(samples["ensemble_average"], samples["hard256_average"])
            ]
        )

    here = Path(__file__).resolve().parent
    sources = directory / "sources"
    sources.mkdir(exist_ok=True)
    summary["source_sha256"] = {}
    for name in [
        "stud_mccfr.cpp",
        "stud_epoch_ensemble.cpp",
        "evaluate_epoch_lbr.cpp",
        "analyze_epoch_lbr.py",
        "local_split_helpers.hpp",
        "stud_rules.hpp",
    ]:
        destination = sources / name
        if not destination.exists():
            source = here / name
            if not source.exists():
                source = here.parent / "cpp_mccfr" / name
            shutil.copy2(source, destination)
        summary["source_sha256"][name] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
    binary = sources / "evaluate_epoch_lbr.exe"
    if not binary.exists():
        shutil.copy2(here / "bin/evaluate_epoch_lbr.exe", binary)
    summary["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for name, label, color in zip(names, labels, colors):
        values = np.asarray(samples[name])
        indices = np.arange(1, len(values) + 1)
        start = min(49, len(values) // 10)
        axes[0].plot(
            indices[start:] * 2,
            (np.cumsum(values) / indices)[start:],
            label=label.replace("\n", " "),
            color=color,
        )
    axes[0].set(
        title=f"Frozen 7-stud {meta.get('training_hands', 100000):,}-hand policies: actual profit earned by PolicyLBR",
        xlabel="Evaluation hands per target (seat-paired)",
        ylabel="Cumulative LBR ante/hand",
    )
    axes[0].legend()
    positions = np.arange(len(names))
    for position, name, color in zip(positions, names, colors):
        stats = summary["policies"][name]
        axes[1].errorbar(
            position,
            stats["mean"],
            yerr=1.96 * stats["se"],
            fmt="o",
            capsize=6,
            color=color,
        )
        axes[1].annotate(
            f'{stats["mean"]:.3f}',
            (position, stats["mean"]),
            xytext=(12, 4),
            textcoords="offset points",
        )
    axes[1].set(
        xticks=positions,
        xticklabels=labels,
        xlim=(-0.5, len(names) - 0.35),
        ylabel="LBR ante/hand (lower is better for target)",
        title=f'{meta["pairs"]:,} independent deal pairs; {meta["particles"]} particles; 95% pair-sampling CI',
    )
    for ax in axes:
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.grid(alpha=0.2)
    fig.savefig(directory / "lbr_comparison.png", dpi=160)
    for ax in axes:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("LBR ante/hand (signed log; linear within +/-1)")
        ax.margins(y=0.12)
    fig.savefig(directory / "lbr_comparison_symlog.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "policies": summary["policies"],
                "paired_differences": summary["paired_differences"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    check = estimate([1.0, 3.0])
    assert check["mean"] == 2 and math.isclose(check["se"], 1)
    assert math.isclose(check["ci95"][0], 0.04)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
