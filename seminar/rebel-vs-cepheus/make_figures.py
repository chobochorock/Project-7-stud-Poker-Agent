"""Regenerate seminar figures from the cited CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

COLORS = {
    "gray": "#667085",
    "blue": "#2677A6",
    "green": "#3CA37D",
    "orange": "#E67E3F",
    "red": "#C44E52",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finish(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_liars_dice() -> None:
    rows = read_csv("rebel_liars_dice.csv")
    games = [row["game"] for row in rows]
    series = [
        ("Full-game FP", "full_game_fp", COLORS["gray"]),
        ("Full-game CFR", "full_game_cfr", COLORS["blue"]),
        ("ReBeL FP", "rebel_fp", COLORS["orange"]),
        ("ReBeL CFR-D", "rebel_cfr_d", COLORS["green"]),
    ]
    x = np.arange(len(games))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for index, (label, key, color) in enumerate(series):
        values = [float(row[key]) for row in rows]
        offset = (index - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, games)
    ax.set_ylabel("Exploitability (log scale, lower is better)")
    ax.set_title("Liar's Dice: 1,024 iterations", pad=46)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "rebel_liars_dice_exploitability.png")


def plot_cfrplus_targets() -> None:
    rows = read_csv("cfrplus_target_iterations.csv")
    labels = [
        "Matching pennies\n1e-3",
        "1000x1000 matrix\n1e-3",
        "1000x1000 matrix\n1e-4",
    ]
    cfr = [int(row["cfr_iterations"]) for row in rows]
    cfrplus = [int(row["cfrplus_iterations"]) for row in rows]
    lower_bound = [row["cfr_is_lower_bound"].lower() == "true" for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars_cfr = ax.bar(x - width / 2, cfr, width, label="CFR", color=COLORS["gray"])
    bars_plus = ax.bar(
        x + width / 2, cfrplus, width, label="CFR+", color=COLORS["blue"]
    )
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Iterations to target (log scale)")
    ax.set_title("CFR+ reaches fixed exploitability targets sooner")
    for bar, value, is_lower in zip(bars_cfr, cfr, lower_bound):
        label = f">{value:,}" if is_lower else f"{value:,}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, value in zip(bars_plus, cfrplus):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "cfrplus_target_iterations.png")


def plot_hunl_scores() -> None:
    rows = read_csv("rebel_hunl_scores.csv")
    labels = [row["opponent"] for row in rows]
    values = [float(row["score_mbb_per_game"]) for row in rows]
    errors = [float(row["stddev"]) for row in rows]
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.bar(labels, values, yerr=errors, capsize=5, color=colors)
    ax.bar_label(
        bars,
        labels=[f"{v:.0f} +/- {e:.0f}" for v, e in zip(values, errors)],
        padding=5,
        fontsize=9,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ReBeL score (mbb/game, higher is better)")
    ax.set_title("ReBeL HUNL match results are not exploitability")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "rebel_hunl_match_scores.png")


def plot_local_7stud() -> None:
    rows = read_csv("local_7stud_240p.csv")
    labels = [row["resolver"].replace(" ", "\n", 1) for row in rows]
    values = [float(row["lbr_ante_per_hand"]) for row in rows]
    lows = [float(row["ci95_low"]) for row in rows]
    highs = [float(row["ci95_high"]) for row in rows]
    errors = np.array(
        [
            [value - low for value, low in zip(values, lows)],
            [high - value for value, high in zip(values, highs)],
        ]
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=6,
        color=[COLORS["gray"], COLORS["blue"], COLORS["green"]],
    )
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_ylabel("Policy-LBR profit (ante/hand, lower is better)")
    ax.set_title("Local 7-stud sampled resolver control: 10,000 hands")
    ax.set_ylim(0, max(highs) * 1.13)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "local_7stud_resolver.png")


def main() -> None:
    plot_liars_dice()
    plot_cfrplus_targets()
    plot_hunl_scores()
    plot_local_7stud()
    print(f"Wrote four figures to {FIGURES}")


if __name__ == "__main__":
    main()
