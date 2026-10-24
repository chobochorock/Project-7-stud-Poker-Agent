"""Paired dynamic-tree/k-means gap plots; use run plot --run PATH --window 100.

Reads complete CSV lines, including during training. Does not touch checkpoints.
Red/blue MA100 bands are sample ranges, not confidence intervals.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np

from ..hand_recall_plot import COLORS, draw_bands, rolling_bands


def plot(run: Path, window: int = 100) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = (run / "per_hand.csv").read_text(encoding="utf-8")
    text = text[: text.rfind("\n") + 1]
    rows = list(csv.DictReader(io.StringIO(text)))
    policies = ("tree_current", "kmeans_current", "tree_average", "kmeans_average")
    x = np.array([int(r["hands"]) for r in rows])
    values = np.array([[float(r[p]) for p in policies] for r in rows])
    if len(rows) < window or not np.array_equal(x, np.arange(1, len(rows) + 1)):
        raise ValueError("Need a contiguous trace at least as long as window")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid gap")
    bands = [rolling_bands(values[:, i], window) for i in range(4)]
    positive = values[values > 0]
    floor = positive.min() / 2 if len(positive) else 1e-6
    config = json.loads((run / "config.json").read_text())["config"]
    for log in (False, True):
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for col, policy in enumerate(("Current", "Average")):
            for arm, (label, color) in enumerate(
                (
                    ("Dynamic tree", COLORS["recall"]),
                    ("Fixed hard256 k-means", COLORS["baseline"]),
                )
            ):
                draw_bands(
                    axes[0, col],
                    x[window - 1 :],
                    bands[col * 2 + arm],
                    color,
                    label,
                    floor if log else None,
                )
            num, den = bands[col * 2][:, 2], bands[col * 2 + 1][:, 2]
            ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            axes[1, col].plot(x[window - 1 :], ratio, color=COLORS["recall"])
            axes[1, col].axhline(1, color=COLORS["baseline"])
            axes[0, col].set_title(f"{policy}: MA{window}")
            axes[0, col].set_ylabel("Local BR-gap (ante)")
            axes[1, col].set_ylabel("Tree mean / k-means mean")
            if log:
                axes[0, col].set_yscale("log")
            else:
                axes[0, col].set_ylim(bottom=0)
        for axis in axes.flat:
            axis.grid(alpha=0.2)
            axis.set_xlabel("Completed hands")
            if log:
                axis.set_xscale("log")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        figure.suptitle(f"Dynamic-tree MCCFR vs fresh k-means | {len(rows):,} hands")
        figure.text(
            0.5,
            0.015,
            f"{config['particles']} particles/root. Different key partitions; matched evaluator.\n"
            "Bands: min-max / central 90%, NOT confidence intervals. Not exploitability. Zeros retained in means.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.97))
        figure.savefig(run / f"local_br_gap_{'log' if log else 'linear'}.png", dpi=150)
        if log:
            positive_means = np.concatenate([b[:, 2] for b in bands])
            positive_means = positive_means[positive_means > 0]
            zoom_floor = (
                10 ** (np.floor(np.log10(positive_means.min())) - 1)
                if len(positive_means)
                else floor
            )
            for axis in axes[0]:
                axis.set_ylim(zoom_floor, max(float(values.max()) * 1.2, zoom_floor * 10))
            footer = figure.texts[-1]
            footer.set_text(
                footer.get_text()
                + f"\nZoom: values below {zoom_floor:g} ante outside view; means unchanged."
            )
            figure.savefig(run / "local_br_gap_log_zoom.png", dpi=150)
        plt.close(figure)
    summary = dict(
        hands=len(rows),
        window=window,
        policies=policies,
        last_window=[b[-1].tolist() for b in bands],
        band_order=["min", "p05", "mean", "p95", "max"],
        last_row=rows[-1],
        log_floor=float(floor),
    )
    (run / "plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PLOT hands={len(rows)} window={window} output={run}")
