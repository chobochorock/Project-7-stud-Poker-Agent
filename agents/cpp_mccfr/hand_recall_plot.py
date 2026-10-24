"""Red/blue Local BR-gap bands; use hand_recall plot --run PATH --window 100.

Bands describe the distribution of samples, not a confidence interval of the
mean. No point scatter. The lower log boundary clips zeros ONLY for drawing;
the linear figure, stored statistics and rolling means retain exact zeros.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLORS = {"recall": "#d62728", "baseline": "#1565c0"}


def rolling_bands(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Return [min, q05, mean, q95, max] for every full trailing window."""
    values = np.asarray(values, dtype=float)
    if window < 1 or values.ndim != 1 or len(values) < window:
        raise ValueError("Need at least one full positive-length rolling window")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite gap")
    views = np.lib.stride_tricks.sliding_window_view(values, window)
    result = np.empty((len(views), 5))
    # Bound temporary quantile memory even for a million hands.
    for start in range(0, len(views), 8192):
        block = views[start : start + 8192]
        q = np.quantile(block, [0, 0.05, 0.95, 1], axis=1)
        result[start : start + len(block)] = np.column_stack(
            (q[0], q[1], block.mean(axis=1), q[2], q[3])
        )
    return result


def draw_bands(axis, x, bands, color, label, floor=None):
    drawn = np.maximum(bands, floor) if floor is not None else bands
    axis.fill_between(
        x,
        drawn[:, 0],
        drawn[:, 4],
        step="post",
        color=color,
        alpha=0.10,
        linewidth=0,
        label=f"{label}: min-max",
    )
    axis.fill_between(
        x,
        drawn[:, 1],
        drawn[:, 3],
        step="post",
        color=color,
        alpha=0.25,
        linewidth=0,
        label=f"{label}: central 90%",
    )
    axis.plot(x, drawn[:, 2], color=color, linewidth=1.25, label=f"{label}: mean")


def plot(run: Path, window: int = 100) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((run / "config.json").read_text(encoding="utf-8"))
    # Numeric columns only: do not convert 64-bit root seeds to imprecise floats.
    data = np.loadtxt(
        run / "per_hand.csv", delimiter=",", skiprows=1, usecols=range(5), ndmin=2
    )
    if len(data) < window or not np.array_equal(
        data[:, 0], np.arange(1, len(data) + 1)
    ):
        raise ValueError("Incomplete/non-contiguous trace or fewer than window hands")
    if not np.isfinite(data).all() or (data[:, 1:] < 0).any():
        raise ValueError("Invalid Local BR-gap trace")
    x = data[window - 1 :, 0]
    bands = [rolling_bands(data[:, column], window) for column in range(1, 5)]
    positive = data[:, 1:][data[:, 1:] > 0]
    floor = float(positive.min() / 2) if len(positive) else 1e-6
    labels = ("Hand k-means + recall", "Hand k-means, current bucket only")
    for log in (False, True):
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for col, policy in enumerate(("Current", "Average")):
            for arm, name in enumerate(("recall", "baseline")):
                draw_bands(
                    axes[0, col],
                    x,
                    bands[col * 2 + arm],
                    COLORS[name],
                    labels[arm],
                    floor if log else None,
                )
            numerator, denominator = bands[col * 2][:, 2], bands[col * 2 + 1][:, 2]
            ratio = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
            axes[1, col].plot(
                x,
                ratio,
                color=COLORS["recall"],
                linewidth=1,
                label="Recall mean / baseline mean",
            )
            axes[1, col].axhline(
                1, color=COLORS["baseline"], linewidth=1, label="Baseline = 1"
            )
            axes[0, col].set_title(f"{policy} strategy: trailing {window} hands")
            axes[0, col].set_ylabel("Local BR-gap (ante)")
            axes[1, col].set_ylabel("Matched moving-mean ratio")
            if log:
                axes[0, col].set_yscale("log")
            else:
                axes[0, col].set_ylim(bottom=0)
        for axis in axes.flat:
            if log:
                axis.set_xscale("log")
            axis.set_xlabel("Completed training hands (post-update)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        suffix = "log" if log else "linear"
        note = (
            f"Log drawing floor={floor:.3g}; zeros remain in statistics."
            if log
            else "Zeros shown without clipping."
        )
        figure.suptitle(
            f"Hand-only k-means memory comparison | {len(data):,} hands recorded"
        )
        figure.text(
            0.5,
            0.015,
            f"{manifest['config']['particles']} particles/root; one root/hand. "
            "Bands are sample ranges, NOT confidence intervals.\n"
            f"{note} Not exploitability or LBR profit.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.97))
        figure.savefig(run / f"local_br_gap_{suffix}.png", dpi=150)
        plt.close(figure)
    panel = np.loadtxt(
        run / "panel.csv", delimiter=",", skiprows=1, usecols=range(5), ndmin=2
    )
    panel_summary = []
    for hand in np.unique(panel[:, 0]):
        rows = panel[panel[:, 0] == hand, 1:]
        if len(rows) != manifest["config"]["panel_roots"]:
            continue
        panel_summary.append(dict(hands=int(hand), means=rows.mean(axis=0).tolist()))
    summary = dict(
        hands=len(data),
        window=window,
        order=[
            "recall_current",
            "baseline_current",
            "recall_average",
            "baseline_average",
        ],
        last_window=[item[-1].tolist() for item in bands],
        band_order=["min", "p05", "mean", "p95", "max"],
        raw_zero_counts=(data[:, 1:] == 0).sum(axis=0).tolist(),
        log_draw_floor=floor,
        fixed_panel=panel_summary,
    )
    (run / "plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PLOT hands={len(data)} window={window} output={run}")
