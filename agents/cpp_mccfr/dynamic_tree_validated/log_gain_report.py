"""Plot a saved log-gain DT and its matched k-means control, without training.

Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain plot --output NEW_DIR
Bands are fixed-root min/max and central 90%, NOT CIs or per-hand moving averages.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np

from . import log_gain
from .report import leaf_support
from ..dynamic_tree.review import POLICIES, checkpoint_header, panel_summary
from ..hand_recall_plot import draw_bands


def analyze(run):
    manifest = log_gain.manifest_at(run)
    config = manifest["config"]
    hands = checkpoint_header(run, manifest["identity"])
    groups = {}
    with (run / "panel.csv").open() as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["hands"]), []).append(row)
    required = {0, hands} | set(
        range(config["panel_every"], hands + 1, config["panel_every"])
    )
    if not required.issubset(groups) or min(groups) != 0 or max(groups) != hands:
        raise ValueError("Incomplete fixed-panel history")
    seeds = [r["root_seed"] for r in groups[0]]
    if len(seeds) != config["panel_roots"] or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid panel root count")
    bands = []
    budgets = sorted(groups)
    for hand in budgets:
        rows = groups[hand]
        expected = config["fixed_gain"] or 1 / (1 + math.log10(max(1, hand)))
        if [r["root_seed"] for r in rows] != seeds or any(
            not math.isclose(float(r["gain_threshold"]), expected, rel_tol=1e-12)
            for r in rows
        ):
            raise ValueError("Panel roots or gain schedule disagree")
        values = np.array([[float(r[p]) for p in POLICIES] for r in rows])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid gap")
        q = np.quantile(values, [0, 0.05, 0.95, 1], axis=0)
        bands.append(np.stack((q[0], q[1], values.mean(axis=0), q[2], q[3]), axis=-1))
    checks = splits = 0
    with (run / "splits.csv").open() as handle:
        for row in csv.DictReader(handle):
            checks += 1
            splits += int(row["accepted"])
            expected = config["fixed_gain"] or 1 / (1 + math.log10(int(row["hands"])))
            if not math.isclose(float(row["gain_threshold"]), expected, rel_tol=1e-12):
                raise ValueError("Split threshold disagrees with hand count")
            if int(row["accepted"]) and float(row["relative_gain"]) + 1e-12 < expected:
                raise ValueError("Split accepted below threshold")
    support = leaf_support(run)
    if support["checks"] != checks or support["splits"] != splits:
        raise ValueError("Split log/checkpoint mismatch")
    summary = dict(
        run=str(run),
        hands=hands,
        config=config,
        leaf_support=support,
        checkpoint_bytes=(run / "checkpoint.bin").stat().st_size,
        final_panel=panel_summary(groups[hands]),
        metric="fixed-root Local BR-gap, ante; not exploitability/LBR",
        bands="root distribution min-max and central90%; not CI or per-hand MA100",
    )
    return summary, np.array(budgets), np.array(bands), groups


def report(run, output):
    if output.exists() or not output.is_relative_to(log_gain.base.DATA.resolve()):
        raise ValueError("Use a NEW family data output directory")
    with log_gain.base.writer_lock(run):
        summary, x, bands, groups = analyze(run)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    for scale in ("linear", "log"):
        use = x > 0 if scale == "log" else np.ones(len(x), dtype=bool)
        xx, yy = x[use], bands[use]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for col, policy in enumerate(("Current", "Average")):
            for arm, color, label in (
                (0, "#d62728", "DT"),
                (1, "#1565c0", "matched k-means"),
            ):
                draw_bands(
                    axes[0, col],
                    xx,
                    yy[:, 2 * col + arm],
                    color,
                    label,
                    0.001 if scale == "log" else None,
                )
            denominator = yy[:, 2 * col + 1, 2]
            ratio = np.divide(
                yy[:, 2 * col, 2],
                denominator,
                out=np.full_like(denominator, np.nan),
                where=denominator > 0,
            )
            axes[1, col].plot(xx, ratio, color="#d62728")
            axes[1, col].axhline(1, color="#1565c0")
            axes[0, col].set_title(policy)
            axes[0, col].set_ylabel("Fixed-panel Local BR-gap (ante)")
            axes[1, col].set_ylabel("DT / k-means mean")
            axes[0, col].legend(fontsize=8)
            axes[0, col].set_ylim(bottom=0.001 if scale == "log" else 0)
            if scale == "log":
                axes[0, col].set_yscale("log")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.set_xlabel("Total training hands")
            if scale == "log":
                ax.set_xscale("log")
        fig.suptitle(
            f"Split threshold: {summary['config']['gain_schedule']} | red DT / blue k-means"
        )
        fig.text(
            0.5,
            0.01,
            "Bands: fixed-root min-max / central 90%, not confidence intervals or per-hand MA.\n"
            "Finite-particle max bias remains; raw means retain zeros; log display floors gaps at 0.001.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(output / f"local_br_gap_{scale}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
        axes,
        ("gain_threshold", "tree_leaves", "peak_rss_bytes"),
        ("Required SSE reduction", "DT leaves", "Process peak working set (GiB)"),
    ):
        values = [float(groups[h][0][key]) for h in x]
        if key == "peak_rss_bytes":
            values = np.asarray(values) / 2**30
        ax.plot(x, values, color="#d62728")
        ax.set_title(title)
        ax.set_xlabel("Total training hands")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "split_resources.png", dpi=150)
    plt.close(fig)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REPORT output={output} matched_kmeans=included")
    return summary
