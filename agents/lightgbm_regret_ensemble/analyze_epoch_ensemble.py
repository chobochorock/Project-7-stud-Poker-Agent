"""Validate and graph a completed or in-progress 7-stud epoch run.

Usage: python agents/lightgbm_regret_ensemble/analyze_epoch_ensemble.py RUN_DIRECTORY
Creates comparison.png, comparison_log.png and analysis.json without altering
training outputs. Log y axes mask nonpositive gaps; no epsilon is added.
Intervals describe variation across 128 fixed roots, not exploitability bounds.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from run_epoch_ensemble import np, plt


def analyze(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))

    def complete_rows(path):
        text = path.read_text(encoding="utf-8")
        text = text[: text.rfind("\n") + 1]
        return np.atleast_1d(
            np.genfromtxt(io.StringIO(text), delimiter=",", names=True)
        )

    rows = complete_rows(directory / "per_hand.csv")
    audit = complete_rows(directory / "boundary_audit.csv")
    if not len(rows) or not len(audit):
        raise ValueError("Training and boundary audit rows are required")
    assert np.array_equal(rows["hand"], np.arange(1, len(rows) + 1))
    for name in rows.dtype.names:
        assert np.isfinite(rows[name]).all(), name
    boundaries = []
    for hand in np.unique(audit["hand"]):
        at_hand = audit[audit["hand"] == hand]
        for models in np.unique(at_hand["epoch_models"]):
            block = at_hand[at_hand["epoch_models"] == models]
            if len(block) != 128:
                continue
            assert np.array_equal(block["root"], np.arange(128))
            record = {"hand": int(hand), "models": int(models)}
            for prefix in ["ensemble", "hard256"]:
                values = block[f"{prefix}_gap_ante"]
                record[prefix] = float(values.mean())
                record[f"{prefix}_se"] = float(
                    values.std(ddof=1) / np.sqrt(len(values))
                )
            difference = block["ensemble_gap_ante"] - block["hard256_gap_ante"]
            record["paired_difference"] = float(difference.mean())
            record["paired_difference_se"] = float(
                difference.std(ddof=1) / np.sqrt(len(block))
            )
            boundaries.append(record)
    post = [r for r in boundaries if r["hand"] == r["models"] * config["epoch_hands"]]
    summaries = [
        json.loads(p.read_text())
        for p in sorted(directory.glob("epoch_*/summary.json"))
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), layout="constrained")
    colors = [
        ("ensemble", "Epoch LightGBM", "#bf354b"),
        ("hard256", "Hard-256 MCCFR", "#168776"),
    ]
    block_size = min(1000, max(1, len(rows) // 20))
    count = len(rows) // block_size
    x = rows["hand"][: count * block_size].reshape(count, block_size).mean(axis=1)
    for prefix, label, color in colors:
        gap = rows[f"{prefix}_root_local_gap_ante"]
        means = gap[: count * block_size].reshape(count, block_size).mean(axis=1)
        axes[0].plot(x, means, color=color, label=label)
        hx = np.array([r["hand"] for r in post])
        hy = np.array([r[prefix] for r in post])
        se = np.array([r[f"{prefix}_se"] for r in post])
        axes[1].plot(hx, hy, "o-", color=color, label=label)
        axes[1].fill_between(
            hx, np.maximum(0, hy - 1.96 * se), hy + 1.96 * se, alpha=0.15, color=color
        )
    pre = [
        r for r in boundaries if r["hand"] == (r["models"] + 1) * config["epoch_hands"]
    ]
    if pre:
        axes[1].scatter(
            [r["hand"] for r in pre],
            [r["ensemble"] for r in pre],
            marker="x",
            color="#bf354b",
            label="Before abstraction",
            zorder=4,
        )
        by_hand = {r["hand"]: r for r in post}
        for r in pre:
            if r["hand"] in by_hand:
                axes[1].plot(
                    [r["hand"], r["hand"]],
                    [r["ensemble"], by_hand[r["hand"]]["ensemble"]],
                    color="#bf354b",
                    alpha=0.5,
                    linestyle=":",
                )
    axes[0].set_title(
        f"7-stud: every-hand first-5th-infoset gap ({block_size}-hand means, {config['particles']} particles)"
    )
    axes[1].set_title(
        "Fixed 128-root audit, 128 particles/root; after abstraction; mean +/- 1.96 SE"
    )
    for ax in axes[:2]:
        ax.set_ylabel("Estimated one-step deviation gain (antes)")
    axes[2].plot(
        rows["hand"],
        rows["epoch_infosets"] / 1e6,
        color="#bf354b",
        label="Disk-backed epoch infosets",
    )
    axes[2].plot(
        rows["hand"],
        rows["hard256_buckets"] / 1e6,
        color="#168776",
        label="Hard-256 regret entries",
    )
    axes[2].set_ylabel("Table entries (millions)")
    axes[2].set_title("Learning-table counts; previous ensemble models are additional")
    for ax in axes:
        ax.set_xlabel("Training hands")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(directory / "comparison.png", dpi=160)
    for ax in axes[:2]:
        ax.set_yscale("log", nonpositive="mask")
        ax.set_ylabel("Estimated deviation gain (antes, log scale)")
        ax.grid(alpha=0.1, which="minor")
    fig.savefig(directory / "comparison_log.png", dpi=160)
    plt.close(fig)
    result = {
        "hands": len(rows),
        "complete": (
            len(rows) == config["hands"]
            and len(summaries) == config["hands"] // config["epoch_hands"]
            and post[-1]["hand"] == config["hands"]
        ),
        "boundaries": boundaries,
        "epochs": summaries,
        "combined_cpp_peak_rss_gib": float(rows["peak_cpp_rss_bytes"].max() / 2**30),
        "ensemble_model_bytes": sum(r["model_bytes"] for r in summaries),
        "baseline_model_bytes": (directory / f"hard256_{int(rows['hand'][-1])}.bin")
        .stat()
        .st_size
        if (directory / f"hard256_{int(rows['hand'][-1])}.bin").exists()
        else None,
        "ensemble_buckets_total": sum(r["buckets"] for r in summaries),
        "saved_row_bytes": sum(p.stat().st_size for p in directory.glob("rows_*.bin")),
        "fit_seconds": sum(r["fit_seconds"] for r in summaries),
        "training_nodes": {
            p: int(rows[f"{p}_training_nodes"][-1]) for p, _, _ in colors
        },
        "last_10k_gap_mean": {
            p: float(rows[f"{p}_root_local_gap_ante"][-10000:].mean())
            for p, _, _ in colors
        },
    }
    independent_path = directory / "independent_audit.csv"
    if independent_path.exists():
        independent = complete_rows(independent_path)
        assert np.array_equal(independent["root"], np.arange(len(independent)))
        audit_summary = {
            "roots": len(independent),
            "seed": int(independent["seed"][0]),
            "particles": int(independent["particles"][0]),
        }
        methods = [
            "ensemble_current",
            "hard256_current",
            "hard256_average",
            "uniform",
        ]
        if "ensemble_average" in independent.dtype.names:
            methods.append("ensemble_average")
        for method in methods:
            values = independent[method]
            assert np.isfinite(values).all() and np.all(values >= 0)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            audit_summary[method] = {
                "mean": mean,
                "se": se,
                "root_sampling_ci95": [mean - 1.96 * se, mean + 1.96 * se],
            }
        for kind in ("current", "average"):
            if f"ensemble_{kind}" not in independent.dtype.names:
                continue
            difference = (
                independent[f"ensemble_{kind}"] - independent[f"hard256_{kind}"]
            )
            mean = float(difference.mean())
            se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
            audit_summary[f"paired_{kind}_difference"] = {
                "mean": mean,
                "se": se,
                "root_sampling_ci95": [mean - 1.96 * se, mean + 1.96 * se],
            }
        audit_summary["matched_models_mean"] = float(
            independent["matched_epoch_models"].mean()
        )
        audit_summary["roots_with_no_matched_model"] = int(
            np.count_nonzero(independent["matched_epoch_models"] == 0)
        )
        result["independent_audit"] = audit_summary
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        ax.barh(
            [m.replace("_", " ") for m in methods],
            [audit_summary[m]["mean"] for m in methods],
            xerr=[1.96 * audit_summary[m]["se"] for m in methods],
            color=["#bf354b", "#168776", "#78b6aa", "#808080", "#d78a9b"][
                : len(methods)
            ],
            capsize=3,
        )
        ax.set_xlabel(
            "Estimated one-step deviation gain (antes); lower is smaller local gap"
        )
        ax.set_title(
            f"Held-out {len(independent)} roots; mean +/- 1.96 SE (not exploitability)"
        )
        fig.savefig(directory / "independent_audit.png", dpi=160)
        plt.close(fig)
    resources = directory / "resource_samples.csv"
    if resources.exists():
        peaks = {}
        with resources.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = int(row["lifetime_peak_working_set_bytes"])
                assert value >= 0
                peaks[row["name"]] = max(peaks.get(row["name"], 0), value)
        result["observed_process_peaks_gib"] = {
            name: value / 2**30 for name, value in peaks.items()
        }
    (directory / "analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("boundaries", "epochs")},
            indent=2,
        )
    )
    print("Last audit:", post[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory.resolve())
