"""Review a completed DT run without loading all per-hand rows into RAM.

Usage (project root): python -m agents.cpp_mccfr.dynamic_tree.review --run PATH
    --output PATH/reviews/NAME
Outputs: summary.json, linear/log gap plots and fixed-panel diagnostics.
MA100 is evaluated every 100 hands; every hand enters one displayed window.
No training/rollouts; checkpoint validation is header/log offsets, not full load.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from ..hand_recall_plot import COLORS, draw_bands

POLICIES = ("tree_current", "kmeans_current", "tree_average", "kmeans_average")
METRICS = POLICIES + tuple(
    p + suffix for p in POLICIES for suffix in ("_crossfit", "_action_se")
)
LOGS = ("per_hand.csv", "panel.csv", "splits.csv")


def checkpoint_header(run: Path, identity: str) -> int:
    with (run / "checkpoint.bin").open("rb") as handle:
        def read_text():
            size = struct.unpack("<I", handle.read(4))[0]
            if size > 100000:
                raise ValueError("Invalid checkpoint string length")
            return handle.read(size).decode("utf-8")

        if read_text() not in ("DYNAMIC_TREE1", "DYNAMIC_TREE_VALIDATED1") or read_text() != identity:
            raise ValueError("Checkpoint identity mismatch")
        hands = struct.unpack("<Q", handle.read(8))[0]
        sizes = struct.unpack("<3Q", handle.read(24))
    if sizes != tuple((run / name).stat().st_size for name in LOGS):
        raise ValueError("Logs do not end at checkpoint; review a completed run")
    return hands


def bands_at_window_ends(gaps: np.ndarray, window: int = 100) -> np.ndarray:
    count = len(gaps) // window
    windows = gaps[:count * window].reshape(count, window, 4)
    quantiles = np.quantile(windows, [0, 0.05, 0.95, 1], axis=1)
    return np.stack(
        (quantiles[0], quantiles[1], windows.mean(axis=1), quantiles[2], quantiles[3]),
        axis=-1,
    )


def panel_summary(rows: list[dict]) -> dict:
    values = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Invalid panel value")
    result = {key: float(value.mean()) for key, value in values.items()}
    result["root_count"] = len(rows)
    result["root_se"] = {
        key: float(value.std(ddof=1) / np.sqrt(len(rows)))
        for key, value in values.items()
    }
    result["paired_tree_minus_kmeans"] = {}
    for policy in ("current", "average"):
        for suffix in ("", "_crossfit"):
            key = policy + suffix
            delta = values["tree_" + key] - values["kmeans_" + key]
            result["paired_tree_minus_kmeans"][key] = {
                "mean": float(delta.mean()),
                "root_sample_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            }
    return result


def analyze(run: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    manifest = json.loads((run / "config.json").read_text())
    config = manifest["config"]
    hands = checkpoint_header(run, manifest["identity"])
    watched = ("checkpoint.bin", "config.json") + LOGS
    before = {name: (run / name).stat() for name in watched}
    project = Path(__file__).resolve().parents[3]
    hashes = {
        name: {
            "current": hashlib.sha256((project / name).read_bytes()).hexdigest() == digest,
            "snapshot": hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == digest,
        }
        for name, digest in config["source_sha256"].items()
    }
    atlas_matches = hashlib.sha256((run / "hand_atlas.bin").read_bytes()).hexdigest() == config["atlas_sha256"]
    blocks, plot_x, plot_bands = [], [], []
    done = 0
    cap_hand = None
    with (run / "per_hand.csv").open() as handle:
        header = next(csv.reader([next(handle)]))
        columns = [i for i, name in enumerate(header) if name != "root_seed"]
        names = [header[i] for i in columns]
        index = {name: i for i, name in enumerate(names)}
        while done < hands:
            count = min(10000, hands - done)
            chunk = np.loadtxt(handle, delimiter=",", usecols=columns, max_rows=count, ndmin=2)
            if len(chunk) != count or not np.isfinite(chunk).all():
                raise ValueError("Incomplete/nonfinite per-hand trace")
            if not np.array_equal(chunk[:, 0], np.arange(done + 1, done + count + 1)):
                raise ValueError("Noncontiguous per-hand trace")
            gaps = chunk[:, 1:5]
            if (gaps < 0).any():
                raise ValueError("Negative plugin gap")
            block = {key: float(chunk[:, index[key]].mean()) for key in METRICS}
            block.update(start_hand=done + 1, end_hand=done + count, count=count)
            block["missing_query_rate"] = {
                p: float(chunk[:, index[p + "_missing"]].sum() / chunk[:, index[p + "_queries"]].sum())
                for p in POLICIES
            }
            blocks.append(block)
            bands = bands_at_window_ends(gaps)
            plot_bands.append(bands)
            plot_x.append(chunk[99:len(bands) * 100:100, 0])
            if config["max_splits"] >= 0 and cap_hand is None:
                hits = np.flatnonzero(chunk[:, index["splits"]] >= config["max_splits"])
                if len(hits):
                    cap_hand = int(chunk[hits[0], 0])
            done += count
            last_row = dict(zip(names, chunk[-1].tolist()))
        if handle.read().strip():
            raise ValueError("Unexpected rows after checkpoint")
    with (run / "panel.csv").open() as handle:
        panel_rows = list(csv.DictReader(handle))
    groups = {}
    for row in panel_rows:
        groups.setdefault(int(row["hands"]), []).append(row)
    expected = set(range(0, hands + 1, config["panel_every"]))
    if set(groups) != expected:
        raise ValueError("Unexpected panel budgets")
    roots = [row["root_seed"] for row in groups[0]]
    if len(roots) != config["panel_roots"] or len(set(roots)) != config["panel_roots"]:
        raise ValueError("Unexpected panel root count")
    for rows in groups.values():
        if [row["root_seed"] for row in rows] != roots:
            raise ValueError("Fixed panel roots changed")
    checks, splits, last_split = 0, 0, None
    with (run / "splits.csv").open() as handle:
        for row in csv.DictReader(handle):
            checks += 1
            if row["accepted"] == "1":
                splits += 1
                last_split = row
    if splits != last_row["splits"] or checks != last_row["checks"]:
        raise ValueError("Split log disagrees with final metrics")
    for name, old in before.items():
        now = (run / name).stat()
        if (old.st_size, old.st_mtime_ns) != (now.st_size, now.st_mtime_ns):
            raise ValueError("Run changed during review")
    summary = {
        "hands": hands, "config": config, "source_hashes": hashes,
        "atlas_matches": atlas_matches, "checkpoint_log_offsets_match": True,
        "checkpoint_bytes": before["checkpoint.bin"].st_size,
        "cap_reached_hand": cap_hand, "checks": checks, "splits": splits,
        "last_split": last_split, "last_row": last_row,
        "blocks_10000": blocks,
        "panels": {str(h): panel_summary(rows) for h, rows in groups.items()},
        "plot_window": 100, "plot_stride": 100, "omitted_partial_window_rows": hands % 100,
        "limitations": "Single training seed; fifth-street roots; finite-particle max bias; not LBR/exploitability. Header validation only, no solver reload.",
    }
    return summary, np.concatenate(plot_x), np.concatenate(plot_bands)


def plots(summary: dict, x: np.ndarray, bands: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = summary["cap_reached_hand"]
    unlimited = all(summary["config"][key] < 0 for key in ("max_splits", "max_depth"))
    tree_label = "Unlimited DT" if unlimited else "Limited DT"
    if summary["config"].get("schema") == "dynamic_tree_validated_1":
        config = summary["config"]
        tree_label = f"{config['split_rule'].title()} DT (gain {config['gain']:g})"
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for col, policy in enumerate(("Current", "Average")):
            for arm, label in enumerate((tree_label, "Fixed hard256 k-means")):
                color = COLORS["recall" if arm == 0 else "baseline"]
                draw_bands(axes[0, col], x, bands[:, 2 * col + arm], color, label, 0.001 if scale == "log" else None)
            num, den = bands[:, 2 * col, 2], bands[:, 2 * col + 1, 2]
            ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            axes[1, col].plot(x, ratio, color=COLORS["recall"], linewidth=0.8)
            axes[1, col].axhline(1, color=COLORS["baseline"])
            axes[0, col].set_title(f"{policy}: trailing MA100, displayed every 100 hands")
            axes[0, col].set_ylabel("Local BR-gap (ante)")
            axes[1, col].set_ylabel("DT mean / k-means mean")
            if scale == "log":
                axes[0, col].set_yscale("log")
                axes[0, col].set_ylim(0.001, bands[:, :, 4].max() * 1.2)
            else:
                axes[0, col].set_ylim(bottom=0)
            axes[0, col].legend(fontsize=8, loc="upper right")
        for axis in axes.flat:
            if cap is not None:
                axis.axvline(cap, color="#222222", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
            axis.set_xlabel("Completed hands")
            if scale == "log":
                axis.set_xscale("log")
        cap_label = f"{cap:,}" if cap is not None else "not reached"
        if summary["config"]["max_splits"] < 0:
            cap_label = "disabled"
        fig.suptitle(f"{tree_label} vs matched k-means | {summary['hands']:,} hands | split cap: {cap_label}")
        fig.text(0.5, 0.02, "Red: DT; blue: k-means; dashed: split cap. Bands: min-max / central 90%, not CI.\n"
                 "32 particles/root; different key partitions. Log view clips below 0.001 ante; all values enter means. Not exploitability.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
        fig.savefig(output / f"local_br_gap_{scale}.png", dpi=150)
        plt.close(fig)
    panel = summary["panels"]
    budgets = np.array([int(h) for h in panel])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, policy in enumerate(("current", "average")):
        for arm, color in (("tree", COLORS["recall"]), ("kmeans", COLORS["baseline"])):
            for row, suffix in enumerate(("", "_crossfit")):
                values = [panel[str(h)][f"{arm}_{policy}{suffix}"] for h in budgets]
                axes[row, col].plot(budgets, values, color=color, label=arm)
        axes[0, col].set_title(policy.title())
        axes[0, col].set_ylabel("Fixed-panel plugin gap (ante)")
        axes[1, col].set_ylabel("Fixed-panel signed crossfit (ante)")
        axes[1, col].axhline(0, color="#444444", linewidth=0.6)
        axes[0, col].legend()
    for axis in axes.flat:
        if cap is not None:
            axis.axvline(cap, color="#222222", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Completed hands")
    fig.suptitle("Same 128 fixed roots, 32 particles/root; sampling diagnostics, not exploitability")
    fig.tight_layout()
    fig.savefig(output / "fixed_panel.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output == run or not output.is_relative_to(run):
        parser.error("Output must be a review subdirectory of the run")
    # Check sampled MA100 against the existing full rolling implementation.
    from ..hand_recall_plot import rolling_bands
    sample = np.random.default_rng(0).uniform(size=(300, 4))
    sample[:100, 0] = 0
    actual = bands_at_window_ends(sample)
    for i in range(4):
        np.testing.assert_allclose(actual[:, i], rolling_bands(sample[:, i], 100)[::100])
    summary, x, bands = analyze(run)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(summary, x, bands, output)
    print(json.dumps({"hands": summary["hands"], "cap_hand": summary["cap_reached_hand"],
                      "last_10k": summary["blocks_10000"][-1],
                      "final_panel": summary["panels"][str(summary["hands"])],
                      "last_row": summary["last_row"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
