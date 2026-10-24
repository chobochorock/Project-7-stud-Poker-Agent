"""Plot committed metrics while disk-backed MCCFR keeps running.

Usage (project root): python -m agents.cpp_mccfr.hand_recall_live_plot --run
    agents/cpp_mccfr/data/hand_recall_kmeans256_disk_1m_seed111 --window 100
Output: RUN/reviews/TIMESTAMP_HANDS/ with CSV snapshots, plots and summaries.
Only meta/metrics are read; no model copy, writer lock, checkpoint or training.
Reads the immutable committed prefix in short transactions, closing each before
the next batch. No single long-lived WAL snapshot is held. Local BR-gap is not
exploitability; crossfit is a signed diagnostic, not an unbiased exact gap.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np

from .hand_recall_disk import FAMILY, HEADER, POLICIES, connect, read_manifest
from .hand_recall_plot import COLORS, plot, rolling_bands


def read_committed_metrics(run: Path, batch_hands: int = 1000) -> tuple[dict, list]:
    """Read an append-only committed prefix without pinning WAL for the full scan."""
    if batch_hands < 1:
        raise ValueError("Batch size must be positive")

    def read(sql, params=()):
        start = time.monotonic()
        with connect(run) as db:
            db.set_progress_handler(lambda: int(time.monotonic() - start > 15), 1000)
            return list(db.execute(sql, params))

    meta_query = "SELECT k,v FROM meta WHERE k!='state'"
    metadata = {key: bytes(value).decode() for key, value in read(meta_query)}
    hands = int(metadata["hands"])
    snapshots = [[], []]
    # The writer only INSERTs metrics after migration; ready prefixes never change.
    if metadata["status"] != "ready":
        raise ValueError("Database migration is not complete")
    for first in range(1, hands + 1, batch_hands):
        rows = read(
            "SELECT line FROM metrics WHERE kind=0 AND hand BETWEEN ? AND ? ORDER BY hand,root",
            (first, min(first + batch_hands - 1, hands)),
        )
        snapshots[0].extend(line for (line,) in rows)
    snapshots[1] = [line for (line,) in read(
        "SELECT line FROM metrics WHERE kind=1 AND hand<=? ORDER BY hand,root", (hands,)
    )]
    latest = {key: bytes(value).decode() for key, value in read(meta_query)}
    if (latest["identity"] != metadata["identity"] or latest["status"] != "ready"
            or int(latest["hands"]) < hands):
        raise ValueError("Run identity/status or committed prefix changed")
    return metadata, snapshots


def parse_rows(lines: list[str]) -> list[dict]:
    rows = []
    for line in lines:
        values = next(csv.reader([line]))
        if len(values) not in (21, len(HEADER)):
            raise ValueError("Unexpected metric schema")
        values += [""] * (len(HEADER) - len(values))
        row = dict(zip(HEADER, values))
        row["hands"] = int(row["hands"])
        for name in POLICIES:
            value = float(row[name])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid Local BR-gap")
        rows.append(row)
    return rows


def validate_trace(rows: list[dict], hands: int) -> None:
    if len(rows) != hands or any(
        row["hands"] != index for index, row in enumerate(rows, 1)
    ):
        raise ValueError("Metrics do not match the committed hand count")


def means(rows: list[dict]) -> dict:
    result = {}
    for policy in POLICIES:
        result[policy] = {"gap": float(np.mean([float(r[policy]) for r in rows]))}
        for field in ("crossfit", "mean_action_se"):
            values = [float(r[f"{policy}_{field}"]) for r in rows if r[f"{policy}_{field}"] != ""]
            result[policy][field] = float(np.mean(values)) if values else None
            result[policy][f"{field}_count"] = len(values)
        queries = sum(int(r[f"{policy}_queries"]) for r in rows)
        result[policy]["missing_query_fraction"] = (
            sum(int(r[f"{policy}_missing"]) for r in rows) / queries if queries else None
        )
    return result


def diagnostics(output: Path, rows: list[dict], window: int) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r[f"{POLICIES[0]}_crossfit"] != ""]
    if len(rows) < window:
        return
    hands = np.array([r["hands"] for r in rows])
    if not np.all(np.diff(hands) == 1):
        raise ValueError("Diagnostic trace has internal gaps")
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, name in enumerate(("Current", "Average")):
        for arm, label in enumerate(("Recall", "Current-bucket baseline")):
            policy = POLICIES[col * 2 + arm]
            color = COLORS["recall" if arm == 0 else "baseline"]
            for row_index, field in enumerate(("crossfit", "mean_action_se")):
                values = np.array([float(r[f"{policy}_{field}"]) for r in rows])
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite diagnostic")
                bands = rolling_bands(values, window)
                axes[row_index, col].plot(hands[window - 1:], bands[:, 2], color=color, label=label)
        axes[0, col].set_title(f"{name} strategy: MA{window}")
        axes[0, col].axhline(0, color="black", linewidth=0.8, linestyle=":")
        axes[0, col].set_ylabel("Signed crossfit deviation (ante)")
        axes[1, col].set_ylabel("Mean action standard error (ante)")
        axes[1, col].set_ylim(bottom=0)
    for axis in axes.flat:
        axis.set_xlabel("Completed training hands")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Sampling diagnostics | negative crossfit values are retained")
    figure.text(0.5, 0.015, "Crossfit is NOT an unbiased exact gap; action SE is NOT a gap confidence interval.\n"
                "Diagnostics start after migration; earlier values are missing, not zero.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    figure.savefig(output / "sampling_diagnostics.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((FAMILY / "data").resolve()) or args.window < 1:
        parser.error("Run must be in this family's data directory; window must be positive")
    manifest = read_manifest(run)
    start = time.monotonic()
    metadata, snapshots = read_committed_metrics(run)
    read_seconds = time.monotonic() - start
    if metadata["identity"] != manifest["identity"] or metadata["status"] != "ready":
        raise ValueError("Database does not match the ready run manifest")
    hands = int(metadata["hands"])
    rows, panels = [parse_rows(lines) for lines in snapshots]
    validate_trace(rows, hands)
    if hands < args.window:
        raise ValueError("Not enough hands for the requested window")
    observed = datetime.now(timezone(timedelta(hours=9)))
    output = run / "reviews" / f"{observed:%Y%m%d_%H%M%S_%f}_{hands}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for filename, records in (("per_hand.csv", rows), ("panel.csv", panels)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "source_run": str(run), "observed_kst": observed.isoformat(),
        "committed_hands": hands, "read_total_seconds": read_seconds,
        "read_mode": "immutable committed prefix; 1000 hands per short transaction",
        "model_db_gib_observed_after_snapshot": (run / "model.sqlite").stat().st_size / 2**30,
        "first_1000": means(rows[:1000]), "last_1000": means(rows[-1000:]),
        "last_window": means(rows[-args.window:]),
        "latest_state_counts": {arm: int(rows[-1][f"{arm}_states"]) for arm in ("recall", "baseline")},
        "fixed_panels": [],
    }
    for hand in sorted({r["hands"] for r in panels}):
        group = [r for r in panels if r["hands"] == hand]
        if len(group) != manifest["config"]["panel_roots"]:
            raise ValueError("Incomplete fixed evaluation panel")
        summary["fixed_panels"].append({"hands": hand, "roots": len(group), "means": means(group)})
    plot(output, args.window)
    diagnostics(output, rows, args.window)
    (output / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
