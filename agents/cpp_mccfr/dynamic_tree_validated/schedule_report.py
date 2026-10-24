"""Plot completed split-schedule branches with the SAME k-means baseline.

Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.schedule_report
    --runs freeze=RUN continue=RUN --output NEW_FAMILY_DATA_DIR
Read-only; pause training first. Uses fixed-panel distributions, NOT per-hand MA100.
Parent panel is included once; all means retain zeros, log display clips at .001.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import schedule
from .report import leaf_support
from ..dynamic_tree.review import checkpoint_header, panel_summary, POLICIES
from ..hand_recall_plot import draw_bands


def panels(path):
    groups = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["hands"]), []).append(row)
    return groups


def analyze(run):
    manifest = schedule.manifest_at(run)
    config = manifest["config"]
    hands = checkpoint_header(run, manifest["identity"])
    spec = config["schedule"]
    own = panels(run / "panel.csv")
    required = {spec["branch_hands"], hands} | set(
        range(
            (spec["branch_hands"] // config["panel_every"] + 1) * config["panel_every"],
            hands + 1,
            config["panel_every"],
        )
    )
    if (
        not required.issubset(own)
        or min(own) != spec["branch_hands"]
        or max(own) != hands
    ):
        raise ValueError("Missing fixed-panel budgets")
    parent = Path(spec["parent"])
    parent_manifest = schedule.manifest_at(parent, scheduled=False)
    if parent_manifest["identity"] != spec["parent_identity"]:
        raise ValueError("Parent identity changed")
    groups = {
        h: r
        for h, r in panels(parent / "panel.csv").items()
        if h < spec["branch_hands"]
    }
    groups.update(own)
    seeds = [row["root_seed"] for row in own[spec["branch_hands"]]]
    if len(seeds) != config["panel_roots"] or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid root count")
    for rows in groups.values():
        if [row["root_seed"] for row in rows] != seeds:
            raise ValueError("Panel roots differ")
    budgets = sorted(groups)
    bands = []
    for h in budgets:
        values = np.array([[float(r[p]) for p in POLICIES] for r in groups[h]])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid gap")
        q = np.quantile(values, [0, 0.05, 0.95, 1], axis=0)
        bands.append(np.stack((q[0], q[1], values.mean(axis=0), q[2], q[3]), axis=-1))
    support = leaf_support(run)
    last = own[hands][-1]
    checks = splits = 0
    with (run / "splits.csv").open() as handle:
        for row in csv.DictReader(handle):
            checks += 1
            splits += int(row["accepted"])
    if (
        support["splits"] != spec["parent_splits"] + splits
        or support["checks"] != spec["parent_checks"] + checks
        or support["splits"] != int(last["splits"])
    ):
        raise ValueError("Split counts disagree")
    if spec["mode"] == "freeze" and (splits or support["monitor_samples"]):
        raise ValueError("Frozen branch grew or retained split monitors")
    return (
        dict(
            hands=hands,
            config=config,
            leaf_support=support,
            last_row=last,
            final_panel=panel_summary(own[hands]),
            checkpoint_bytes=(run / "checkpoint.bin").stat().st_size,
        ),
        np.array(budgets),
        np.array(bands),
        groups,
    )


def compare(runs, output):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(schedule.base.DATA.resolve()):
        raise ValueError("Use a NEW family data output")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, curves = {}, {}
    reference = None
    reference_config = None
    for label, run in runs.items():
        with schedule.base.writer_lock(run):
            summary, x, bands, groups = analyze(run)
        cfg = summary["config"]
        conditions = [
            cfg[k]
            for k in ("seed", "eval_seed", "particles", "panel_roots", "atlas_sha256")
        ]
        conditions += [cfg["schedule"]["parent_checkpoint_sha256"]]
        if reference_config is not None and reference_config != conditions:
            raise ValueError("Unmatched initialization/evaluation conditions")
        reference_config = conditions
        if reference is not None:
            for hand in groups.keys() & reference.keys():
                for a, b in zip(groups[hand], reference[hand]):
                    for key in a:
                        if key.startswith("kmeans_") or key == "root_seed":
                            if a[key] != b[key]:
                                raise ValueError(
                                    f"Matched baseline differs at {hand}: {key}"
                                )
        reference = groups
        result[label] = summary
        curves[label] = (x, bands)
    output.mkdir(parents=True)
    for label, (x, bands) in curves.items():
        for scale in ("linear", "log"):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
            for col, policy in enumerate(("Current", "Average")):
                for arm, color, name in (
                    (0, "#d62728", label),
                    (1, "#1565c0", "matched k-means"),
                ):
                    draw_bands(
                        axes[0, col],
                        x,
                        bands[:, 2 * col + arm],
                        color,
                        name,
                        0.001 if scale == "log" else None,
                    )
                denominator = bands[:, 2 * col + 1, 2]
                ratio = np.divide(
                    bands[:, 2 * col, 2],
                    denominator,
                    out=np.full_like(denominator, np.nan),
                    where=denominator > 0,
                )
                axes[1, col].plot(x, ratio, color="#d62728")
                axes[1, col].axhline(1, color="#1565c0")
                axes[0, col].set_title(policy)
                axes[0, col].set_ylabel("Fixed-panel Local BR-gap (ante)")
                axes[1, col].set_ylabel("DT / k-means mean")
                axes[0, col].legend(fontsize=8)
                if scale == "log":
                    axes[0, col].set_yscale("log")
                    axes[0, col].set_ylim(bottom=0.001)
                else:
                    axes[0, col].set_ylim(bottom=0)
            for ax in axes.flat:
                ax.axvline(
                    result[label]["config"]["schedule"]["branch_hands"],
                    color="black",
                    linestyle="--",
                )
                ax.grid(alpha=0.2)
                ax.set_xlabel("Total training hands")
                if scale == "log":
                    ax.set_xscale("log")
                    ax.set_xlim(left=max(1, x[x > 0].min()))
            fig.suptitle(f"{label}: split-schedule ablation; red DT / blue k-means")
            fig.text(
                0.5,
                0.01,
                "Each point is a fixed-root panel, NOT a per-hand moving average. Bands: min-max / central 90%, not CIs.\n"
                "Local BR-gap is not exploitability; finite-particle max bias remains. Dashed line: fork.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 0.95))
            fig.savefig(output / f"{label}_{scale}.png", dpi=150)
            plt.close(fig)
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for col, policy in enumerate(("Current", "Average")):
            for (label, (x, bands)), color in zip(
                curves.items(), ("#d62728", "#008060", "#aa6500")
            ):
                axes[col].plot(x, bands[:, 2 * col, 2], color=color, label=label)
            x, bands = max(curves.values(), key=lambda item: item[0][-1])
            axes[col].plot(
                x, bands[:, 2 * col + 1, 2], color="#1565c0", label="matched k-means"
            )
            axes[col].set_title(policy)
            axes[col].set_xlabel("Total training hands")
            axes[col].set_ylabel("Fixed-panel mean Local BR-gap (ante)")
            axes[col].grid(alpha=0.2)
            axes[col].legend()
            if scale == "log":
                axes[col].set_xscale("log")
                axes[col].set_yscale("log")
                axes[col].set_xlim(left=max(1, x[x > 0].min()))
        fig.suptitle(
            "Split stop vs continue; shared 1M prefix if using the documented runs\nPanel means, not exploitability; distribution bands are in the per-branch figures"
        )
        fig.tight_layout()
        fig.savefig(output / f"comparison_{scale}.png", dpi=150)
        plt.close(fig)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REPORT output={output} matched_kmeans_parity=PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True, help="LABEL=RUN, one to three"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.runs:
        label, path = value.split("=", 1)
        if (
            not label
            or not all(c.isalnum() or c in "_-" for c in label)
            or label in runs
        ):
            parser.error("Use unique simple labels")
        runs[label] = Path(path).resolve()
    if not 1 <= len(runs) <= 3:
        parser.error("Compare one to three branches")
    compare(runs, args.output)


if __name__ == "__main__":
    main()
