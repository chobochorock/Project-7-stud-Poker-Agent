"""Compare shadow-split runs with their co-trained k-means arm and dynamic-tree references.

Usage (project root):
    python -m agents.cpp_mccfr.shadow_split.report --output agents/cpp_mccfr/data/shadow_split/<new> \
        --runs summary_z066=<run> own_z066=<run> ... \
        [--references dt_centered=agents/cpp_mccfr/data/dynamic_tree_validated/<run> ...] \
        [--lbrs summary_z066=<run>/<lbr dir> dt_centered=<dt lbr dir> ...] [--window 100] [--max-hands N]
Reads complete CSV lines (also during training). Writes local_br_gap_log.png (MA log-log, red
family = shadow arms, other colours = references, blue = k-means), ratio_log.png (arm / k-means
MA means), growth.png (leaves, splits, shadow MiB), panel.csv (128-root panel means), lbr.csv
and summary.json. Runs must share the training seed, eval seed and particles; references that
differ are labelled unmatched. Bands are sample ranges, NOT confidence intervals. Local BR-gap
and LBR are not exploitability.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from ..hand_recall_plot import COLORS, rolling_bands
from .run import DATA

POLICIES = ("current", "average")
PALETTE = ["#d62728", "#ff7f0e", "#8c564b", "#e377c2", "#9467bd", "#bcbd22", "#7f7f7f",
           "#17becf", "#2ca02c", "#000000", "#aec7e8", "#ffbb78"]


def read_rows(run: Path, name: str = "per_hand.csv", max_hands: int | None = None) -> list[dict]:
    text = (run / name).read_text(encoding="utf-8")
    text = text[: text.rfind("\n") + 1]
    rows = list(csv.DictReader(io.StringIO(text)))
    return [r for r in rows if max_hands is None or int(r["hands"]) <= max_hands]


def condition(run: Path) -> dict:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))["config"]
    return {name: config.get(name) for name in ("seed", "eval_seed", "particles", "panel_roots", "panel_every")}


def pairs(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items or []:
        label, _, path = item.partition("=")
        if not label or not path:
            raise ValueError(f"Use label=path, got {item}")
        result[label] = Path(path).resolve()
    return result


def panel_means(rows: list[dict]) -> dict[int, dict]:
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row["hands"]), []).append(row)
    return {
        hands: {
            column: float(np.mean([float(r[column]) for r in group]))
            for column in ("tree_current", "kmeans_current", "tree_average", "kmeans_average")
        } | {"roots": len(group)}
        for hands, group in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--references", nargs="*", default=[])
    parser.add_argument("--lbrs", nargs="*", default=[])
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--max-hands", type=int, help="truncate every trace (e.g. a reference still training)")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(DATA.resolve()):
        raise SystemExit(f"Use a new output directory under {DATA}")
    runs, references, lbrs = pairs(args.runs), pairs(args.references), pairs(args.lbrs)
    output.mkdir(parents=True)
    first = next(iter(runs.values()))
    base_condition = condition(first)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, summary = {}, {"runs": {}, "references": {}, "lbr": {}, "condition": base_condition}
    for group, items in (("runs", runs), ("references", references)):
        for label, run in items.items():
            rows = read_rows(run, max_hands=args.max_hands)
            matched = condition(run) == base_condition
            data[label] = dict(group=group, rows=rows, matched=matched)
            last = rows[-1]
            summary[group][label] = dict(
                run=str(run), matched_condition=matched, hands=int(last["hands"]),
                leaves=int(last["tree_leaves"]), kmeans_states=int(last["kmeans_states"]),
                splits=int(last["splits"]), tree_nodes=int(last["tree_nodes"]),
                kmeans_nodes=int(last["kmeans_nodes"]), peak_rss_mib=int(last["peak_rss_bytes"]) / 2**20,
                public_roots=int(last.get("public_roots", 0) or 0),
                shadow_mib=int(last.get("shadow_bytes", 0) or 0) / 2**20,
                panel=panel_means(read_rows(run, "panel.csv", args.max_hands)),
            )
    # The co-trained k-means arms must agree where hands overlap (same seed and code).
    kmeans = {label: [r["kmeans_current"] for r in item["rows"]] for label, item in data.items() if item["matched"]}
    shortest = min(len(v) for v in kmeans.values())
    identical = len({tuple(v[:shortest]) for v in kmeans.values()}) == 1
    summary["kmeans_arms_identical_over_common_hands"] = identical

    window = args.window
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    ratio_figure, ratio_axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    floor = 1e-3
    for col, policy in enumerate(POLICIES):
        drawn_kmeans = False
        for index, (label, item) in enumerate(data.items()):
            rows = item["rows"]
            if len(rows) < window:
                continue
            x = np.array([int(r["hands"]) for r in rows])[window - 1 :]
            tree = rolling_bands(np.array([float(r["tree_" + policy]) for r in rows]), window)[:, 2]
            base = rolling_bands(np.array([float(r["kmeans_" + policy]) for r in rows]), window)[:, 2]
            colour = PALETTE[index % len(PALETTE)]
            style = "-" if item["group"] == "runs" else "--"
            suffix = "" if item["matched"] else " (unmatched)"
            axes[col].plot(x, np.maximum(tree, floor), style, color=colour, linewidth=1.1, label=label + suffix)
            if not drawn_kmeans and item["matched"]:
                axes[col].plot(x, np.maximum(base, floor), color=COLORS["baseline"], linewidth=1.4,
                               label="hard256 k-means (co-trained)")
                drawn_kmeans = True
            ratio = np.divide(tree, base, out=np.full_like(tree, np.nan), where=base > 0)
            ratio_axes[col].plot(x, ratio, style, color=colour, linewidth=1.0, label=label + suffix)
        for axis in (axes[col], ratio_axes[col]):
            axis.set_xscale("log")
            axis.grid(alpha=0.2)
            axis.set_xlabel("Completed hands")
        axes[col].set_yscale("log")
        axes[col].set_title(f"{policy}: MA{window} mean Local BR-gap (ante)")
        ratio_axes[col].set_yscale("log")
        ratio_axes[col].axhline(1, color=COLORS["baseline"])
        ratio_axes[col].set_title(f"{policy}: arm / its k-means (MA{window} means)")
        axes[col].legend(fontsize=7)
        ratio_axes[col].legend(fontsize=7)
    note = ("One-step gap at the first 5th-street decision, 32 particles; MA means of per-hand values; "
            f"drawn values below {floor} ante clipped for log display only. Not exploitability, not LBR.")
    figure.text(0.5, 0.01, note, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output / "local_br_gap_log.png", dpi=140)
    ratio_figure.tight_layout()
    ratio_figure.savefig(output / "ratio_log.png", dpi=140)
    plt.close("all")

    growth, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for index, (label, item) in enumerate(data.items()):
        rows = item["rows"][:: max(1, len(item["rows"]) // 2000)]
        x = [int(r["hands"]) for r in rows]
        colour = PALETTE[index % len(PALETTE)]
        axes[0].plot(x, [int(r["tree_leaves"]) for r in rows], color=colour, label=label)
        axes[1].plot(x, [int(r["splits"]) for r in rows], color=colour, label=label)
        axes[2].plot(x, [int(r.get("shadow_bytes", 0) or 0) / 2**20 for r in rows], color=colour, label=label)
    first_rows = next(iter(data.values()))["rows"][:: max(1, len(next(iter(data.values()))["rows"]) // 2000)]
    axes[0].plot([int(r["hands"]) for r in first_rows], [int(r["kmeans_states"]) for r in first_rows],
                 color=COLORS["baseline"], label="k-means states")
    for axis, title in zip(axes, ("policy tables (leaves / states)", "splits", "shadow MiB (training only)")):
        axis.set_title(title)
        axis.set_xlabel("Completed hands")
        axis.grid(alpha=0.2)
        axis.set_yscale("symlog")
        axis.legend(fontsize=7)
    growth.tight_layout()
    growth.savefig(output / "growth.png", dpi=140)
    plt.close("all")

    with (output / "panel.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "group", "hands", "roots", "tree_current", "kmeans_current", "tree_average", "kmeans_average"])
        for group in ("runs", "references"):
            for label, item in summary[group].items():
                for hands, values in item["panel"].items():
                    writer.writerow([label, group, hands, values["roots"], values["tree_current"],
                                     values["kmeans_current"], values["tree_average"], values["kmeans_average"]])
    with (output / "lbr.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "training_hands", "seeds", "pairs", "policy", "mean", "ci_low", "ci_high"])
        for label, directory in lbrs.items():
            meta = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            hands = meta.get("training_hands")
            if hands is None:
                hands = json.loads((directory / str(meta["seeds"][0]) / "evaluation.json").read_text())["training_hands"]
            summary["lbr"][label] = dict(directory=str(directory), training_hands=hands, seeds=meta["seeds"],
                                         policies={k: v["mean"] for k, v in meta["policies"].items()},
                                         paired=meta["paired_model_minus_baseline"])
            rows = list(meta["policies"].items()) + [(f"paired_{k}", v) for k, v in meta["paired_model_minus_baseline"].items()]
            for policy, stats in rows:
                writer.writerow([label, hands, " ".join(map(str, meta["seeds"])), meta["pairs_per_seed"], policy,
                                 stats["mean"], stats["bootstrap_ci95"][0], stats["bootstrap_ci95"][1]])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {label: {m: v for m, v in item.items() if m != "panel"} for label, item in summary[k].items()}
                      for k in ("runs", "references")}, indent=1))
    print("kmeans_arms_identical_over_common_hands", identical)


if __name__ == "__main__":
    main()
