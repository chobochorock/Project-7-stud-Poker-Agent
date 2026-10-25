"""Summarize a completed EVPA pilot without rerunning training.

Usage: python -m agents.evpa.report agents/evpa/data/<run>
Output: RESULTS.md and paired_comparison.json alongside the original measurements.
Uncertainty: paired bootstrap over held-out public roots after averaging clustering
seeds. It does not cover training-split or network-training uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def plot_gap_bands(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = {"card_kmeans": {}, "ev_network": {}}
    with (folder / "online_gap.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] in traces:
                key = (row["root"], row["seed"])
                traces[row["arm"]].setdefault(key, []).append(
                    float(row["current_preupdate_root_local_gap"])
                )
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), constrained_layout=True, sharex=True
    )
    for arm, color in [("card_kmeans", "#1768bf"), ("ev_network", "#cf303b")]:
        matrix = np.asarray(list(traces[arm].values()))
        window = min(100, matrix.shape[1])
        smooth = np.convolve(
            matrix.mean(axis=0), np.ones(window) / window, mode="valid"
        )
        for ax in axes:
            ax.plot(
                np.arange(window, matrix.shape[1] + 1), smooth, color=color, label=arm
            )
            for start in range(0, matrix.shape[1], window):
                end = min(start + window, matrix.shape[1])
                values = matrix[:, start:end].ravel()
                low, q05, q95, high = np.quantile(values, [0, 0.05, 0.95, 1])
                if ax is axes[0]:
                    ax.fill_between(
                        [start + 1, end], low, high, color=color, alpha=0.06
                    )
                    ax.fill_between([start + 1, end], q05, q95, color=color, alpha=0.17)
            ax.grid(alpha=0.2)
            ax.set_ylabel("Root Local BR-gap (antes)")
    axes[0].set_title(
        "100-iteration boxes: min/max (light), central 90% (dark), MA100 mean (line)"
    )
    axes[0].legend()
    axes[1].set(
        xlabel="Full CFR iterations", title="Same MA100 means without the bands"
    )
    fig.suptitle(
        "Held-out mini-stud roots: current policy before each update\n"
        "Bands mix root / clustering-seed / within-window variation; NOT confidence intervals"
    )
    fig.savefig(folder / "local_gap_bands.png", dpi=160)
    plt.close(fig)


def summarise(folder):
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "completed":
        raise ValueError("run is not complete")
    with (folder / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if int(r["iteration"]) == run["iterations"]
        ]
    with (folder / "pruning_audit.csv").open(encoding="utf-8", newline="") as stream:
        audits = list(csv.DictReader(stream))
    with (folder / "teacher.csv").open(encoding="utf-8", newline="") as stream:
        teachers = list(csv.DictReader(stream))
    arms = [
        "exact",
        "card_kmeans",
        "ev_oracle",
        "ev_network",
        "ev_network_pruned",
        "ev_prune_only",
    ]
    roots = sorted({int(r["root"]) for r in rows})

    def root_means(arm, field):
        return np.array(
            [
                np.mean(
                    [
                        float(r[field])
                        for r in rows
                        if r["arm"] == arm and int(r["root"]) == root
                    ]
                )
                for root in roots
            ]
        )

    baseline = root_means("card_kmeans", "average_exploitability")
    rng = np.random.default_rng(20261017)
    resamples = rng.integers(0, len(roots), size=(10000, len(roots)))
    comparisons = {}
    for arm in arms:
        difference = root_means(arm, "average_exploitability") - baseline
        boot = difference[resamples].mean(axis=1)
        comparisons[arm] = {
            "paired_difference_vs_card_kmeans": float(difference.mean()),
            "paired_bootstrap_95_interval": np.quantile(boot, [0.025, 0.975]).tolist(),
            "roots_better_than_card_kmeans": int((difference < -1e-10).sum()),
            "roots": len(roots),
        }
    (folder / "paired_comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    summary = run["summary"]
    lines = [
        "# EVPA-inspired mini-stud pilot results",
        "",
        "This is a component pilot, not the full EVPA paper reproduction or a 52-card seven-stud result.",
        "",
        f"- Public-root split: {len(run['split']['train'])} train / {len(run['split']['valid'])} validation / {len(roots)} test; seed {run['seed']}.",
        f"- Teacher: {run['teacher_iterations']} full signed CFR iterations per root. Online: {run['iterations']} iterations per case; K={run['k']} per player.",
        f"- Models: {run['members']} independently initialized {run['hidden']}x{run['hidden']} MLPs; {run['steps']} optimizer steps each.",
        "- Prefix policy is forced CHECK with H4 discard/reveal. Test roots are equally weighted, not whole-game on-policy reach-weighted.",
        "- Each learned/pruned evaluation uses a frozen network; test teacher values are diagnostic/oracle-only.",
        "",
        "## Final measurements",
        "",
        "All losses are in antes; lower is better. Average and current policies are separate.",
        "",
        "| Arm | Average exploitability | Current exploitability | Average root gap | Rows | Kept actions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        s = summary[arm]
        lines.append(
            f"| {arm} | {s['average_exploitability']:.6f} | {s['current_exploitability']:.6f} | {s['average_root_local_gap']:.6f} | {s['bucket_rows']:.1f} | {s['retained_fraction']:.1%} |"
        )
    lines += [
        "",
        "## Paired differences",
        "",
        "Difference = arm minus matched card k-means. Intervals resample test roots after averaging clustering seeds.",
        "They do not include training-split or neural-training variation and are exploratory, not full-game guarantees.",
        "",
        "| Arm | Difference | 95% paired bootstrap interval | Better roots |",
        "|---|---:|---|---:|",
    ]
    for arm in arms:
        c = comparisons[arm]
        lo, hi = c["paired_bootstrap_95_interval"]
        lines.append(
            f"| {arm} | {c['paired_difference_vs_card_kmeans']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {c['roots_better_than_card_kmeans']}/{len(roots)} |"
        )
    lines += ["", "## Prediction and pruning audit", ""]
    for split, values in run["ev_errors"].items():
        lines.append(
            f"- {split}: EV RMSE={values['rmse_pot_units']:.6f} pot units, MAE={values['mae_pot_units']:.6f}; fraction inside ensemble min/max={values['ensemble_range_coverage']:.1%} (not a confidence interval)."
        )
    for arm in ("ev_network_pruned", "ev_prune_only"):
        rate = np.mean(
            [
                float(r["reference_best_excluded_fraction"])
                for r in audits
                if r["arm"] == arm
            ]
        )
        lines.append(
            f"- {arm}: reference-best action excluded at {rate:.1%} of original infosets (tolerance 0.01 root pots; finite teacher reference)."
        )
    teacher_loss = np.asarray(
        [float(r["exploitability"]) for r in teachers if r["split"] == "test"]
    )
    lines.append(
        f"- Held-out teacher exploitability: mean={teacher_loss.mean():.6f}, max={teacher_loss.max():.6f} antes. Labels are not exact Nash values."
    )
    lines += [
        "",
        "## Costs and limitations",
        "",
        f"- Teacher/root/label phase: {run['teacher_seconds']:.2f} s; network fitting/checkpoint phase: {run['network_training_seconds']:.2f} s; full pilot elapsed: {run['elapsed_seconds']:.2f} s.",
        f"- EV inference checkpoint: {(folder/'ev_models.pt').stat().st_size / 2**20:.3f} MiB; offline label archive: {(folder/'labels.npz').stat().st_size / 2**20:.3f} MiB.",
        "- Learning curves match ONLINE iterations only. Teacher and network costs must be amortized or charged separately for an efficiency comparison.",
        "- Pruning masks restrict policies, but the pilot still computes the full matrix tree. No pruning speedup is claimed.",
        "- Full original actions remain available to the exact BR evaluator. Bucket-legal masks are conservative unions, unlike a verbatim EVPA implementation.",
        "- Extra original-type raw-regret arrays are diagnostic storage, not compressed policy storage. Maps, masks, joint matrices and networks also consume memory.",
        "- Root gap is a one-decision deviation at the first actor, not exploitability or an LBR match. L/C/H are diagnostics, not exploitability lower bounds.",
        "",
        "## Artifacts",
        "",
        "![Comparison](comparison.png)",
        "",
        "`online_gap.csv` contains every iteration's pre-update root gap. `metrics.csv` contains exact checkpoint metrics.",
        "`ev_models.pt` and `policy_root*.npz` are inference exports, not resumable optimizer/experiment checkpoints.",
        "See [configuration and executable command](../../README.md).",
        "",
    ]
    lines += [
        "![Local gap bands](local_gap_bands.png)",
        "",
        "The 100-iteration boxes show min/max and central 90% across held-out roots, clustering seeds and iterations within each box; lines are MA100 means, not confidence intervals.",
        "",
    ]
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    plot_gap_bands(folder)
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(summarise(parser.parse_args().folder), indent=2))
