"""Shared k-means overlays and frozen-checkpoint evaluation (NO training).

Usage: python -m agents.autoencoder_abstraction.comparison --help
Default: evaluate legacy seed-7 hard256 checkpoints at 10k intervals to 1M,
using AE/VQ's roots, particles, seeds and LBR settings. Never overwrite a run.
Plots use log-log gaps, gap/baseline ratios, and measured LBR endpoints.
Slopes describe observed checkpoints, not asymptotes or forecast crossings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np

from agents.autoencoder_abstraction.experiment import FAMILY, ROOT
from agents.state_action_embedding.experiment import write_json


BASELINE = FAMILY / "data/kmeans_reference_matched_20260924"
LEGACY = ROOT / "agents/lightgbm_regret_ensemble/data/seven_stud_1m_seed7_20260919_204332"
ATLAS = ROOT / "agents/cpp_mccfr/data/power256_selfplay100m_eps20_v1.bin"
GAP_FIELDS = ("eval_roots", "eval_particles", "eval_seed")
LBR_FIELDS = ("lbr_pairs", "lbr_particles", "eval_seed")


def curve(summary: dict) -> dict:
    """Convert a training summary into the common measured-curve format."""
    return {"local_gap": summary["local_gap"], "lbr": [
        dict(hands=int(summary["training"]["hands"]), policy=p, **metric)
        for p, metric in summary["lbr"].items()
    ]}


def log_trend(metrics: list[dict], start: int = 100000, stop: int = 1000000) -> dict:
    """OLS slope on observed positive checkpoint means, without extrapolation."""
    selected = sorted((g for g in metrics if start <= g["hands"] <= stop
                       and np.isfinite(g["mean"]) and g["mean"] > 0), key=lambda g: g["hands"])
    if len(selected) < 3:
        return {"slope": None, "checkpoints": len(selected)}
    x = np.array([g["hands"] for g in selected], dtype=float)
    y = np.array([g["mean"] for g in selected])
    if len(np.unique(x)) != len(x):
        raise ValueError("Duplicate training budgets in a curve")
    slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
    return {"slope": slope, "factor_per_doubling": 2 ** slope,
            "start_hands": int(x[0]), "end_hands": int(x[-1]),
            "endpoint_reduction_percent": float(100 * (1 - y[-1] / y[0])),
            "checkpoints": len(selected)}


def gap_ratios(metrics: list[dict], baseline: list[dict]) -> list[dict]:
    """Exact budget matches only: no interpolation or division by zero."""
    reference = {g["hands"]: g["mean"] for g in baseline if np.isfinite(g["mean"]) and g["mean"] > 0}
    return [dict(hands=g["hands"], ratio=g["mean"] / reference[g["hands"]])
            for g in metrics if g["hands"] > 0 and g["hands"] in reference
            and np.isfinite(g["mean"]) and g["mean"] > 0]


def plot_comparison(runs: dict[str, dict], evaluation: dict, output: Path,
                    baseline_dir: Path = BASELINE, *,
                    trend_start: int = 100000, trend_stop: int = 1000000) -> dict:
    """Always include the saved baseline; warn visibly if absent/incompatible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = dict(runs)
    has_archived_points = any(run.get("points_only", False) for run in runs.values())
    maximum = max(g["hands"] for run in runs.values() for g in run["local_gap"])
    baseline_path = baseline_dir / "summary.json"
    metadata_path = baseline_dir / "run.json"
    gap_matched = lbr_matched = False
    if baseline_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            raise ValueError("Incomplete baseline evaluation")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        runs = {"k-means (legacy)": baseline, **runs}
        gap_matched = all(evaluation[f] == metadata["arguments"][f] for f in GAP_FIELDS)
        lbr_matched = all(evaluation[f] == metadata["arguments"][f] for f in LBR_FIELDS)
        note = "Matched evaluation; legacy training data and seat partition differ"
        if not gap_matched or not lbr_matched:
            note = "WARNING: reference evaluation settings differ; no matched ratio for mismatched metric"
    else:
        note = "WARNING: k-means reference missing; run python -m agents.autoencoder_abstraction.comparison"
    if note.startswith("WARNING"):
        warnings.warn(note, stacklevel=2)
    result = {"baseline": str(baseline_dir), "gap_evaluation_matched": gap_matched,
              "lbr_evaluation_matched": lbr_matched, "note": note, "trends": {}, "gap_ratios": {}}
    colors = {"k-means (legacy)": "#222222", "AE": "#2563eb", "VQ-VAE": "#c2410c",
              "K512 Mem16 (archived)": "#16816b", "K64 (archived)": "#8a4ca8"}
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    for col, policy in enumerate(("current", "average")):
        reference = [g for g in runs.get("k-means (legacy)", {}).get("local_gap", [])
                     if g["policy"] == policy and 0 < g["hands"] <= maximum]
        for label, run in runs.items():
            color = colors.get(label, "#16816b")
            metrics = sorted((g for g in run["local_gap"] if g["policy"] == policy
                              and 0 < g["hands"] <= maximum), key=lambda g: g["hands"])
            positive = [g for g in metrics if np.isfinite(g["mean"]) and g["mean"] > 0]
            x = [g["hands"] for g in positive]
            style = "--" if label == "k-means (legacy)" else "-"
            key = f"{label}/{policy}"
            result["trends"][key] = log_trend(metrics, trend_start, trend_stop)
            slope = result["trends"][key]["slope"]
            trend_label = label if slope is None else f"{label} (slope {slope:.2f})"
            points_only = run.get("points_only", False) or len(x) == 1
            means = np.array([g["mean"] for g in positive])
            lower = np.array([g["ci95_low"] for g in positive])
            upper = np.array([g["ci95_high"] for g in positive])
            if points_only:
                axes[0, col].errorbar(x, means,
                    yerr=[means - np.where(lower > 0, lower, np.nan), upper - means],
                    color=color, fmt="D", linestyle="none", capsize=3, label=trend_label)
            else:
                axes[0, col].plot(x, means, color=color, linestyle=style, label=trend_label)
                axes[0, col].fill_between(x, np.where(lower > 0, lower, np.nan),
                                         upper, color=color, alpha=.07)
            if gap_matched and label != "k-means (legacy)":
                ratio = gap_ratios(metrics, reference)
                result["gap_ratios"][key] = ratio
                if ratio:
                    axes[1, col].plot([g["hands"] for g in ratio], [g["ratio"] for g in ratio],
                                      label=label, color=color)
            endpoints = sorted((g for g in run["lbr"] if g["policy"] == policy
                                and 0 < g["hands"] <= maximum), key=lambda g: g["hands"])
            # LBR can be negative: keep its y-axis linear instead of hiding signs.
            axes[2, col].errorbar([g["hands"] for g in endpoints], [g["mean"] for g in endpoints],
                yerr=[g["ci95_high"] - g["mean"] for g in endpoints],
                color=color, marker="o", linestyle="none", capsize=5, label=label)
        axes[0, col].set(title=f"{policy.title()} policy: Local BR-gap (log-log)", ylabel="Root gap (ante, log scale)")
        axes[0, col].set_yscale("log")
        axes[1, col].axhline(1, color="#222222", linestyle="--", label="K256 = 1")
        axes[1, col].set_yscale("log")
        axes[1, col].set(title="Gap / legacy K256: shared measured budgets only", ylabel="Gap / K256 gap (log scale)")
        if not gap_matched:
            axes[1, col].text(.5, .5, "No matched baseline ratio", transform=axes[1, col].transAxes, ha="center")
        axes[2, col].set(title="LBR: measured endpoints only (95% CI)", ylabel="LBR profit (ante/hand)")
        axes[2, col].axhline(0, color="#777777", linewidth=.7)
        for axis in axes[:, col]:
            axis.set_xscale("log")
            axis.set_xlabel("Hands / archived budget label (log scale)" if has_archived_points
                            else "Training hands (log scale)")
            axis.grid(alpha=.2, which="both")
            axis.legend(fontsize=8)
    figure.suptitle(f"Abstraction / k-means comparison\n{note}", fontsize=11)
    figure.text(.5, .005, "0 hands / nonpositive gaps omitted only on log axes; no smoothing or extrapolation.\n"
                f"Slopes: {trend_start:,}-{trend_stop:,} log-log fit, not forecasts. Local gap and LBR are NOT exploitability.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, .94))
    figure.savefig(output, dpi=170)
    plt.close(figure)
    write_json(output.with_suffix(".json"), result)
    return result


def main():
    from agents.autoencoder_abstraction.run_cfr import SOURCES, build, mean_ci, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=BASELINE)
    parser.add_argument("--hands", type=int, default=1000000)
    args = parser.parse_args()
    if args.hands < 10000 or args.hands > 1000000 or args.hands % 10000:
        parser.error("--hands must be a multiple of 10000, in [10000, 1000000]")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    reference = json.loads((FAMILY / "data/ae_cfr_1m/run.json").read_text(encoding="utf-8"))
    settings = {f: reference["arguments"][f] for f in set(GAP_FIELDS + LBR_FIELDS)}
    settings.update(hands=args.hands, eval_every=10000, seed=7)
    legacy = json.loads((LEGACY / "config.json").read_text(encoding="utf-8"))
    assert sha256(ATLAS) == legacy["source_sha256"][ATLAS.name]
    rules = ROOT / "environments/seven_stud/stud_rules.hpp"
    assert sha256(rules) == legacy["source_sha256"][rules.name]
    hashes = {str(LEGACY / f"hard256_{hand}.bin"): sha256(LEGACY / f"hard256_{hand}.bin")
              for hand in range(10000, args.hands + 1, 10000)}
    binary = build()
    subprocess.run([str(binary), "--engine-test"], check=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"arguments": settings, "status": "evaluating", "evaluation_only": True,
                "command": subprocess.list2cmdline([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
                "checkpoint_sha256": hashes, "atlas_sha256": sha256(ATLAS),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES},
                "reference_run": str(FAMILY / "data/ae_cfr_1m"),
                "limits": "Legacy actor-shared keys and epsilon-0.2 self-play atlas, not AE/VQ low-fold data/seat keys"}
    write_json(args.out_dir / "run.json", metadata)
    command = [str(binary), "--evaluate-kmeans", str(ATLAS), str(LEGACY), str(args.out_dir / "solver"),
               str(args.hands), "10000", *[str(settings[f]) for f in
               ("eval_roots", "eval_particles", "lbr_pairs", "lbr_particles", "eval_seed")]]
    started = time.perf_counter()
    with (args.out_dir / "run.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
    gap = np.genfromtxt(args.out_dir / "solver/local_gap.csv", delimiter=",", names=True)
    summary = {"local_gap": [], "lbr": []}
    for hand in np.unique(gap["hands"]):
        for policy in ("current", "average"):
            summary["local_gap"].append(dict(hands=int(hand), policy=policy,
                **mean_ci(gap[policy][gap["hands"] == hand])))
    for folder in sorted((args.out_dir / "solver").glob("lbr_*")):
        hand = int(folder.name.removeprefix("lbr_"))
        pairs = np.genfromtxt(folder / "lbr_pairs.csv", delimiter=",", names=True)
        for policy in ("current", "average"):
            values = (pairs[f"{policy}_seat0"] + pairs[f"{policy}_seat1"]) / 2
            summary["lbr"].append(dict(hands=hand, policy=policy, **mean_ci(values)))
    write_json(args.out_dir / "summary.json", summary)
    metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
    write_json(args.out_dir / "run.json", metadata)
    print(f"RESULTS {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
