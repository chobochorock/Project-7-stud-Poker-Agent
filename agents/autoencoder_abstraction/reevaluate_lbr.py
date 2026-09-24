"""Evaluate frozen VQ 10M and archived k-means on new paired LBR deal streams.

Usage: python -m agents.autoencoder_abstraction.reevaluate_lbr --out-dir NEW_DIR
Defaults: 5 new seeds x 2000 seat-swapped deal pairs per current/average policy.
No training or adaptive stopping. Original seed 307 is excluded from results.
LBR profit is ante/hand, not exact exploitability. See CFR.md for limitations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .analyze_1m import rows
from .analyze_10m import paired_values
from .run_cfr import FAMILY, ROOT, SOURCES, build, mean_ci, sha256
from agents.state_action_embedding.experiment import write_json


REVIEW = FAMILY / "data/vq_10m_review_20260924"
VQ = FAMILY / "data/vq_cfr_10m"
SEEDS = [2026092401, 2026092402, 2026092403, 2026092404, 2026092405]
CONTROLS = {
    "memory16_10m_current": "K512 10M",
    "memory16_30m_current": "K512 30M",
    "power64_100m_current": 'K64 "100M"',
}


def summarize(values: np.ndarray, bootstrap: int = 2000) -> dict:
    """Equal-size evaluation seeds x paired-deal means; no outlier removal."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite seed x deal-pair matrix")
    result = mean_ci(values.ravel())
    rng = np.random.default_rng(20260924)
    estimates = []
    for first in range(0, bootstrap, 50):
        count = min(50, bootstrap - first)
        means = np.zeros(count)
        for block in values:
            indices = rng.integers(len(block), size=(count, len(block)))
            means += block[indices].mean(axis=1) / len(values)
        estimates.extend(means)
    result["bootstrap_ci95"] = np.quantile(estimates, [.025, .975]).tolist()
    result["seed_means"] = values.mean(axis=1).tolist()
    result["min_pair"] = float(values.min())
    result["max_pair"] = float(values.max())
    result["pairs_absolute_at_least_100"] = int((np.abs(values) >= 100).sum())
    result["largest_pair_contribution_to_mean"] = float(
        values.ravel()[np.argmax(np.abs(values))] / values.size
    )
    return result


def plot(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["models"])
    colors = ["#c94712", "#15826b", "#3264a8", "#854ba3"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, policy in zip(axes, ("current", "average")):
        for index, label in enumerate(labels):
            metric = summary["models"][label][policy]
            mean = metric["mean"]
            low, high = metric["bootstrap_ci95"]
            axis.errorbar(index, mean, yerr=[[mean-low], [high-mean]],
                          fmt="o", capsize=5, color=colors[index], markersize=7)
            axis.scatter(index + np.linspace(-.13, .13, len(metric["seed_means"])),
                         metric["seed_means"], marker="x", color=colors[index], alpha=.6)
        axis.axhline(1, color="#777777", linestyle="--", label="Reference = 1 ante/hand")
        axis.axhline(0, color="#aaaaaa", linewidth=.8)
        axis.set_xticks(range(len(labels)), labels, rotation=15)
        axis.set_title(f"{policy.title()} policy")
        axis.set_ylabel("LBR attacker profit (ante/hand); lower is better")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Frozen-policy LBR: {len(summary['seeds'])} new evaluation seeds x "
        f"{summary['pairs_per_seed']:,} deal pairs\n"
        "Dots: pooled means / 95% paired bootstrap; crosses: seed means\n"
        "Same deals and 64 particles; no retraining; not exact exploitability",
        fontsize=12,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if (args.pairs < 2 or len(set(args.seeds)) != len(args.seeds)
            or any(s < 0 or s >= 2**64 or s == 307 for s in args.seeds)):
        raise ValueError("Use positive pair count and distinct new uint64 seeds")

    previous = json.loads((REVIEW / "run.json").read_text(encoding="utf-8"))
    original = json.loads((VQ / "run.json").read_text(encoding="utf-8"))
    vq_model, vq_atlas = VQ / "solver/policy_10000000.bin", VQ / "neural_atlas.bin"
    models = {"VQ 10M": dict(model=str(vq_model), atlas=str(vq_atlas),
                            bucket="neural", budget_label=10000000)}
    for key, label in CONTROLS.items():
        models[label] = previous["controls"][key]
    input_hashes = {str(Path(item[field])): sha256(Path(item[field]))
                    for item in models.values() for field in ("model", "atlas")}
    assert input_hashes[str(vq_atlas)] == original["atlas_sha256"]
    for key in CONTROLS:
        for path, digest in previous["controls"][key]["input_sha256"].items():
            assert input_hashes[path] == digest

    binary = build()
    subprocess.run([str(binary), "--engine-test"], check=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", evaluation_only=True, seeds=args.seeds,
                    pairs_per_seed=args.pairs, lbr_particles=64, gap_roots=128,
                    gap_particles=32, bootstrap_replicates=2000,
                    primary="VQ 10M current LBR; fixed budget independent of outcome",
                    command=subprocess.list2cmdline([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
                    input_sha256=input_hashes,
                    source_sha256={str(p.relative_to(ROOT)): sha256(p)
                                   for p in [*SOURCES, Path(__file__).resolve()]},
                    models=models, runs=[])
    write_json(args.out_dir / "run.json", metadata)
    started = time.perf_counter()

    # Reproduce the old neural evaluation before using the new loading branch.
    smoke = args.out_dir / "reproduce_seed307"
    command = [str(binary), "--evaluate-checkpoint", str(vq_atlas), str(vq_model),
               str(smoke), "10000000", "neural", "128", "32", "500", "64", "307"]
    with (args.out_dir / "reproduce_seed307.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert rows(smoke / "lbr_pairs.csv") == rows(VQ / "solver/lbr_pairs.csv")
    assert rows(smoke / "local_gap.csv") == [g for g in rows(VQ / "solver/local_gap.csv")
                                             if int(g["hands"]) == 10000000]
    metadata["seed307_exact_reproduction"] = True
    write_json(args.out_dir / "run.json", metadata)
    excluded = {g["deal_seed"] for g in rows(smoke / "lbr_pairs.csv")}
    streams, values = {}, {label: {p: [] for p in ("current", "average")} for label in models}
    for seed in args.seeds:
        for index, (label, model) in enumerate(models.items()):
            output = args.out_dir / f"seed_{seed}_model_{index}"
            command = [str(binary), "--evaluate-checkpoint", model["atlas"], model["model"],
                       str(output), str(model["budget_label"]), model["bucket"],
                       "128", "32", str(args.pairs), "64", str(seed)]
            print(f"EVALUATING model={label} seed={seed} pairs={args.pairs}", flush=True)
            before = time.perf_counter()
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            pair_rows = rows(output / "lbr_pairs.csv")
            assert len(pair_rows) == args.pairs
            stream = [(g["pair"], g["deal_seed"]) for g in pair_rows]
            if seed not in streams:
                ids = {deal for _, deal in stream}
                assert len(ids) == args.pairs and not ids.intersection(excluded)
                excluded.update(ids)
                streams[seed] = stream
            assert stream == streams[seed]
            for policy in ("current", "average"):
                payoffs = paired_values(pair_rows, policy)
                assert np.isfinite(payoffs).all() and (np.abs(payoffs) <= 1000).all()
                values[label][policy].append(payoffs)
            metadata["runs"].append(dict(model=label, seed=seed, output=str(output),
                                         command=command, seconds=time.perf_counter()-before))
            write_json(args.out_dir / "run.json", metadata)
            print(f"COMPLETE model={label} seed={seed} runs={len(metadata['runs'])}/{len(models)*len(args.seeds)}", flush=True)
    summary = dict(seeds=args.seeds, pairs_per_seed=args.pairs, models={},
                   paired_vq_minus_control={}, original_seed307_excluded=True)
    for label in models:
        summary["models"][label] = {}
        for policy in ("current", "average"):
            values[label][policy] = np.array(values[label][policy])
            summary["models"][label][policy] = summarize(values[label][policy])
        if label != "VQ 10M":
            summary["paired_vq_minus_control"][label] = {
                policy: summarize(values["VQ 10M"][policy] - values[label][policy])
                for policy in ("current", "average")
            }
    assert all(sha256(Path(path)) == digest for path, digest in input_hashes.items())
    write_json(args.out_dir / "summary.json", summary)
    plot(summary, args.out_dir / "lbr_comparison.png")
    metadata.update(status="complete", wall_seconds=time.perf_counter()-started,
                    model_hashes_unchanged=True, unique_matched_deal_streams=True)
    write_json(args.out_dir / "run.json", metadata)
    print(f"RESULTS {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
