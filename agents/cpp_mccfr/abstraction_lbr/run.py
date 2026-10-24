"""Build frozen adapters and play paired Policy-LBR matches (no training).

Usage (project root): python -m agents.cpp_mccfr.abstraction_lbr.run --output NEW_DIR
Defaults: 2 new seeds x250 deal pairs x2 seats = 1,000 hands per policy.
Current/average, matched k-means and uniform controls are all included.
Hand-recall must be paused. Model files are never rewritten or copied.
See README.md for metric limits and the mechanically generated include files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from ..hand_recall_disk import metadata, read_manifest

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PROJECT = FAMILY.parents[1]
RUNS = {
    "limited_dt": ("tree", FAMILY / "data/dynamic_tree/seed111"),
    "unlimited_dt": ("tree", FAMILY / "data/dynamic_tree_unlimited/seed111_1m"),
    "hand_recall": ("disk", FAMILY / "data/hand_recall_kmeans256_disk_1m_seed111"),
}
NAMES = ("model_current", "baseline_current", "model_average", "baseline_average", "uniform")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_hash(handle)


def stream_hash(handle) -> str:
    value = hashlib.sha256()
    for chunk in iter(lambda: handle.read(2**20), b""):
        value.update(chunk)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build(output: Path) -> dict:
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ not found")
    sources = {
        "tree": FAMILY / "dynamic_tree_unlimited/solver.cpp",
        "disk": FAMILY / "hand_recall_disk.cpp",
    }
    directory = output / "build"
    directory.mkdir()
    binaries = {}
    prefix = Path(compiler).resolve().parents[1] / "opt"
    for kind, source in sources.items():
        text = source.read_text(encoding="utf-8")
        signature = "int main(int argc, char** argv)"
        include = '#include "../hand_recall.cpp"' if kind == "tree" else '#include "hand_recall.cpp"'
        if text.count(signature) != 1 or text.count(include) != 1:
            raise ValueError("Engine entrypoint/include changed; inspect before rebuilding")
        text = text.replace(signature, f"int unused_{kind}_training_main(int argc, char** argv)")
        text = text.replace(include, f'#include "{(FAMILY / "hand_recall.cpp").as_posix()}"')
        (directory / f"{kind}_engine.inc").write_text(text, encoding="utf-8")
        binary = directory / f"evaluate_{kind}.exe"
        command = [compiler, "-O3", "-std=c++17", "-DNOMINMAX", "-static-libstdc++", "-static-libgcc",
                   "-I", str(directory), str(HERE / f"{kind}.cpp"), "-o", str(binary), "-lpsapi"]
        if kind == "disk":
            command += ["-I", str(prefix / "include"), str(prefix / "lib/libsqlite3.a")]
        print(f"BUILD kind={kind}", flush=True)
        with (directory / f"{kind}_build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[kind] = binary
    return binaries


def estimate(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite seed x pair matrix")
    mean = float(values.mean())
    se = float(np.sqrt(np.sum(values.var(axis=1, ddof=1) / values.shape[1])) / len(values))
    rng = np.random.default_rng(20261004)
    boot = np.zeros(2000)
    for seed_values in values:
        for first in range(0, len(boot), 100):
            idx = rng.integers(len(seed_values), size=(100, len(seed_values)))
            boot[first:first + 100] += seed_values[idx].mean(axis=1) / len(values)
    return dict(mean=mean, se=se, ci95=[mean - 1.96*se, mean + 1.96*se],
                bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
                seed_means=values.mean(axis=1).tolist(), min_pair=float(values.min()),
                max_pair=float(values.max()), pairs_abs_ge_100=int((np.abs(values) >= 100).sum()))


def analyze(output: Path, seeds: list[int], pairs: int) -> dict:
    samples, streams, result = {}, {}, {"models": {}, "seeds": seeds, "pairs_per_seed": pairs}
    training_hands = {}
    for label in RUNS:
        samples[label] = {name: [] for name in NAMES}
        for seed in seeds:
            directory = output / f"{label}_{seed}"
            meta = json.loads((directory / "evaluation.json").read_text())
            assert meta["pairs"] == pairs and meta["seed"] == seed
            assert training_hands.setdefault(label, meta["training_hands"]) == meta["training_hands"]
            assert result.setdefault("particles", meta["particles"]) == meta["particles"]
            with (directory / "paired_payoffs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == pairs and [int(r["pair"]) for r in rows] == list(range(pairs))
            stream = [r["deal_seed"] for r in rows]
            assert len(set(stream)) == pairs
            assert streams.setdefault(seed, stream) == stream
            for name in NAMES:
                seats = np.array([[float(r[name + "_seat0"]), float(r[name + "_seat1"])] for r in rows])
                assert np.isfinite(seats).all() and (np.abs(seats) <= 1000).all()
                values = seats.mean(axis=1)
                assert np.isclose(values.mean(), meta["policies"][name]["mean"], atol=1e-10)
                samples[label][name].append(values)
        samples[label] = {k: np.array(v) for k, v in samples[label].items()}
        result["models"][label] = {name: estimate(values) for name, values in samples[label].items()}
        result["models"][label]["paired_model_minus_baseline"] = {
            p: estimate(samples[label]["model_" + p] - samples[label]["baseline_" + p])
            for p in ("current", "average")
        }
    assert len(set().union(*(set(v) for v in streams.values()))) == len(seeds) * pairs
    for name in ("baseline_current", "baseline_average", "uniform"):
        np.testing.assert_array_equal(samples["limited_dt"][name], samples["unlimited_dt"][name])
    np.testing.assert_array_equal(samples["limited_dt"]["uniform"], samples["hand_recall"]["uniform"])
    result["paired_unlimited_minus_limited"] = {
        p: estimate(samples["unlimited_dt"]["model_" + p] - samples["limited_dt"]["model_" + p])
        for p in ("current", "average")
    }
    result["matched_deals_and_duplicate_controls_identical"] = True
    result["training_hands"] = training_hands
    save(output / "summary.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Limited DT\n1M", "Unlimited DT\n1M", f"Hand recall\n{training_hands['hand_recall']:,}",
              "Summary-key\nk-means 1M", f"Exact-history\nk-means {training_hands['hand_recall']:,}", "Uniform"]
    groups = [("limited_dt", "model"), ("unlimited_dt", "model"), ("hand_recall", "model"),
              ("limited_dt", "baseline"), ("hand_recall", "baseline"), ("limited_dt", "uniform")]
    colors = ["#c62828", "#eb6a24", "#287a43", "#1565c0", "#00a0bd", "#666666"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, policy in zip(axes, ("current", "average")):
        for i, ((run, arm), color) in enumerate(zip(groups, colors)):
            name = "uniform" if arm == "uniform" else arm + "_" + policy
            stats = result["models"][run][name]
            lo, hi = stats["bootstrap_ci95"]
            ax.vlines(i, lo, hi, color=color, linewidth=2)
            ax.scatter(i, stats["mean"], color=color, s=45)
            ax.scatter(i + np.linspace(-.1,.1,len(seeds)), stats["seed_means"], color=color, marker="x", alpha=.5)
            ax.annotate(f'{stats["mean"]:.2f}', (i, stats["mean"]), xytext=(4, 7), textcoords="offset points", fontsize=9)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_title(policy.title() + " policy")
        ax.set_ylabel("LBR attacker profit (ante/hand); lower is better for target")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Frozen 7-stud: {len(seeds)*pairs*2:,} hands/policy, {result['particles']} LBR particles\n95% seed-stratified paired-deal bootstrap; crosses: evaluation-seed means; not exploitability")
    fig.tight_layout()
    fig.savefig(output / "lbr_comparison.png", dpi=150)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026100401, 2026100402])
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(FAMILY / "data"):
        parser.error("Use a new output inside cpp_mccfr/data")
    if (args.pairs < 2 or args.particles < 1 or len(set(args.seeds)) != len(args.seeds)
            or any(seed <= 0 or seed >= 2**64 for seed in args.seeds)):
        parser.error("Invalid evaluation budgets/seeds")
    output.mkdir(parents=True)
    manifest = dict(status="preparing", training=False, pairs=args.pairs, seeds=args.seeds,
                    particles=args.particles, runs={}, command=subprocess.list2cmdline(sys.argv))
    inputs = {}
    for label, (kind, run) in RUNS.items():
        config = read_manifest(run)
        assert all(digest(PROJECT / name) == value for name,value in config["config"]["source_sha256"].items())
        assert digest(run / "hand_atlas.bin") == config["config"]["atlas_sha256"]
        file = run / ("checkpoint.bin" if kind == "tree" else "model.sqlite")
        inputs[str(file)] = [file.stat().st_size, file.stat().st_mtime_ns]
        manifest["runs"][label] = dict(kind=kind, run=str(run), config=config,
                                       file_stat=inputs[str(file)])
        if kind == "disk":
            manifest["runs"][label]["metadata"] = metadata(run)
    save(output / "run.json", manifest)
    binaries = build(output)
    manifest["binaries"] = {k: dict(path=str(v), sha256=digest(v)) for k,v in binaries.items()}
    manifest["adapter_source_sha256"] = {p.name: digest(p) for p in HERE.iterdir() if p.suffix in (".py", ".cpp", ".hpp")}
    save(output / "run.json", manifest)
    if args.build_only:
        print("BUILT " + str(output), flush=True)
        return
    started = time.perf_counter()
    for label, (kind, run) in RUNS.items():
        with (output / f"{label}_self_test.log").open("w") as log:
            subprocess.run([str(binaries[kind]), "--self-test", str(run)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"SELF_TEST model={label} passed", flush=True)
        for seed in args.seeds:
            destination = output / f"{label}_{seed}"
            command = [str(binaries[kind]), str(run), str(destination), str(args.pairs), str(args.particles), str(seed)]
            print(f"EVALUATE model={label} seed={seed} pairs={args.pairs}", flush=True)
            with destination.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"COMPLETE model={label} seed={seed}", flush=True)
    assert all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == state for p,state in inputs.items())
    summary = analyze(output, args.seeds, args.pairs)
    manifest.update(status="complete", wall_seconds=time.perf_counter()-started, input_file_stats_unchanged=True)
    save(output / "run.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
