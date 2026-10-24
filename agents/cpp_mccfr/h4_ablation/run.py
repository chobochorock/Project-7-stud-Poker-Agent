"""Learn twelve discard/reveal actions against frozen DT/KM betting policies.

Usage (project root): python -m agents.cpp_mccfr.h4_ablation.run --help
See README.md. Generated adapters reuse the original LBR verbatim except its
H4 prior; no original training source or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ..dynamic_tree_validated import run as base
from ..dynamic_tree_validated.schedule import manifest_at
from ..abstraction_lbr.run import estimate

HERE = Path(__file__).resolve().parent
DATA = base.legacy.FAMILY / "data/h4_ablation"
DEFAULT_SOURCE = base.DATA / "centered_seed222_freeze1m_10m"
NAMES = ("dt_heuristic", "dt_learned", "km_heuristic", "km_learned")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Original adapter changed; inspect before rebuilding: " + old)
    return text.replace(old, new)


def adapted_source():
    """Anchored mechanical extraction, with fail-closed source shape checks."""
    source = (base.legacy.FAMILY / "stud_mccfr.cpp").read_text(encoding="utf-8")
    start = "template <typename TargetPolicy>\nclass PolicyLBR {"
    end = "\nvoid apply_h4_choices("
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("LBR extraction markers changed")
    lbr = source[source.index(start):source.index(end)]
    lbr = lbr.replace("PolicyLBR", "H4PolicyLBR")
    old = """const int h4_ways = h4_multiplicity(
                {particle.target_hidden[0], particle.target_hidden[1]},
                target_public.front(),
                particle.target_discarded);"""
    new = """const double h4_ways = target_.learned_h4()
                ? target_.h4_weight(
                    {particle.target_hidden[0], particle.target_hidden[1]},
                    target_public.front(), particle.target_discarded, target_seat)
                : h4_multiplicity(
                    {particle.target_hidden[0], particle.target_hidden[1]},
                    target_public.front(), particle.target_discarded);"""
    lbr = replace_once(lbr, old, new)
    first = "template <typename TargetPolicy>\ndouble play_hand_policy_lbr("
    last = "\nstd::array<std::vector<PowerVector>, 3> collect_power_samples("
    if source.count(first) != 1 or source.count(last) != 1:
        raise ValueError("Match extraction markers changed")
    match = source[source.index(first):source.index(last)]
    match = match.replace("play_hand_policy_lbr", "play_hand_h4_lbr")
    match = match.replace("PolicyLBR<TargetPolicy>", "H4PolicyLBR<TargetPolicy>")
    first = "double h4_continuation_value("
    last = "\nvoid train_h4_root("
    if source.count(first) != 1 or source.count(last) != 1:
        raise ValueError("Continuation extraction markers changed")
    continuation = source[source.index(first):source.index(last)]
    continuation = replace_once(continuation, "MCCFR& continuation", "TargetPolicy& continuation")
    continuation = replace_once(continuation, "h4_continuation_value", "h4_frozen_value")
    adapter = (base.HERE / "lbr.cpp").read_text(encoding="utf-8")
    if adapter.count("struct FrozenModel {") != 1 or adapter.count("\nint main(") != 1:
        raise ValueError("Frozen adapter extraction markers changed")
    adapter = adapter[adapter.index("struct FrozenModel {"):adapter.index("\nint main(")]
    return adapter + lbr + match + "\ntemplate <class TargetPolicy>\n" + continuation


def build():
    hashes = dict(base.hashes())
    for path in (HERE / "experiment.cpp", Path(__file__)):
        hashes[str(path.relative_to(base.legacy.PROJECT))] = base.legacy.sha256(path)
    directory = DATA / "build" / base.identity(hashes)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / "h4_generated.hpp"
    header.write_text(adapted_source(), encoding="utf-8")
    binary = directory / "h4_ablation.exe"
    if not binary.exists():
        compiler = shutil.which("g++")
        if not compiler:
            raise ValueError("g++ is required")
        command = [compiler, "-O3", "-std=c++17", "-static-libstdc++", "-static-libgcc",
                   "-I", str(directory), str(HERE / "experiment.cpp"), "-o",
                   str(binary.with_suffix(".building.exe")), "-lpsapi"]
        with (directory / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        os.replace(binary.with_suffix(".building.exe"), binary)
    return binary, hashes


def execute(command, log):
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + subprocess.list2cmdline(command) + "\n")
        handle.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            if process.wait():
                raise RuntimeError(f"Experiment failed; see {log}")


def analyze(output, seeds):
    samples = {name: [] for name in NAMES}
    for seed in seeds:
        with (output / f"lbr_{seed}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for name in NAMES:
            samples[name].append([(float(r[name + "_seat0"]) +
                                   float(r[name + "_seat1"])) / 2 for r in rows])
    samples = {name: np.asarray(values) for name, values in samples.items()}
    result = {"unit": "LBR attacker ante/hand; lower is better for target",
              "policies": {name: estimate(values) for name, values in samples.items()},
              "paired_differences": {}}
    for label, left, right in (
        ("dt_learned_minus_heuristic", "dt_learned", "dt_heuristic"),
        ("km_learned_minus_heuristic", "km_learned", "km_heuristic"),
        ("learned_dt_minus_km", "dt_learned", "km_learned"),
        ("heuristic_dt_minus_km", "dt_heuristic", "km_heuristic"),
    ):
        result["paired_differences"][label] = estimate(samples[left] - samples[right])
    save(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    plot(output, result)


def plot(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family, color in (("dt", "red"), ("km", "blue")):
        x = np.array([0, 1]) + (-.06 if family == "dt" else .06)
        values = [result["policies"][f"{family}_{mode}"] for mode in ("heuristic", "learned")]
        y = np.array([v["mean"] for v in values])
        bounds = np.array([v["bootstrap_ci95"] for v in values]).T
        axes[0].errorbar(x, y, yerr=np.maximum(0, [y-bounds[0], bounds[1]-y]),
                         marker="o", color=color, capsize=4, label=family.upper())
    axes[0].set_xticks([0, 1], ["Heuristic H4", "Learned H4 (average)"])
    axes[0].set_ylabel("LBR attacker ante/hand (lower is better)")
    axes[0].set_title("Frozen 10M average betting; bootstrap 95% CI")
    axes[0].legend()
    panel = output / "panel.csv"
    if panel.exists():
        rows = list(csv.DictReader(panel.open()))
        for arm, color, label in ((0, "red", "DT"), (1, "blue", "k-means")):
            selected = [r for r in rows if int(r["arm"]) == arm]
            axes[1].plot([int(r["hands"]) for r in selected],
                         [float(r["gap"]) for r in selected], color=color, label=label)
        axes[1].set(xscale="symlog", yscale="log", xlabel="H4 training deals/model",
                    ylabel="H4 Local BR-gap (ante)", title="Fixed roots, MC plug-in max; NOT 5th-street gap")
        axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "train", "lbr"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026100851)
    parser.add_argument("--panel-every", type=int, default=10000)
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[2026100841, 2026100842])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.run.resolve()
    if not output.is_relative_to(DATA.resolve()):
        parser.error(f"Use output under {DATA}")
    if min(args.hands, args.panel_every, args.roots, args.particles) < 1 or args.pairs < 2:
        parser.error("Budgets must be positive; pairs >= 2")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error("Use unique evaluation seeds")
    if args.mode == "train" and (args.particles < 2 or args.particles % 2):
        parser.error("The cross-fitted training panel needs even particles >= 2")
    with base.writer_lock(source):
        manifest = manifest_at(source)
        paths = [source / name for name in ("checkpoint.bin", "hand_atlas.bin", "config.json")]
        before = {str(p): base.legacy.sha256(p) for p in paths}
        binary, hashes = build()
        output.mkdir(parents=True, exist_ok=True)
        common = [str(binary), args.mode, str(source), manifest["identity"], str(output)]
        if args.mode == "test":
            execute(common, output / "test.log")
        elif args.mode == "train":
            config = {"source": str(source), "source_files": before, "code_hashes": hashes,
                      "seed": args.seed, "panel_every": args.panel_every, "roots": args.roots,
                      "particles": args.particles, "betting_policy": "frozen average, DT and matched k-means",
                      "h4": "seat-specific suit-canonical 12-action signed regrets; uniform initial; SIMPLE average",
                      "budget": "initial deals per model; 24 frozen betting terminal rollouts per deal/model",
                      "scope": "H4-only self-play, not joint betting retraining or full-game equilibrium"}
            path = output / "config.json"
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError("Resume configuration mismatch")
            save(path, config)
            execute(common + list(map(str, [args.hands, args.seed, args.panel_every,
                                           args.roots, args.particles])), output / "train.log")
        else:
            config = json.loads((output / "config.json").read_text())
            if config["source_files"] != before or config["code_hashes"] != hashes:
                raise ValueError("Evaluation source/code mismatch")
            for seed in args.eval_seeds:
                if (output / f"lbr_{seed}.csv").exists():
                    raise ValueError("Evaluation exists; use fresh seeds")
                execute(common + list(map(str, [args.pairs, args.particles, seed])), output / "lbr.log")
            save(output / "lbr_config.json", dict(pairs=args.pairs, particles=args.particles,
                                                 seeds=args.eval_seeds, binary_sha256=base.legacy.sha256(binary)))
            analyze(output, args.eval_seeds)
        if before != {str(p): base.legacy.sha256(p) for p in paths}:
            raise RuntimeError("Frozen source files changed during experiment")
        save(output / "input_integrity.json", {"unchanged": True, "sha256": before})


if __name__ == "__main__":
    main()
