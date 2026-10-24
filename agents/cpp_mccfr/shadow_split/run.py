"""Shadow-Regret Splitting on C++ 7-stud v3 with a co-trained hard256 k-means arm.

Usage (project root):
    python -m agents.cpp_mccfr.shadow_split.run build --check
    python -m agents.cpp_mccfr.shadow_split.run train --run agents/cpp_mccfr/data/shadow_split/<new> \
        --hands 50000 --seed 222 --key-mode summary --rule z --z 0.66
    python -m agents.cpp_mccfr.shadow_split.run lbr --run <run> --output <run>/lbr_<seeds> --seeds A B
    python -m agents.cpp_mccfr.shadow_split.report --runs label=<run> ... --output <new folder>
Train output: checkpoint.bin (both arms, shadows, RNG), per_hand.csv / panel.csv (one-step
Local BR-gap of the FIRST 5th-street decision for both arms, current and average; same
evaluator, roots and particles as dynamic_tree_validated), splits.csv (every split test),
bin_edges.csv, config.json, invocations.jsonl, console.log, source/ copy.
Metrics are Local BR-gap and Policy-LBR in ante; neither is exploitability.
See README.md for the method, key modes, defaults and their provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess

from .. import hand_recall as legacy
from ..dynamic_tree_validated import run as validated

HERE = Path(__file__).resolve().parent
DATA = legacy.FAMILY / "data" / "shadow_split"
SOURCES = tuple(validated.SOURCES) + (HERE / "solver.cpp", HERE / "lbr.cpp", Path(__file__))
KEY_MODES = {"own": 0, "exact": 1, "summary": 2}
RULES = {"time_uniform": 0, "z": 1}
INHERIT = {"A": 0, "B": 2, "reset": 3}
SPLIT_OPTIONS = (
    "key_mode", "rule", "rho_scale", "z", "delta", "span", "eps_min", "kappa", "eps_unit",
    "grace", "shadow_after", "inherit", "max_splits", "freeze_after", "shadow_gib",
)


def hashes() -> dict:
    return {str(p.relative_to(legacy.PROJECT)): legacy.sha256(p) for p in SOURCES}


def identity(config: dict) -> str:
    return validated.identity(config)


def build(check: bool = False, lbr: bool = False) -> Path:
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ is required")
    kind = "lbr" if lbr else "solver"
    output = legacy.FAMILY / "bin" / f"shadow_split_{kind}_{identity(hashes())[:16]}.exe"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = output.with_suffix(".building.exe")
        subprocess.run(
            [compiler, "-O3", "-std=c++17", "-static-libstdc++", "-static-libgcc",
             str(HERE / f"{kind}.cpp"), "-o", str(temporary), "-lpsapi"],
            cwd=legacy.PROJECT, check=True,
        )
        os.replace(temporary, output)
    if check and not lbr:
        subprocess.run([str(output), "--self-test", str(legacy.ATLAS)], check=True)
    return output


def manifest_for(args) -> dict:
    options = {name: getattr(args, name) for name in (
        "seed", "eval_seed", "particles", "panel_every", "panel_roots") + SPLIT_OPTIONS}
    config = dict(
        schema="shadow_split_7stud_1",
        **options,
        source_sha256=hashes(),
        atlas_sha256=legacy.ATLAS_SHA,
        algorithm="signed alternating external MCCFR; traversal-frozen policies; SIMPLE average; no linear weights",
        budget="one fresh fifth-street deal and two traversals per model per hand",
        rules="C++ stud v3 HU, heuristic H4, ante1000, stack1000ante",
        tree=("one card tree per public node; leaves keep per-(feature, bin, action) signed regret and "
              "strategy shadows; split when Lambda - rho > eps, Lambda = (g(L)+g(R)-g(L+R))/W, g = max+"),
        public_node={"own": "seat, street, legal mask, own actions on earlier streets, exact current street",
                     "exact": "seat, street, legal mask, exact betting history",
                     "summary": "hard256 baseline betting summary + seat (same partition as the baseline, "
                                "without its card cluster)"}[args.key_mode],
        features=("power vector dims 0-17 + pot, opponent invested, aggression, history length "
                  "(summary mode splits on 0-17 only); 8 quantile bins per street from 20k no-fold "
                  "uniform-play states, seed 2026100801"),
        radius={"z": "z * (max_a sqrt(QL) + max_a sqrt(QR) + max_a sqrt(QP)) / W; Q = sum over traversals "
                     "of squared per-traversal totals; heuristic per-test rule, no simultaneous guarantee",
                "time_uniform": "rho_scale * time-uniform empirical-Bernstein radius with span; valid only "
                                "when a leaf gets one visit per traversal (own/exact keys)"}[args.rule],
        provenance=("z 0.66 = Toy rho x 0.1 (sqrt(2 L) * 0.1 with L ~ 22), chosen after Toy Leduc results; "
                    "inheritance B best in Toy at 1e7; eps_unit = initial pot 2 ante; grace 100 as Toy"),
        baseline="fresh existing hard256 betting-summary partition plus explicit seat (dynamic_tree_validated arm)",
        initialization="zero regrets and average sums; no pretrained strategy",
        data_bias="atlas fit from old epsilon0.2 self-play100k seed37301; bins from no-fold uniform play",
        metric="post-update fifth-street one-step Local BR-gap, ante, NOT exploitability/LBR",
    )
    return dict(identity=identity(config), config=config)


def train(args) -> int:
    run = args.run.resolve()
    if not run.is_relative_to(DATA.resolve()):
        raise ValueError(f"Use a new directory under {DATA}")
    if legacy.sha256(legacy.ATLAS) != legacy.ATLAS_SHA:
        raise ValueError("Frozen atlas changed")
    manifest = manifest_for(args)
    if args.resume:
        if json.loads((run / "config.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume configuration/source mismatch")
        if legacy.sha256(run / "hand_atlas.bin") != legacy.ATLAS_SHA:
            raise ValueError("Run-local atlas changed")
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Nonempty directory: use another run or --resume")
    binary = build()
    run.mkdir(parents=True, exist_ok=True)
    with validated.writer_lock(run):
        if not args.resume:
            if any(p.name != "writer.lock" for p in run.iterdir()):
                raise ValueError("Run changed while acquiring lock")
            shutil.copy2(legacy.ATLAS, run / "hand_atlas.bin")
            shutil.copy2(legacy.ATLAS.with_suffix(".json"), run / "atlas_provenance.json")
            for source in SOURCES:
                target = run / "source" / source.relative_to(legacy.PROJECT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            (run / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        command = [str(binary), str(run), manifest["identity"]]
        command += [str(getattr(args, name)) for name in (
            "hands", "seed", "eval_seed", "particles", "panel_every", "panel_roots", "checkpoint_every")]
        command += [str(int(args.resume)), str(KEY_MODES[args.key_mode]), str(RULES[args.rule])]
        command += [repr(float(getattr(args, name))) for name in (
            "rho_scale", "z", "delta", "span", "eps_min", "kappa", "eps_unit")]
        command += [str(args.grace), str(args.shadow_after), str(INHERIT[args.inherit]),
                    str(args.max_splits), str(args.freeze_after), repr(float(args.shadow_gib)),
                    str(args.max_kmeans_states), str(int(args.reserve_gib * 2**30)),
                    str(args.max_tree_states)]
        with (run / "invocations.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), command=command,
                                      exe_sha256=legacy.sha256(binary))) + "\n")
        with (run / "console.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(
                command, cwd=legacy.PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            ) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    return process.wait()
                except KeyboardInterrupt:
                    print("Finishing the hand and saving. Please wait.", flush=True)
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    remaining, _ = process.communicate()
                    print(remaining, end="")
                    log.write(remaining)
                    return process.returncode


def evaluate_lbr(args) -> None:
    run, output = args.run.resolve(), args.output.resolve()
    manifest = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (config["schema"] != "shadow_split_7stud_1" or config["source_sha256"] != hashes()
            or legacy.sha256(run / "hand_atlas.bin") != config["atlas_sha256"]):
        raise ValueError("Model configuration/source/atlas mismatch")
    if (output.exists() or not output.is_relative_to(DATA.resolve()) or args.pairs < 2
            or args.particles < 1 or len(set(args.seeds)) != len(args.seeds)
            or any(seed <= 0 or seed >= 2**64 for seed in args.seeds)):
        raise ValueError("Use a new family data output and valid LBR budgets/seeds")
    binary = build(lbr=True)
    output.mkdir(parents=True)
    from ..abstraction_lbr.run import estimate
    import csv
    import numpy as np

    names = ("model_current", "baseline_current", "model_average", "baseline_average", "uniform")
    samples = {name: [] for name in names}
    hands = None
    before = (run / "checkpoint.bin").stat()
    with validated.writer_lock(run):
        subprocess.run([str(binary), "--self-test", str(run), manifest["identity"]], check=True)
        for seed in args.seeds:
            destination = output / str(seed)
            command = [str(binary), str(run), str(destination), str(args.pairs),
                       str(args.particles), str(seed), manifest["identity"]]
            with (output / f"{seed}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (destination / "paired_payoffs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != args.pairs:
                raise ValueError("Incomplete LBR output")
            hands = json.loads((destination / "evaluation.json").read_text())["training_hands"]
            for name in names:
                samples[name].append([(float(row[name + "_seat0"]) + float(row[name + "_seat1"])) / 2
                                      for row in rows])
            print(f"LBR seed={seed} hands_per_policy={2*args.pairs} output={destination}", flush=True)
    after = (run / "checkpoint.bin").stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during evaluation")
    arrays = {name: np.asarray(value) for name, value in samples.items()}
    summary = dict(
        run=str(run), training_hands=hands, seeds=args.seeds, pairs_per_seed=args.pairs,
        particles=args.particles, metric="LBR attacker ante/hand, not exploitability",
        deal_stream="abstraction_lbr convention (deal seed = CLI seed), as dynamic_tree_validated",
        policies={name: estimate(value) for name, value in arrays.items()},
        paired_model_minus_baseline={p: estimate(arrays["model_" + p] - arrays["baseline_" + p])
                                     for p in ("current", "average")},
        input_unchanged=True, exe_sha256=legacy.sha256(binary),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("build").add_argument("--check", action="store_true")
    training = sub.add_parser("train")
    training.add_argument("--run", type=Path, required=True)
    for name, default in dict(hands=50000, seed=222, eval_seed=307, particles=32, panel_every=10000,
                              panel_roots=128, checkpoint_every=5000, max_kmeans_states=500000,
                              max_tree_states=5000000,
                              grace=100, shadow_after=100).items():
        training.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    training.add_argument("--key-mode", choices=tuple(KEY_MODES), default="summary",
                          help="public node: summary = baseline betting summary; own; exact history")
    training.add_argument("--rule", choices=tuple(RULES), default="z")
    training.add_argument("--z", type=float, default=0.66, help="z rule multiplier (Toy rho x 0.1 equivalent)")
    training.add_argument("--rho-scale", type=float, default=1.0, help="time_uniform rule multiplier")
    training.add_argument("--delta", type=float, default=0.05)
    training.add_argument("--span", type=float, default=2000.0, help="increment range in ante (time_uniform)")
    training.add_argument("--eps-min", type=float, default=0.0)
    training.add_argument("--kappa", type=float, default=1.0)
    training.add_argument("--eps-unit", type=float, default=2.0, help="eps(t)=max(eps_min, kappa*eps_unit/sqrt(t))")
    training.add_argument("--inherit", choices=tuple(INHERIT), default="B")
    training.add_argument("--max-splits", type=int, default=-1, help="-1 unlimited, 0 no-split control")
    training.add_argument("--freeze-after", type=int, default=0)
    training.add_argument("--shadow-gib", type=float, default=4.0, help="shadow memory cap; later leaves stay unshadowed")
    training.add_argument("--reserve-gib", type=float, default=10)
    training.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("lbr")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--pairs", type=int, default=250)
    evaluation.add_argument("--particles", type=int, default=64)
    evaluation.add_argument("--seeds", type=int, nargs="+", required=True)
    return result


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if args.command == "build":
        print(build(args.check))
    elif args.command == "lbr":
        evaluate_lbr(args)
    else:
        for name in ("hands", "seed", "eval_seed", "panel_every", "panel_roots", "checkpoint_every",
                     "max_kmeans_states", "max_tree_states", "grace"):
            if getattr(args, name) < 1:
                cli.error(f"{name} must be positive")
        if not (args.particles >= 2 and args.z >= 0 and args.rho_scale >= 0 and 0 < args.delta < 1
                and args.span > 0 and args.eps_min >= 0 and args.kappa >= 0 and args.eps_unit >= 0
                and args.max_splits >= -1 and args.freeze_after >= 0 and args.shadow_gib >= 0
                and math.isfinite(args.reserve_gib) and args.reserve_gib >= 0):
            cli.error("Invalid split/evaluation/memory settings")
        if shutil.disk_usage(legacy.FAMILY).free < args.reserve_gib * 2**30:
            cli.error("Insufficient free disk space")
        raise SystemExit(train(args))


if __name__ == "__main__":
    main()
