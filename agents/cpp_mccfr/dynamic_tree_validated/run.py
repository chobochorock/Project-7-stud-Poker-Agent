"""Compare raw, action-centered, and held-out-validated dynamic tree splits.

Usage (project root): python -m agents.cpp_mccfr.dynamic_tree_validated.run --help
Train: ... run train --run agents/cpp_mccfr/data/dynamic_tree_validated/seed111 --hands 100000
Output: model/RNG/monitor checkpoint, per-hand gaps, fixed panels and split log.
Split count/depth default to -1 (unlimited). Resource stops remain separate.
This is a heuristic imperfect-recall split method, NOT proved CFR+IRA.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import shutil
import signal
import subprocess

from .. import hand_recall as legacy

HERE = Path(__file__).resolve().parent
DATA = legacy.FAMILY / "data" / "dynamic_tree_validated"
SOURCES = tuple(p for p in legacy.SOURCES if p.suffix != ".py") + (
    HERE / "solver.cpp",
    HERE / "lbr.cpp",
    HERE.parent / "abstraction_lbr/evaluate.hpp",
    Path(__file__),
)


def hashes() -> dict:
    return {str(p.relative_to(legacy.PROJECT)): legacy.sha256(p) for p in SOURCES}


def identity(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def build(check: bool = False, lbr: bool = False) -> Path:
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ is required")
    kind = "lbr" if lbr else "solver"
    output = legacy.FAMILY / "bin" / f"dynamic_tree_validated_{kind}_{identity(hashes())[:16]}.exe"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = output.with_suffix(".building.exe")
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-static-libstdc++",
                "-static-libgcc",
                str(HERE / f"{kind}.cpp"),
                "-o",
                str(temporary),
                "-lpsapi",
            ],
            cwd=legacy.PROJECT,
            check=True,
        )
        os.replace(temporary, output)
    if check:
        subprocess.run([str(output), "--self-test", str(legacy.ATLAS)], check=True)
    return output


@contextmanager
def writer_lock(run: Path):
    with (run / "writer.lock").open("a+b") as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def train(args) -> int:
    run = args.run.resolve()
    if not run.is_relative_to(DATA.resolve()):
        raise ValueError(f"Use a new directory under {DATA}")
    if legacy.sha256(legacy.ATLAS) != legacy.ATLAS_SHA:
        raise ValueError("Frozen atlas changed")
    options = {
        name: getattr(args, name)
        for name in (
            "seed",
            "eval_seed",
            "particles",
            "panel_every",
            "panel_roots",
            "window",
            "min_child",
            "max_splits",
            "max_depth",
            "gain",
            "min_rmse",
            "split_rule",
            "validation_z",
            "freeze_after",
        )
    }
    config = dict(
        schema="dynamic_tree_validated_1",
        **options,
        source_sha256=hashes(),
        atlas_sha256=legacy.ATLAS_SHA,
        algorithm="signed alternating external MCCFR; traversal-frozen policies; SIMPLE average",
        budget="one fresh fifth-street deal and two traversals per model per hand",
        rules="C++ stud v3 HU, heuristic H4, ante1000, stack1000ante",
        tree="root per street/seat/legal mask; 34D observable summary; local CART with optional centered labels and chronological holdout",
        baseline="fresh existing hard256 betting-summary partition plus explicit seat",
        monitor="one uniformly sampled visit per leaf/hand, recent bounded ring; no evaluation labels",
        initialization="zero regrets and average sums; no pretrained strategy",
        data_bias="atlas fit from old epsilon0.2 self-play100k seed37301; tree sees own evolving policy",
        metric="post-update fifth-street one-step Local BR-gap, ante, NOT exploitability/LBR",
        inheritance="fractional regret/average mass; approximate history allocation; no perfect recall guarantee",
    )
    manifest = dict(identity=identity(config), config=config)
    if args.resume:
        if json.loads((run / "config.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume configuration/source mismatch")
        if legacy.sha256(run / "hand_atlas.bin") != legacy.ATLAS_SHA:
            raise ValueError("Run-local atlas changed")
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Nonempty directory: use another run or --resume")
    binary = build()
    run.mkdir(parents=True, exist_ok=True)
    with writer_lock(run):
        if not args.resume:
            if any(p.name != "writer.lock" for p in run.iterdir()):
                raise ValueError("Run changed while acquiring lock")
            shutil.copy2(legacy.ATLAS, run / "hand_atlas.bin")
            shutil.copy2(
                legacy.ATLAS.with_suffix(".json"), run / "atlas_provenance.json"
            )
            for source in SOURCES:
                target = run / "source" / source.relative_to(legacy.PROJECT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            (run / "config.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        command = [str(binary), str(run), manifest["identity"]]
        command += [
            str(getattr(args, name))
            for name in (
                "hands",
                "seed",
                "eval_seed",
                "particles",
                "panel_every",
                "panel_roots",
                "checkpoint_every",
            )
        ]
        command += [str(int(args.resume))]
        command += [
            str(getattr(args, name))
            for name in (
                "window",
                "min_child",
                "max_splits",
                "max_depth",
                "gain",
                "min_rmse",
                "max_kmeans_states",
            )
        ]
        command += [str(int(args.reserve_gib * 2**30))]
        command += [str({"raw": 0, "centered": 1, "validated": 2}[args.split_rule]),
                    str(args.validation_z), str(args.freeze_after)]
        with (run / "invocations.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    dict(
                        utc=datetime.now(timezone.utc).isoformat(),
                        command=command,
                        exe_sha256=legacy.sha256(binary),
                    )
                )
                + "\n"
            )
        with (run / "console.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(
                command,
                cwd=legacy.PROJECT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("build").add_argument("--check", action="store_true")
    training = sub.add_parser("train")
    training.add_argument("--run", type=Path, required=True)
    for name, default in dict(
        hands=100000,
        seed=111,
        eval_seed=307,
        particles=32,
        panel_every=10000,
        panel_roots=128,
        checkpoint_every=1000,
        window=256,
        min_child=32,
        max_kmeans_states=500000,
    ).items():
        training.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    training.add_argument(
        "--max-splits", type=int, default=-1,
        help="Total split limit: -1 unlimited (default), 0 disables splitting",
    )
    training.add_argument(
        "--max-depth", type=int, default=-1,
        help="Tree depth limit: -1 unlimited (default), 0 keeps roots only",
    )
    training.add_argument("--gain", type=float, default=0.20)
    training.add_argument(
        "--min-rmse", type=float, default=1.0, help="Parent regret RMSE gate, ante"
    )
    training.add_argument("--reserve-gib", type=float, default=10)
    training.add_argument("--split-rule", choices=("raw", "centered", "validated"), default="validated")
    training.add_argument("--validation-z", type=float, default=2.0,
                          help="Heuristic heldout improvement/SE gate, not a sequential CI")
    training.add_argument("--freeze-after", type=int, default=0,
                          help="Stop splitting after this hand; 0 means never freeze")
    training.add_argument("--resume", action="store_true")
    plotting = sub.add_parser("plot")
    plotting.add_argument("--run", type=Path, required=True)
    plotting.add_argument("--window", type=int, default=100)
    evaluation = sub.add_parser("lbr")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--pairs", type=int, default=250)
    evaluation.add_argument("--particles", type=int, default=64)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[2026100701, 2026100702])
    return result


def evaluate_lbr(args) -> None:
    run, output = args.run.resolve(), args.output.resolve()
    manifest = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (config["schema"] != "dynamic_tree_validated_1" or config["source_sha256"] != hashes()
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
    before = (run / "checkpoint.bin").stat()
    with writer_lock(run):
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
            for name in names:
                samples[name].append([(float(row[name + "_seat0"]) + float(row[name + "_seat1"])) / 2 for row in rows])
            print(f"LBR seed={seed} hands_per_policy={2*args.pairs} output={destination}", flush=True)
    after = (run / "checkpoint.bin").stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during evaluation")
    arrays = {name: np.asarray(value) for name, value in samples.items()}
    summary = dict(run=str(run), seeds=args.seeds, pairs_per_seed=args.pairs,
                   particles=args.particles, metric="LBR attacker ante/hand, not exploitability",
                   policies={name: estimate(value) for name, value in arrays.items()},
                   paired_model_minus_baseline={p: estimate(arrays["model_" + p] - arrays["baseline_" + p]) for p in ("current", "average")},
                   input_unchanged=True, exe_sha256=legacy.sha256(binary))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if args.command == "build":
        print(build(args.check))
    elif args.command == "lbr":
        evaluate_lbr(args)
    elif args.command == "plot":
        from ..dynamic_tree.plot import plot

        if args.window < 1:
            cli.error("window must be positive")
        plot(args.run.resolve(), args.window)
    else:
        for name in (
            "hands",
            "seed",
            "eval_seed",
            "panel_every",
            "panel_roots",
            "checkpoint_every",
            "max_kmeans_states",
        ):
            if getattr(args, name) < 1:
                cli.error(f"{name} must be positive")
        if not (
            args.particles >= 2
            and args.window >= 4
            and args.min_child >= 2
            and 2 * args.min_child <= args.window
            and -1 <= args.max_splits <= 2**63 - 1
            and -1 <= args.max_depth <= 2**63 - 1
            and args.window <= 100000
            and 0 < args.gain <= 1
            and math.isfinite(args.min_rmse)
            and args.min_rmse >= 0
            and math.isfinite(args.reserve_gib)
            and args.reserve_gib >= 0
            and math.isfinite(args.validation_z)
            and args.validation_z >= 0
            and 0 <= args.freeze_after < 2**64
            and (args.split_rule != "validated" or args.window >= 4 * args.min_child)
        ):
            cli.error("Invalid split/evaluation/memory settings")
        if shutil.disk_usage(legacy.FAMILY).free < args.reserve_gib * 2**30:
            cli.error("Insufficient free disk space")
        raise SystemExit(train(args))


if __name__ == "__main__":
    main()
