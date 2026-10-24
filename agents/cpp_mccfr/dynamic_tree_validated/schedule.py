"""Fork an existing DT into frozen/continuing split experiments; never alters the parent.

Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.schedule --help
See SCHEDULE.md for two 1M -> 10M commands, slower-split parameters and evaluation.
Fork performs a read-only policy panel and saves models/RNG, but does not train.
The compatible checkpoint and old solver stay unchanged; schedule hashes are additional.
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

from . import run as base
from .report import leaf_support
from ..dynamic_tree.review import checkpoint_header

SOURCES = (Path(__file__).resolve(), base.HERE / "schedule.cpp")


def schedule_hashes():
    return {
        str(p.relative_to(base.legacy.PROJECT)): base.legacy.sha256(p) for p in SOURCES
    }


def manifest_at(path, scheduled=True):
    manifest = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (
        manifest["identity"] != base.identity(config)
        or config["schema"] != "dynamic_tree_validated_1"
        or config["source_sha256"] != base.hashes()
        or base.legacy.sha256(path / "hand_atlas.bin") != config["atlas_sha256"]
    ):
        raise ValueError("Source, atlas or identity mismatch")
    if scheduled and config.get("schedule_source_sha256") != schedule_hashes():
        raise ValueError("Schedule source mismatch")
    sources = dict(config["source_sha256"], **config.get("schedule_source_sha256", {}))
    for name, digest in sources.items():
        if base.legacy.sha256(path / "source" / name) != digest:
            raise ValueError(f"Snapshot mismatch: {name}")
    return manifest


def build():
    digest = base.identity(dict(base.hashes(), **schedule_hashes()))[:16]
    output = base.legacy.FAMILY / "bin" / f"dt_split_schedule_{digest}.exe"
    if not output.exists():
        compiler = shutil.which("g++")
        if not compiler:
            raise ValueError("g++ is required")
        temporary = output.with_suffix(f".{os.getpid()}.building.exe")
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-static-libstdc++",
                "-static-libgcc",
                str(base.HERE / "schedule.cpp"),
                "-o",
                str(temporary),
                "-lpsapi",
            ],
            check=True,
        )
        os.replace(temporary, output)
    return output


def execute(
    run, manifest, hands, checkpoint_every, max_states, reserve_gib, parent=None
):
    config = manifest["config"]
    binary = build()
    command = [str(binary), str(run), manifest["identity"], str(hands)]
    command += [
        str(config[k])
        for k in ("seed", "eval_seed", "particles", "panel_every", "panel_roots")
    ]
    command += [str(checkpoint_every), "0" if parent else "1"]
    command += [
        str(config[k])
        for k in ("window", "min_child", "max_splits", "max_depth", "gain", "min_rmse")
    ]
    command += [
        str(max_states),
        str(int(reserve_gib * 2**30)),
        str({"raw": 0, "centered": 1, "validated": 2}[config["split_rule"]]),
        str(config["validation_z"]),
        str(config["freeze_after"]),
        str(config["schedule"]["min_split_visits"]),
        str(config["schedule"]["split_check_visits"]),
    ]
    if parent:
        command += [str(parent), config["schedule"]["parent_identity"]]
    with (run / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                dict(
                    utc=datetime.now(timezone.utc).isoformat(),
                    command=command,
                    exe_sha256=base.legacy.sha256(binary),
                )
            )
            + "\n"
        )
    with (run / "console.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=base.legacy.PROJECT,
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
                print("Finishing the hand and saving; please wait.", flush=True)
                process.send_signal(signal.CTRL_BREAK_EVENT)
                remaining, _ = process.communicate()
                log.write(remaining)
                print(remaining, end="")
                return process.returncode


def fork(args):
    parent, destination = args.source.resolve(), args.run.resolve()
    if destination.exists() or not destination.is_relative_to(base.DATA.resolve()):
        raise ValueError(f"Use a NEW output directory under {base.DATA}")
    binary = build()
    subprocess.run([str(binary), "--self-test"], check=True)
    with base.writer_lock(parent):
        original = manifest_at(parent, scheduled=False)
        config = original["config"].copy()
        if (
            "schedule" in config
            or config["freeze_after"]
            or config["max_splits"] >= 0
            or config["max_depth"] >= 0
        ):
            raise ValueError("Fork an original uncapped, unfrozen DT run")
        hands = checkpoint_header(parent, original["identity"])
        if hands < 1:
            raise ValueError("Parent must contain training")
        minimum = (
            config["window"] if args.min_split_visits is None else args.min_split_visits
        )
        interval = (
            config["window"]
            if args.split_check_visits is None
            else args.split_check_visits
        )
        if any(
            n < config["window"] or n % config["window"] for n in (minimum, interval)
        ):
            raise ValueError(
                "Split visits must be positive multiples of the monitor window"
            )
        if args.mode == "freeze" and (args.min_split_visits or args.split_check_visits):
            raise ValueError("Split timing parameters only apply to continue mode")
        if args.eval_every < 1:
            raise ValueError("eval-every must be positive")
        if (
            shutil.disk_usage(base.DATA).free
            < (parent / "checkpoint.bin").stat().st_size + args.reserve_gib * 2**30
        ):
            raise ValueError(
                "Insufficient space for an independent branch and disk reserve"
            )
        support = leaf_support(parent)
        digest = base.legacy.sha256(parent / "checkpoint.bin")
        config.update(
            panel_every=args.eval_every,
            freeze_after=hands if args.mode == "freeze" else 0,
            initialization=f"fork of {parent} at {hands} hands; full regrets/averages/RNG inherited",
            metric="fixed fifth-street panel only; no per-hand evaluations; ante; not exploitability/LBR",
            schedule_source_sha256=schedule_hashes(),
            schedule=dict(
                version=1,
                mode=args.mode,
                parent=str(parent),
                parent_identity=original["identity"],
                parent_checkpoint_sha256=digest,
                branch_hands=hands,
                parent_splits=support["splits"],
                parent_checks=support["checks"],
                parent_leaves=support["leaves"],
                min_split_visits=minimum,
                split_check_visits=interval,
            ),
        )
        manifest = dict(identity=base.identity(config), config=config)
        destination.mkdir(parents=True)
        for name in ("hand_atlas.bin", "atlas_provenance.json"):
            shutil.copy2(parent / name, destination / name)
        shutil.copytree(parent / "source", destination / "source")
        for source in SOURCES:
            target = destination / "source" / source.relative_to(base.legacy.PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (destination / "config.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with base.writer_lock(destination):
            result = execute(
                destination, manifest, hands, 100000, 500000, args.reserve_gib, parent
            )
        if result:
            raise ValueError(
                f"Fork failed, preserving partial output for inspection: {destination}"
            )
        checkpoint_header(destination, manifest["identity"])
        if digest != base.legacy.sha256(parent / "checkpoint.bin"):
            raise ValueError("Parent checkpoint changed")
    print(f"FORK_READY mode={args.mode} hands={hands} run={destination}")


def train(args):
    run = args.run.resolve()
    if not run.is_relative_to(base.DATA.resolve()):
        raise ValueError("Use the family data directory")
    if args.hands < 1 or args.checkpoint_every < 1 or args.max_kmeans_states < 1:
        raise ValueError("Training budgets must be positive")
    with base.writer_lock(run):
        manifest = manifest_at(run)
        if shutil.disk_usage(run).free < args.reserve_gib * 2**30:
            raise ValueError("Insufficient disk reserve")
        # C++ restore trims only uncommitted suffixes in THIS branch, not its parent.
        return execute(
            run,
            manifest,
            args.hands,
            args.checkpoint_every,
            args.max_kmeans_states,
            args.reserve_gib,
        )


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    creation = commands.add_parser("fork", help="Create a branch without training")
    creation.add_argument("--source", type=Path, required=True)
    creation.add_argument("--run", type=Path, required=True)
    creation.add_argument("--mode", choices=("freeze", "continue"), required=True)
    creation.add_argument("--eval-every", type=int, default=10000)
    creation.add_argument(
        "--min-split-visits",
        type=int,
        help="Distinct monitored hands since leaf creation; multiple of window",
    )
    creation.add_argument(
        "--split-check-visits",
        type=int,
        help="Check every N monitored leaf hands; multiple of window",
    )
    creation.add_argument("--reserve-gib", type=float, default=10)
    training = commands.add_parser(
        "train", help="Resume a branch to a TOTAL hand count"
    )
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--hands", type=int, default=10000000)
    training.add_argument("--checkpoint-every", type=int, default=100000)
    training.add_argument("--max-kmeans-states", type=int, default=500000)
    training.add_argument("--reserve-gib", type=float, default=10)
    evaluation = commands.add_parser("lbr", help="Read-only frozen evaluation")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--pairs", type=int, default=250)
    evaluation.add_argument("--particles", type=int, default=64)
    evaluation.add_argument(
        "--seeds", type=int, nargs="+", default=[2026100811, 2026100812]
    )
    return cli


def main():
    args = parser().parse_args()
    if hasattr(args, "reserve_gib") and (
        not math.isfinite(args.reserve_gib) or args.reserve_gib < 0
    ):
        raise ValueError("reserve-gib must be finite and nonnegative")
    if args.command == "fork":
        fork(args)
    elif args.command == "train":
        raise SystemExit(train(args))
    else:
        manifest_at(args.run.resolve())
        base.evaluate_lbr(args)


if __name__ == "__main__":
    main()
