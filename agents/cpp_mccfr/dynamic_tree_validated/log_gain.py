"""Train a fresh centered DT to 1M hands with an inverse-log split threshold.

Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain train
Output: agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_1m/.
Runs matched hard256 k-means too; fixed-root Local BR-gap every 10k hands.
See LOG_GAIN.md for resume, fixed-gain control, plots, LBR and limitations.
This is a heuristic split schedule, not a Hoeffding test or CFR guarantee.
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

SOURCES = (Path(__file__).resolve(), base.HERE / "log_gain.cpp")
DEFAULT_RUN = base.DATA / "log_gain_seed222_1m"


def hashes():
    return {
        str(p.relative_to(base.legacy.PROJECT)): base.legacy.sha256(p) for p in SOURCES
    }


def manifest_at(run):
    manifest = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (
        manifest["identity"] != base.identity(config)
        or config["schema"] != "dynamic_tree_validated_1"
        or config["source_sha256"] != base.hashes()
        or config.get("log_gain_source_sha256") != hashes()
        or base.legacy.sha256(run / "hand_atlas.bin") != config["atlas_sha256"]
    ):
        raise ValueError("Source, configuration or atlas mismatch")
    for name, digest in dict(base.hashes(), **hashes()).items():
        if base.legacy.sha256(run / "source" / name) != digest:
            raise ValueError(f"Source snapshot mismatch: {name}")
    return manifest


def build(check=False):
    digest = base.identity(dict(base.hashes(), **hashes()))[:16]
    output = base.legacy.FAMILY / "bin" / f"dt_log_gain_{digest}.exe"
    if not output.exists():
        compiler = shutil.which("g++")
        if not compiler:
            raise ValueError("g++ is required")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".{os.getpid()}.building.exe")
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-static-libstdc++",
                "-static-libgcc",
                str(base.HERE / "log_gain.cpp"),
                "-o",
                str(temporary),
                "-lpsapi",
            ],
            check=True,
        )
        os.replace(temporary, output)
    if check:
        subprocess.run([str(output), "--self-test"], check=True)
    return output


def configuration(args):
    return dict(
        schema="dynamic_tree_validated_1",
        **{
            k: getattr(args, k)
            for k in (
                "seed",
                "eval_seed",
                "particles",
                "panel_every",
                "panel_roots",
                "window",
                "min_child",
                "min_rmse",
                "fixed_gain",
            )
        },
        split_rule="centered",
        max_splits=-1,
        max_depth=-1,
        freeze_after=0,
        validation_z=2.0,
        gain=args.fixed_gain or 1.0,
        gain_schedule="fixed"
        if args.fixed_gain is not None
        else "1/(1+log10(max(1,t)))",
        t_unit="total training hands, not traversals or leaf visits; survives resume",
        source_sha256=base.hashes(),
        log_gain_source_sha256=hashes(),
        atlas_sha256=base.legacy.ATLAS_SHA,
        algorithm="signed alternating external MCCFR; traversal-frozen policies; SIMPLE average",
        budget="one fresh fifth-street deal and two traversals per model per hand",
        rules="C++ stud v3 HU, heuristic H4, ante1000, stack1000ante",
        tree="root per street/seat/legal mask; 34D observable summary; centered-label local CART",
        monitor="one uniformly sampled visit per leaf/hand; recent window; checks every window leaf hands",
        initialization="zero regrets and average sums; no pretrained strategy or tree",
        baseline="fresh existing hard256 betting-summary partition plus explicit seat",
        data_bias="atlas from old epsilon0.2 self-play100k seed37301; tree labels from own evolving policy",
        metric="fixed fifth-street Local BR-gap panel only; no per-hand evaluations; ante; not exploitability/LBR",
        inheritance="fractional regret/average mass; approximate history allocation; no perfect recall guarantee",
    )


def validate(args):
    for name in (
        "hands",
        "seed",
        "eval_seed",
        "panel_every",
        "panel_roots",
        "checkpoint_every",
        "max_kmeans_states",
    ):
        if not 1 <= getattr(args, name) < 2**64:
            raise ValueError(f"{name} must be a positive uint64")
    if not (
        2 <= args.particles <= 2**31 - 1
        and 2 <= args.panel_roots <= 2**31 - 1
        and 4 <= args.window <= 100000
        and 2 <= args.min_child <= args.window // 2
        and math.isfinite(args.min_rmse)
        and args.min_rmse >= 0
        and math.isfinite(args.reserve_gib)
        and args.reserve_gib >= 0
        and (args.fixed_gain is None or 0 < args.fixed_gain <= 1)
    ):
        raise ValueError("Invalid monitor, gain, evaluation or disk setting")


def train(args):
    validate(args)
    run = args.run.resolve()
    if not run.is_relative_to(base.DATA.resolve()) or run == base.DATA.resolve():
        raise ValueError(f"Use a run folder under {base.DATA}")
    if base.legacy.sha256(base.legacy.ATLAS) != base.legacy.ATLAS_SHA:
        raise ValueError("Frozen atlas changed")
    if shutil.disk_usage(base.legacy.FAMILY).free < args.reserve_gib * 2**30:
        raise ValueError("Insufficient disk reserve")
    config = configuration(args)
    manifest = dict(identity=base.identity(config), config=config)
    if args.resume:
        if manifest_at(run) != manifest:
            raise ValueError(
                "Resume configuration mismatch; retain the original learning settings"
            )
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Nonempty run directory: use --resume or a new run")
    binary = build()
    run.mkdir(parents=True, exist_ok=True)
    with base.writer_lock(run):
        if not args.resume:
            if any(p.name != "writer.lock" for p in run.iterdir()):
                raise ValueError("Run changed while acquiring lock")
            shutil.copy2(base.legacy.ATLAS, run / "hand_atlas.bin")
            shutil.copy2(
                base.legacy.ATLAS.with_suffix(".json"), run / "atlas_provenance.json"
            )
            for source in (*base.SOURCES, *SOURCES):
                target = run / "source" / source.relative_to(base.legacy.PROJECT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            (run / "config.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        command = [str(binary), str(run), manifest["identity"]]
        command += [
            str(getattr(args, k))
            for k in (
                "hands",
                "seed",
                "eval_seed",
                "particles",
                "panel_every",
                "panel_roots",
                "checkpoint_every",
            )
        ]
        command += [
            str(int(args.resume)),
            str(args.window),
            str(args.min_child),
            str(args.min_rmse),
            str(args.fixed_gain or 0),
            str(args.max_kmeans_states),
            str(int(args.reserve_gib * 2**30)),
        ]
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


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("build").add_argument("--check", action="store_true")
    training = commands.add_parser(
        "train", help="Fresh one-shot run; explicit --resume after interruption"
    )
    training.add_argument("--run", type=Path, default=DEFAULT_RUN)
    for name, default in dict(
        hands=1000000,
        seed=222,
        eval_seed=307,
        particles=32,
        panel_every=10000,
        panel_roots=128,
        checkpoint_every=10000,
        window=256,
        min_child=32,
        max_kmeans_states=500000,
    ).items():
        training.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    training.add_argument(
        "--min-rmse", type=float, default=1.0, help="Ante; centered-label parent RMSE"
    )
    training.add_argument(
        "--fixed-gain",
        type=float,
        help="Optional fixed-threshold control; omit for inverse-log",
    )
    training.add_argument(
        "--reserve-gib", type=float, default=10, help="Disk reserve, NOT RAM"
    )
    training.add_argument("--resume", action="store_true")
    plotting = commands.add_parser(
        "plot", help="Read-only plot of a stopped, saved run"
    )
    plotting.add_argument("--run", type=Path, default=DEFAULT_RUN)
    plotting.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("lbr", help="Read-only fixed-policy LBR")
    evaluation.add_argument("--run", type=Path, default=DEFAULT_RUN)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--pairs", type=int, default=250)
    evaluation.add_argument("--particles", type=int, default=64)
    evaluation.add_argument(
        "--seeds", type=int, nargs="+", default=[2026100821, 2026100822]
    )
    return cli


def main():
    args = parser().parse_args()
    if args.command == "build":
        print(build(args.check))
    elif args.command == "lbr":
        manifest_at(args.run.resolve())
        base.evaluate_lbr(args)
    elif args.command == "plot":
        from .log_gain_report import report

        report(args.run.resolve(), args.output.resolve())
    else:
        result = train(args)
        if result == 0:
            from .log_gain_report import report

            output = args.run.resolve() / f"review_{args.hands}"
            if not output.exists():
                report(args.run.resolve(), output)
            else:
                print(f"REPORT already exists, preserved: {output}")
        raise SystemExit(result)


if __name__ == "__main__":
    main()
