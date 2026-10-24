"""Build/run hand-only k-means recall comparison; no training on import.

Usage (project root): python -m agents.cpp_mccfr.hand_recall --help
Train: ... hand_recall train --run agents/cpp_mccfr/data/hand_recall_1m
Plot:  ... hand_recall plot --run agents/cpp_mccfr/data/hand_recall_1m
Metric: posterior one-step Local BR-gap in ante, not exploitability/LBR.
See HAND_RECALL.md for the two keys, evaluation timing, and memory limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

FAMILY = Path(__file__).resolve().parent
PROJECT = FAMILY.parents[1]
ATLAS = FAMILY / "data/power256_selfplay100m_eps20_v1.bin"
ATLAS_SHA = "2944bed6ef966cf49a412c1ca3aca981f07436b6fb370cb7fb77a954737170a0"
EXE = FAMILY / "bin/hand_recall.exe"
SOURCES = (
    FAMILY / "hand_recall.cpp",
    FAMILY / "hand_recall.py",
    FAMILY / "hand_recall_plot.py",
    FAMILY / "stud_mccfr.cpp",
    FAMILY / "stud_rules.hpp",
    FAMILY / "local_split_helpers.hpp",
    FAMILY.parent / "lightgbm_regret_ensemble/stud_epoch_ensemble.cpp",
    PROJECT / "environments/seven_stud/stud_rules.hpp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(check: bool = False) -> None:
    EXE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-static-libstdc++",
            "-static-libgcc",
            str(FAMILY / "hand_recall.cpp"),
            "-o",
            str(EXE),
            "-lpsapi",
        ],
        cwd=PROJECT,
        check=True,
    )
    if check:
        subprocess.run([str(EXE), "--self-test"], check=True)


def train(args: argparse.Namespace) -> int:
    run = args.run.resolve()
    if not run.is_relative_to((FAMILY / "data").resolve()):
        raise ValueError("Run must be inside agents/cpp_mccfr/data/")
    if sha256(ATLAS) != ATLAS_SHA:
        raise ValueError("Frozen k-means atlas changed; review its provenance first")
    config = dict(
        schema="hand_recall_1",
        seed=args.seed,
        eval_seed=args.eval_seed,
        particles=args.particles,
        panel_every=args.panel_every,
        panel_roots=args.panel_roots,
        checkpoint_every=args.checkpoint_every,
        atlas_sha256=ATLAS_SHA,
        source_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in SOURCES},
        algorithm="signed alternating external-sampling MCCFR; SIMPLE average",
        budget="one fresh fifth-street deal + two player traversals per arm per hand",
        rules="C++ seven-stud v3; heuristic H4; ante1000; stack1000ante",
        initialization="zero regret and strategy mass; uniform unseen information keys",
        arms=dict(
            recall="past+current hand buckets; exact public action/chip history; seat",
            baseline="current hand bucket only; same exact public history and seat",
        ),
        metric="first fifth-street infoset conditional-particle one-step deviation gap, ante",
        timing="post-update; independent root every hand; current and average separate",
        sampling="frozen atlas fit on 100k old-model self-play hands, epsilon0.2, seed37301",
        approximation="existing 18D features, 128 completions max, hard256 per street",
        comparison="matched fresh k-means control; not a continuation of old checkpoints",
    )
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = run / "config.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["config"] != config or saved["identity"] != identity:
            raise ValueError(
                "Resume source/seed/atlas/evaluator mismatch; do not overwrite"
            )
        if sha256(run / "hand_atlas.bin") != ATLAS_SHA:
            raise ValueError("Run-local atlas is damaged")
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Nonempty run directory; choose a new directory or --resume")

    # Compile before creating a run, so a compiler failure cannot leave a fake run.
    build()
    if not args.resume:
        run.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ATLAS, run / "hand_atlas.bin")
        shutil.copy2(ATLAS.with_suffix(".json"), run / "atlas_provenance.json")
        for source in SOURCES:
            destination = run / "source" / source.relative_to(PROJECT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest_path.write_text(
            json.dumps(dict(identity=identity, config=config), indent=2) + "\n",
            encoding="utf-8",
        )
    invocation = dict(
        hands=args.hands,
        max_states=args.max_states,
        resume=args.resume,
        exe_sha256=sha256(EXE),
    )
    with (run / "invocations.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(invocation) + "\n")
    command = [
        str(EXE),
        str(run / "hand_atlas.bin"),
        str(run),
        str(args.hands),
        str(args.seed),
        str(args.eval_seed),
        str(args.particles),
        str(args.panel_every),
        str(args.panel_roots),
        str(args.checkpoint_every),
        str(args.max_states),
        str(int(args.resume)),
        identity,
    ]
    print(
        "Both arms train; one independent evaluated information set per hand.",
        flush=True,
    )
    print(
        "No old checkpoints loaded. Metrics: Local BR-gap, NOT exploitability.",
        flush=True,
    )
    with (run / "console.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT,
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                print(
                    "Interrupted. Resume from last atomic checkpoint; its suffix is replayed."
                )
                return 130


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser(
        "build", help="Compile only; --check runs non-training checks"
    )
    builder.add_argument("--check", action="store_true")
    trainer = sub.add_parser("train", help="Start/resume BOTH comparison arms")
    trainer.add_argument("--run", type=Path, required=True)
    trainer.add_argument("--hands", type=int, default=1_000_000)
    trainer.add_argument("--seed", type=int, default=111)
    trainer.add_argument("--eval-seed", type=int, default=307)
    trainer.add_argument("--particles", type=int, default=32)
    trainer.add_argument("--panel-every", type=int, default=10_000)
    trainer.add_argument("--panel-roots", type=int, default=128)
    trainer.add_argument("--checkpoint-every", type=int, default=10_000)
    trainer.add_argument(
        "--max-states",
        type=int,
        default=5_000_000,
        help="Combined state-count safety stop; NOT a RAM bound",
    )
    trainer.add_argument("--resume", action="store_true")
    plotter = sub.add_parser("plot", help="Plot existing traces only; never train")
    plotter.add_argument("--run", type=Path, required=True)
    plotter.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    if args.command == "build":
        build(args.check)
    elif args.command == "plot":
        from .hand_recall_plot import plot

        plot(args.run.resolve(), args.window)
    else:
        for key in (
            "hands",
            "seed",
            "eval_seed",
            "panel_every",
            "panel_roots",
            "checkpoint_every",
            "max_states",
        ):
            if getattr(args, key) < 1:
                parser.error(f"--{key.replace('_', '-')} must be positive")
        if args.particles < 2:
            parser.error("--particles must be >=2")
        raise SystemExit(train(args))


if __name__ == "__main__":
    main()
