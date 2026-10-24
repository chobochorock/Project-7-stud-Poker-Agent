"""Bounded-RAM hand-recall MCCFR; SQLite is the resumable checkpoint and log.

Usage (project root): python -m agents.cpp_mccfr.hand_recall_disk --help
Prepare: ... prepare --source-run OLD_RUN --run NEW_RUN (no training)
Train:   ... train --run NEW_RUN --hands 1000000 (automatically resumes)
Plot:    ... plot --run NEW_RUN --window 100
See HAND_RECALL_DISK.md. Local BR-gap is not exploitability. Disk can fill up.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import tempfile

from . import hand_recall as legacy

FAMILY, PROJECT = legacy.FAMILY, legacy.PROJECT
SOURCES = (*legacy.SOURCES, FAMILY / "hand_recall_disk.cpp", Path(__file__))
POLICIES = ("recall_current", "baseline_current", "recall_average", "baseline_average")
HEADER = ["hands", *POLICIES, "root_seed"]
HEADER += [f"{policy}_{name}" for policy in POLICIES for name in ("queries", "missing")]
HEADER += [
    f"{arm}_{name}"
    for name in ("states", "nodes", "payload_bytes")
    for arm in ("recall", "baseline")
]
HEADER += ["peak_rss_bytes"]
HEADER += [
    f"{policy}_{name}"
    for policy in POLICIES
    for name in ("crossfit", "mean_action_se", "uniform_fallback")
]


def identity(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def source_hashes() -> dict:
    return {str(path.relative_to(PROJECT)): legacy.sha256(path) for path in SOURCES}


def build(check: bool = False) -> Path:
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ is required")
    prefix = Path(compiler).resolve().parents[1] / "opt"
    header, library = prefix / "include/sqlite3.h", prefix / "lib/libsqlite3.a"
    if not header.exists() or not library.exists():
        raise ValueError(f"Installed SQLite development files missing: {prefix}")
    digest = identity(
        dict(sources=source_hashes(), sqlite=legacy.sha256(library), compiler=compiler)
    )
    exe = FAMILY / "bin" / f"hand_recall_disk_{digest[:16]}.exe"
    if not exe.exists():
        exe.parent.mkdir(parents=True, exist_ok=True)
        temporary = exe.with_suffix(".building.exe")
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-static-libstdc++",
                "-static-libgcc",
                "-I",
                str(header.parent),
                str(FAMILY / "hand_recall_disk.cpp"),
                str(library),
                "-lpsapi",
                "-o",
                str(temporary),
            ],
            cwd=PROJECT,
            check=True,
        )
        os.replace(temporary, exe)
    if check:
        with tempfile.TemporaryDirectory(
            prefix="disk_parity_", dir=FAMILY / "data"
        ) as directory:
            subprocess.run(
                [str(exe), "--self-test", directory, str(legacy.ATLAS)], check=True
            )
    return exe


@contextmanager
def run_lock(run: Path):
    """OS-released single-writer lock also covers preparation and CSV export."""
    with (run / "writer.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise ValueError(
                "This run is already being used by another process"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def read_manifest(run: Path) -> dict:
    result = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if identity(result["config"]) != result["identity"]:
        raise ValueError("Manifest identity mismatch")
    return result


@contextmanager
def connect(run: Path):
    connection = sqlite3.connect((run / "model.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def metadata(run: Path) -> dict:
    with connect(run) as db:
        return {
            key: bytes(value).decode()
            for key, value in db.execute("SELECT k,v FROM meta WHERE k!='state'")
        }


def export_metrics(run: Path) -> None:
    """Stream a committed snapshot; legacy diagnostic fields stay blank."""
    with connect(run) as db:
        db.execute("BEGIN")
        for kind, filename in enumerate(("per_hand.csv", "panel.csv")):
            target = run / filename
            temporary = target.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(HEADER)
                for (line,) in db.execute(
                    "SELECT line FROM metrics WHERE kind=? ORDER BY hand,root", (kind,)
                ):
                    row = next(csv.reader([line]))
                    if len(row) not in (21, len(HEADER)):
                        raise ValueError("Unexpected metric schema")
                    writer.writerow(row + [""] * (len(HEADER) - len(row)))
            os.replace(temporary, target)
    print(f"EXPORTED committed metrics: {run}")


def invoke(exe: Path, run: Path, manifest: dict, args, mode: str) -> int:
    config = manifest["config"]
    command = [str(exe), mode, str(run), manifest["identity"], str(args.hands)]
    command += [
        str(config[key])
        for key in ("seed", "eval_seed", "particles", "panel_every", "panel_roots")
    ]
    command += [
        str(args.commit_every),
        str(args.db_cache_mib),
        str(args.node_cache_entries),
        str(int(args.reserve_gib * 2**30)),
        config["source_run"],
        config["source_identity"],
        str(args.report_every),
    ]
    with (run / "invocations.jsonl").open("a", encoding="utf-8") as out:
        out.write(
            json.dumps(
                dict(
                    utc=datetime.now(timezone.utc).isoformat(),
                    command=command,
                    exe_sha256=legacy.sha256(exe),
                )
            )
            + "\n"
        )
    with (run / "console.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=PROJECT,
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
                print(
                    "Saving the completed hand; please wait. Forced termination resumes the last commit."
                )
                process.send_signal(signal.CTRL_BREAK_EVENT)
                remaining, _ = process.communicate()
                print(remaining, end="")
                log.write(remaining)
                return process.returncode


def verify_legacy(source: Path) -> dict:
    manifest = read_manifest(source)
    if manifest["config"]["schema"] != "hand_recall_1":
        raise ValueError("Only the hand-recall RAM checkpoint can be migrated")
    if legacy.sha256(source / "hand_atlas.bin") != legacy.ATLAS_SHA:
        raise ValueError("Source atlas differs from the frozen atlas")
    for name, digest in manifest["config"]["source_sha256"].items():
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid source manifest path")
        archived, current = source / "source" / relative, PROJECT / relative
        if legacy.sha256(archived) != digest:
            raise ValueError(f"Archived source hash mismatch: {relative}")
        old, new = archived.read_text(encoding="utf-8"), current.read_text(
            encoding="utf-8"
        )
        if relative.name == "hand_recall.cpp":
            new = new.replace("#ifndef HAND_RECALL_NO_MAIN\n", "").removesuffix(
                "#endif\n"
            )
        if old != new:
            raise ValueError(
                f"Algorithm/evaluator changed since source checkpoint: {relative}"
            )
    return manifest


def prepare(run: Path, args) -> int:
    source = args.source_run.resolve()
    if source == run:
        raise ValueError("Use a NEW run directory; never overwrite the RAM run")
    original = verify_legacy(source)
    config = dict(
        original["config"],
        schema="hand_recall_disk_1",
        source_run=str(source),
        source_identity=original["identity"],
        source_checkpoint_sha256=legacy.sha256(source / "checkpoint.bin"),
        source_sha256=source_hashes(),
        storage="SQLite WITHOUT ROWID; lossless sparse-double payload; atomic nodes/RNG/metrics",
        diagnostics="original gap unchanged; signed split-sample crossfit + action SE from migration onwards",
    )
    manifest = dict(identity=identity(config), config=config)
    path = run / "config.json"
    if path.exists():
        if read_manifest(run) != manifest:
            raise ValueError("Existing disk run has a different source/configuration")
        if (run / "model.sqlite").exists() and metadata(run).get("status") == "ready":
            print("Already prepared; use train to resume. No checkpoint overwritten.")
            return 0
    else:
        if any(item.name != "writer.lock" for item in run.iterdir()):
            raise ValueError("New run directory must be empty")
        available = shutil.disk_usage(run).free
        if (
            available
            < int(args.reserve_gib * 2**30)
            + 2 * (source / "checkpoint.bin").stat().st_size
        ):
            raise ValueError(
                "Not enough free space for conservative migration estimate and reserve"
            )
        shutil.copy2(source / "hand_atlas.bin", run / "hand_atlas.bin")
        shutil.copy2(source / "atlas_provenance.json", run / "atlas_provenance.json")
        for file in SOURCES:
            destination = run / "source" / file.relative_to(PROJECT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return invoke(build(), run, manifest, args, "import")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser(
        "build",
        help="Compile; --check runs tiny parity/rollback checks, not a benchmark",
    )
    builder.add_argument("--check", action="store_true")
    for name in ("prepare", "train", "plot", "status"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True, type=Path)
        if name == "plot":
            command.add_argument("--window", type=int, default=100)
        if name not in ("prepare", "train"):
            continue
        if name == "prepare":
            command.add_argument("--source-run", required=True, type=Path)
        command.add_argument(
            "--hands",
            type=int,
            default=1_000_000,
            help="Total target, NOT additional hands",
        )
        command.add_argument("--db-cache-mib", type=int, default=256)
        command.add_argument("--node-cache-entries", type=int, default=32768)
        command.add_argument("--commit-every", type=int, default=25)
        command.add_argument("--reserve-gib", type=float, default=20)
        command.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()
    if args.command == "build":
        print(build(args.check))
        return
    run = args.run.resolve()
    if not run.is_relative_to((FAMILY / "data").resolve()):
        parser.error("Run must be inside agents/cpp_mccfr/data/")
    if args.command in ("prepare", "train"):
        for name in (
            "hands",
            "db_cache_mib",
            "node_cache_entries",
            "commit_every",
            "report_every",
        ):
            if getattr(args, name) < 1:
                parser.error(f"{name} must be positive")
        if not 0 <= args.reserve_gib < float("inf"):
            parser.error("reserve-gib must be finite and nonnegative")
    if args.command == "prepare":
        run.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        print(
            json.dumps(
                dict(
                    metadata=metadata(run),
                    db_gib=(run / "model.sqlite").stat().st_size / 2**30,
                    free_gib=shutil.disk_usage(run).free / 2**30,
                ),
                indent=2,
            )
        )
        return
    with run_lock(run):
        if args.command == "prepare":
            code = prepare(run, args)
        elif args.command == "train":
            manifest = read_manifest(run)
            if manifest["config"]["source_sha256"] != source_hashes():
                raise ValueError(
                    "Source changed since preparation; review before resuming"
                )
            if legacy.sha256(run / "hand_atlas.bin") != legacy.ATLAS_SHA:
                raise ValueError("Run atlas changed")
            code = invoke(build(), run, manifest, args, "train")
        else:
            code = 0
        if (run / "model.sqlite").exists() and metadata(run).get("status") == "ready":
            export_metrics(run)
        if args.command == "plot":
            from .hand_recall_plot import plot

            plot(run, args.window)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
