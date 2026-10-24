"""Continue a frozen log-gain run in bounded chunks with safe file-based handoff.

Usage (project root): python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain_chunks --help
fork preserves the original run. train resumes model/RNG/monitor without changing
learning settings; stop requests CTRL_BREAK after the C++ save handler is ready.
Data, console, per-chunk plots and the latest checkpoint stay in the run folder.
Requires the already installed psutil; low available RAM pauses, never caps splits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading

import psutil

from . import log_gain, log_gain_report, run as base
from ..dynamic_tree.review import checkpoint_header

DEFAULT_RUN = base.DATA / "log_gain_seed222_10m"


def event(run: Path, **values) -> None:
    record = dict(utc=datetime.now(timezone.utc).isoformat(), **values)
    with (run / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    temporary = run / "chunk_status.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, run / "chunk_status.json")


def family_path(path: Path) -> Path:
    path = path.resolve()
    if path == base.DATA.resolve() or not path.is_relative_to(base.DATA.resolve()):
        raise ValueError(f"Use a run folder under {base.DATA}")
    return path


def fork(source: Path, destination: Path) -> None:
    source, destination = family_path(source), family_path(destination)
    if destination.exists():
        raise ValueError("Fork destination must not exist")
    with base.writer_lock(source):
        manifest = log_gain.manifest_at(source)
        hands = checkpoint_header(source, manifest["identity"])
        digest = base.legacy.sha256(source / "checkpoint.bin")
        needed = (source / "checkpoint.bin").stat().st_size + 10 * 2**30
        if shutil.disk_usage(base.DATA).free < needed:
            raise ValueError("Insufficient space for fork and disk reserve")
        destination.mkdir()
        for name in (
            "config.json",
            "checkpoint.bin",
            "hand_atlas.bin",
            "atlas_provenance.json",
            "per_hand.csv",
            "panel.csv",
            "splits.csv",
        ):
            shutil.copy2(source / name, destination / name)
        shutil.copytree(source / "source", destination / "source")
        if base.legacy.sha256(destination / "checkpoint.bin") != digest:
            raise ValueError("Fork checkpoint hash mismatch")
        event(
            destination,
            state="forked",
            source=str(source),
            hands=hands,
            parent_checkpoint_sha256=digest,
            model_identity=manifest["identity"],
        )
    print(f"FORK_READY hands={hands} run={destination}", flush=True)


def command_for(run: Path, manifest: dict, target: int, args) -> list[str]:
    config = manifest["config"]
    command = [str(log_gain.build()), str(run), manifest["identity"], str(target)]
    command += [
        str(config[k])
        for k in ("seed", "eval_seed", "particles", "panel_every", "panel_roots")
    ]
    command += [str(args.checkpoint_every), "1"]
    command += [str(config[k]) for k in ("window", "min_child", "min_rmse")]
    command += [
        str(config["fixed_gain"] or 0),
        str(args.max_kmeans_states),
        str(int(args.reserve_gib * 2**30)),
    ]
    return command


def execute(run: Path, manifest: dict, target: int, args) -> int:
    command = command_for(run, manifest, target, args)
    with (run / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                dict(
                    utc=datetime.now(timezone.utc).isoformat(),
                    command=command,
                    exe_sha256=base.legacy.sha256(Path(command[0])),
                    controller_sha256=base.legacy.sha256(Path(__file__)),
                )
            )
            + "\n"
        )
    messages = queue.Queue()
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

            def read_output():
                for line in process.stdout:
                    messages.put(line)
                messages.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            ready = False
            stop_reason = None
            signaled = False
            event(
                run,
                state="running",
                pid=process.pid,
                target=target,
                checkpoint_every=args.checkpoint_every,
                min_free_gib=args.min_free_gib,
            )
            while True:
                try:
                    try:
                        line = messages.get(timeout=0.25)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        break
                    if line:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        ready = ready or line.startswith("CONFIG ")
                    if (run / "STOP_REQUESTED").exists():
                        stop_reason = stop_reason or "handoff_request"
                    if psutil.virtual_memory().available < args.min_free_gib * 2**30:
                        stop_reason = stop_reason or "low_available_ram"
                    if (
                        ready
                        and stop_reason
                        and not signaled
                        and process.poll() is None
                    ):
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                        signaled = True
                        event(
                            run,
                            state="saving",
                            pid=process.pid,
                            target=target,
                            reason=stop_reason,
                        )
                        print(
                            f"SAFE_STOP reason={stop_reason}; waiting for checkpoint",
                            flush=True,
                        )
                except KeyboardInterrupt:
                    stop_reason = "keyboard_interrupt"
            reader.join()
            result = process.wait()
    hands = checkpoint_header(run, manifest["identity"])
    event(
        run,
        state="chunk_complete" if result == 0 and not stop_reason else "paused",
        hands=hands,
        target=target,
        returncode=result,
        reason=stop_reason,
        checkpoint_bytes=(run / "checkpoint.bin").stat().st_size,
    )
    return 2 if stop_reason and result == 0 else result


def train(args) -> int:
    run = family_path(args.run)
    for name in ("hands", "chunk_hands", "checkpoint_every", "max_kmeans_states"):
        if not 1 <= getattr(args, name) < 2**64:
            raise ValueError(f"Invalid {name}")
    for name in ("reserve_gib", "min_free_gib"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"Invalid {name}")
    control = run / "controller"
    control.mkdir(exist_ok=True)
    with base.writer_lock(control):
        with base.writer_lock(run):
            manifest = log_gain.manifest_at(run)
            hands = checkpoint_header(run, manifest["identity"])
            if args.hands < hands:
                raise ValueError("Target is below saved hand count")
            # Starting this command is explicit permission to resume a paused run.
            (run / "STOP_REQUESTED").unlink(missing_ok=True)
        while hands < args.hands:
            if (run / "STOP_REQUESTED").exists():
                event(run, state="paused", hands=hands, reason="handoff_between_chunks")
                return 2
            if psutil.virtual_memory().available < args.min_free_gib * 2**30:
                event(
                    run,
                    state="paused",
                    hands=hands,
                    reason="low_available_ram_before_chunk",
                )
                return 2
            if shutil.disk_usage(run).free < args.reserve_gib * 2**30:
                event(run, state="paused", hands=hands, reason="low_disk_before_chunk")
                return 2
            target = min(args.hands, (hands // args.chunk_hands + 1) * args.chunk_hands)
            with base.writer_lock(run):
                result = execute(run, manifest, target, args)
            if result:
                return result
            hands = checkpoint_header(run, manifest["identity"])
            if hands != target:
                raise ValueError("Worker reported success before its target")
            output = run / f"review_{hands}"
            if not output.exists():
                log_gain_report.report(run, output)
            event(run, state="milestone", hands=hands, report=str(output))
        event(run, state="complete", hands=hands)
    return 0


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    creation = sub.add_parser("fork")
    creation.add_argument("--source", type=Path, default=log_gain.DEFAULT_RUN)
    creation.add_argument("--run", type=Path, default=DEFAULT_RUN)
    training = sub.add_parser("train")
    training.add_argument("--run", type=Path, default=DEFAULT_RUN)
    for name, default in dict(
        hands=10000000,
        chunk_hands=1000000,
        checkpoint_every=100000,
        max_kmeans_states=500000,
    ).items():
        training.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    training.add_argument("--reserve-gib", type=float, default=10)
    training.add_argument(
        "--min-free-gib",
        type=float,
        default=3,
        help="Pause/save below this available system RAM; no split cap",
    )
    stop = sub.add_parser("stop")
    stop.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "fork":
        fork(args.source, args.run)
    elif args.command == "stop":
        path = family_path(args.run)
        if not (path / "config.json").is_file():
            raise ValueError("Run does not exist")
        (path / "STOP_REQUESTED").touch()
        print(
            "STOP_REQUESTED; wait for PAUSED/checkpoint before starting another writer"
        )
    else:
        raise SystemExit(train(args))
