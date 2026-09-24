"""Evaluate a saved 7-Stud Deep CFR policy."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--port", type=int, default=28731)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--stack-ante", type=int, default=1000)
    parser.add_argument("--start-street", type=int, choices=(5, 6, 7), default=5)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--opponent", choices=("heuristic", "policy-lbr"), default="heuristic"
    )
    parser.add_argument("--belief-particles", type=int, default=240)
    parser.add_argument("--report-every", type=int, default=0)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=ROOT / "agents/cpp_mccfr" / "deep_cfr_evaluate.exe",
    )
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"model does not exist: {args.model}")
    if (
        args.hands <= 0
        or args.hands % 2
        or args.belief_particles <= 0
        or args.progress_seconds <= 0
    ):
        parser.error("hands must be positive and even")
    if args.build or not args.evaluator.exists():
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                str(ROOT / "agents/cpp_mccfr" / "deep_cfr_evaluate.cpp"),
                "-o",
                str(args.evaluator),
                "-lws2_32",
            ],
            check=True,
        )

    server = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(ROOT / "agents/deep_cfr/deep_cfr_ipc_server.py"),
            "--port",
            str(args.port),
            "--model",
            str(args.model),
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--seed",
            str(args.seed),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    ready = server.stdout.readline() if server.stdout else ""
    if '"server": "ready"' not in ready:
        server.kill()
        raise RuntimeError(f"inference server failed: {ready.strip()}")
    evaluated = subprocess.run(
        [
            str(args.evaluator),
            "--port",
            str(args.port),
            "--hands",
            str(args.hands),
            "--ante",
            str(args.ante),
            "--stack-ante",
            str(args.stack_ante),
            "--seed",
            str(args.seed),
            "--start-street",
            str(args.start_street),
            "--opponent",
            args.opponent,
            "--belief-particles",
            str(args.belief_particles),
            "--report-every",
            str(args.report_every),
            "--progress-seconds",
            str(args.progress_seconds),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    server.communicate(timeout=30)
    if server.returncode:
        raise RuntimeError(f"inference server exited {server.returncode}")
    print(evaluated.stdout, end="")


if __name__ == "__main__":
    main()
