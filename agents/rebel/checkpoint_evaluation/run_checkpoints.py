from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)
HERE = Path(__file__).resolve().parent
HEADER = struct.Struct("<8sIIQ")
MAGIC = b"RBV7S3\0\0"


def run(
    command: list[str], *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def combine_datasets(chunks: list[Path], output: Path) -> int:
    dimensions = 0
    count = 0
    payloads: list[bytes] = []
    for path in chunks:
        data = path.read_bytes()
        magic, version, current_dimensions, current_count = HEADER.unpack_from(data)
        if magic != MAGIC or version != 3:
            raise RuntimeError(f"invalid V7 chunk: {path}")
        if dimensions and dimensions != current_dimensions:
            raise RuntimeError("V7 chunk dimension mismatch")
        dimensions = current_dimensions
        record_size = (dimensions + 2) * 4
        payload = data[HEADER.size :]
        if len(payload) != current_count * record_size:
            raise RuntimeError(f"truncated V7 chunk: {path}")
        payloads.append(payload)
        count += current_count
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        target.write(HEADER.pack(MAGIC, 3, dimensions, count))
        for payload in payloads:
            target.write(payload)
    return count


def wait_for_server(log_path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("value server exited before accepting connections")
        if log_path.exists() and '"server": "ready"' in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.1)
    raise RuntimeError("value server did not start")


def parse_json_output(text: str) -> dict[str, object]:
    start = text.find("{")
    if start < 0:
        raise RuntimeError("evaluator returned no JSON")
    return json.loads(text[start:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[10000])
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "agents/cpp_mccfr/data/made_call_r1000_k512_epsheur20_memory16_30m.bin"
        ),
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("agents/cpp_mccfr/data/power512_epsheur20_memory16_v1.bin"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--lbr-hands", type=int, default=10000)
    parser.add_argument("--quick-lbr-hands", type=int, default=0)
    parser.add_argument("--lbr-particles", type=int, default=240)
    parser.add_argument("--full-lbr-checkpoints", type=int, nargs="*", default=[])
    parser.add_argument("--search-particles", type=int, default=64)
    parser.add_argument("--label-iterations", type=int, default=32)
    parser.add_argument("--sixth-iterations", type=int, default=32)
    parser.add_argument("--seventh-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=76001)
    parser.add_argument("--port", type=int, default=28761)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(set(args.checkpoints))
    if (
        not checkpoints
        or checkpoints[0] <= 0
        or any(value % 10000 for value in checkpoints)
    ):
        parser.error("checkpoints must be positive multiples of 10000")
    run_dir = HERE / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = HERE / "generate_v7_hands.exe"
    evaluator = HERE / "stud_rebel_recursive.exe"
    if not args.skip_build:
        run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                str(HERE / "generate_v7_hands.cpp"),
                "-o",
                str(generator),
            ]
        )
        run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                str(ROOT / "agents/cpp_mccfr/stud_rebel_recursive.cpp"),
                "-lws2_32",
                "-o",
                str(evaluator),
            ]
        )

    results_path = run_dir / "lbr_checkpoints.json"
    results: list[dict[str, object]] = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else []
    )
    full_lbr_checkpoints = set(args.full_lbr_checkpoints)
    if any(value not in checkpoints for value in full_lbr_checkpoints):
        parser.error("full LBR checkpoints must also appear in --checkpoints")
    if args.quick_lbr_hands < 0 or args.lbr_hands <= 0:
        parser.error("LBR hand counts must be positive")

    def evaluation_hands(checkpoint: int) -> int:
        if args.quick_lbr_hands and checkpoint not in full_lbr_checkpoints:
            return args.quick_lbr_hands
        return args.lbr_hands

    completed = {
        int(row["training_hands"])
        for row in results
        if int(row.get("lbr_hands", 0)) >= evaluation_hands(int(row["training_hands"]))
        and int(row.get("lbr_particles", 0)) == args.lbr_particles
    }
    chunks: list[Path] = []
    generated_hands = 0
    for checkpoint in checkpoints:
        while generated_hands < checkpoint:
            chunk_hands = min(10000, checkpoint - generated_hands)
            chunk = (
                run_dir
                / f"v7_chunk_{generated_hands:08d}_{generated_hands + chunk_hands:08d}.bin"
            )
            if not chunk.exists():
                run(
                    [
                        str(generator),
                        "--model",
                        str(args.model),
                        "--atlas",
                        str(args.atlas),
                        "--hands",
                        str(chunk_hands),
                        "--group-offset",
                        str(generated_hands),
                        "--particles",
                        str(args.search_particles),
                        "--iterations",
                        str(args.label_iterations),
                        "--samples-per-root",
                        "4",
                        "--report-every",
                        "1000",
                        "--output",
                        str(chunk),
                        "--seed",
                        str(args.seed + generated_hands),
                    ]
                )
            chunks.append(chunk)
            generated_hands += chunk_hands
        if checkpoint in completed:
            continue

        dataset = run_dir / f"v7_{checkpoint:08d}_hands.bin"
        sample_count = combine_datasets(chunks, dataset)
        value_model = run_dir / f"v7_{checkpoint:08d}_hands.pt"
        run(
            [
                str(args.python),
                "-B",
                str(HERE / "python_entry.py"),
                str(ROOT / "agents/rebel/train_rebel_v7.py"),
                "--input",
                str(dataset),
                "--output",
                str(value_model),
                "--hidden",
                "128",
                "--layers",
                "2",
                "--epochs",
                str(args.train_epochs),
                "--batch-size",
                "256",
                "--threads",
                "1",
                "--device",
                args.device,
                "--seed",
                str(args.seed + checkpoint + 1),
            ]
        )
        metadata = json.loads(
            value_model.with_suffix(".json").read_text(encoding="utf-8")
        )
        dataset.unlink()

        server_log_path = run_dir / f"value_server_{checkpoint:08d}.log"
        server_log = server_log_path.open("w", encoding="utf-8")
        server = subprocess.Popen(
            [
                str(args.python),
                "-B",
                str(HERE / "python_entry.py"),
                str(ROOT / "agents/rebel/rebel_value_server.py"),
                "--model",
                str(value_model),
                "--port",
                str(args.port),
                "--threads",
                "1",
            ],
            cwd=ROOT,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server(server_log_path, server)
            checkpoint_lbr_hands = evaluation_hands(checkpoint)
            evaluated = run(
                [
                    str(evaluator),
                    "--bucket",
                    "power-memory16",
                    "--model",
                    str(args.model),
                    "--atlas",
                    str(args.atlas),
                    "--value-port",
                    str(args.port),
                    "--hands",
                    str(checkpoint_lbr_hands),
                    "--lbr-particles",
                    str(args.lbr_particles),
                    "--sixth-particles",
                    str(args.search_particles),
                    "--sixth-iterations",
                    str(args.sixth_iterations),
                    "--sixth-prior",
                    "100",
                    "--seventh-particles",
                    str(args.search_particles),
                    "--seventh-iterations",
                    str(args.seventh_iterations),
                    "--seventh-prior",
                    "100",
                    "--progress-seconds",
                    "30",
                    "--seed",
                    str(args.seed + checkpoint + 2),
                ],
                capture=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
            server_log.close()

        report = parse_json_output(evaluated.stdout)
        (run_dir / f"lbr_{checkpoint:08d}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        results = [row for row in results if int(row["training_hands"]) != checkpoint]
        results.append(
            {
                "implementation": "partial-recursive-rebel-v7",
                "training_hands": checkpoint,
                "training_samples": sample_count,
                "validation_mae_ante": metadata["best_validation_mae_ante"],
                "lbr_ante_per_hand": report["average_profit_ante_for_lbr"],
                "standard_error": report["paired_standard_error_ante"],
                "ci95": report["ci95_ante_for_lbr"],
                "lbr_hands": checkpoint_lbr_hands,
                "lbr_particles": args.lbr_particles,
            }
        )
        results.sort(key=lambda row: int(row["training_hands"]))
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        run(
            [
                str(args.python),
                "-B",
                str(HERE / "plot_lbr_curve.py"),
                "--input",
                str(results_path),
                "--output",
                str(run_dir / "lbr_curve.png"),
            ]
        )

    print(
        json.dumps(
            {
                "status": "complete",
                "implementation": "partial-recursive-rebel-v7",
                "results": str(results_path),
                "graph": str(run_dir / "lbr_curve.png"),
            }
        )
    )


if __name__ == "__main__":
    main()
