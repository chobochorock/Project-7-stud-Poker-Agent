"""Train/evaluate frozen AE or VQ buckets with the existing C++ signed MCCFR.

Usage (project root): python -m agents.autoencoder_abstraction.run_cfr --help
Input: three trusted power-scope checkpoint.pt files and their dataset.npz.
Output: new data/<run>/ with frozen atlas, final policy, CSV, plots and metadata.
One hand = one fresh fifth-street root, traversed for each player. Not CFR+.
Local BR-gap and observed LBR profit are approximate, NOT exploitability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from agents.autoencoder_abstraction.experiment import FAMILY, ROOT, assign
from agents.state_action_embedding.experiment import write_json


SOURCES = [
    FAMILY / "neural_mccfr.cpp",
    ROOT / "agents/cpp_mccfr/stud_mccfr.cpp",
    ROOT / "agents/lightgbm_regret_ensemble/stud_epoch_ensemble.cpp",
    ROOT / "environments/seven_stud/stud_rules.hpp",
    ROOT / "agents/cpp_mccfr/local_split_helpers.hpp",
    Path(__file__).resolve(),
    FAMILY / "experiment.py",
    FAMILY / "comparison.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> Path:
    binary = FAMILY / "bin/neural_mccfr.exe"
    if not binary.exists() or any(
        p.suffix in (".cpp", ".hpp") and p.stat().st_mtime > binary.stat().st_mtime
        for p in SOURCES
    ):
        binary.parent.mkdir(exist_ok=True)
        subprocess.run([
            "g++", "-O2", "-std=c++17", "-DNOMINMAX", str(SOURCES[0]),
            "-o", str(binary), "-lpsapi",
        ], check=True)
    return binary


def export_and_check(args, binary: Path) -> dict:
    atlas = args.out_dir / "neural_atlas.bin"
    with np.load(args.source_dir / "dataset.npz", allow_pickle=False) as data:
        rows = data["rows"]
    fixture_type = np.dtype([("street", "u1"), ("code", "<u2"), ("power", "<f4", (18,))])
    fixture = np.empty(len(rows), fixture_type)
    fixture["street"], fixture["power"] = rows["street"], rows["power"]
    sources = {}
    with atlas.open("wb") as output:
        output.write(b"NAEAT01\0")
        for street in (5, 6, 7):
            path = args.source_dir / f"power_{args.method}_s{street}_seed{args.representation_seed}/checkpoint.pt"
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            if any(checkpoint[key] != expected for key, expected in {
                "method": args.method, "scope": "power", "street": street,
                "features": 18, "seed": args.representation_seed, "format_version": 1,
            }.items()):
                raise ValueError(f"Incompatible checkpoint: {path}")
            network = checkpoint["model"]
            hidden = checkpoint["architecture"]["hidden"]
            latent = checkpoint["architecture"]["latent"]
            codes = checkpoint["codes"]
            centers = checkpoint["centers"] if args.method == "ae_kmeans" else network["codebook.weight"]
            arrays = [network[key] for key in (
                "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias"
            )] + [centers]
            shapes = [(hidden, 18), (hidden,), (latent, hidden), (latent,), (codes, latent)]
            output.write(struct.pack("<4I", 18, hidden, latent, codes))
            for tensor, shape in zip(arrays, shapes):
                array = np.asarray(tensor, dtype="<f4")
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"Invalid weights: {path}")
                output.write(array.tobytes())
            selected = rows["street"] == street
            fixture["code"][selected] = assign(checkpoint, rows["power"][selected])
            sources[str(path.resolve())] = sha256(path)
    # Temporary parity inputs are reproducible from the retained source dataset.
    with tempfile.TemporaryDirectory(prefix="parity_", dir=args.out_dir) as temporary:
        test_path = Path(temporary) / "rows.bin"
        with test_path.open("wb") as output:
            output.write(struct.pack("<I", len(fixture)))
            output.write(fixture.tobytes())
        result = subprocess.run([str(binary), "--self-test", str(atlas), str(test_path)],
                                check=True, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        (args.out_dir / "validation.txt").write_text(result.stdout, encoding="utf-8")
        malformed = Path(temporary) / "bad.bin"
        original = atlas.read_bytes()
        for corrupt in (original[:-1], original + b"extra", original[:24] + struct.pack("<f", float("nan")) + original[28:]):
            malformed.write_bytes(corrupt)
            failed = subprocess.run([str(binary), "--self-test", str(malformed), str(test_path)],
                                    capture_output=True, text=True)
            if failed.returncode == 0:
                raise AssertionError("Malformed neural atlas accepted")
    return {"source_checkpoints": sources, "parity_rows": len(fixture), "bucket_mismatches": 0,
            "atlas_sha256": sha256(atlas), "dataset_sha256": sha256(args.source_dir / "dataset.npz")}


def mean_ci(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    half_width = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width,
            "samples": len(values)}


def report(out_dir: Path) -> dict:
    from agents.autoencoder_abstraction.comparison import curve, plot_comparison

    solver = out_dir / "solver"
    gap = np.genfromtxt(solver / "local_gap.csv", delimiter=",", names=True)
    payoffs = np.genfromtxt(solver / "lbr_pairs.csv", delimiter=",", names=True)
    progress = np.atleast_1d(np.genfromtxt(solver / "progress.csv", delimiter=",", names=True))
    with (solver / "lbr_queries.csv").open(newline="", encoding="utf-8") as stream:
        queries = {row["policy"]: row for row in csv.DictReader(stream)}
    summary = {"training": {key: float(progress[-1][key]) for key in progress.dtype.names},
               "lbr": {}, "local_gap": []}
    for policy in ("current", "average"):
        for hands in np.unique(gap["hands"]):
            metric = mean_ci(gap[policy][gap["hands"] == hands])
            summary["local_gap"].append({"hands": int(hands), "policy": policy, **metric})
        paired = (payoffs[f"{policy}_seat0"] + payoffs[f"{policy}_seat1"]) / 2
        metric = mean_ci(paired)
        query = queries[policy]
        metric["policy_queries"] = int(query["queries"])
        metric["missing_buckets"] = int(query["misses"])
        metric["missing_rate"] = int(query["misses"]) / max(1, int(query["queries"]))
        summary["lbr"][policy] = metric
    write_json(out_dir / "summary.json", summary)
    arguments = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))["arguments"]
    label = "AE" if arguments["method"] == "ae_kmeans" else "VQ-VAE"
    plot_comparison({label: curve(summary)}, arguments, out_dir / "metrics.png")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ae_kmeans", "vqvae"), required=True)
    parser.add_argument("--source-dir", type=Path, default=FAMILY / "data/power_lowfold_10k_20260923")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--representation-seed", type=int, default=11)
    parser.add_argument("--seed", type=int, default=7, help="MCCFR root/action seed")
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--eval-roots", type=int, default=128)
    parser.add_argument("--eval-particles", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=307)
    parser.add_argument("--lbr-pairs", type=int, default=500, help="Two seat-swapped hands per pair, per policy")
    parser.add_argument("--lbr-particles", type=int, default=64)
    args = parser.parse_args()
    for name in ("hands", "eval_every", "eval_particles", "lbr_particles"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eval_roots < 2 or args.lbr_pairs < 2 or min(args.seed, args.eval_seed, args.representation_seed) < 0:
        parser.error("At least 2 evaluation roots/pairs and nonnegative seeds required")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    binary = build()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    metadata = {"arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "command": subprocess.list2cmdline([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES},
                "algorithm": "signed external-sampling MCCFR, alternating traversers per root",
                "average": "existing external-sampling non-traverser strategy_sum; no linear weighting",
                "initial_policy": "zero regrets, uniform legal actions",
                "partition": "seat + street + frozen card code + exact legal mask + cumulative betting context",
                "checkpoint": "tables and frozen atlas only, NOT exact RNG/optimizer resume",
                "status": "validating"}
    write_json(args.out_dir / "run.json", metadata)
    with threadpool_limits(limits=4):
        metadata.update(export_and_check(args, binary))
    metadata["status"] = "training"
    write_json(args.out_dir / "run.json", metadata)
    command = [str(binary), str(args.out_dir / "neural_atlas.bin"), str(args.out_dir / "solver"),
               str(args.hands), str(args.seed), str(args.eval_every), str(args.eval_roots),
               str(args.eval_particles), str(args.lbr_pairs), str(args.lbr_particles), str(args.eval_seed)]
    with (args.out_dir / "run.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    report(args.out_dir)
    metadata.update(status="complete", wall_seconds=time.perf_counter() - started,
                    policy_sha256=sha256(args.out_dir / "solver" / f"policy_{args.hands}.bin"))
    write_json(args.out_dir / "run.json", metadata)
    print(f"RESULTS {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
