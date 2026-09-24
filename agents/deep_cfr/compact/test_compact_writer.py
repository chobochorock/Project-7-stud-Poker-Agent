"""Compare dense and compact C++ samples from an identical traversal."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)
sys.path.insert(0, str(ROOT))

from train_deep_cfr_7th import read_records
from compact_reservoir import CompactReservoir, read_compact_records


def generate(path: Path, compact: bool) -> dict:
    command = [
        str(HERE / "deep_cfr_compact_traverse.exe"),
        "--uniform-policy",
        "1",
        "--traverser",
        "0",
        "--traversals",
        "2",
        "--iteration",
        "1",
        "--ante",
        "1000",
        "--stack-ante",
        "1000",
        "--start-street",
        "5",
        "--seed",
        "43101",
        "--output",
        str(path),
        "--compact-output",
        "1" if compact else "0",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    dense_path = HERE / ".writer_dense.bin"
    compact_path = HERE / ".writer_compact.bin"
    try:
        dense_result = generate(dense_path, False)
        compact_result = generate(compact_path, True)
        dense = read_records(dense_path)
        compact = read_compact_records(compact_path)
        assert len(dense) == len(compact)
        for field in ("iteration", "player", "kind", "legal_mask", "target"):
            assert np.array_equal(dense[field], compact[field])

        reservoir = CompactReservoir(len(compact), 5)
        reservoir.add(compact, np.random.default_rng(1))
        assert np.array_equal(dense["state"], reservoir.dense(np.arange(len(compact))))
        for key in (
            "nodes",
            "network_queries",
            "samples",
            "advantage_samples",
            "strategy_samples",
            "trajectory_br_gap_mean_ante",
            "trajectory_br_gap_standard_error_ante",
            "trajectory_br_gap_by_street_mean_ante",
        ):
            assert dense_result[key] == compact_result[key]
        print(
            json.dumps(
                {
                    "compact_writer_self_test": "ok",
                    "records": len(compact),
                    "dense_bytes": dense_path.stat().st_size,
                    "compact_bytes": compact_path.stat().st_size,
                    "file_compression_ratio": (
                        dense_path.stat().st_size / compact_path.stat().st_size
                    ),
                }
            )
        )
    finally:
        for name in ("dense", "compact"):
            mapped = locals().get(name)
            memory_map = getattr(mapped, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        dense_path.unlink(missing_ok=True)
        compact_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
