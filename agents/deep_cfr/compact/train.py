"""Run the existing Deep CFR algorithm with lossless compact reservoirs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)
sys.path.insert(0, str(ROOT))

import train_deep_cfr_7th as base
from compact_reservoir import (
    CompactReservoir,
    compact_fit_metrics,
    read_compact_records,
)


base_run = base.run


def compact_run(args) -> None:
    default_generator = ROOT / "agents/cpp_mccfr" / "deep_cfr_traverse.exe"
    if args.generator == default_generator:
        args.generator = HERE / "deep_cfr_compact_traverse.exe"
    base.Reservoir = CompactReservoir
    base.fit_metrics = compact_fit_metrics
    base.read_records = read_compact_records
    args.compact_output = True
    base_run(args)
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "trainer": "deep-cfr-stud-compact-reservoir",
            "reservoir_storage": "lossless-token-v1",
            "reservoir_bytes_per_slot": CompactReservoir(1, 5).bytes_per_slot,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    base.run = compact_run
    base.main()
