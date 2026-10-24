"""Particle-count sensitivity: the same deals scored with 256 vs 64 PolicyLBR particles.

Usage (project root):
    python -m agents.cpp_mccfr.neural_distill.particle_check --run RUN --high lbr_particles256 --low lbr_confirm \
        --seed 155250792 --pairs 1000
For every policy label in RUN/HIGH, takes the first PAIRS deals of RUN/LOW/<label>_<seed> (the low-particle run
must have the identical deal stream; checked) and reports student-minus-teacher (current/average) under both
particle counts with deal-pair bootstrap 95% intervals. Writes RUN/summary_<HIGH>.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from agents.cpp_mccfr.neural_distill.run import estimate


def rows(directory: Path, pairs: int) -> list[dict]:
    with (directory / "paired_payoffs.csv").open(encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    if len(data) < pairs:
        raise ValueError(f"{directory}: {len(data)} pairs, need {pairs}")
    return data[:pairs]


def column(data: list[dict], name: str) -> np.ndarray:
    return np.array([[(float(r[f"{name}_seat0"]) + float(r[f"{name}_seat1"])) / 2 for r in data]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--high", default="lbr_particles256")
    parser.add_argument("--low", default="lbr_confirm")
    parser.add_argument("--seed", type=int, default=155250792)
    parser.add_argument("--pairs", type=int, default=1000)
    args = parser.parse_args()
    result = {"high": args.high, "low": args.low, "seed": args.seed, "pairs": args.pairs, "policies": {}}
    for directory in sorted((args.run / args.high).glob(f"*_{args.seed}")):
        if not (directory / "evaluation.json").is_file():
            continue
        label = directory.name[: -len(f"_{args.seed}")]
        high = rows(directory, args.pairs)
        low = rows(args.run / args.low / directory.name, args.pairs)
        if [r["deal_seed"] for r in high] != [r["deal_seed"] for r in low]:
            raise ValueError(f"{label}: deal streams differ between {args.high} and {args.low}")
        entry = {}
        for tag, data in (("high", high), ("low", low)):
            for policy in ("current", "average"):
                model, teacher = column(data, f"model_{policy}"), column(data, f"baseline_{policy}")
                entry[f"{tag}_{policy}"] = dict(student=estimate(model), teacher=estimate(teacher),
                                                minus_teacher=estimate(model - teacher))
        result["policies"][label] = entry
    (args.run / f"summary_{args.high}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for label, entry in result["policies"].items():
        for key, value in entry.items():
            d = value["minus_teacher"]
            print(f"{label:22s} {key:13s} student {value['student']['mean']:.3f} teacher {value['teacher']['mean']:.3f} "
                  f"diff {d['mean']:+.3f} [{d['bootstrap_ci95'][0]:+.3f}, {d['bootstrap_ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
