"""Head-to-head of small-K checkpoints against fixed references (fast complement to Policy-LBR).

Usage (project root):
    python -m agents.cpp_mccfr.small_k_plateau.league --tag NAME [--arms k1 k16 ...] [--modes average current]
        [--budgets 10k 1m 10m] [--hands 20000] [--seed 1307] [--run RUN]
Input:  RUN/<arm>/k<K>_<budget>.bin checkpoints and atlases; the reference K512 10M table; the frozen
        agents/evaluation/league/bin/evaluate_league.exe (7-stud v3 match engine, no training, no resolving).
Output: RUN/league/NAME/{roster.tsv, matches/matches.csv, matches/paired_payoffs.csv, vs_reference.csv}.
Metric: mean payoff of the checkpoint against the reference in ante/hand over seat-swapped deal pairs
    (positive = the checkpoint wins). This is one-on-one EV against ONE fixed opponent; it is not exploitability,
    not LBR, and its ranking need not match either. SE is over deal pairs (evaluation noise only).
References: ref_k512_10m_average, ref_k512_10m_current (existing reference table), heuristic.
The round robin also plays checkpoints against each other; those rows stay in matches.csv.
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = PROJECT / "agents/cpp_mccfr/data/small_k_plateau_20261010"
EXE = PROJECT / "agents/evaluation/league/bin/evaluate_league.exe"
REF_MODEL = PROJECT / "agents/cpp_mccfr/data/made_call_r1000_k512_epsheur20_memory16_10m.bin"
REF_ATLAS = PROJECT / "agents/cpp_mccfr/data/power512_epsheur20_memory16_v1.bin"


def budget(tag: str) -> int:
    return int(tag[:-1]) * (1000 if tag.endswith("k") else 1000000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--modes", nargs="+", default=["average"], choices=("average", "current"))
    parser.add_argument("--budgets", nargs="*", default=[])
    parser.add_argument("--hands", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1307)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / "league" / args.tag
    out.mkdir(parents=True, exist_ok=False)
    rows = [("ref_k512_10m_average", "mccfr", "average", REF_MODEL, REF_ATLAS),
            ("ref_k512_10m_current", "mccfr", "current", REF_MODEL, REF_ATLAS),
            ("heuristic", "heuristic", "current", "", "")]
    arms = args.arms or sorted(p.name for p in run.iterdir() if re.fullmatch(r"k\d+(_s\d+)?", p.name))
    for arm in arms:
        k = re.match(r"k(\d+)", arm).group(1)
        atlas = run / arm / f"power{k}_epsheur20_memory16_v1.bin"
        for model in sorted((run / arm).glob(f"k{k}_*.bin"), key=lambda p: budget(p.stem.split("_")[1])):
            tag = model.stem.split("_")[1]
            if args.budgets and tag not in args.budgets:
                continue
            for mode in args.modes:
                rows.append((f"{arm}_{tag}_{mode}", "mccfr", mode, model, atlas))
    with (out / "roster.tsv").open("w", encoding="utf-8", newline="") as handle:
        for name, kind, mode, model, atlas in rows:
            handle.write("\t".join([name, kind, mode, str(model), str(atlas), "0", "0"]) + "\n")
    with (out / "league.log").open("w") as log:
        subprocess.run([str(EXE), "--self-test", str(out / "roster.tsv")], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(EXE), str(out / "roster.tsv"), str(out / "matches"), str(args.hands), str(args.seed)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)

    pairs = {}
    with (out / "matches" / "paired_payoffs.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = (float(row["a_seat0_ante"]) + float(row["a_seat1_ante"])) / 2
            pairs.setdefault((row["a"], row["b"]), []).append(value)
    references = [name for name, *_ in rows[:3]]
    with (out / "vs_reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["player", "reference", "mean_ante_per_hand", "se", "pairs"])
        for name, *_ in rows[3:]:
            for reference in references:
                if (name, reference) in pairs:
                    values = np.array(pairs[(name, reference)])
                elif (reference, name) in pairs:
                    values = -np.array(pairs[(reference, name)])
                else:
                    continue
                writer.writerow([name, reference, f"{values.mean():.6f}", f"{values.std(ddof=1) / np.sqrt(len(values)):.6f}", len(values)])
    print(f"wrote {out / 'vs_reference.csv'} ({len(rows)} players)")


if __name__ == "__main__":
    main()
