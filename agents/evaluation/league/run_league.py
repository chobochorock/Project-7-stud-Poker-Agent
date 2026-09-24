"""Run a frozen 7-stud round robin and create payoff matrices and confidence intervals.

From the project root:
  python agents/evaluation/league/run_league.py --build --hands 10000 --seed 1307
  python agents/evaluation/league/run_league.py --build --hands 20 --out-dir NEW_SMOKE_DIR
  python agents/evaluation/league/run_league.py --analyze SAVED_RUN_DIR

Each matchup uses HANDS/2 shared deals played in both seats. All matchups reuse
the same deal list. Rankings are pool-dependent chip profits, not exploitability.
Current and average views share frozen tables. No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAMILY = ROOT / "agents/lightgbm_regret_ensemble"
CORE = ROOT / "agents/cpp_mccfr"
sys.path.insert(0, str(FAMILY))
from run_epoch_ensemble import np, plt
from matplotlib.colors import SymLogNorm

DEFAULT_RUN = FAMILY / "data/seven_stud_1m_seed7_20260919_204332"
EXE = HERE / "bin/evaluate_league.exe"
MASK64 = (1 << 64) - 1


def make_roster(run):
    data = CORE / "data"
    atlas256 = data / "power256_selfplay100m_eps20_v1.bin"
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["hands"] != 1000000 or config["epoch_hands"] != 10000:
        raise ValueError("This comparison roster requires the completed 1M/10k run")
    players = [
        dict(
            id="ensemble_1m_current",
            label="Ensemble 1M",
            kind="ensemble",
            mode="current",
            model=str(run),
            atlas=str(atlas256),
            hands=1000000,
            epoch_hands=10000,
        )
    ]
    # One final representative per archived hard-policy variant; no smoke models.
    specs = [
        ("hard256_1m", "K256 1M", run / "hard256_1000000.bin", atlas256),
        (
            "power64_100m",
            "K64 100M",
            data / "root_mccfr_ante1000_100m.bin",
            data / "power64_v1.bin",
        ),
        (
            "memory16_30m",
            "Mem16 30M",
            data / "made_call_r1000_k512_epsheur20_memory16_30m.bin",
            data / "power512_epsheur20_memory16_v1.bin",
        ),
        (
            "memory16_10m",
            "Mem16 10M",
            data / "made_call_r1000_k512_epsheur20_memory16_10m.bin",
            data / "power512_epsheur20_memory16_v1.bin",
        ),
        (
            "memory81_10m",
            "Mem81 10M",
            data / "made_call_r1000_k512_pool250k_memory81_10m.bin",
            data / "power512_heuristic_pool_250k_v1.bin",
        ),
        (
            "power512_10m",
            "K512 10M",
            data / "made_call_pair_r1000_k512_eps20_10m.bin",
            data / "power512_selfplay100m_eps20_v1.bin",
        ),
        (
            "tree_100m_nodes",
            "Tree 100M nodes",
            data / "power_tree_100m_nodes.bin",
            data / "power_tree_100m_nodes.atlas",
        ),
    ]
    for name, label, model, atlas in specs:
        for mode in ("current", "average"):
            players.append(
                dict(
                    id=f"{name}_{mode}",
                    label=f"{label} {mode[0].upper()}",
                    kind="mccfr",
                    mode=mode,
                    model=str(model),
                    atlas=str(atlas),
                    hands=0,
                    epoch_hands=0,
                )
            )
    for name in ("heuristic", "uniform"):
        players.append(
            dict(
                id=name,
                label=name.title(),
                kind=name,
                mode="fixed",
                model="-",
                atlas="-",
                hands=0,
                epoch_hands=0,
            )
        )
    for player in players:
        for field in ("atlas", "model"):
            if player[field] != "-" and not Path(player[field]).exists():
                raise FileNotFoundError(player[field])
    return players


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def estimate(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent deal-pair values required")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return dict(mean=mean, se=se, ci95=[mean - 1.96 * se, mean + 1.96 * se])


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    players = manifest["players"]
    names = [p["id"] for p in players]
    pair_count = manifest["hands_per_match"] // 2
    matchups = list(itertools.combinations(range(len(players)), 2))
    samples = np.empty((len(matchups), pair_count))
    counts = np.zeros(len(matchups), dtype=int)
    outcomes = np.zeros((len(matchups), 3), dtype=int)
    with (directory / "matches/paired_payoffs.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            match = int(row["match"])
            if not 0 <= match < len(matchups):
                raise ValueError("Unexpected matchup index")
            i, j = matchups[match]
            pair = int(row["pair"])
            expected_seed = manifest["seed"] ^ (
                ((pair + 1) * 0xD1B54A32D192ED03) & MASK64
            )
            if (
                pair != counts[match]
                or pair >= pair_count
                or row["a"] != names[i]
                or row["b"] != names[j]
                or int(row["deal_seed"]) != expected_seed
            ):
                raise ValueError("Missing/duplicate/mismatched pair or deal seed")
            values = [float(row[f"a_seat{s}_ante"]) for s in (0, 1)]
            if not all(math.isfinite(v) and abs(v) <= 1000 for v in values):
                raise ValueError("Invalid payoff")
            samples[match, pair] = sum(values) / 2
            outcomes[match] += [
                sum(v > 0 for v in values),
                sum(v == 0 for v in values),
                sum(v < 0 for v in values),
            ]
            counts[match] += 1
    if not np.all(counts == pair_count):
        raise ValueError("Incomplete league; refusing to rank partial opponents")
    with (directory / "matches/matches.csv").open(newline="") as source:
        engine = list(csv.DictReader(source))
    if len(engine) != len(matchups):
        raise ValueError("Missing completed match summaries")

    matrix = np.full((len(players), len(players)), np.nan)
    overall = np.zeros((len(players), pair_count))
    learned = np.zeros_like(overall)
    learned_counts = np.zeros(len(players), dtype=int)
    records = []
    is_learned = [p["kind"] in ("ensemble", "mccfr") for p in players]
    for match, (i, j) in enumerate(matchups):
        stats = estimate(samples[match])
        source = engine[match]
        if (
            int(source["match"]) != match
            or source["a"] != names[i]
            or source["b"] != names[j]
            or int(source["hands"]) != 2 * pair_count
            or not math.isclose(
                float(source["mean_ante_a"]), stats["mean"], abs_tol=1e-8
            )
        ):
            raise ValueError("Engine/CSV mismatch")
        matrix[i, j] = stats["mean"]
        matrix[j, i] = -stats["mean"]
        overall[i] += samples[match]
        overall[j] -= samples[match]
        if is_learned[j]:
            learned[i] += samples[match]
            learned_counts[i] += 1
        if is_learned[i]:
            learned[j] -= samples[match]
            learned_counts[j] += 1
        records.append(
            dict(
                match=match,
                a=names[i],
                b=names[j],
                **stats,
                wins_a=int(outcomes[match, 0]),
                ties=int(outcomes[match, 1]),
                losses_a=int(outcomes[match, 2]),
                engine=source,
            )
        )
    assert np.allclose(np.nan_to_num(matrix) + np.nan_to_num(matrix).T, 0)
    assert np.allclose(overall.sum(axis=0), 0, atol=1e-8)
    ranking = [
        dict(
            id=name,
            label=players[i]["label"],
            overall=estimate(overall[i] / (len(players) - 1)),
            learned_opponents=estimate(learned[i] / learned_counts[i]),
        )
        for i, name in enumerate(names)
    ]
    ranking.sort(key=lambda r: r["learned_opponents"]["mean"], reverse=True)
    result = dict(
        complete=True,
        hands_per_match=2 * pair_count,
        matches=len(matchups),
        total_hands=len(matchups) * 2 * pair_count,
        ranking=ranking,
        matchups=records,
        interval_scope="pointwise normal-approximation 95% deal-pair sampling intervals; not training-seed uncertainty or multiplicity-adjusted",
        score_scope="equal weight per opponent; learned-opponent ranking excludes heuristic/uniform rivals, not their own scores; pool-dependent, not exploitability",
    )
    (directory / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "payoff_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(["row_player"] + names)
        for i, name in enumerate(names):
            writer.writerow(
                [name] + ["" if i == j else matrix[i, j] for j in range(len(names))]
            )

    labels = [p["label"] for p in players]
    fig, ax = plt.subplots(figsize=(15, 13), layout="constrained")
    limit = max(0.1, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap="RdBu",
        norm=SymLogNorm(linthresh=0.1, vmin=-limit, vmax=limit),
    )
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=labels,
        yticklabels=labels,
        title=f"7-stud frozen-policy league: row player's profit (ante/hand)\n{2 * pair_count:,} hands per matchup; C=current, A=average; * pointwise 95% pair CI excludes zero",
    )
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    for record, (i, j) in zip(records, matchups):
        significant = record["ci95"][0] > 0 or record["ci95"][1] < 0
        for row, col in ((i, j), (j, i)):
            value = matrix[row, col]
            rgba = im.cmap(im.norm(value))
            color = "black" if sum(rgba[:3]) / 3 > 0.55 else "white"
            ax.text(
                col,
                row,
                f"{value:.2f}" + ("*" if significant else ""),
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )
    fig.colorbar(
        im, ax=ax, shrink=0.75, label="Row-player ante/hand (signed-log color scale)"
    )
    fig.savefig(directory / "payoff_matrix.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True, layout="constrained")
    for ax, key, title in zip(
        axes,
        ("learned_opponents", "overall"),
        ("Against learned opponents only", "Against all other participants"),
    ):
        for i, row in enumerate(ranking):
            s = row[key]
            ax.errorbar(
                s["mean"],
                i,
                xerr=1.96 * s["se"],
                fmt="o",
                capsize=3,
                color="#bf354b" if row["id"] == "ensemble_1m_current" else "#168776",
            )
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set(title=title, xlabel="Profit (ante/hand), 95% deal-block CI")
        ax.grid(alpha=0.2)
    axes[0].set_yticks(range(len(ranking)), [r["label"] for r in ranking])
    axes[0].invert_yaxis()
    fig.savefig(directory / "ranking.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "complete": True,
                "matches": len(matchups),
                "total_hands": result["total_hands"],
                "ranking": ranking,
                "ensemble_matchups": [
                    r for r in records if r["a"] == "ensemble_1m_current"
                ],
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1307)
    parser.add_argument("--ensemble-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--analyze", type=Path)
    args = parser.parse_args()
    assert estimate([1, 3])["mean"] == 2 and estimate([1, 3])["se"] == 1
    if args.analyze:
        analyze(args.analyze.resolve())
        return
    if args.hands < 4 or args.hands % 2 or not 0 <= args.seed <= MASK64:
        parser.error("--hands must be even and >=4; --seed must be uint64")
    players = make_roster(args.ensemble_run.resolve())
    if args.build:
        EXE.parent.mkdir(exist_ok=True)
        compiler = shutil.which("g++") or "C:/ProgramData/mingw64/mingw64/bin/g++.exe"
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-DNOMINMAX",
                "-static-libstdc++",
                "-static-libgcc",
                str(HERE / "evaluate_league.cpp"),
                "-o",
                str(EXE),
                "-lpsapi",
            ],
            check=True,
        )
    if not EXE.is_file():
        parser.error("Build the evaluator with --build first")
    directory = (
        args.out_dir or HERE / "data" / datetime.now().strftime("league_%Y%m%d_%H%M%S")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for player in players:
        paths = [
            Path(player[field]) for field in ("model", "atlas") if player[field] != "-"
        ]
        if player["kind"] == "ensemble":
            paths = [Path(player["atlas"])] + [
                Path(player["model"]) / f"epoch_{i:03}/model.bin" for i in range(1, 101)
            ]
        for path in paths:
            if str(path) not in hashes:
                hashes[str(path)] = sha256(path)
    sources = directory / "sources"
    sources.mkdir()
    for path in [
        Path(__file__),
        HERE / "evaluate_league.cpp",
        FAMILY / "stud_epoch_ensemble.cpp",
        CORE / "stud_mccfr.cpp",
        CORE / "stud_rules.hpp",
        CORE / "local_split_helpers.hpp",
        EXE,
    ]:
        shutil.copy2(path, sources / path.name)
    manifest = dict(
        seed=args.seed,
        hands_per_match=args.hands,
        players=players,
        input_sha256=hashes,
        source_sha256={p.name: sha256(p) for p in sources.iterdir()},
        scope="heads-up 7-stud v3; fixed heuristic H4; 5th-7th; ante=1000; stack=1000 antes; frozen policies",
        exclusions="online resolving, neural/Python adapters, incompatible/smoke/duplicate checkpoints, and soft-bucket variants with nonserialized runtime flags; selection fixed before evaluation",
        budget_note="archived 10M/30M/100M names are not equal training budgets; Tree 100M counts training nodes, not hands",
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    roster = directory / "roster.tsv"
    with roster.open("w", newline="", encoding="utf-8") as target:
        for p in players:
            fields = [
                str(p[k])
                for k in (
                    "id",
                    "kind",
                    "mode",
                    "model",
                    "atlas",
                    "hands",
                    "epoch_hands",
                )
            ]
            if any("\t" in v or "\n" in v or "\r" in v for v in fields):
                raise ValueError("Tabs/newlines are not supported in roster fields")
            target.write("\t".join(fields) + "\n")
    start = time.perf_counter()
    with (directory / "progress.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            [
                str(EXE),
                str(roster),
                str(directory / "matches"),
                str(args.hands),
                str(args.seed),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f"League failed; see {directory / 'progress.log'}")
    manifest["evaluation_wall_seconds_including_load_and_tests"] = (
        time.perf_counter() - start
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    analyze(directory)
    print("RESULTS", directory)


if __name__ == "__main__":
    main()
