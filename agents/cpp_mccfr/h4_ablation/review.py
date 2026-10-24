"""Validate completed H4 checkpoints/evaluations and draw matched diagnostic plots.

Usage: python -m agents.cpp_mccfr.h4_ablation.review --run agents/cpp_mccfr/data/h4_ablation/seed2026100851_100k
Input is read-only; writes review.json and comparison.png inside the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from .run import DATA, NAMES, analyze, save


def checkpoint_stats(path):
    record = struct.Struct("<Q24dQ")
    result = {}
    with path.open("rb") as handle:
        size, = struct.unpack("<I", handle.read(4))
        assert handle.read(size) == b"H4_FROZEN_SIGNED1"
        hands, seed = struct.unpack("<QQ", handle.read(16))
        for name in ("dt", "km"):
            count, = struct.unpack("<Q", handle.read(8))
            visits, entropy, maxima = [], [], []
            keys = set()
            negative_regrets = 0
            seat_visits = [0, 0]
            for _ in range(count):
                row = record.unpack(handle.read(record.size))
                key, regret, average, touches = row[0], np.array(row[1:13]), np.array(row[13:25]), row[25]
                assert key not in keys and touches > 0
                keys.add(key)
                assert np.isfinite(regret).all() and np.isfinite(average).all()
                assert (average >= 0).all() and np.isclose(average.sum(), touches)
                probability = average / touches
                visits.append(touches)
                entropy.append(float(-(probability * np.log(np.maximum(probability, 1e-300))).sum()))
                maxima.append(float(probability.max()))
                negative_regrets += int((regret < 0).sum())
                seat_visits[key % 2] += touches
            assert seat_visits == [hands, hands]
            result[name] = dict(states=count, mean_visits=float(np.mean(visits)),
                                visit_quantiles=np.quantile(visits, [0, .25, .5, .75, .95, 1]).tolist(),
                                states_one_visit=int((np.array(visits) == 1).sum()),
                                mean_policy_entropy=float(np.mean(entropy)),
                                mean_max_action_probability=float(np.mean(maxima)),
                                negative_regret_entries=negative_regrets)
        assert not handle.read(1)
    return dict(hands=hands, seed=seed, checkpoint_bytes=path.stat().st_size, models=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    if not output.is_relative_to(DATA.resolve()):
        parser.error("Expected a H4 experiment directory")
    config = json.loads((output / "lbr_config.json").read_text())
    metadata = []
    for seed in config["seeds"]:
        with (output / f"lbr_{seed}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == config["pairs"]
        assert len({r["deal_seed"] for r in rows}) == len(rows)
        assert [int(r["pair"]) for r in rows] == list(range(len(rows)))
        for name in NAMES:
            values = np.array([[float(r[name+"_seat0"]), float(r[name+"_seat1"])] for r in rows])
            assert np.isfinite(values).all() and (np.abs(values) <= 1000).all()
        metadata.append(json.loads((output / f"lbr_{seed}.json").read_text()))
    result = checkpoint_stats(output / "h4_checkpoint.bin")
    result["evaluation_metadata"] = metadata
    result["integrity"] = json.loads((output / "input_integrity.json").read_text())
    result["betting_query_misses"] = {name: sum(m["policies"][name]["misses"] for m in metadata) for name in NAMES}
    result["actual_h4_unseen"] = {name: sum(m["policies"][name]["h4_unseen"] for m in metadata) for name in NAMES}
    save(output / "review.json", result)
    analyze(output, config["seeds"])
    summary = json.loads((output / "summary.json").read_text())
    with (output / "panel.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len({(r["hands"], r["arm"]) for r in rows}) == len(rows)
    assert int(rows[-1]["hands"]) == result["hands"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm, family, color in ((0, "dt", "red"), (1, "km", "blue")):
        stats = [summary["policies"][f"{family}_{mode}"] for mode in ("heuristic", "learned")]
        values = np.array([s["mean"] for s in stats])
        bounds = np.array([s["bootstrap_ci95"] for s in stats]).T
        axes[0, 0].errorbar(np.array([0,1]) + (arm-.5)*.12, values,
                            yerr=[values-bounds[0],bounds[1]-values], color=color,
                            marker="o", capsize=4, label=family.upper())
        selected = [r for r in rows if int(r["arm"]) == arm and int(r["hands"]) > 0]
        x = [int(r["hands"]) for r in selected]
        axes[0, 1].plot(x, [float(r["gap"]) for r in selected], color=color, label=family.upper())
        axes[1, 0].plot(x, [float(r["km_ratio"]) for r in selected], color=color, label=family.upper())
        axes[1, 1].plot(x, [float(r["crossfit_gap"]) for r in selected], color=color, label=family.upper())
    axes[0, 0].set_xticks([0,1], ["Heuristic H4", "Learned average H4"])
    axes[0, 0].set(ylabel="LBR attacker ante/hand", title="Fixed average betting; bootstrap 95% CI")
    axes[0, 1].set(xscale="log", yscale="log", xlabel="H4 training deals/model", ylabel="H4 Local BR-gap (ante)",
                   title="64 fixed H4 roots x 32 conditional samples")
    axes[1, 0].set(xscale="log", xlabel="H4 training deals/model", ylabel="Gap / matched k-means gap",
                   title="Matched-budget ratio; average H4")
    axes[1, 0].axhline(1, color="gray", linewidth=.7)
    axes[1, 1].set(xscale="log", xlabel="H4 training deals/model", ylabel="Cross-fitted action advantage (ante)",
                   title="Signed noise diagnostic, NOT the true max gap")
    axes[1, 1].axhline(0, color="gray", linewidth=.7)
    for axis in axes.flat:
        axis.legend()
        axis.grid(alpha=.2)
    fig.suptitle("12-action H4 only, downstream DT/k-means 10M fixed; NOT fifth-street gap")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
