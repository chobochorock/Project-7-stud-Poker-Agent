"""Audit the completed 100k/1M AE and VQ runs without training or playing hands.

Usage: python -m agents.autoencoder_abstraction.analyze_1m --out-dir NEW_DIRECTORY
Inputs: the four named runs in this family's data directory.
Output: verification/paired statistics JSON and a comparison plot.
Confidence intervals cover evaluation deals, not learning seeds or LBR bias.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from agents.autoencoder_abstraction.comparison import BASELINE, curve, plot_comparison
from agents.autoencoder_abstraction.run_cfr import FAMILY, mean_ci, sha256
from agents.state_action_embedding.experiment import write_json

RUNS = {
    "ae_100k": "cfr_ae_100k_seed7_20260923",
    "ae_1m": "ae_cfr_1m",
    "vq_100k": "cfr_vq_100k_seed7_20260923",
    "vq_1m": "vq_cfr_1m",
}


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_run(folder: Path) -> dict:
    metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    args = metadata["arguments"]
    hands = args["hands"]
    assert metadata["status"] == "complete", folder
    assert metadata["bucket_mismatches"] == 0
    assert sha256(folder / "neural_atlas.bin") == metadata["atlas_sha256"]
    model = folder / "solver" / f"policy_{hands}.bin"
    assert sha256(model) == metadata["policy_sha256"]
    with model.open("rb") as stream:
        magic, plus, abstraction, street, keys = struct.unpack("<8sBBBQ", stream.read(19))
    assert (magic, plus, abstraction, street) == (b"MCCFRV5\0", 0, 6, 5)
    assert keys == summary["training"]["buckets"]
    assert f"DONE hands={hands} " in (folder / "run.log").read_text(encoding="utf-8")
    progress = rows(folder / "solver/progress.csv")
    assert int(progress[-1]["hands"]) == hands
    assert int(progress[-1]["traversals"]) == 2 * hands
    for field in ("hands", "traversals", "node_visits", "buckets"):
        values = [int(row[field]) for row in progress]
        assert all(a <= b for a, b in zip(values, values[1:])), field
        assert values[-1] == summary["training"][field]
    gap = rows(folder / "solver/local_gap.csv")
    steps = sorted({int(row["hands"]) for row in gap})
    expected_steps = sorted(set(range(0, hands + 1, args["eval_every"])) | {hands})
    assert steps == expected_steps
    assert len(gap) == len(steps) * args["eval_roots"]
    root_ids = None
    for step in steps:
        batch = [row for row in gap if int(row["hands"]) == step]
        ids = [(row["root"], row["root_seed"]) for row in batch]
        assert len(set(ids)) == args["eval_roots"]
        if root_ids is not None:
            assert ids == root_ids
        root_ids = ids
        for policy in ("current", "average"):
            values = np.array([float(row[policy]) for row in batch])
            assert np.isfinite(values).all() and (values >= 0).all()
            saved = next(item for item in summary["local_gap"]
                         if item["hands"] == step and item["policy"] == policy)
            np.testing.assert_allclose(values.mean(), saved["mean"], rtol=1e-12)
    pairs = rows(folder / "solver/lbr_pairs.csv")
    assert len(pairs) == args["lbr_pairs"]
    assert [int(row["pair"]) for row in pairs] == list(range(len(pairs)))
    assert len({row["deal_seed"] for row in pairs}) == len(pairs)
    payoffs = {}
    for policy in ("current", "average"):
        seats = np.array([[float(row[f"{policy}_seat{seat}"]) for seat in (0, 1)] for row in pairs])
        assert np.isfinite(seats).all() and (np.abs(seats) <= 1000).all()
        payoffs[policy] = seats.mean(axis=1)
        recomputed = mean_ci(payoffs[policy])
        for field in ("mean", "ci95_low", "ci95_high"):
            np.testing.assert_allclose(recomputed[field], summary["lbr"][policy][field], rtol=1e-12)
    return dict(metadata=metadata, summary=summary, progress=progress, gap=gap,
                pairs=pairs, payoffs=payoffs, model_bytes=model.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    runs = {name: load_run(FAMILY / "data" / folder) for name, folder in RUNS.items()}
    reference = runs["ae_100k"]
    for run in runs.values():
        for field in ("seed", "representation_seed", "eval_every", "eval_roots", "eval_particles",
                      "eval_seed", "lbr_pairs", "lbr_particles"):
            assert run["metadata"]["arguments"][field] == reference["metadata"]["arguments"][field]
        assert [row["deal_seed"] for row in run["pairs"]] == [row["deal_seed"] for row in reference["pairs"]]
        for source, digest in reference["metadata"]["source_sha256"].items():
            if source.endswith((".cpp", ".hpp")):
                assert run["metadata"]["source_sha256"][source] == digest
    result = {"runs": {}, "paired_lbr_differences": {}, "checks": {}}
    for name, run in runs.items():
        result["runs"][name] = {
            "path": str(FAMILY / "data" / RUNS[name]), "model_bytes": run["model_bytes"],
            "training": run["summary"]["training"], "lbr": run["summary"]["lbr"],
            "wall_seconds": run["metadata"]["wall_seconds"],
            "gap_rows": len(run["gap"]), "lbr_pairs": len(run["pairs"]),
        }
        result["runs"][name]["final_gap"] = {
            p: next(g for g in run["summary"]["local_gap"]
                    if g["policy"] == p and g["hands"] == run["metadata"]["arguments"]["hands"])
            for p in ("current", "average")
        }
    for method in ("ae", "vq"):
        old, new = runs[f"{method}_100k"], runs[f"{method}_1m"]
        assert old["metadata"]["atlas_sha256"] == new["metadata"]["atlas_sha256"]
        assert old["gap"] == new["gap"][:len(old["gap"])]
        for old_row, new_row in zip(old["progress"], new["progress"]):
            for field in ("hands", "traversals", "node_visits", "buckets"):
                assert old_row[field] == new_row[field]
        result["checks"][f"{method}_100k_prefix_exact"] = True
        result["paired_lbr_differences"][f"{method}_1m_minus_100k"] = {
            p: mean_ci(new["payoffs"][p] - old["payoffs"][p]) for p in ("current", "average")
        }
    result["paired_lbr_differences"]["vq_minus_ae_at_1m"] = {
        p: mean_ci(runs["vq_1m"]["payoffs"][p] - runs["ae_1m"]["payoffs"][p])
        for p in ("current", "average")
    }
    result["checks"].update(hashes_headers_budgets=True, csv_statistics=True, evaluation_deals_matched=True)
    baseline_metadata = json.loads((args.baseline_dir / "run.json").read_text(encoding="utf-8"))
    assert baseline_metadata["status"] == "complete"
    for field in ("eval_roots", "eval_particles", "eval_seed", "lbr_pairs", "lbr_particles"):
        assert baseline_metadata["arguments"][field] == reference["metadata"]["arguments"][field]
    baseline_gap = rows(args.baseline_dir / "solver/local_gap.csv")
    assert [(g["hands"], g["root"], g["root_seed"]) for g in baseline_gap] == [
        (g["hands"], g["root"], g["root_seed"]) for g in runs["ae_1m"]["gap"]]
    initial = [g for g in baseline_gap if int(g["hands"]) == 0]
    assert initial == [g for g in reference["gap"] if int(g["hands"]) == 0]
    result["paired_gap_minus_kmeans"] = {}
    for budget, hand in (("100k", 100000), ("1m", 1000000)):
        baseline_pairs = rows(args.baseline_dir / f"solver/lbr_{hand}/lbr_pairs.csv")
        assert [g["deal_seed"] for g in baseline_pairs] == [g["deal_seed"] for g in reference["pairs"]]
        for method in ("ae", "vq"):
            run = runs[f"{method}_{budget}"]
            result["paired_lbr_differences"][f"{method}_minus_kmeans_at_{budget}"] = {
                p: mean_ci(run["payoffs"][p] - np.array([
                    (float(g[f"{p}_seat0"]) + float(g[f"{p}_seat1"])) / 2 for g in baseline_pairs]))
                for p in ("current", "average")
            }
            result["paired_gap_minus_kmeans"][f"{method}_at_{budget}"] = {
                p: mean_ci(np.array([float(g[p]) for g in run["gap"] if int(g["hands"]) == hand])
                           - np.array([float(g[p]) for g in baseline_gap if int(g["hands"]) == hand]))
                for p in ("current", "average")
            }
    result["checks"]["kmeans_roots_initial_policy_and_lbr_deals_matched"] = True
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "verification.json", result)
    curves = {}
    for method, label in (("ae", "AE"), ("vq", "VQ-VAE")):
        curves[label] = curve(runs[f"{method}_1m"]["summary"])
        curves[label]["lbr"].extend(curve(runs[f"{method}_100k"]["summary"])["lbr"])
    plot_comparison(curves, reference["metadata"]["arguments"],
                    args.out_dir / "comparison.png", args.baseline_dir)
    print(f"Verified 4 runs and matched k-means evaluation: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
