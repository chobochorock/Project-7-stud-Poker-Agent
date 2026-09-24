"""Verify VQ 10M, compare its 1M prefix, and evaluate archived k-means controls.

Usage: python -m agents.autoencoder_abstraction.analyze_10m --out-dir NEW_DIR
No training. Reuses the C++ root-gap/LBR evaluator for K512 Mem16 10M/30M
and K64 100M, plus a K256 1M reproducibility check. Existing outputs rejected.
Models have different data/initialization/key partitions: not an encoder ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from agents.autoencoder_abstraction.analyze_1m import load_run, rows
from agents.autoencoder_abstraction.comparison import (
    ATLAS, BASELINE, GAP_FIELDS, LBR_FIELDS, LEGACY, curve, log_trend, plot_comparison,
)
from agents.autoencoder_abstraction.run_cfr import FAMILY, ROOT, SOURCES, build, mean_ci, sha256
from agents.state_action_embedding.experiment import write_json


CONTROLS = {
    "memory16_10m_current": (10000000, "power-memory16", "K512 Mem16 (archived)"),
    "memory16_30m_current": (30000000, "power-memory16", "K512 Mem16 (archived)"),
    "power64_100m_current": (100000000, "power", "K64 (archived)"),
}


def paired_values(pair_rows: list[dict], policy: str) -> np.ndarray:
    return np.array([(float(g[f"{policy}_seat0"]) + float(g[f"{policy}_seat1"])) / 2
                     for g in pair_rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    runs = {name: load_run(FAMILY / "data" / folder) for name, folder in {
        "vq_100k": "cfr_vq_100k_seed7_20260923", "vq_1m": "vq_cfr_1m",
        "vq_10m": "vq_cfr_10m", "ae_1m": "ae_cfr_1m",
    }.items()}
    old, new = runs["vq_1m"], runs["vq_10m"]
    settings = new["metadata"]["arguments"]
    for run in runs.values():
        for field in (*GAP_FIELDS, *LBR_FIELDS, "eval_every", "seed", "representation_seed"):
            assert run["metadata"]["arguments"][field] == settings[field]
        assert [g["deal_seed"] for g in run["pairs"]] == [g["deal_seed"] for g in new["pairs"]]
    assert old["metadata"]["atlas_sha256"] == new["metadata"]["atlas_sha256"]
    assert old["gap"] == new["gap"][:len(old["gap"])]
    for a, b in zip(old["progress"], new["progress"]):
        for field in ("hands", "traversals", "node_visits", "buckets"):
            assert a[field] == b[field]
    for path in SOURCES:
        if path.suffix in (".cpp", ".hpp") and path.name != "neural_mccfr.cpp":
            relative = str(path.relative_to(ROOT))
            assert sha256(path) == new["metadata"]["source_sha256"][relative]
    result = {"checks": {"hashes_headers_csv_statistics": True, "vq_1m_prefix_exact": True},
              "evaluation_settings": {f: settings[f] for f in set(GAP_FIELDS + LBR_FIELDS)},
              "runs": {}, "controls": {}, "paired_10m_minus_1m": {},
              "paired_10m_minus_control": {}}
    for name, run in runs.items():
        budget = run["metadata"]["arguments"]["hands"]
        result["runs"][name] = {"training": run["summary"]["training"],
            "model_bytes": run["model_bytes"], "wall_seconds": run["metadata"]["wall_seconds"],
            "gap_rows": len(run["gap"]), "lbr": run["summary"]["lbr"],
            "final_gap": {p: next(g for g in run["summary"]["local_gap"]
                                  if g["hands"] == budget and g["policy"] == p)
                          for p in ("current", "average")}}
    final_rows = [g for g in new["gap"] if int(g["hands"]) == 10000000]
    for policy in ("current", "average"):
        earlier = np.array([float(g[policy]) for g in old["gap"] if int(g["hands"]) == 1000000])
        result["paired_10m_minus_1m"][policy] = {
            "gap": mean_ci(np.array([float(g[policy]) for g in final_rows]) - earlier),
            "lbr": mean_ci(new["payoffs"][policy] - old["payoffs"][policy]),
        }
        values = new["payoffs"][policy]
        worst = int(np.argmax(np.abs(values)))
        result["runs"]["vq_10m"][f"{policy}_lbr_tail"] = {
            "min_pair": float(values.min()), "max_pair": float(values.max()),
            "largest_absolute_pair_id": worst, "largest_pair_value": float(values[worst]),
            "largest_pair_contribution_to_mean": float(values[worst] / len(values)),
            "pairs_absolute_at_least_100": int((np.abs(values) >= 100).sum()),
        }
    binary = build()
    subprocess.run([str(binary), "--engine-test"], check=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "agents/evaluation/league/data/league_1m_30m_100m_20260920/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {g["id"]: g for g in manifest["players"]}
    inputs["hard256_1m_check"] = {"model": str(LEGACY / "hard256_1000000.bin"), "atlas": str(ATLAS)}
    controls = {"hard256_1m_check": (1000000, "power-memory16", "reproducibility"), **CONTROLS}
    curves = {"VQ-VAE": curve(new["summary"]), "AE": curve(runs["ae_1m"]["summary"])}
    for name in ("vq_100k", "vq_1m"):
        curves["VQ-VAE"]["lbr"].extend(curve(runs[name]["summary"])["lbr"])
    metadata = {"status": "evaluating", "evaluation_only": True,
        "command": subprocess.list2cmdline([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [*SOURCES, Path(__file__).resolve()]},
        "evaluation_settings": result["evaluation_settings"], "controls": {}}
    write_json(args.out_dir / "run.json", metadata)
    for name, (budget, bucket, label) in controls.items():
        item = inputs[name]
        model, atlas = Path(item["model"]), Path(item["atlas"])
        hashes = {str(p): sha256(p) for p in (model, atlas)}
        if name in CONTROLS:
            for path, digest in hashes.items():
                assert digest == manifest["input_sha256"][path]
        output = args.out_dir / name
        command = [str(binary), "--evaluate-checkpoint", str(atlas), str(model), str(output),
                   str(budget), bucket, *[str(settings[f]) for f in
                   ("eval_roots", "eval_particles", "lbr_pairs", "lbr_particles", "eval_seed")]]
        started = time.perf_counter()
        print(f"EVALUATION control={name} roots={settings['eval_roots']} pairs={settings['lbr_pairs']}", flush=True)
        with (args.out_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
        gap, pairs = rows(output / "local_gap.csv"), rows(output / "lbr_pairs.csv")
        assert [(g["root"], g["root_seed"]) for g in gap] == [(g["root"], g["root_seed"]) for g in final_rows]
        assert [(g["pair"], g["deal_seed"]) for g in pairs] == [(g["pair"], g["deal_seed"]) for g in new["pairs"]]
        if name == "hard256_1m_check":
            assert gap == [g for g in rows(BASELINE / "solver/local_gap.csv") if int(g["hands"]) == budget]
            assert pairs == rows(BASELINE / "solver/lbr_1000000/lbr_pairs.csv")
            result["checks"]["new_entrypoint_reproduces_k256_gap_and_lbr_exactly"] = True
        metrics = {"local_gap": [], "lbr": [], "points_only": True}
        result["paired_10m_minus_control"][name] = {}
        for policy in ("current", "average"):
            values = np.array([float(g[policy]) for g in gap])
            payoffs = paired_values(pairs, policy)
            assert np.isfinite(values).all() and (values >= 0).all()
            assert np.isfinite(payoffs).all() and (np.abs(payoffs) <= 1000).all()
            metrics["local_gap"].append(dict(hands=budget, policy=policy, **mean_ci(values)))
            metrics["lbr"].append(dict(hands=budget, policy=policy, **mean_ci(payoffs)))
            result["paired_10m_minus_control"][name][policy] = {
                "gap": mean_ci(np.array([float(g[policy]) for g in final_rows]) - values),
                "lbr": mean_ci(new["payoffs"][policy] - payoffs),
            }
        result["controls"][name] = metrics
        if name in CONTROLS:
            curves.setdefault(label, {"local_gap": [], "lbr": [], "points_only": True})
            for kind in ("local_gap", "lbr"):
                curves[label][kind].extend(metrics[kind])
        metadata["controls"][name] = {"model": str(model), "atlas": str(atlas), "budget_label": budget,
            "bucket": bucket, "input_sha256": hashes, "command": command,
            "evaluation_seconds": time.perf_counter() - started}
        write_json(args.out_dir / "run.json", metadata)
    for policy in ("current", "average"):
        metrics = [g for g in new["summary"]["local_gap"] if g["policy"] == policy]
        result["runs"]["vq_10m"][f"{policy}_trends"] = {
            "100k_to_1m": log_trend(metrics),
            "1m_to_10m": log_trend(metrics, 1000000, 10000000),
        }
    plot_comparison(curves, settings, args.out_dir / "comparison.png",
                    trend_start=1000000, trend_stop=10000000)
    write_json(args.out_dir / "verification.json", result)
    metadata["status"] = "complete"
    write_json(args.out_dir / "run.json", metadata)
    print(f"RESULTS {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
