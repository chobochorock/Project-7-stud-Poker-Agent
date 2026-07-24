from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from agent.mccfr_agent import mccfr_bucket_signature, mccfr_bucket_vector
from poker_env import BETTING_RULES_VERSION


def _cluster_allocation(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    if total < len(group_sizes):
        raise ValueError(f"clusters must be at least the {len(group_sizes)} history groups")
    total = min(total, sum(group_sizes.values()))
    allocation = {signature: 1 for signature in group_sizes}
    for _ in range(total - len(allocation)):
        eligible = [
            signature
            for signature, size in group_sizes.items()
            if allocation[signature] < size
        ]
        signature = max(
            eligible,
            key=lambda item: group_sizes[item] / allocation[item],
        )
        allocation[signature] += 1
    return allocation


def compress_table(
    input_path: Path,
    output_path: Path,
    *,
    clusters: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    config = source.get("config", {})
    if int(config.get("betting_rules_version", 1)) != BETTING_RULES_VERSION:
        raise ValueError("Source MCCFR table uses obsolete betting rules")
    if config.get("start_street", "7th_hidden") != "7th_hidden":
        raise ValueError("The first compressor only supports a seventh-street table")

    records: list[tuple[list[Any], dict[str, Any]]] = [
        (json.loads(key), node) for key, node in source["nodes"].items()
    ]
    vectors = np.asarray(
        [mccfr_bucket_vector(payload) for payload, _ in records], dtype=np.float32
    )
    feature_mean = vectors.mean(axis=0)
    feature_scale = vectors.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    vectors = (vectors - feature_mean) / feature_scale

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (payload, _) in enumerate(records):
        grouped[mccfr_bucket_signature(payload)].append(index)
    allocation = _cluster_allocation(
        {signature: len(indices) for signature, indices in grouped.items()}, clusters
    )

    output_groups = []
    zero_mass_clusters = 0
    for group_number, signature in enumerate(sorted(grouped)):
        indices = np.asarray(grouped[signature], dtype=np.int64)
        values = vectors[indices]
        count = allocation[signature]
        masses = np.asarray(
            [
                sum(records[index][1].get("strategy_sum", {}).values())
                for index in indices
            ],
            dtype=np.float64,
        )
        weights = 1.0 + np.log1p(masses)
        if count == 1:
            centers = np.average(values, axis=0, weights=weights)[None, :]
            assignments = np.zeros(len(indices), dtype=np.int32)
        else:
            model = MiniBatchKMeans(
                n_clusters=count,
                batch_size=min(max(batch_size, count * 3), len(indices)),
                n_init=3,
                max_iter=100,
                random_state=seed + group_number,
            ).fit(values, sample_weight=weights)
            centers = model.cluster_centers_
            assignments = model.predict(values)

        nodes = [
            {"regrets": defaultdict(float), "strategy_sum": defaultdict(float)}
            for _ in range(count)
        ]
        support = [0] * count
        for record_index, cluster in zip(indices, assignments):
            node = records[int(record_index)][1]
            support[int(cluster)] += 1
            for field in ("regrets", "strategy_sum"):
                for action, value in node.get(field, {}).items():
                    nodes[int(cluster)][field][action] += float(value)
        zero_mass_clusters += sum(
            sum(node["strategy_sum"].values()) == 0.0 for node in nodes
        )
        output_groups.append(
            {
                "signature": json.loads(signature),
                "centers": centers.astype(float).tolist(),
                "support": support,
                "nodes": [
                    {
                        "regrets": dict(node["regrets"]),
                        "strategy_sum": dict(node["strategy_sum"]),
                    }
                    for node in nodes
                ],
            }
        )

    result = {
        "version": 1,
        "betting_rules_version": BETTING_RULES_VERSION,
        "metadata": {
            "source": str(input_path.resolve()),
            "source_buckets": len(records),
            "clusters": sum(allocation.values()),
            "history_groups": len(grouped),
            "compression_ratio": len(records) / sum(allocation.values()),
            "zero_mass_clusters": zero_mass_clusters,
            "seed": seed,
        },
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_scale": feature_scale.astype(float).tolist(),
        "groups": output_groups,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, separators=(",", ":")), encoding="utf-8"
    )
    temporary.replace(output_path)
    return result["metadata"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress a 7th-street MCCFR table with k-means.")
    parser.add_argument("--input", type=Path, default=Path("models/mccfr_7th.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/mccfr_7th_kmeans.json")
    )
    parser.add_argument("--clusters", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = compress_table(
        args.input,
        args.output,
        clusters=args.clusters,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
