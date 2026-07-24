from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


ACTIONS = np.asarray(
    ("CHECK", "BBING", "DDADANG", "QUARTER", "HALF", "CALL", "FOLD")
)
QUANTILES = np.asarray((0, 0.01, 0.1, 0.5, 0.9, 0.99, 1))


def analyze_model_dir(model_dir: Path, output: Path | None = None) -> dict[str, object]:
    if output is None:
        output = model_dir / "cluster_analysis.json"
    reports = {
        "spherical_kmeans": _analyze_artifact(
            model_dir / "spherical_kmeans.npz", "spherical_kmeans"
        ),
        "diagonal_gmm": _analyze_artifact(
            model_dir / "diagonal_gmm.npz", "diagonal_gmm"
        ),
    }
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return reports


def _analyze_artifact(path: Path, kind: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as artifact:
        support = artifact["support"].astype(np.float64)
        component_q = artifact["component_q"].astype(np.float64)
        component_policy = artifact["component_policy"].astype(np.float64)
        share = support / support.sum()
        average = support.mean()
        positive = share > 0
        effective_entropy = float(
            np.exp(-(share[positive] * np.log(share[positive])).sum())
        )
        effective_simpson = float(1 / np.square(share).sum())

        q_order = np.argsort(component_q, axis=1)
        q_top = q_order[:, -1]
        q_margin = component_q[np.arange(len(support)), q_top] - component_q[
            np.arange(len(support)), q_order[:, -2]
        ]
        policy_top = np.argmax(component_policy, axis=1)
        policy_entropy = -(
            component_policy * np.log(np.maximum(component_policy, 1e-12))
        ).sum(axis=1)

        geometry_rows: list[dict[str, float | int]]
        geometry_summary: dict[str, object]
        if kind == "spherical_kmeans":
            centers = artifact["centers"].astype(np.float64)
            similarity = centers @ centers.T
            np.fill_diagonal(similarity, -np.inf)
            nearest = np.argmax(similarity, axis=1)
            nearest_value = similarity[np.arange(len(support)), nearest]
            geometry_rows = [
                {
                    "nearest_cluster": int(nearest[index]),
                    "nearest_cosine": float(nearest_value[index]),
                }
                for index in range(len(support))
            ]
            geometry_summary = {
                "nearest_cosine_quantiles": _quantiles(nearest_value)
            }
        else:
            means = artifact["means"].astype(np.float64)
            variances = artifact["variances"].astype(np.float64)
            distance_squared = np.square(means[:, None] - means[None, :]).sum(axis=2)
            np.fill_diagonal(distance_squared, np.inf)
            nearest = np.argmin(distance_squared, axis=1)
            nearest_value = np.sqrt(distance_squared[np.arange(len(support)), nearest])
            anisotropy = variances.max(axis=1) / variances.min(axis=1)
            geometric_std = np.exp(0.5 * np.log(variances).mean(axis=1))
            weights = artifact["weights"].astype(np.float64)
            geometry_rows = [
                {
                    "nearest_cluster": int(nearest[index]),
                    "nearest_distance": float(nearest_value[index]),
                    "mixture_weight": float(weights[index]),
                    "geometric_std": float(geometric_std[index]),
                    "anisotropy": float(anisotropy[index]),
                }
                for index in range(len(support))
            ]
            geometry_summary = {
                "nearest_distance_quantiles": _quantiles(nearest_value),
                "mixture_weight_quantiles": _quantiles(weights),
                "variance_quantiles": _quantiles(variances.ravel()),
                "anisotropy_quantiles": _quantiles(anisotropy),
            }

        rows = []
        for index in range(len(support)):
            rows.append(
                {
                    "cluster": index,
                    "support": float(support[index]),
                    "support_share": float(share[index]),
                    "q_top_action": str(ACTIONS[q_top[index]]),
                    "q_margin": float(q_margin[index]),
                    "q_range": float(component_q[index].max() - component_q[index].min()),
                    "policy_top_action": str(ACTIONS[policy_top[index]]),
                    "policy_top_probability": float(component_policy[index, policy_top[index]]),
                    "policy_entropy": float(policy_entropy[index]),
                    **geometry_rows[index],
                }
            )
        csv_path = path.with_name(f"{path.stem}_clusters.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        report = {
            "components": len(support),
            "active_components": int((support > 0).sum()),
            "effective_components_entropy": effective_entropy,
            "effective_components_simpson": effective_simpson,
            "support_quantiles": _quantiles(support),
            "below_0_1x_average_support": int((support < average * 0.1).sum()),
            "below_0_01x_average_support": int((support < average * 0.01).sum()),
            "q_top_actions": _counts(ACTIONS[q_top]),
            "policy_top_actions": _counts(ACTIONS[policy_top]),
            "largest_clusters": [
                {
                    "cluster": int(index),
                    "support": float(support[index]),
                    "share": float(share[index]),
                }
                for index in np.argsort(support)[-10:][::-1]
            ],
            "component_csv": str(csv_path.resolve()),
            **geometry_summary,
        }
        return report


def _quantiles(values: np.ndarray) -> dict[str, float]:
    labels = ("min", "p01", "p10", "median", "p90", "p99", "max")
    return {
        label: float(value)
        for label, value in zip(labels, np.quantile(values, QUANTILES))
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    names, counts = np.unique(values, return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze saved clustering artifacts.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    analyze_model_dir(args.model_dir, args.output)


if __name__ == "__main__":
    main()
