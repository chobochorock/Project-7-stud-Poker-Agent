"""Fit TARA forests from interval buffers (sklearn) and the B2 fixed centroid partition.

Library module used by run.py; no training on import. Usage (project root):
    python -m agents.cpp_mccfr.tara.fit --buffer <run>/buffer_3.bin --budget 2048 --output <run>/forest_4.bin
Buffer (written by solver.cpp at the end of interval t): per public node the legal mask, street,
visit count, regret mass (sum of per-visit L2 norms) and a uniform reservoir of visits
(card features phi, 22 floats; unweighted sampled regret r, 8 floats).
Forest (read by solver.cpp): per public node one multi-output DecisionTreeRegressor
(criterion squared_error; label = r over the node's legal actions; one row per sampled visit),
leaves allocated across public nodes in proportion to regret mass (D'Hondt, at least 1 each).
--partition random keeps the rows and leaf allocation but splits on random features/thresholds
(control for whether regret-fitted splits matter; cf. random partitions in f-RCFR, D'Orazio et al. 2020).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

FEATURES = 22
ACTIONS = 8


def _text(handle) -> bytes:
    (size,) = struct.unpack("<I", handle.read(4))
    return handle.read(size)


def read_buffer(path: Path) -> list[dict]:
    slots = []
    with path.open("rb") as handle:
        if _text(handle) != b"TARABUFFER1":
            raise ValueError("invalid buffer file")
        (count,) = struct.unpack("<I", handle.read(4))
        row = np.dtype([("x", "<f4", FEATURES), ("r", "<f4", ACTIONS)])
        for _ in range(count):
            key = _text(handle)
            mask, street = struct.unpack("<BB", handle.read(2))
            visits, mass = struct.unpack("<Qd", handle.read(16))
            (rows,) = struct.unpack("<I", handle.read(4))
            data = np.frombuffer(handle.read(rows * row.itemsize), dtype=row, count=rows)
            slots.append(dict(key=key, mask=mask, street=street, visits=visits, mass=mass,
                              x=np.array(data["x"]), r=np.array(data["r"])))
        if handle.read(1):
            raise ValueError("trailing buffer data")
    # The C++ writes in hash-map order, which depends on insertion history (e.g. a mid-interval resume).
    return sorted(slots, key=lambda s: s["key"])


def legal(mask: int) -> list[int]:
    return [a for a in range(ACTIONS) if mask >> a & 1]


def dhondt(mass: list[float], caps: list[int], total: int) -> list[int]:
    """At least one leaf each, at most caps[i], then the largest mass / leaves first."""
    alloc = [1] * len(mass)
    left = total - len(mass)
    while left > 0:
        best, value = -1, -1.0
        for i, (m, k) in enumerate(zip(mass, alloc)):
            if k < caps[i] and m / k > value:
                best, value = i, m / k
        if best < 0:
            break
        alloc[best] += 1
        left -= 1
    return alloc


def floor32(values: np.ndarray) -> np.ndarray:
    """Largest float32 <= each float64 threshold, so float32 x <= t32 iff x <= threshold."""
    t32 = values.astype(np.float32)
    over = t32.astype(np.float64) > values
    t32[over] = np.nextafter(t32[over], np.float32(-np.inf))
    return t32


def _write_text(out, data: bytes) -> None:
    out.write(struct.pack("<I", len(data)))
    out.write(data)


def random_tree(x: np.ndarray, k: int, min_samples_leaf: int, rng: np.random.Generator) -> list:
    """Control partition (f-RCFR-style): same rows and leaf count, but a random feature and a random
    admissible threshold per split, ignoring the regret labels. Largest leaf is split first."""
    nodes = [(-1, 0.0, -1, -1)]
    members = {0: np.arange(len(x))}
    while len(members) < k:
        candidates = [(len(rows), node) for node, rows in members.items() if len(rows) >= 2 * min_samples_leaf]
        split = False
        for _, node in sorted(candidates, reverse=True):
            rows = members[node]
            for feature in rng.permutation(x.shape[1]):
                values = np.sort(x[rows, feature].astype(np.float64))
                # threshold between order statistics msl-1 and n-msl keeps min_samples_leaf on both sides
                lo, hi = min_samples_leaf - 1, len(values) - min_samples_leaf
                gaps = [i for i in range(lo, hi) if values[i] < values[i + 1]]
                if not gaps:
                    continue
                i = gaps[rng.integers(len(gaps))]
                threshold = float(floor32(np.array([(values[i] + values[i + 1]) / 2]))[0])
                left = rows[x[rows, feature] <= np.float32(threshold)]
                right = rows[x[rows, feature] > np.float32(threshold)]
                if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                    continue
                nodes[node] = (int(feature), threshold, len(nodes), len(nodes) + 1)
                nodes += [(-1, 0.0, -1, -1), (-1, 0.0, -1, -1)]
                del members[node]
                members[len(nodes) - 2], members[len(nodes) - 1] = left, right
                split = True
                break
            if split:
                break
        if not split:
            break
    return nodes


def fit_slot(slot: dict, k: int, min_samples_leaf: int, seed: int, partition: str, index: int) -> list:
    """Tree with at most k leaves for one public node (a single leaf when k <= 1 or too few rows)."""
    from sklearn.tree import DecisionTreeRegressor

    if k <= 1 or len(slot["r"]) < 2 * min_samples_leaf:
        return [(-1, 0.0, -1, -1)]
    if partition == "random":  # per-node stream: nested in k and independent of the other nodes
        return random_tree(slot["x"], k, min_samples_leaf, np.random.default_rng([seed, index]))
    model = DecisionTreeRegressor(criterion="squared_error", max_leaf_nodes=k,
                                  min_samples_leaf=min_samples_leaf, random_state=seed)
    model.fit(slot["x"], slot["r"][:, legal(slot["mask"])].astype(np.float64))
    tree = model.tree_
    thresholds = floor32(tree.threshold)
    return [(-1, 0.0, -1, -1) if tree.children_left[i] == -1 else
            (int(tree.feature[i]), float(thresholds[i]), int(tree.children_left[i]), int(tree.children_right[i]))
            for i in range(tree.node_count)]


def fit_forest(slots: list[dict], budget: int, min_samples_leaf: int, seed: int,
               partition: str = "learned", rounds: int = 6) -> tuple[list, dict]:
    """D'Hondt allocation by regret mass; a node whose tree cannot realize its allocation (too few distinct
    rows) is capped at what it realized and the freed leaves are reallocated (up to `rounds` passes)."""
    caps = [max(1, len(s["r"]) // max(1, min_samples_leaf)) for s in slots]
    total = max(budget, len(slots))
    cache: dict[tuple[int, int], list] = {}
    for done in range(1, rounds + 1):
        alloc = dhondt([s["mass"] for s in slots], caps, total)
        trees, short = [], False
        for i, (slot, k) in enumerate(zip(slots, alloc)):
            if (i, k) not in cache:
                cache[i, k] = fit_slot(slot, k, min_samples_leaf, seed, partition, i)
            nodes = cache[i, k]
            realized = sum(1 for node in nodes if node[0] < 0)
            if realized < k:
                caps[i], short = realized, True
            trees.append(nodes)
        if not short:
            break
    records = [(slot["key"], slot["mask"], nodes) for slot, nodes in zip(slots, trees)]
    leaves = sum(1 for nodes in trees for node in nodes if node[0] < 0)
    fitted = sum(1 for nodes in trees if len(nodes) > 1)
    stats = dict(partition=partition, public_nodes=len(slots), fitted_public_nodes=fitted, leaves=leaves,
                 allocated=int(sum(alloc)), allocation_rounds=done, budget=budget, min_samples_leaf=min_samples_leaf,
                 rows=int(sum(len(s["r"]) for s in slots)), visits=int(sum(s["visits"] for s in slots)),
                 mass=float(sum(s["mass"] for s in slots)))
    return records, stats


def write_forest(path: Path, records: list, centroids: dict | None = None, kind: int | None = None) -> None:
    """kind 0: fitted trees (records); 1: B2 centroids; 2: B0 hard256 atlas cluster (no payload)."""
    kind = (1 if centroids else 0) if kind is None else kind
    with path.open("wb") as out:
        _write_text(out, b"TARAFOREST1")
        out.write(struct.pack("<i", kind))
        if centroids:
            for street in (5, 6, 7):
                mean, scale, centres = centroids[street]
                out.write(np.asarray(mean, "<f4").tobytes())
                out.write(np.asarray(scale, "<f4").tobytes())
                out.write(struct.pack("<I", len(centres)))
                out.write(np.asarray(centres, "<f4").tobytes())
        out.write(struct.pack("<I", len(records)))
        for key, mask, nodes in records:
            _write_text(out, key)
            out.write(struct.pack("<BI", mask, len(nodes)))
            for index, (feature, threshold, left, right) in enumerate(nodes):
                out.write(struct.pack("<ifiii", feature, threshold, left, right, index if feature < 0 else -1))


def fit_centroids(slots: list[dict], clusters: int, seed: int) -> dict:
    """B2: one standardized k-means per street on the same card features, shared by its public nodes."""
    from sklearn.cluster import KMeans

    result = {}
    for street in (5, 6, 7):
        x = np.concatenate([s["x"] for s in slots if s["street"] == street]).astype(np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale <= 0] = 1.0
        z = (x - mean) / scale
        k = min(clusters, len(np.unique(z, axis=0)))
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
        result[street] = (mean, scale, model.cluster_centers_)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partition", choices=("learned", "random"), default="learned")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    records, stats = fit_forest(read_buffer(args.buffer), args.budget, args.min_samples_leaf, args.seed,
                                args.partition)
    write_forest(args.output, records)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
