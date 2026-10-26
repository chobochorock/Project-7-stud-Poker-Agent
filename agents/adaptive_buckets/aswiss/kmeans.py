"""Card-feature k-means abstraction matched per time step to the bucket counts of a split run.

Usage: python kmeans.py TREE_DIR FINAL_BUCKET_MAP OUT_MAP [seed]
Within each time-step group g (the coarsest buckets), cluster the infosets' card features (own known
rank counts, own public rank counts, opponent public rank counts, previous-street betting one-hot) into
K_g clusters, where K_g is the number of final buckets the split run produced inside g. Plain Lloyd
k-means with k-means++ seeding on standardized features; K_g is capped by the number of distinct
feature rows.
"""
import sys
import numpy as np

tree_dir, final_map, out, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(seed)
coarse = np.fromfile(f"{tree_dir}/bucket_timestep.bin", np.int32)
final = np.fromfile(final_map, np.int32)
X = np.fromfile(f"{tree_dir}/features.bin").reshape(len(coarse), -1)
X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
result = np.empty(len(coarse), np.int32); next_id = 0; total_k = 0
for g in np.unique(coarse):
    members = np.flatnonzero(coarse == g)
    k = len(np.unique(final[members]))
    rows, inverse = np.unique(X[members], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    weight = np.bincount(inverse).astype(float)
    k = min(k, len(rows))
    centres = [rows[rng.choice(len(rows), p=weight / weight.sum())]]
    for _ in range(1, k):
        d = np.min(((rows[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1) * weight
        centres.append(rows[rng.choice(len(rows), p=d / d.sum())] if d.sum() > 0 else rows[rng.integers(len(rows))])
    C = np.array(centres)
    for _ in range(50):
        assign = ((rows[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([np.average(rows[assign == j], axis=0, weights=weight[assign == j]) if (assign == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C): break
        C = newC
    used = np.unique(assign); remap = {j: i for i, j in enumerate(used)}
    result[members] = next_id + np.array([remap[j] for j in assign[inverse]])
    next_id += len(used); total_k += len(used)
result.tofile(out)
print(f"k-means buckets={total_k} (split run final buckets={len(np.unique(final))})")
