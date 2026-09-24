"""Fit AE+k-means and VQ-VAE buckets on the existing C++ 7-stud scope.

Usage (project root, system Python with torch/sklearn/matplotlib):
    python -m agents.autoencoder_abstraction.experiment collect --out-dir RUN
    python -m agents.autoencoder_abstraction.experiment train --out-dir RUN
    python -m agents.autoencoder_abstraction.experiment report --out-dir RUN
Data: pre-action observations, specified low-fold policy, hand-level 80/10/10 split.
Metrics: held-out compression/behavior-return diagnostics, NOT CFR or exploitability.
See README.md for scope, frozen bucket API, information loss and checkpoint usage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from agents.state_action_embedding.experiment import MemoryMeter, write_json


FAMILY = Path(__file__).resolve().parent
ROOT = FAMILY.parents[1]
FOLD = np.array([0.02625, 0.06125, 0.0875])
ROW = np.dtype([
    ("hand", "<u4"), ("actor", "u1"), ("street", "u1"),
    ("mask", "u1"), ("action", "u1"), ("power", "<f4", (18,)),
    ("cards", "u1", (12,)), ("history", "u1", (24,)),
    ("scalars", "<f4", (7,)), ("outcome", "<f4"),
])


def collect(args):
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    binary = FAMILY / "bin" / "collect.exe"
    binary.parent.mkdir(exist_ok=True)
    subprocess.run([
        "g++", "-O2", "-std=c++17", str(FAMILY / "collect.cpp"), "-o", str(binary)
    ], check=True)
    subprocess.run([str(binary), "--self-test"], check=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw = args.out_dir / "collection.pending.bin"
    started = time.perf_counter()
    subprocess.run([
        str(binary), str(raw), str(args.hands), str(args.data_seed), "128"
    ], check=True)
    with raw.open("rb") as stream:
        if stream.read(8) != b"NAE7D01\0":
            raise ValueError("Invalid collection header")
        data = np.fromfile(stream, dtype=ROW)
    if (raw.stat().st_size - 8) % ROW.itemsize:
        raise ValueError("Truncated collection")
    if not len(data) or len(np.unique(data["hand"])) != args.hands:
        raise ValueError("Incomplete collection")
    if np.any((data["mask"] & (1 << data["action"])) == 0):
        raise ValueError("Illegal action in collection")
    partition = np.zeros(args.hands, np.uint8)
    order = np.random.default_rng(args.data_seed + 1).permutation(args.hands)
    partition[order[int(.8 * args.hands):int(.9 * args.hands)]] = 1
    partition[order[int(.9 * args.hands):]] = 2
    np.savez_compressed(args.out_dir / "dataset.npz", rows=data, split=partition[data["hand"]])
    with np.load(args.out_dir / "dataset.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["rows"], data)
    raw.unlink()  # This run's verified staging file, never an earlier dataset.
    observed = []
    for street in (5, 6, 7):
        selected = data[data["street"] == street]
        eligible = selected[(selected["mask"] & 128) != 0]
        count = len(eligible)
        observed.append({
            "street": street, "decisions": len(selected), "fold_eligible": count,
            "folds": int((eligible["action"] == 7).sum()),
            "empirical_fold": float((eligible["action"] == 7).mean()),
            "target_fold": float(FOLD[street - 5]),
            "actions": np.bincount(selected["action"], minlength=8).tolist(),
        })
    write_json(args.out_dir / "dataset.json", {
        "environment": "existing C++ heads-up 7-stud v3, fixed heuristic H4",
        "ante": 1000, "stack_antes": 1000, "hands": args.hands,
        "data_seed": args.data_seed, "power_completion_limit": 128,
        "sampling": "per-decision fold schedule; uniform over remaining legal actions",
        "rows": len(data), "split_hands": np.bincount(partition, minlength=3).tolist(),
        "fold_audit": observed, "seconds": time.perf_counter() - started,
        "bytes": (args.out_dir / "dataset.npz").stat().st_size,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (FAMILY / "collect.cpp", ROOT / "agents/cpp_mccfr/stud_mccfr.cpp",
                         ROOT / "environments/seven_stud/stud_rules.hpp")},
    })
    print(json.dumps(observed, indent=2), flush=True)


def inputs(rows, scope):
    if scope == "power":
        return rows["power"].copy()
    # The existing 1832D Deep-CFR observation, not an omniscient state.
    return np.concatenate([
        np.eye(2, dtype=np.float32)[rows["actor"]],
        np.eye(3, dtype=np.float32)[rows["street"] - 5],
        np.eye(53, dtype=np.float32)[rows["cards"]].reshape(len(rows), -1),
        np.eye(49, dtype=np.float32)[rows["history"]].reshape(len(rows), -1),
        rows["scalars"],
        ((rows["mask"][:, None] >> np.arange(8)) & 1).astype(np.float32),
    ], axis=1)


class AutoEncoder(nn.Module):
    def __init__(self, features=18, hidden=64, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VQVAE(AutoEncoder):
    def __init__(self, features=18, hidden=64, latent=8, codes=256):
        super().__init__(features, hidden, latent)
        self.codebook = nn.Embedding(codes, latent)

    def quantize(self, z):
        distances = (z.square().sum(1, keepdim=True)
            + self.codebook.weight.square().sum(1)[None, :]
            - 2 * z @ self.codebook.weight.T)
        ids = distances.argmin(1)
        return self.codebook(ids), ids

    def loss(self, x, beta=0.25):
        z = self.encoder(x)
        quantized, ids = self.quantize(z)
        straight_through = z + (quantized - z).detach()
        reconstructed = self.decoder(straight_through)
        reconstruction = F.mse_loss(reconstructed, x)
        codebook = F.mse_loss(quantized, z.detach())
        commitment = F.mse_loss(z, quantized.detach())
        return reconstruction + codebook + beta * commitment, reconstruction, ids

    def forward(self, x):
        quantized, _ = self.quantize(self.encoder(x))
        return self.decoder(quantized)


def kmeans(x, codes, seed):
    # Same sklearn solver/settings for raw, random and AE latent baselines.
    return KMeans(n_clusters=codes, random_state=seed, n_init=3, max_iter=100).fit(x)


def encode(model, x):
    with torch.inference_mode():
        return np.concatenate([model.encoder(torch.from_numpy(batch)).numpy()
            for batch in np.array_split(x, max(1, (len(x) + 2047) // 2048))])


def fit_network(x, split, method, seed, args):
    torch.manual_seed(seed)
    architecture = dict(features=x.shape[1], hidden=args.hidden, latent=args.latent)
    if method == "vqvae":
        architecture["codes"] = args.codes
        model = VQVAE(**architecture)
    else:
        model = AutoEncoder(**architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(seed)
    train = np.flatnonzero(split == 0)
    validation = torch.from_numpy(x[split == 1])
    curve, best, best_state, best_step = [], float("inf"), None, None
    with MemoryMeter() as memory:
        for step in range(args.steps):
            if method == "vqvae" and step == args.warmup:
                chosen = np.random.default_rng(seed + 1000).choice(
                    train, min(len(train), args.fit_cap), replace=False)
                centers = kmeans(encode(model, x[chosen]), args.codes, seed).cluster_centers_
                with torch.no_grad():
                    model.codebook.weight.copy_(torch.from_numpy(centers))
            batch = torch.from_numpy(x[rng.choice(train, args.batch)])
            optimizer.zero_grad(set_to_none=True)
            if method == "vqvae" and step >= args.warmup:
                loss, _, _ = model.loss(batch)
            else:
                loss = F.mse_loss(model.decoder(model.encoder(batch)), batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step >= args.warmup and ((step + 1) % 100 == 0 or step == args.steps - 1):
                with torch.no_grad():
                    error = float(F.mse_loss(model(validation), validation))
                curve.append({"step": step + 1, "train_loss": float(loss), "validation_mse": error})
                if error < best:
                    best, best_state, best_step = error, copy.deepcopy(model.state_dict()), step + 1
    model.load_state_dict(best_state)
    model.eval()
    return model, architecture, curve, memory.result(), best_step


def assign(checkpoint, x):
    if x.ndim != 2 or x.shape[1] != checkpoint["features"] or not np.isfinite(x).all():
        raise ValueError("Wrong or non-finite abstraction input")
    x = np.asarray(x, np.float32)
    method = checkpoint["method"]
    if method == "raw_kmeans":
        z = x
    else:
        model = (VQVAE if method == "vqvae" else AutoEncoder)(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        z = encode(model, x)
        if method == "vqvae":
            with torch.no_grad():
                return np.concatenate([model.quantize(torch.from_numpy(batch))[1].numpy()
                    for batch in np.array_split(z, max(1, (len(z) + 2047) // 2048))])
    centers = checkpoint["centers"].numpy()
    return pairwise_distances_argmin(z, centers)


def bucket_keys(rows, ids, retained_context=None):
    """Final solver keys must retain betting context, not just these card codes.

    Pass the solver's existing betting-context keys unchanged. Actor/street/legal
    mask are hard partitions. No perfect-recall guarantee is made for compressed IDs.
    """
    if retained_context is None or len(rows) != len(ids) or len(rows) != len(retained_context):
        raise ValueError("Explicit retained betting context required for every row")
    return [(int(row["actor"]), int(row["street"]), int(row["mask"]), int(code), context)
        for row, code, context in zip(rows, ids, retained_context)]


def diagnostics(rows, split, ids, codes):
    train, test = split == 0, split == 2
    power = rows["power"]
    counts = np.bincount(ids[train], minlength=codes)
    total = np.zeros((codes, 18), np.float64)
    np.add.at(total, ids[train], power[train])
    centers = total / counts.clip(1)[:, None]
    centers[counts == 0] = power[train].mean(0)
    probabilities = counts[counts > 0] / counts.sum()
    # Reward is used ONLY in this held-out probe, never encoder/clustering fitting.
    # Use one deterministic decision per hand/actor/street, avoiding length weighting.
    _, unique = np.unique(rows["hand"].astype(np.int64) * 2 + rows["actor"], return_index=True)
    anchor = np.zeros(len(rows), bool)
    anchor[unique] = True
    fit, evaluate = train & anchor, test & anchor
    base_key = rows["actor"].astype(np.int64) * 256 + rows["mask"]
    key = base_key * codes + ids
    def average_by_key(keys, count):
        mass = np.bincount(keys[fit], minlength=count)
        rewards = np.bincount(keys[fit], weights=rows["outcome"][fit], minlength=count)
        estimate = rewards / mass.clip(1)
        estimate[mass == 0] = rows["outcome"][fit].mean()
        return estimate, mass
    base_estimate, _ = average_by_key(base_key, 512)
    estimate, mass = average_by_key(key, 512 * codes)
    prediction = estimate[key[evaluate]]
    unseen = mass[key[evaluate]] == 0
    prediction[unseen] = base_estimate[base_key[evaluate]][unseen]
    return {
        "power_bucket_mse": float(np.mean((power[test] - centers[ids[test]]) ** 2)),
        "used_codes_train": int((counts > 0).sum()),
        "effective_codes_train": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "used_codes_test": int(len(np.unique(ids[test]))),
        "largest_code_fraction": float(counts.max() / counts.sum()),
        "return_rmse_antes": float(np.sqrt(np.mean((prediction - rows["outcome"][evaluate]) ** 2))),
        "group_mean_return_rmse_antes": float(np.sqrt(np.mean(
            (base_estimate[base_key[evaluate]] - rows["outcome"][evaluate]) ** 2))),
        "unseen_reward_bucket_fraction": float(unseen.mean()),
        "test_return_anchors": int(evaluate.sum()), "test_rows": int(test.sum()),
    }


def train(args):
    with np.load(args.out_dir / "dataset.npz", allow_pickle=False) as stored:
        rows, split = stored["rows"], stored["split"]
    for seed in args.seeds:
        for street in (5, 6, 7):
            selected = rows["street"] == street
            street_rows, partition = rows[selected], split[selected]
            x = inputs(street_rows, args.scope)
            train_indices = np.flatnonzero(partition == 0)
            if len(train_indices) < args.codes or not all((partition == i).any() for i in range(3)):
                raise ValueError("Insufficient data for requested codes/splits")
            chosen = np.random.default_rng(seed).choice(
                train_indices, min(len(train_indices), args.fit_cap), replace=False)
            for method in ("raw_kmeans", "random_kmeans", "ae_kmeans", "vqvae"):
                run = args.out_dir / f"{args.scope}_{method}_s{street}_seed{seed}"
                run.mkdir(exist_ok=False)
                started = time.perf_counter()
                checkpoint = dict(method=method, features=x.shape[1], scope=args.scope,
                    street=street, seed=seed, codes=args.codes, format_version=1)
                curve, memory, best_step = [], None, None
                if method in ("ae_kmeans", "vqvae"):
                    model, architecture, curve, memory, best_step = fit_network(x, partition, method, seed, args)
                    checkpoint.update(architecture=architecture, model=model.state_dict())
                    z = encode(model, x[chosen])
                elif method == "random_kmeans":
                    torch.manual_seed(seed)
                    architecture = dict(features=x.shape[1], hidden=args.hidden, latent=args.latent)
                    model = AutoEncoder(**architecture).eval()
                    checkpoint.update(architecture=architecture, model=model.state_dict())
                    z = encode(model, x[chosen])
                else:
                    z = x[chosen]
                if method != "vqvae":
                    checkpoint["centers"] = torch.from_numpy(kmeans(z, args.codes, seed).cluster_centers_)
                ids = assign(checkpoint, x)
                path = run / "checkpoint.pt"
                torch.save(checkpoint, path)
                loaded = torch.load(path, weights_only=True, map_location="cpu")
                np.testing.assert_array_equal(ids[:37], assign(loaded, x[:37]))
                metrics = diagnostics(street_rows, partition, ids, args.codes)
                metrics.update(method=method, scope=args.scope, seed=seed, street=street,
                    steps=args.steps if method in ("ae_kmeans", "vqvae") else 0,
                    warmup=args.warmup, hidden=args.hidden, latent=args.latent, codes=args.codes,
                    fit_rows=len(chosen), best_step=best_step, curve=curve, memory=memory,
                    seconds=time.perf_counter() - started, checkpoint_bytes=path.stat().st_size,
                    torch=str(torch.__version__), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                write_json(run / "metrics.json", metrics)
                print(f"RESULT scope={args.scope} method={method} street={street} seed={seed} "
                    f"mse={metrics['power_bucket_mse']:.6f} used_codes={metrics['used_codes_train']} "
                    f"seconds={metrics['seconds']:.2f}", flush=True)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    results = [json.loads(path.read_text()) for path in sorted(args.out_dir.glob(f"{args.scope}_*/metrics.json"))]
    if not results:
        raise ValueError("No completed metrics")
    summary = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    measurements = ("power_bucket_mse", "effective_codes_train", "return_rmse_antes")
    for method in ("raw_kmeans", "random_kmeans", "ae_kmeans", "vqvae"):
        values = []
        for street in (5, 6, 7):
            group = [item for item in results if item["method"] == method and item["street"] == street]
            if not group:
                continue
            item = {"method": method, "street": street, "seeds": len(group)}
            for metric in (*measurements, "used_codes_train", "largest_code_fraction", "seconds", "checkpoint_bytes"):
                item[metric] = {"mean": float(np.mean([r[metric] for r in group])),
                    "seed_sd": float(np.std([r[metric] for r in group]))}
            summary.append(item)
            values.append(item)
        for axis, metric in zip(axes, measurements):
            axis.errorbar([item["street"] for item in values],
                [item[metric]["mean"] for item in values],
                yerr=[item[metric]["seed_sd"] for item in values], marker="o", label=method, capsize=3)
            axis.set_xticks([5, 6, 7])
            axis.set_xlabel("Street")
            axis.set_title(metric)
            axis.grid(alpha=.2)
    baseline = [next(item["group_mean_return_rmse_antes"] for item in results
        if item["street"] == street) for street in (5, 6, 7)]
    axes[2].plot([5, 6, 7], baseline, "--", color="black", label="actor/legal-mask mean")
    for axis, title in zip(axes, ("Power bucket MSE (lower better)",
        "Effective occupied codes", "Behavior-return RMSE (antes)")):
        axis.set_title(title)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{args.scope} | held-out bucket diagnostics, NOT CFR strength | seed SD, not CI")
    fig.savefig(args.out_dir / f"{args.scope}_comparison.png", dpi=150)
    plt.close(fig)
    write_json(args.out_dir / f"{args.scope}_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train", "report"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=10000)
    parser.add_argument("--data-seed", type=int, default=20260922)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--scope", choices=("power", "infoset"), default="power")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=8)
    parser.add_argument("--codes", type=int, default=256)
    parser.add_argument("--fit-cap", type=int, default=20000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.hands, args.steps, args.batch, args.hidden, args.latent, args.codes, args.threads) < 1:
        parser.error("Counts must be positive")
    if not 0 <= args.warmup < args.steps or args.fit_cap < args.codes:
        parser.error("Invalid warmup or fit cap")
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        {"collect": collect, "train": train, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
