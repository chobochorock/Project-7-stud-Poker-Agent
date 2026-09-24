"""Compare inductive forward Skip-gram and causal CPC on seven-poker v3.

Usage (project root, Python 3.12 with the existing Toy PyTorch dependencies):
    python -m agents.state_action_embedding.experiment collect --out-dir RUN
    python -m agents.state_action_embedding.experiment train --out-dir RUN
    python -m agents.state_action_embedding.experiment train --out-dir RUN --method cpc
    python -m agents.state_action_embedding.experiment memory --out-dir RUN

Input: fresh heads-up cash hands, uniformly random legal betting/discard actions.
Output: compressed player-view sequences, inference checkpoints, JSON measurements.
Metrics: frozen linear probes on held-out hands; not exploitability or policy strength.
Limit: betting events only; no strategic opponent model or reinforcement learning.
See README.md for the shared-target benchmark and memory measurement definitions.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)
try:
    import torch
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT.parent / "Toy-Card-Game-Agent" / ".deep_cfr_deps"))
    import torch

import numpy as np
from torch import nn
from torch.nn import functional as F

from agent.base import PokerAgent
from environments.seven_stud.poker_env import (
    ALL_CARDS,
    BETTING_ACTIONS,
    BETTING_RULES_VERSION,
    PokerGame,
)


CARD_IDS = {str(card): index for index, card in enumerate(ALL_CARDS)}
STREETS = ("5th", "6th", "7th_hidden")
STACK = 1000
FEATURES = 4 * 52 + 3 + 2 + 8 + 8 + 14
MIB = 1024**2


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def encode_event(state: dict, action: str, actor_is_self: bool) -> np.ndarray:
    """Current observation + public action only; deliberately ignore history."""
    opponent = state["opponents"][0]
    cards = np.zeros((4, 52), dtype=np.float32)
    groups = (
        state["my_hidden_cards"],
        state["my_public_cards"],
        [state["my_discarded_card"]] if state["my_discarded_card"] else [],
        opponent["public_cards"],
    )
    for row, group in zip(cards, groups):
        for card in group:
            row[CARD_IDS[card]] = 1
    scalars = np.array(
        [
            state["pot"] / STACK,
            state["current_highest_bet"] / STACK,
            state["my_chips"] / STACK,
            state["my_invested"] / STACK,
            state["my_round_bet"] / STACK,
            opponent["chips"] / STACK,
            opponent["invested"] / STACK,
            opponent["round_bet"] / STACK,
            state["call_amount"] / STACK,
            state["raise_count"] / 6,
            state["my_bet_count"] / 3,
            state["raise_cap"] / 3,
            float(state["my_is_all_in"]),
            float(opponent["is_all_in"]),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            cards.ravel(),
            np.eye(3, dtype=np.float32)[STREETS.index(state["street"])],
            np.eye(2, dtype=np.float32)[int(actor_is_self)],
            np.eye(8, dtype=np.float32)[BETTING_ACTIONS.index(action)],
            np.array([a in state["valid_actions"] for a in BETTING_ACTIONS]),
            scalars,
        ]
    ).astype(np.float32)


class RecordingGame(PokerGame):
    """Record both fixed player perspectives without changing any game rules."""

    def __init__(self):
        super().__init__(["p0", "p1"], log_file=None, game_mode="cash")
        self.events = [[], []]
        self.actions = []
        self.streets = []

    def apply_action(self, player, action):
        legal = self.get_valid_actions(player)
        for seat, viewer in enumerate(self.players):
            state = self.get_ai_state(viewer, legal)
            self.events[seat].append(encode_event(state, action, viewer is player))
        self.actions.append(BETTING_ACTIONS.index(action))
        self.streets.append(STREETS.index(self.street))
        return super().apply_action(player, action)


class UniformAgent(PokerAgent):
    def __init__(self, name: str, seed: int):
        super().__init__(name)
        self.rng = random.Random(seed)

    def choose_action(self, state, valid_actions):
        return self.rng.choice(valid_actions)

    def choose_discard_and_reveal(self, hidden_cards):
        return tuple(self.rng.sample(range(len(hidden_cards)), 2))


def collect(args) -> None:
    path = args.out_dir / "trajectories.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    started = time.perf_counter()
    random.seed(args.seed)
    split_rng = np.random.default_rng(args.seed + 1)
    partition = np.empty(args.hands, dtype=np.int8)
    order = split_rng.permutation(args.hands)
    partition[order[: int(args.hands * 0.8)]] = 0
    partition[order[int(args.hands * 0.8) : int(args.hands * 0.9)]] = 1
    partition[order[int(args.hands * 0.9) :]] = 2
    sequences, actions, streets, returns, splits = [], [], [], [], []
    agents = {f"p{i}": UniformAgent(f"p{i}", args.seed + i + 2) for i in range(2)}
    game = RecordingGame()
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in range(args.hands):
            game.events = [[], []]
            game.actions = []
            game.streets = []
            result = game.play_hand(agents)
            assert sum(result["final_chips"].values()) == 2 * STACK
            for seat in range(2):
                sequences.append(np.asarray(game.events[seat], dtype=np.float32))
                actions.append(game.actions.copy())
                streets.append(game.streets.copy())
                returns.append((result["final_chips"][f"p{seat}"] - STACK) / STACK)
                splits.append(partition[hand])
    lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int16)
    # ponytail: padded in-RAM corpus; use packed/memory-mapped sequences at scale.
    padded = np.zeros((len(sequences), int(lengths.max()), FEATURES), np.float32)
    action_array = np.full(padded.shape[:2], -1, dtype=np.int8)
    street_array = action_array.copy()
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
        action_array[index, : len(sequence)] = actions[index]
        street_array[index, : len(sequence)] = streets[index]
    np.savez_compressed(
        path,
        x=padded,
        lengths=lengths,
        actions=action_array,
        streets=street_array,
        returns=np.asarray(returns, np.float32),
        split=np.asarray(splits, np.int8),
        hand=np.repeat(np.arange(args.hands), 2),
    )
    write_json(
        args.out_dir / "dataset.json",
        {
            "game": "project seven-poker v3; heads-up cash; uniform legal actions",
            "betting_rules_version": BETTING_RULES_VERSION,
            "seed": args.seed,
            "hands": args.hands,
            "features": FEATURES,
            "betting_events_per_hand_mean": float(lengths.mean()),
            "max_length": int(lengths.max()),
            "player_view_tokens": int(lengths.sum()),
            "split_hands": [int((partition == i).sum()) for i in range(3)],
            "generation_seconds": time.perf_counter() - started,
            "compressed_bytes": path.stat().st_size,
            "dense_feature_bytes": padded.nbytes,
            "environment_sha256": hashlib.sha256(
                (ROOT / "environments/seven_stud/poker_env.py").read_bytes()
            ).hexdigest(),
        },
    )
    print(f"Collected {args.hands} hands into {path}", flush=True)


class Representation(nn.Module):
    def __init__(self, method: str, dim: int = 64, context: int = 16):
        super().__init__()
        self.method, self.dim, self.context = method, dim, context
        self.encoder = nn.Sequential(
            nn.Linear(FEATURES, 128), nn.ReLU(), nn.Linear(128, dim), nn.LayerNorm(dim)
        )
        if method == "cpc":
            self.position = nn.Parameter(torch.zeros(context, dim))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                dim, 4, dim * 2, dropout=0.0, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                layer, 2, enable_nested_tensor=False
            )
        self.predictors = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in (1, 2)]
        )

    def summarize(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(len(x), device=x.device)
        if self.method != "cpc":
            return self.encoder(x[rows, lengths - 1])
        encoded = self.encoder(x)
        width = x.shape[1]
        mask = torch.ones(width, width, device=x.device, dtype=torch.bool).triu(1)
        padding = torch.arange(width, device=x.device)[None, :] >= lengths[:, None]
        hidden = self.transformer(
            encoded + self.position[:width], mask=mask, src_key_padding_mask=padding
        )
        return hidden[rows, lengths - 1]


def contexts(data, rows, positions, context):
    lengths = np.minimum(positions + 1, context)
    x = np.zeros((len(rows), int(lengths.max()), FEATURES), np.float32)
    for index, (row, pos, length) in enumerate(zip(rows, positions, lengths)):
        x[index, :length] = data["x"][row, pos + 1 - length : pos + 1]
    return x, lengths


def pair_groups(data, partition=0):
    groups = []
    for horizon in (1, 2):
        for street in range(3):
            pairs = [
                (row, pos)
                for row in np.flatnonzero(data["split"] == partition)
                for pos in range(int(data["lengths"][row]) - horizon)
                if data["streets"][row, pos + horizon] == street
            ]
            if pairs:
                groups.append((horizon, np.asarray(pairs)))
    if not groups:
        raise ValueError("No training pairs: collect more hands")
    return groups


def contrastive_loss(model, x, lengths, targets, hand_ids, horizon):
    query = F.normalize(
        model.predictors[horizon - 1](model.summarize(x, lengths)), dim=-1
    )
    target = F.normalize(model.encoder(targets), dim=-1)
    logits = query @ target.T / 0.1
    diagonal = torch.eye(len(x), dtype=torch.bool, device=x.device)
    same_hand = hand_ids[:, None] == hand_ids[None, :]
    logits = logits.masked_fill(same_hand & ~diagonal, -torch.inf)
    labels = torch.arange(len(x), device=x.device)
    return F.cross_entropy(logits, labels), (logits.argmax(1) == labels).float().mean()


def rss_bytes() -> int:
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "rss",
                    "paged_peak",
                    "paged",
                    "nonpaged_peak",
                    "nonpaged",
                    "pagefile",
                    "pagefile_peak",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(process()), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError()
        return counters.rss
    if Path("/proc/self/statm").exists():
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
    raise RuntimeError("RSS measurement is supported on Windows and Linux")


class MemoryMeter:
    """10 ms sampled process working set; CUDA allocator peaks when available."""

    def __enter__(self):
        gc.collect()
        self.stop = threading.Event()
        self.baseline = self.peak = rss_bytes()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def sample():
            while not self.stop.wait(0.01):
                self.peak = max(self.peak, rss_bytes())

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.cuda_allocated = (
            torch.cuda.max_memory_allocated() / MIB
            if torch.cuda.is_available()
            else None
        )
        self.cuda_reserved = (
            torch.cuda.max_memory_reserved() / MIB
            if torch.cuda.is_available()
            else None
        )
        self.peak = max(self.peak, rss_bytes())
        self.seconds = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()

    def result(self):
        return {
            "rss_baseline_mib": self.baseline / MIB,
            "rss_sampled_peak_mib": self.peak / MIB,
            "rss_peak_delta_mib": (self.peak - self.baseline) / MIB,
            "seconds": self.seconds,
            "cuda_peak_allocated_mib": self.cuda_allocated,
            "cuda_peak_reserved_mib": self.cuda_reserved,
        }


def frozen_features(model, data, device):
    rng = np.random.default_rng(20260919)
    rows = np.flatnonzero(data["lengths"] >= 2)
    # One non-final anchor per player/hand: no overweighting long betting sequences.
    positions = np.array([rng.integers(data["lengths"][row] - 1) for row in rows])
    output = {"encoder": [], "mean_history": []}
    if model.method == "cpc":
        output["context"] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            r, p = rows[start : start + 256], positions[start : start + 256]
            batch, lengths = contexts(data, r, p, model.context)
            x = torch.from_numpy(batch).to(device)
            lens = torch.from_numpy(lengths).to(device)
            local = model.encoder(x)
            output["encoder"].append(
                local[torch.arange(len(r), device=device), lens - 1].cpu().numpy()
            )
            mask = torch.arange(x.shape[1], device=device)[None, :] < lens[:, None]
            mean = (local * mask[..., None]).sum(1) / lens[:, None]
            output["mean_history"].append(mean.cpu().numpy())
            if "context" in output:
                output["context"].append(model.summarize(x, lens).cpu().numpy())
    return (
        {name: np.concatenate(values) for name, values in output.items()},
        data["returns"][rows],
        data["actions"][rows, positions + 1].astype(np.int64),
        data["split"][rows],
        rows,
        positions,
    )


def linear_probe(features, returns, actions, split, steps):
    """Identical linear heads; select checkpoints by validation, never test."""
    train, valid, test = (split == i for i in range(3))
    if not all(mask.any() for mask in (train, valid, test)):
        raise ValueError("Probe requires train, validation and test hands")
    mean, std = features[train].mean(0), features[train].std(0).clip(1e-4)
    x = torch.from_numpy(((features - mean) / std).astype(np.float32))
    rewards = torch.from_numpy(returns)
    labels = torch.from_numpy(actions)
    torch.manual_seed(731)
    head = nn.Linear(x.shape[1], 9)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=0.001)
    best = [math.inf, math.inf]
    predictions = [None, None]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = head(x[train])
        loss = F.mse_loss(output[:, 0], rewards[train]) + F.cross_entropy(
            output[:, 1:], labels[train]
        )
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                output = head(x)
                criteria = [
                    F.mse_loss(output[valid, 0], rewards[valid]).item(),
                    F.cross_entropy(output[valid, 1:], labels[valid]).item(),
                ]
                for index, value in enumerate(criteria):
                    if value < best[index]:
                        best[index] = value
                        predictions[index] = output[test].clone()
    return {
        "return_rmse_chips": float(
            F.mse_loss(predictions[0][:, 0], rewards[test]).sqrt()
        )
        * STACK,
        "next_action_nll": float(F.cross_entropy(predictions[1][:, 1:], labels[test])),
        "next_action_accuracy": float(
            (predictions[1][:, 1:].argmax(1) == labels[test]).float().mean()
        ),
        "test_player_views": int(test.sum()),
    }


def parameter_stats(model, optimizer=None):
    count = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in model.encoder.parameters())
    state_bytes = (
        0
        if optimizer is None
        else sum(
            value.numel() * value.element_size()
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
    )
    return {
        "parameters": count,
        "encoder_parameters": encoder,
        "parameter_mib": sum(p.numel() * p.element_size() for p in model.parameters())
        / MIB,
        "encoder_parameter_mib": sum(
            p.numel() * p.element_size() for p in model.encoder.parameters()
        )
        / MIB,
        "optimizer_state_mib": state_bytes / MIB,
    }


def load_model(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Representation(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def train(args):
    output = args.out_dir / f"{args.method}_seed{args.seed}"
    output.mkdir(exist_ok=False)
    with np.load(args.out_dir / "trajectories.npz", allow_pickle=False) as stored:
        data = dict(stored)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Representation(args.method, args.dim, args.context).to(device)
    groups = pair_groups(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    curve = []
    with MemoryMeter() as memory:
        for step in range(args.steps):
            horizon, pairs = groups[rng.integers(len(groups))]
            rows, pos = pairs[rng.integers(len(pairs), size=args.batch)].T
            batch, lengths = contexts(data, rows, pos, args.context)
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy = contrastive_loss(
                model,
                torch.from_numpy(batch).to(device),
                torch.from_numpy(lengths).to(device),
                torch.from_numpy(data["x"][rows, pos + horizon]).to(device),
                torch.from_numpy(data["hand"][rows]).to(device),
                horizon,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step % 100 == 0 or step == args.steps - 1:
                item = {
                    "step": step + 1,
                    "loss": loss.item(),
                    "retrieval": accuracy.item(),
                }
                curve.append(item)
                print(f"{args.method} seed={args.seed} {item}", flush=True)
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "architecture": {
                "method": args.method,
                "dim": args.dim,
                "context": args.context,
            },
            "model": model.state_dict(),
            "seed": args.seed,
            "steps": args.steps,
            "features": FEATURES,
            "rules_version": BETTING_RULES_VERSION,
        },
        checkpoint,
    )
    features, rewards, actions, split, rows, pos = frozen_features(model, data, device)
    probes = {
        name: linear_probe(value, rewards, actions, split, args.probe_steps)
        for name, value in features.items()
    }
    probes["raw_current"] = linear_probe(
        data["x"][rows, pos], rewards, actions, split, args.probe_steps
    )
    torch.manual_seed(args.seed)
    random_model = Representation("skipgram", args.dim, args.context).to(device)
    random_features = frozen_features(random_model, data, device)[0]["encoder"]
    probes["random_encoder"] = linear_probe(
        random_features, rewards, actions, split, args.probe_steps
    )
    train_mask, test_mask = split == 0, split == 2
    prior = (np.bincount(actions[train_mask], minlength=8) + 1).astype(float)
    prior /= prior.sum()
    probes["constant"] = {
        "return_rmse_chips": float(
            np.sqrt(np.mean((rewards[test_mask] - rewards[train_mask].mean()) ** 2))
        )
        * STACK,
        "next_action_nll": float(-np.log(prior[actions[test_mask]]).mean()),
        "next_action_accuracy": float((actions[test_mask] == prior.argmax()).mean()),
        "test_player_views": int(test_mask.sum()),
    }
    write_json(
        output / "metrics.json",
        {
            "method": args.method,
            "seed": args.seed,
            "steps": args.steps,
            "batch": args.batch,
            "context": args.context,
            "dim": args.dim,
            "device": device,
            "torch": str(torch.__version__),
            "dtype": "float32",
            "threads": torch.get_num_threads(),
            "training_memory": memory.result(),
            "model": parameter_stats(model, optimizer),
            "probes": probes,
            "curve": curve,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(json.dumps(probes, indent=2), flush=True)


def memory_benchmark(args):
    """Run this command in a fresh process for each method/length/phase."""
    path = (
        args.out_dir
        / f"memory_{args.method}_{args.phase}_L{args.context}_B{args.batch}.json"
    )
    if path.exists():
        raise FileExistsError(path)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with MemoryMeter() as memory:
        model = Representation(args.method, args.dim, args.context).to(device)
        x = torch.randn(args.batch, args.context, FEATURES, device=device)
        lengths = torch.full(
            (args.batch,), args.context, dtype=torch.long, device=device
        )
        optimizer = None
        if args.phase == "train":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            target = torch.randn(args.batch, FEATURES, device=device)
            hands = torch.arange(args.batch, device=device)
            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = contrastive_loss(model, x, lengths, target, hands, 1)
                loss.backward()
                optimizer.step()
        else:
            model.eval()
            with torch.inference_mode():
                for _ in range(20):
                    if args.phase == "encoder":
                        model.encoder(x[:, -1])
                    else:
                        model.summarize(x, lengths)
    write_json(
        path,
        {
            "method": args.method,
            "phase": args.phase,
            "context": args.context,
            "batch": args.batch,
            "device": device,
            "dtype": "float32",
            "torch": str(torch.__version__),
            "memory": memory.result(),
            "model": parameter_stats(model, optimizer),
        },
    )
    print(path, flush=True)


def retrieval_audit(args):
    """Diagnostic held-out retrieval; not a cross-model quality score."""
    path = args.out_dir / "retrieval_audit.json"
    if path.exists():
        raise FileExistsError(path)
    with np.load(args.out_dir / "trajectories.npz", allow_pickle=False) as stored:
        data = dict(stored)
    groups = pair_groups(data, partition=2)
    results = []
    for checkpoint in sorted(args.out_dir.glob("*_seed*/checkpoint.pt")):
        model = load_model(checkpoint)
        rng = np.random.default_rng(9123)
        accuracies = {"full_context": [], "last_event_only": []}
        with torch.inference_mode():
            for _ in range(60):
                horizon, pairs = groups[rng.integers(len(groups))]
                rows, positions = pairs[rng.integers(len(pairs), size=args.batch)].T
                for name, context in (
                    ("full_context", model.context),
                    ("last_event_only", 1),
                ):
                    batch, lengths = contexts(data, rows, positions, context)
                    _, accuracy = contrastive_loss(
                        model,
                        torch.from_numpy(batch),
                        torch.from_numpy(lengths),
                        torch.from_numpy(data["x"][rows, positions + horizon]),
                        torch.from_numpy(data["hand"][rows]),
                        horizon,
                    )
                    accuracies[name].append(accuracy.item())
        results.append(
            {
                "run": checkpoint.parent.name,
                "batch": args.batch,
                "batches": 60,
                **{name: float(np.mean(values)) for name, values in accuracies.items()},
            }
        )
    write_json(path, results)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train", "memory", "audit"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("skipgram", "cpc"), default="skipgram")
    parser.add_argument("--hands", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--phase", choices=("train", "encoder", "context"), default="train"
    )
    args = parser.parse_args()
    if (
        min(
            args.hands,
            args.steps,
            args.probe_steps,
            args.batch,
            args.dim,
            args.context,
            args.threads,
        )
        < 1
    ):
        parser.error("Counts and dimensions must be positive")
    if args.hands < 30 or args.dim % 4:
        parser.error("Need hands >= 30 and dim divisible by 4")
    if args.batch < 2 and (args.command == "train" or args.phase == "train"):
        parser.error("Contrastive training needs batch >= 2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    {
        "collect": collect,
        "train": train,
        "memory": memory_benchmark,
        "audit": retrieval_audit,
    }[args.command](args)


if __name__ == "__main__":
    main()
