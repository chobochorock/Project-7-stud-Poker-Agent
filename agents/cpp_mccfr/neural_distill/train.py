"""Train a student network to follow a frozen C++ MCCFR teacher on custom 7-stud states.

Usage (project root, Torch runtime; see README.md):
    $py -m agents.cpp_mccfr.neural_distill.train --dataset RUN/dataset.bin --architecture transformer \
        --seed 11 --steps 30000 --output RUN/transformer_seed11
Input: fixed-size records written by export.cpp (observer tokens, public numerics, legal mask, teacher
    normalized regrets, teacher current/average policies). Hands with hand % 20 == 0 are validation.
Output: weights.ndnet (read by net.hpp), checkpoint.pt, metrics.json, parity_reference.bin.
Objective: regret head = per-state MSE (mean over legal actions) to a per-infoset positive rescaling of
    the teacher regrets (regret matching is invariant to positive per-infoset factors). Default
    --target positive: R / max legal positive regret, clipped to [-1, 1] (no positive regret: R / max|R|).
    --target maxabs: R / max legal |R| (the first version; a free FOLD's large negative regret then
    sets the scale and compresses the regrets that decide the policy). Average head = cross-entropy
    to the teacher average strategy (reported also as KL = CE - teacher entropy). Student current = regret matching on the regret head, student
    average = legal softmax of the average head.
Metric here: total variation and best-action agreement with the teacher on validation states. The
    game-level metric (Policy-LBR ante/hand) is measured by evaluate.cpp, not here.
Limit: imitation of one abstraction teacher on a teacher-generated state distribution; no CFR updates.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "PROJECT_LAYOUT.json").is_file())
TORCH_DEPS = PROJECT.parent / "Toy-Card-Game-Agent/.deep_cfr_deps"
if TORCH_DEPS.is_dir():
    sys.path.insert(0, str(TORCH_DEPS))

import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TOKEN_FIELDS = 8
MAX_TOKENS = 48
MAX_EVENTS = 32
NUMERICS = 20
FLAT_SIZE = 2 + 12 * 18 + MAX_EVENTS * 14 + NUMERICS
KEY_WIDTHS = (3, 512, 16, 3, 4, 4, 4, 2, 6, 3)
KEY_FLAT_SIZE = sum(KEY_WIDTHS) + 8
VOCAB = (8, 14, 5, 32, 4, 3, 9, 3)
FIELD_NAMES = ("type", "rank", "suit", "slot", "street", "actor", "action", "misc")
RECORD = np.dtype(
    [
        ("hand", "<u4"),
        ("viewer", "u1"),
        ("street", "u1"),
        ("found", "u1"),
        ("mask", "u1"),
        ("ntok", "u1"),
        ("behavior", "u1"),
        ("epsilon", "u1"),
        ("reserved", "u1"),
        ("tokens", "i1", (MAX_TOKENS, TOKEN_FIELDS)),
        ("numerics", "<f4", (NUMERICS,)),
        ("regret", "<f4", (8,)),
        ("current", "<f4", (8,)),
        ("average", "<f4", (8,)),
        ("scale", "<f4"),
    ]
)
assert RECORD.itemsize == 576


def load_dataset(path: Path) -> np.ndarray:
    """Read-only memory map; batches copy only the rows they use."""
    with open(path, "rb") as handle:
        header = handle.read(16)
    if header[:8] != b"NDDATA1\0":
        raise ValueError("Not a neural_distill dataset")
    (count,) = struct.unpack("<Q", header[8:16])
    if Path(path).stat().st_size != 16 + count * RECORD.itemsize:
        raise ValueError("Record count does not match the file size")
    return np.memmap(path, dtype=RECORD, mode="r", offset=16, shape=(count,))


def load_keys(path: Path, count: int) -> np.ndarray:
    """Teacher abstraction keys written by export.cpp --keys-only (16 bytes per record, same order)."""
    with open(path, "rb") as handle:
        header = handle.read(16)
    if header[:8] != b"NDKEYS1\0" or struct.unpack("<Q", header[8:16])[0] != count:
        raise ValueError("keys sidecar does not match the dataset")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=16, shape=(count, 16))


def key_flat(keys: np.ndarray) -> np.ndarray:
    """Mirror of tokens.hpp::key_flat: one-hot of street, power cluster, memory bucket and betting summary."""
    keys = np.asarray(keys).astype(np.int64)
    values = [keys[:, 0], keys[:, 1] | (keys[:, 2] << 8), *[keys[:, i] for i in range(3, 11)]]
    x = np.zeros((len(keys), KEY_FLAT_SIZE), np.float32)
    rows = np.arange(len(keys))
    base = 0
    for value, width in zip(values, KEY_WIDTHS):
        if (value < 0).any() or (value >= width).any():
            raise ValueError("key feature out of range")
        x[rows, base + value] = 1.0
        base += width
    x[:, base:] = ((keys[:, 11:12] >> np.arange(8)) & 1).astype(np.float32)
    return x


KEYS = None


def masks_from(data: np.ndarray) -> np.ndarray:
    return ((data["mask"][:, None] >> np.arange(8)) & 1).astype(bool)


def flat_from_tokens(tokens: np.ndarray, numerics: np.ndarray) -> np.ndarray:
    """Vectorised mirror of tokens.hpp::flat_from_tokens."""
    n = len(tokens)
    x = np.zeros((n, FLAT_SIZE), np.float32)
    rows = np.arange(n)[:, None]
    kind = tokens[:, :, 0].astype(np.int64)
    rank = tokens[:, :, 1].astype(np.int64)
    suit = tokens[:, :, 2].astype(np.int64)
    slot = tokens[:, :, 3].astype(np.int64)
    street = tokens[:, :, 4].astype(np.int64)
    actor = tokens[:, :, 5].astype(np.int64)
    action = tokens[:, :, 6].astype(np.int64)
    misc = tokens[:, :, 7].astype(np.int64)
    r, c = np.nonzero(kind == 1)
    x[r, misc[r, c] - 1] = 1.0
    first = np.array([0, 0, 0, 3, 7, 11, 0, 0])
    card_base, event_base = 2, 2 + 12 * 18
    r, c = np.nonzero((kind >= 2) & (kind <= 5))
    base = card_base + (first[kind[r, c]] + slot[r, c]) * 18
    x[r, base + rank[r, c] - 1] = 1.0
    x[r, base + 13 + suit[r, c] - 1] = 1.0
    x[r, base + 17] = 1.0
    r, c = np.nonzero(kind == 6)
    base = event_base + slot[r, c] * 14
    x[r, base + street[r, c] - 1] = 1.0
    x[r, base + 3 + actor[r, c] - 1] = 1.0
    x[r, base + 5 + action[r, c] - 1] = 1.0
    x[r, base + 13] = 1.0
    x[:, event_base + MAX_EVENTS * 14 :] = numerics
    del rows
    return x


class Block(nn.Module):
    def __init__(self, width: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, ffn)
        self.ff2 = nn.Linear(ffn, width)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        dh = width // self.heads
        qkv = self.qkv(self.ln1(x)).view(batch, length, 3, self.heads, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
        scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.out(attended)
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class TransformerStudent(nn.Module):
    def __init__(self, width: int = 64, layers: int = 2, heads: int = 4, ffn: int = 256):
        super().__init__()
        self.config = dict(architecture="transformer", width=width, layers=layers, heads=heads, ffn=ffn, hidden=0)
        self.embedding = nn.ModuleList(nn.Embedding(size, width) for size in VOCAB)  # one table per token field
        self.query = nn.Linear(NUMERICS, width)
        self.blocks = nn.ModuleList(Block(width, heads, ffn) for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        self.regret = nn.Linear(width, 8)
        self.average = nn.Linear(width, 8)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, numerics: torch.Tensor, flat: torch.Tensor | None = None):
        x = sum(table(tokens[:, :, i]) for i, table in enumerate(self.embedding))
        last = lengths - 1
        rows = torch.arange(len(tokens))
        x = x.clone()
        x[rows, last] = x[rows, last] + self.query(numerics)
        pad = torch.arange(tokens.shape[1])[None, :] >= lengths[:, None]
        for block in self.blocks:
            x = block(x, pad)
        h = self.final_norm(x[rows, last])
        return self.regret(h), self.average(h)


class MlpStudent(nn.Module):
    def __init__(self, hidden: int = 128, with_key: bool = False):
        super().__init__()
        name = "mlp_key" if with_key else "mlp"
        self.config = dict(architecture=name, width=0, layers=2, heads=0, ffn=0, hidden=hidden)
        self.fc1 = nn.Linear(FLAT_SIZE + (KEY_FLAT_SIZE if with_key else 0), hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.regret = nn.Linear(hidden, 8)
        self.average = nn.Linear(hidden, 8)

    def forward(self, tokens, lengths, numerics, flat: torch.Tensor):
        h = F.gelu(self.fc2(F.gelu(self.fc1(flat))))
        return self.regret(h), self.average(h)


def regret_matching(regret: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    positive = torch.where(legal, regret.clamp(min=0), torch.zeros_like(regret))
    total = positive.sum(1, keepdim=True)
    uniform = legal.float() / legal.float().sum(1, keepdim=True)
    return torch.where(total > 0, positive / total.clamp(min=1e-30), uniform)


def legal_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal, float("-inf")).softmax(1)


def losses(model, batch):
    regret, logits = model(batch["tokens"], batch["lengths"], batch["numerics"], batch.get("flat"))
    legal = batch["legal"]
    squared = torch.where(legal, (regret - batch["regret"]) ** 2, torch.zeros_like(regret))
    mse = (squared.sum(1) / legal.sum(1)).mean()
    log_p = logits.masked_fill(~legal, float("-inf")).log_softmax(1)
    ce = -(batch["average"] * torch.where(legal, log_p, torch.zeros_like(log_p))).sum(1).mean()
    return mse, ce, regret, logits


def regret_targets(rows: np.ndarray, kind: str) -> np.ndarray:
    """Positive per-infoset rescaling of the stored teacher regrets (stored = R / max legal |R|, scale = max|R|)."""
    stored = rows["regret"].astype(np.float64)
    if kind == "maxabs":
        return stored.astype(np.float32)
    legal = masks_from(rows)
    regret = stored * rows["scale"].astype(np.float64)[:, None]
    positive = np.where(legal, np.maximum(regret, 0.0), 0.0).max(1)
    target = np.where(positive[:, None] > 0, np.clip(regret / np.maximum(positive, 1e-300)[:, None], -1.0, 1.0), stored)
    return np.where(legal, target, 0.0).astype(np.float32)


TARGET_KIND = "positive"


def make_batch(data: np.ndarray, index: np.ndarray, architecture: str) -> dict:
    rows = data[index]
    width = int(rows["ntok"].max())
    batch = {
        "tokens": torch.from_numpy(rows["tokens"][:, :width].astype(np.int64)),
        "lengths": torch.from_numpy(rows["ntok"].astype(np.int64)),
        "numerics": torch.from_numpy(rows["numerics"].copy()),
        "legal": torch.from_numpy(masks_from(rows)),
        "regret": torch.from_numpy(regret_targets(rows, TARGET_KIND)),
        "average": torch.from_numpy(rows["average"].copy()),
        "current": torch.from_numpy(rows["current"].copy()),
    }
    if architecture in ("mlp", "mlp_key"):
        flat = flat_from_tokens(rows["tokens"], rows["numerics"])
        if architecture == "mlp_key":
            flat = np.concatenate([flat, key_flat(KEYS[index])], axis=1)
        batch["flat"] = torch.from_numpy(flat)
    return batch


@torch.no_grad()
def evaluate(model, data: np.ndarray, index: np.ndarray, architecture: str) -> dict:
    model.eval()
    totals = dict(mse=0.0, ce=0.0, kl_average=0.0, tv_current=0.0, tv_average=0.0, agree_current=0.0, agree_average=0.0)
    by_street = {}
    count = 0
    for start in range(0, len(index), 4096):
        part = index[start : start + 4096]
        batch = make_batch(data, part, architecture)
        mse, ce, regret, logits = losses(model, batch)
        legal = batch["legal"]
        current = regret_matching(regret, legal)
        average = legal_softmax(logits, legal)
        tv_c = 0.5 * (current - batch["current"]).abs().sum(1)
        tv_a = 0.5 * (average - batch["average"]).abs().sum(1)
        neg = torch.full_like(current, -1.0)
        agree_c = torch.where(legal, current, neg).argmax(1) == torch.where(legal, batch["current"], neg).argmax(1)
        agree_a = torch.where(legal, average, neg).argmax(1) == torch.where(legal, batch["average"], neg).argmax(1)
        n = len(part)
        totals["mse"] += mse.item() * n
        totals["ce"] += ce.item() * n
        teacher = batch["average"].double()
        entropy = -(teacher * torch.where(teacher > 0, teacher.log(), torch.zeros_like(teacher))).sum(1)
        totals["kl_average"] += (ce.item() * n - entropy.sum().item())
        totals["tv_current"] += tv_c.sum().item()
        totals["tv_average"] += tv_a.sum().item()
        totals["agree_current"] += agree_c.float().sum().item()
        totals["agree_average"] += agree_a.float().sum().item()
        streets = data[part]["street"]
        for street in np.unique(streets):
            sel = torch.from_numpy(streets == street)
            entry = by_street.setdefault(int(street), [0, 0.0, 0.0])
            entry[0] += int(sel.sum())
            entry[1] += tv_c[sel].sum().item()
            entry[2] += tv_a[sel].sum().item()
        count += n
    model.train()
    result = {key: value / count for key, value in totals.items()}
    result["by_street"] = {
        str(s): dict(states=c, tv_current=a / c, tv_average=b / c) for s, (c, a, b) in sorted(by_street.items())
    }
    result["states"] = count
    return result


def export_weights(model, path: Path) -> None:
    config = model.config
    tensors = [(name, value.detach().float().contiguous().numpy()) for name, value in model.state_dict().items()]
    with path.open("wb") as out:
        out.write(b"NDNET1\0\0")
        name = config["architecture"].encode()
        out.write(struct.pack("<I", len(name)) + name)
        for key in ("width", "layers", "heads", "ffn", "hidden"):
            out.write(struct.pack("<I", int(config[key])))
        out.write(struct.pack("<I", len(tensors)))
        for tensor_name, array in tensors:
            encoded = tensor_name.encode()
            out.write(struct.pack("<I", len(encoded)) + encoded)
            out.write(struct.pack("<I", array.ndim))
            for dim in array.shape:
                out.write(struct.pack("<I", dim))
            out.write(array.astype("<f4").tobytes())


@torch.no_grad()
def write_parity_reference(model, data: np.ndarray, index: np.ndarray, architecture: str, path: Path) -> None:
    model.eval()
    batch = make_batch(data, index, architecture)
    regret, logits = model(batch["tokens"], batch["lengths"], batch["numerics"], batch.get("flat"))
    with path.open("wb") as out:
        out.write(struct.pack("<I", len(index)))
        for i, row in enumerate(index):
            out.write(struct.pack("<Q", int(row)))
            out.write(regret[i].numpy().astype("<f4").tobytes())
            out.write(logits[i].numpy().astype("<f4").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--architecture", choices=("transformer", "mlp", "mlp_key"), required=True)
    parser.add_argument("--keys", type=Path, help="mlp_key only: teacher key sidecar from export.cpp --keys-only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=30000, help="optimizer steps")
    parser.add_argument("--batch", type=int, default=512, help="decision states per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128, help="MLP hidden width")
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--target", choices=("positive", "maxabs"), default="positive", help="regret-head target scaling")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    global TARGET_KIND
    TARGET_KIND = args.target
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_dataset(args.dataset)
    global KEYS
    if args.architecture == "mlp_key":
        if args.keys is None:
            parser.error("--keys is required for mlp_key")
        KEYS = load_keys(args.keys, len(data))
    validation = data["hand"] % 20 == 0
    train_index = np.flatnonzero(~validation)
    val_index = np.flatnonzero(validation)
    monitor = np.sort(np.random.default_rng(7).choice(val_index, min(50000, len(val_index)), replace=False))
    model = (
        TransformerStudent(args.width, args.layers, args.heads, args.ffn)
        if args.architecture == "transformer"
        else MlpStudent(args.hidden, with_key=args.architecture == "mlp_key")
    )
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
    )
    print(json.dumps(dict(event="start", parameters=parameters, train=len(train_index), validation=len(val_index))), flush=True)
    history, started = [], time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = make_batch(data, np.sort(rng.choice(train_index, args.batch, replace=False)), args.architecture)
        mse, ce, _, _ = losses(model, batch)
        loss = mse + ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, data, monitor, args.architecture)
            metrics.update(step=step, train_mse=mse.item(), train_ce=ce.item(), seconds=time.perf_counter() - started)
            history.append(metrics)
            print(json.dumps({k: v for k, v in metrics.items() if k != "by_street"}), flush=True)
    train_seconds = time.perf_counter() - started
    final = evaluate(model, data, val_index, args.architecture)
    export_weights(model, args.output / "weights.ndnet")
    torch.save(dict(config=model.config, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}),
               args.output / "checkpoint.pt")
    parity = val_index[:2000]
    write_parity_reference(model, data, parity, args.architecture, args.output / "parity_reference.bin")
    summary = dict(
        architecture=args.architecture,
        parameters=parameters,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        train_states=int(len(train_index)),
        validation_states=int(len(val_index)),
        train_seconds=train_seconds,
        final_validation=final,
        history=history,
        source_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        objective=f"regret head per-state MSE to '{args.target}'-scaled teacher regrets + average head CE",
    )
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(event="done", **{k: final[k] for k in ("tv_current", "tv_average", "agree_current", "agree_average")})), flush=True)


if __name__ == "__main__":
    main()
