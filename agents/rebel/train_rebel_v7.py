"""Train a scalar 7th-street ReBeL infostate-value model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import struct
import sys
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)
TOY_ROOT = ROOT.parent / "Toy-Card-Game-Agent"
for dependency in (TOY_ROOT / ".deep_cfr_deps", TOY_ROOT / ".open_spiel"):
    if dependency.exists():
        sys.path.insert(0, str(dependency))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HEADER = struct.Struct("<8sIIQ")
MAGIC_BY_VERSION = {2: b"RBV7S2\0\0", 3: b"RBV7S3\0\0"}


class ValueDataset(Dataset):
    def __init__(
        self,
        records: np.memmap,
        dimensions: int,
        indices: np.ndarray,
        target_mean: float,
        target_scale: float,
    ) -> None:
        self.records = records
        self.dimensions = dimensions
        self.indices = indices
        self.target_mean = target_mean
        self.target_scale = target_scale

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records[int(self.indices[index])]
        features = torch.from_numpy(np.array(row[1 : 1 + self.dimensions], copy=True))
        target = torch.tensor(
            [(float(row[1 + self.dimensions]) - self.target_mean) / self.target_scale],
            dtype=torch.float32,
        )
        return features, target


class ValueNet(nn.Module):
    def __init__(self, dimensions: int, hidden: int, layers: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        width = dimensions
        for _ in range(layers):
            modules.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        modules.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DenormalizedValue(nn.Module):
    def __init__(self, model: nn.Module, mean: float, scale: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs) * self.target_scale + self.target_mean


def open_dataset(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as source:
        header = source.read(HEADER.size)
    if len(header) != HEADER.size:
        raise RuntimeError("truncated V7 dataset header")
    magic, version, dimensions, count = HEADER.unpack(header)
    if magic != MAGIC_BY_VERSION.get(version) or not dimensions or not count:
        raise RuntimeError("invalid V7 dataset header")
    expected = HEADER.size + count * (dimensions + 2) * 4
    if path.stat().st_size != expected:
        raise RuntimeError(
            f"V7 dataset size mismatch: {path.stat().st_size} != {expected}"
        )
    records = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=HEADER.size,
        shape=(count, dimensions + 2),
    )
    return records, dimensions, count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mean: float,
    scale: float,
    device: torch.device,
) -> tuple[float, float]:
    absolute = squared = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for features, normalized in loader:
            features = features.to(device)
            targets = normalized.to(device) * scale + mean
            predictions = model(features) * scale + mean
            difference = predictions - targets
            absolute += difference.abs().sum().item()
            squared += difference.square().sum().item()
            count += difference.numel()
    return absolute / count, (squared / count) ** 0.5


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch has no CUDA support")

    records, dimensions, count = open_dataset(args.input)
    targets = records[:, 1 + dimensions]
    target_mean = float(np.mean(targets, dtype=np.float64))
    target_scale = max(float(np.std(targets, dtype=np.float64)), 1.0)
    groups = records[:, 0].astype(np.int64)
    unique_groups = np.unique(groups)
    np.random.shuffle(unique_groups)
    validation_groups = max(1, int(unique_groups.size * args.validation_fraction))
    validation_mask = np.isin(groups, unique_groups[:validation_groups])
    validation_indices = np.flatnonzero(validation_mask)
    training_indices = np.flatnonzero(~validation_mask)
    if not training_indices.size:
        raise RuntimeError("not enough V7 samples for a training split")

    training = ValueDataset(
        records, dimensions, training_indices, target_mean, target_scale
    )
    validation = ValueDataset(
        records, dimensions, validation_indices, target_mean, target_scale
    )
    training_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = ValueNet(dimensions, args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.SmoothL1Loss()
    best_epoch = 0
    best_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, targets_batch in training_loader:
            features = features.to(device)
            targets_batch = targets_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        mae, rmse = evaluate(
            model, validation_loader, target_mean, target_scale, device
        )
        if mae < best_mae:
            best_epoch = epoch
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": total_loss / max(1, batches),
                    "validation_mae_ante": mae,
                    "validation_rmse_ante": rmse,
                }
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("V7 training produced no checkpoint")
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported = DenormalizedValue(
        model.to("cpu").eval(), target_mean, target_scale
    ).eval()
    torch.jit.script(exported).save(str(args.output))
    baseline_mae = float(np.mean(np.abs(targets - target_mean), dtype=np.float64))
    metadata = {
        "trainer": "rebel-v7-value",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "samples": int(count),
        "dimensions": int(dimensions),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_mean_ante": target_mean,
        "target_scale_ante": target_scale,
        "constant_baseline_mae_ante": baseline_mae,
        "best_epoch": best_epoch,
        "best_validation_mae_ante": best_mae,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=73002)
    args = parser.parse_args()
    if (
        min(
            args.hidden,
            args.layers,
            args.epochs,
            args.batch_size,
            args.threads,
        )
        <= 0
    ):
        parser.error("model and training sizes must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    train(args)


if __name__ == "__main__":
    main()
