"""Lossless token storage for the 1,832-wide Deep CFR tensor."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


DIMENSIONS = 1832
ACTIONS = 8
CARD_SLOTS = 12
CARD_TOKENS = 53
HISTORY_SLOTS = 24
HISTORY_TOKENS = 49
CARD_OFFSET = 5
HISTORY_OFFSET = CARD_OFFSET + CARD_SLOTS * CARD_TOKENS
CONTINUOUS_OFFSET = HISTORY_OFFSET + HISTORY_SLOTS * HISTORY_TOKENS
CONTINUOUS_FEATURES = 7
LEGAL_OFFSET = CONTINUOUS_OFFSET + CONTINUOUS_FEATURES
FORMAT_MAGIC = 0x434F4D50  # "COMP"
FORMAT_VERSION = 1
SAMPLE_HEADER = struct.Struct("<8sIIIQ")
COMPACT_RECORD_DTYPE = np.dtype(
    [
        ("iteration", "<f4"),
        ("player", "u1"),
        ("kind", "u1"),
        ("legal_mask", "u1"),
        ("reserved", "u1"),
        ("street", "u1"),
        ("cards", "u1", (CARD_SLOTS,)),
        ("history", "u1", (HISTORY_SLOTS,)),
        ("continuous", "<f4", (CONTINUOUS_FEATURES,)),
        ("target", "<f4", (ACTIONS,)),
    ]
)


def read_compact_records(path: Path) -> np.memmap:
    with path.open("rb") as source:
        magic, version, dimensions, actions, count = SAMPLE_HEADER.unpack(
            source.read(SAMPLE_HEADER.size)
        )
    if not magic.startswith(b"DCFRC1") or version != 1:
        raise RuntimeError(f"invalid compact sample file: {path}")
    if dimensions != DIMENSIONS or actions != ACTIONS:
        raise RuntimeError("compact sample tensor schema mismatch")
    expected = SAMPLE_HEADER.size + count * COMPACT_RECORD_DTYPE.itemsize
    if path.stat().st_size != expected:
        raise RuntimeError("truncated compact Deep CFR sample file")
    return np.memmap(
        path,
        mode="r",
        offset=SAMPLE_HEADER.size,
        dtype=COMPACT_RECORD_DTYPE,
        shape=(count,),
    )


class CompactReservoir:
    """Uniform reservoir whose categorical features are stored as exact tokens."""

    def __init__(self, capacity: int, start_street: int):
        del start_street
        self.capacity = capacity
        self.viewers = np.empty(capacity, dtype=np.uint8)
        self.streets = np.empty(capacity, dtype=np.uint8)
        self.cards = np.empty((capacity, CARD_SLOTS), dtype=np.uint8)
        self.history = np.empty((capacity, HISTORY_SLOTS), dtype=np.uint8)
        self.continuous = np.empty((capacity, CONTINUOUS_FEATURES), dtype=np.float32)
        self.legal_masks = np.empty(capacity, dtype=np.uint8)
        self.targets = np.empty((capacity, ACTIONS), dtype=np.float32)
        self.iterations = np.empty(capacity, dtype=np.float32)
        self._size = 0
        self._seen = 0
        self.seen_by_street = [0, 0, 0]

    @property
    def size(self) -> int:
        return self._size

    @property
    def seen(self) -> int:
        return self._seen

    @property
    def sizes(self) -> list[int]:
        return np.bincount(self.streets[: self._size], minlength=3).astype(int).tolist()

    @property
    def allocated_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.viewers,
                self.streets,
                self.cards,
                self.history,
                self.continuous,
                self.legal_masks,
                self.targets,
                self.iterations,
            )
        )

    @property
    def bytes_per_slot(self) -> int:
        return self.allocated_bytes // self.capacity

    def _store(
        self,
        slots: np.ndarray,
        states: np.ndarray,
        targets: np.ndarray,
        iterations: np.ndarray,
    ) -> None:
        if not len(slots):
            return
        self.viewers[slots] = np.argmax(states[:, :2], axis=1)
        self.streets[slots] = np.argmax(states[:, 2:5], axis=1)
        self.cards[slots] = np.argmax(
            states[:, CARD_OFFSET:HISTORY_OFFSET].reshape(-1, CARD_SLOTS, CARD_TOKENS),
            axis=2,
        )
        self.history[slots] = np.argmax(
            states[:, HISTORY_OFFSET:CONTINUOUS_OFFSET].reshape(
                -1, HISTORY_SLOTS, HISTORY_TOKENS
            ),
            axis=2,
        )
        self.continuous[slots] = states[:, CONTINUOUS_OFFSET:LEGAL_OFFSET]
        legal = states[:, LEGAL_OFFSET:] > 0.5
        self.legal_masks[slots] = np.sum(
            legal.astype(np.uint8) << np.arange(ACTIONS, dtype=np.uint8),
            axis=1,
            dtype=np.uint8,
        )
        self.targets[slots] = targets
        self.iterations[slots] = iterations

    def add(self, records: np.ndarray, rng: np.random.Generator) -> None:
        if "state" not in (records.dtype.names or ()):
            self._add_compact(records, rng)
            return
        count = len(records)
        if not count:
            return
        states = records["state"]
        streets = np.argmax(states[:, 2:5], axis=1)
        counts = np.bincount(streets, minlength=3)
        self.seen_by_street = [
            old + int(added) for old, added in zip(self.seen_by_street, counts)
        ]

        fill = min(count, self.capacity - self._size)
        if fill:
            slots = np.arange(self._size, self._size + fill)
            self._store(
                slots,
                states[:fill],
                records["target"][:fill],
                records["iteration"][:fill],
            )
            self._size += fill

        remaining = count - fill
        if remaining:
            highs = np.arange(
                self._seen + fill + 1,
                self._seen + count + 1,
                dtype=np.int64,
            )
            chosen_slots = rng.integers(0, highs)
            selected = np.flatnonzero(chosen_slots < self.capacity)
            if len(selected):
                selected_slots = chosen_slots[selected]
                reversed_slots = selected_slots[::-1]
                _, last_from_end = np.unique(reversed_slots, return_index=True)
                keep = len(selected) - 1 - last_from_end
                source = fill + selected[keep]
                self._store(
                    selected_slots[keep],
                    states[source],
                    records["target"][source],
                    records["iteration"][source],
                )

        self._seen += count

    def _store_compact(self, slots: np.ndarray, records: np.ndarray) -> None:
        if not len(slots):
            return
        self.viewers[slots] = records["player"]
        self.streets[slots] = records["street"]
        self.cards[slots] = records["cards"]
        self.history[slots] = records["history"]
        self.continuous[slots] = records["continuous"]
        self.legal_masks[slots] = records["legal_mask"]
        self.targets[slots] = records["target"]
        self.iterations[slots] = records["iteration"]

    def _add_compact(self, records: np.ndarray, rng: np.random.Generator) -> None:
        count = len(records)
        if not count:
            return
        counts = np.bincount(records["street"], minlength=3)
        self.seen_by_street = [
            old + int(added) for old, added in zip(self.seen_by_street, counts)
        ]

        fill = min(count, self.capacity - self._size)
        if fill:
            slots = np.arange(self._size, self._size + fill)
            self._store_compact(slots, records[:fill])
            self._size += fill

        remaining = count - fill
        if remaining:
            highs = np.arange(
                self._seen + fill + 1,
                self._seen + count + 1,
                dtype=np.int64,
            )
            chosen_slots = rng.integers(0, highs)
            selected = np.flatnonzero(chosen_slots < self.capacity)
            if len(selected):
                selected_slots = chosen_slots[selected]
                _, last_from_end = np.unique(selected_slots[::-1], return_index=True)
                keep = len(selected) - 1 - last_from_end
                self._store_compact(
                    selected_slots[keep], records[fill + selected[keep]]
                )

        self._seen += count

    def dense(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        rows = np.arange(len(indices))
        result = np.zeros((len(indices), DIMENSIONS), dtype=np.float32)
        result[rows, self.viewers[indices]] = 1.0
        result[rows, 2 + self.streets[indices]] = 1.0
        for slot in range(CARD_SLOTS):
            result[
                rows,
                CARD_OFFSET
                + slot * CARD_TOKENS
                + self.cards[indices, slot].astype(np.int64),
            ] = 1.0
        for slot in range(HISTORY_SLOTS):
            result[
                rows,
                HISTORY_OFFSET
                + slot * HISTORY_TOKENS
                + self.history[indices, slot].astype(np.int64),
            ] = 1.0
        result[:, CONTINUOUS_OFFSET:LEGAL_OFFSET] = self.continuous[indices]
        result[:, LEGAL_OFFSET:] = (
            (self.legal_masks[indices, None] >> np.arange(ACTIONS)) & 1
        ).astype(np.float32)
        return result

    def batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.integers(0, self._size, size=min(batch_size, self._size))
        return self.dense(indices), self.targets[indices], self.iterations[indices]

    def payload(self, prefix: str) -> dict[str, np.ndarray]:
        size = self._size
        return {
            f"{prefix}_meta": np.array(
                [
                    FORMAT_MAGIC,
                    FORMAT_VERSION,
                    size,
                    self._seen,
                    *self.seen_by_street,
                ],
                dtype=np.int64,
            ),
            f"{prefix}_viewers": self.viewers[:size],
            f"{prefix}_streets": self.streets[:size],
            f"{prefix}_cards": self.cards[:size],
            f"{prefix}_history": self.history[:size],
            f"{prefix}_continuous": self.continuous[:size],
            f"{prefix}_legal_masks": self.legal_masks[:size],
            f"{prefix}_targets": self.targets[:size],
            f"{prefix}_iterations": self.iterations[:size],
        }

    def restore(self, archive: np.lib.npyio.NpzFile, prefix: str) -> None:
        meta = archive[f"{prefix}_meta"]
        if (
            len(meta) != 7
            or int(meta[0]) != FORMAT_MAGIC
            or int(meta[1]) != FORMAT_VERSION
        ):
            raise RuntimeError("checkpoint is not compact reservoir v1")
        self._size, self._seen = map(int, meta[2:4])
        self.seen_by_street = list(map(int, meta[4:]))
        if self._size > self.capacity:
            raise RuntimeError("saved reservoir exceeds configured capacity")
        size = self._size
        for name in (
            "viewers",
            "streets",
            "cards",
            "history",
            "continuous",
            "legal_masks",
            "targets",
            "iterations",
        ):
            getattr(self, name)[:size] = archive[f"{prefix}_{name}"]


def compact_fit_metrics(
    model,
    reservoir: CompactReservoir,
    device,
    target_scale: float = 1.0,
) -> list[dict[str, float | int]]:
    import torch

    with torch.no_grad():
        return _compact_fit_metrics(model, reservoir, device, target_scale, torch)


def _compact_fit_metrics(
    model, reservoir: CompactReservoir, device, target_scale: float, torch
) -> list[dict[str, float | int]]:
    model.to(device).eval()
    result = []
    for street in range(3):
        indices = np.flatnonzero(reservoir.streets[: reservoir.size] == street)
        size = len(indices)
        squared_error = legal_count = agreement = 0.0
        for begin in range(0, size, 1024):
            batch = indices[begin : begin + 1024]
            states = torch.from_numpy(reservoir.dense(batch)).to(device)
            targets = torch.from_numpy(reservoir.targets[batch]).to(device)
            predictions = model(states) * target_scale
            legal = states[:, -ACTIONS:] > 0
            squared_error += torch.square(predictions - targets)[legal].sum().item()
            legal_count += legal.sum().item()
            floor = torch.full_like(predictions, -torch.inf)
            agreement += (
                (
                    torch.where(legal, predictions, floor).argmax(1)
                    == torch.where(legal, targets, floor).argmax(1)
                )
                .sum()
                .item()
            )
        result.append(
            {
                "street": street + 5,
                "samples": size,
                "legal_mse": squared_error / max(1, legal_count),
                "best_action_agreement": agreement / max(1, size),
            }
        )
    return result
