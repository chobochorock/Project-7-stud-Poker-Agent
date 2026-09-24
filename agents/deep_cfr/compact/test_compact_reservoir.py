"""Minimal lossless round-trip check for compact Deep CFR storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compact_reservoir import (
    ACTIONS,
    CARD_OFFSET,
    CARD_SLOTS,
    CARD_TOKENS,
    CONTINUOUS_OFFSET,
    HISTORY_OFFSET,
    HISTORY_SLOTS,
    HISTORY_TOKENS,
    LEGAL_OFFSET,
    CompactReservoir,
)

import sys

sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "PROJECT_LAYOUT.json").is_file()
        )
    ),
)
from train_deep_cfr_7th import DIMENSIONS, RECORD_DTYPE


def records(count: int) -> np.ndarray:
    rng = np.random.default_rng(17)
    result = np.zeros(count, dtype=RECORD_DTYPE)
    result["iteration"] = rng.integers(1, 100, size=count)
    rows = np.arange(count)
    result["state"][rows, rng.integers(0, 2, size=count)] = 1
    result["state"][rows, 2 + rng.integers(0, 3, size=count)] = 1
    for slot in range(CARD_SLOTS):
        result["state"][
            rows,
            CARD_OFFSET + slot * CARD_TOKENS + rng.integers(0, CARD_TOKENS, size=count),
        ] = 1
    for slot in range(HISTORY_SLOTS):
        result["state"][
            rows,
            HISTORY_OFFSET
            + slot * HISTORY_TOKENS
            + rng.integers(0, HISTORY_TOKENS, size=count),
        ] = 1
    result["state"][:, CONTINUOUS_OFFSET:LEGAL_OFFSET] = rng.random(
        (count, LEGAL_OFFSET - CONTINUOUS_OFFSET), dtype=np.float32
    )
    legal = rng.integers(0, 2, size=(count, ACTIONS), dtype=np.uint8)
    legal[:, 0] = 1
    result["state"][:, LEGAL_OFFSET:] = legal
    result["target"] = rng.normal(size=(count, ACTIONS)).astype(np.float32)
    return result


def main() -> None:
    source = records(64)
    reservoir = CompactReservoir(64, 5)
    reservoir.add(source, np.random.default_rng(19))
    restored_dense = reservoir.dense(np.arange(64))
    assert restored_dense.shape == (64, DIMENSIONS)
    assert np.array_equal(restored_dense, source["state"])
    assert np.array_equal(reservoir.targets, source["target"])

    path = Path(__file__).with_name(".test_compact_reservoir.npz")
    try:
        np.savez(path, **reservoir.payload("test"))
        restored = CompactReservoir(64, 5)
        with np.load(path) as archive:
            restored.restore(archive, "test")
        assert np.array_equal(restored.dense(np.arange(64)), source["state"])
        assert np.array_equal(restored.targets, source["target"])
    finally:
        path.unlink(missing_ok=True)

    reservoir.add(records(256), np.random.default_rng(23))
    assert reservoir.size == 64
    assert reservoir.seen == 320
    sampled = reservoir.dense(np.arange(64))
    assert np.all(sampled[:, :2].sum(axis=1) == 1)
    assert np.all(sampled[:, 2:5].sum(axis=1) == 1)
    assert np.all(
        sampled[:, CARD_OFFSET:HISTORY_OFFSET]
        .reshape(64, CARD_SLOTS, CARD_TOKENS)
        .sum(axis=2)
        == 1
    )
    assert np.all(
        sampled[:, HISTORY_OFFSET:CONTINUOUS_OFFSET]
        .reshape(64, HISTORY_SLOTS, HISTORY_TOKENS)
        .sum(axis=2)
        == 1
    )

    dense_bytes = (DIMENSIONS + ACTIONS + 1) * 4 + 1
    print(
        {
            "self_test": "ok",
            "dense_bytes_per_slot": dense_bytes,
            "compact_bytes_per_slot": reservoir.bytes_per_slot,
            "compression_ratio": dense_bytes / reservoir.bytes_per_slot,
        }
    )


if __name__ == "__main__":
    main()
